import asyncio
import json
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.agent.task_control import TaskController
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    active_tool_names: set[str] = Field(default_factory=set, exclude=True)
    _current_base64_image: Optional[str] = None
    _current_image_mime_type: Optional[str] = None
    finish_status: Optional[str] = None
    finish_reason: Optional[str] = None
    final_answer: Optional[str] = None
    task_controller: TaskController = Field(default_factory=TaskController)
    max_context_chars: int = 120000
    keep_recent_messages: int = 24
    _termination_rejection: Optional[str] = None

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    def get_tool_params_for_step(self) -> List[dict]:
        """Return the tool schemas exposed to the LLM for the current step."""
        return self.available_tools.to_params()

    @staticmethod
    def _message_text_size(message: Message) -> int:
        return len(message.content or "") + len(message.base64_image or "")

    @staticmethod
    def _compact_text(value: str, limit: int) -> str:
        text = value or ""
        if len(text) <= limit:
            return text
        head = int(limit * 0.6)
        tail = max(0, limit - head)
        return (
            text[:head]
            + "\n\n...[observation compacted; middle omitted]...\n\n"
            + text[-tail:]
        )

    @staticmethod
    def _history_line(message: Message) -> str:
        role = getattr(message.role, "value", message.role)
        content = " ".join((message.content or "").split())
        if message.tool_calls:
            names = ", ".join(call.function.name for call in message.tool_calls)
            prefix = f"assistant tool calls: {names}"
        elif role == "tool":
            prefix = f"tool {message.name or 'unknown'}"
        else:
            prefix = str(role)
        return f"- {prefix}: {content[:700]}".rstrip()

    def _compact_memory_if_needed(self) -> None:
        messages = self.memory.messages
        if not messages:
            return

        # Old screenshots are expensive and cease to be actionable after the page
        # changes. Keep only the newest visual observations.
        for message in messages[:-8]:
            if message.base64_image:
                message.base64_image = None
                message.image_mime_type = None

        total_chars = sum(self._message_text_size(message) for message in messages)
        if total_chars <= self.max_context_chars:
            return

        first_user_index = next(
            (
                index
                for index, message in enumerate(messages)
                if getattr(message.role, "value", message.role) == "user"
            ),
            None,
        )
        pinned_indices = {
            index
            for index, message in enumerate(messages)
            if getattr(message.role, "value", message.role) == "system"
        }
        if first_user_index is not None:
            pinned_indices.add(first_user_index)

        tail_start = max(0, len(messages) - self.keep_recent_messages)
        while tail_start < len(messages) and getattr(
            messages[tail_start].role, "value", messages[tail_start].role
        ) == "tool":
            tail_start += 1

        old_messages = [
            message
            for index, message in enumerate(messages[:tail_start])
            if index not in pinned_indices
        ]
        history_lines = [self._history_line(message) for message in old_messages[-40:]]
        compacted_history = Message.user_message(
            "COMPACTED EXECUTION HISTORY (facts and tool outcomes only):\n"
            + ("\n".join(history_lines) if history_lines else "- No retained details.")
        )

        rebuilt = [
            messages[index]
            for index in sorted(pinned_indices)
            if index < tail_start
        ]
        rebuilt.append(compacted_history)
        rebuilt.extend(messages[tail_start:])
        self.memory.messages = rebuilt
        logger.info(
            "Compacted agent memory from {} to {} messages ({} input chars before compaction)",
            len(messages),
            len(rebuilt),
            total_chars,
        )

    def _step_prompt(self) -> Optional[Message]:
        parts = []
        progress = self.task_controller.progress_text()
        if progress:
            parts.append(progress)
        if self.task_controller.last_recovery_directive:
            parts.append(self.task_controller.last_recovery_directive)
            self.task_controller.last_recovery_directive = ""
        if self.next_step_prompt:
            parts.append(self.next_step_prompt)
        if not parts:
            return None
        return Message.user_message("\n\n".join(parts))

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        self._compact_memory_if_needed()
        request_messages = list(self.messages)
        step_prompt = self._step_prompt()
        if step_prompt:
            # The control prompt is transient. Persisting the same text on every
            # ReAct iteration caused quadratic context growth in long browser runs.
            request_messages.append(step_prompt)

        try:
            tool_params = self.get_tool_params_for_step()
            self.active_tool_names = {
                tool["function"]["name"]
                for tool in tool_params
                if isinstance(tool, dict) and "function" in tool
            }

            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=request_messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=tool_params,
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # Log response info
        logger.info(f"✨ {self.name}'s response: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                if content:
                    allowed, rejection = self.task_controller.validate_termination(
                        "success",
                        evidence_ids=[],
                        explicit=False,
                        final_answer=content,
                    )
                    if allowed:
                        self.finish_status = "success"
                        self.final_answer = content.strip()
                        self.state = AgentState.FINISHED
                    else:
                        self.task_controller.last_recovery_directive = (
                            "COMPLETION GATE: " + rejection
                        )
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            self._current_base64_image = None
            self._current_image_mime_type = None

            result = await self.execute_tool(command)
            result = self.task_controller.compact_observation(
                command.function.name, result
            )

            if self.max_observe:
                result = self._compact_text(result, int(self.max_observe))

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
                image_mime_type=self._current_image_mime_type,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

            if self.state == AgentState.FINISHED:
                logger.info("Terminate accepted; skipping remaining batched tool calls")
                break

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        if self.active_tool_names and name not in self.active_tool_names:
            return (
                f"Error: Tool '{name}' is currently gated off by the tool router. "
                "Use the exposed tools for this step."
            )

        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")

            circuit_breaker_message = self.task_controller.preflight_tool(name, args)
            if circuit_breaker_message:
                logger.warning(circuit_breaker_message)
                return f"Error: {circuit_breaker_message}"

            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # Handle special tools
            await self._handle_special_tool(
                name=name,
                result=result,
                tool_input=args,
            )

            if name.lower() == Terminate().name and self._termination_rejection:
                rejection = self._termination_rejection
                self._termination_rejection = None
                return f"Error: Termination gate rejected success. {rejection}"

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image
                self._current_image_mime_type = getattr(
                    result, "image_mime_type", None
                )

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            recovery_directive = self.task_controller.record_tool_result(
                name, args, observation
            )
            if self.task_controller.last_evidence_receipt:
                observation = (
                    f"{observation}\n\n"
                    f"{self.task_controller.last_evidence_receipt}"
                )
            if recovery_directive:
                observation = f"{observation}\n\n{recovery_directive}"

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if name.lower() == Terminate().name:
            tool_input = kwargs.get("tool_input") or {}
            if isinstance(tool_input, dict):
                status = tool_input.get("status")
                allowed, rejection = self.task_controller.validate_termination(
                    status,
                    evidence_ids=tool_input.get("evidence_ids"),
                    explicit=True,
                    reason=tool_input.get("reason", ""),
                    final_answer=tool_input.get("final_answer", ""),
                )
                if not allowed:
                    self._termination_rejection = rejection
                    logger.warning(f"Termination gate rejected success: {rejection}")
                    return

                self.finish_status = status
                self.finish_reason = tool_input.get("reason")
                final_answer = tool_input.get("final_answer")
                if isinstance(final_answer, str) and final_answer.strip():
                    self.final_answer = final_answer.strip()

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(
        self,
        request: Optional[str] = None,
        *,
        task_objective: Optional[str] = None,
    ) -> str:
        """Run the agent with cleanup when done."""
        objective = task_objective if task_objective is not None else request
        if not objective:
            objective = next(
                (
                    message.content
                    for message in self.memory.messages
                    if getattr(message.role, "value", message.role) == "user"
                    and message.content
                ),
                "",
            )
        guide = self.task_controller.initialize(objective or "")
        if guide:
            self.memory.add_message(Message.system_message(guide))
        try:
            return await super().run(request)
        finally:
            await self.cleanup()
