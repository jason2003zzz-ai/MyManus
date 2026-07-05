import argparse
import asyncio

from app.agent.manus import Manus
from app.logger import logger


async def run_cli(prompt: str | None = None):
    """Run MyManus once from the command line."""
    agent = None
    try:
        user_prompt = prompt if prompt else input("Enter your prompt: ")
        if not user_prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        agent = await Manus.create()
        await agent.run(user_prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        if agent is not None:
            await agent.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Run MyManus agent with a prompt")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["web"],
        help="Start the MyManus web interface",
    )
    parser.add_argument(
        "--prompt", type=str, required=False, help="Input prompt for the agent"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the web interface",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7788,
        help="Port for the web interface",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for the web interface",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "web":
        from app.web.server import run_web

        run_web(host=args.host, port=args.port, reload=args.reload)
        return

    asyncio.run(run_cli(args.prompt))


if __name__ == "__main__":
    main()
