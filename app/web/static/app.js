const state = {
  currentRunId: null,
  parentRunId: null,
  conversation: [],
  forceNewConversation: false,
  socket: null,
  running: false,
  status: "idle",
  skills: [],
  attachments: [],
  selectedSkillIds: new Set(),
  editingSkillId: null,
};

const els = {
  workspace: document.querySelector(".workspace"),
  form: document.querySelector("#runForm"),
  prompt: document.querySelector("#promptInput"),
  promptPanel: document.querySelector(".prompt-panel"),
  continuePanel: document.querySelector("#continuePanel"),
  continueLabel: document.querySelector("#continueLabel"),
  newConversationButton: document.querySelector("#newConversationButton"),
  runPanel: document.querySelector(".run-panel"),
  runHeader: document.querySelector(".run-header"),
  runButton: document.querySelector("#runButton"),
  cancelButton: document.querySelector("#cancelButton"),
  clearButton: document.querySelector("#clearButton"),
  clearRunsButton: document.querySelector("#clearRunsButton"),
  openSkillsButton: document.querySelector("#openSkillsButton"),
  attachmentInput: document.querySelector("#attachmentInput"),
  attachmentsSummary: document.querySelector("#attachmentsSummary"),
  attachmentsList: document.querySelector("#attachmentsList"),
  skillsSummary: document.querySelector("#skillsSummary"),
  skillsSelectedLine: document.querySelector("#skillsSelectedLine"),
  skillsDialog: document.querySelector("#skillsDialog"),
  skillsDialogMeta: document.querySelector("#skillsDialogMeta"),
  closeSkillsDialogButton: document.querySelector("#closeSkillsDialogButton"),
  skillsDoneButton: document.querySelector("#skillsDoneButton"),
  skillsList: document.querySelector("#skillsList"),
  newSkillButton: document.querySelector("#newSkillButton"),
  skillEditor: document.querySelector("#skillEditor"),
  skillName: document.querySelector("#skillNameInput"),
  skillContent: document.querySelector("#skillContentInput"),
  saveSkillButton: document.querySelector("#saveSkillButton"),
  deleteSkillButton: document.querySelector("#deleteSkillButton"),
  cancelSkillButton: document.querySelector("#cancelSkillButton"),
  stream: document.querySelector("#eventStream"),
  result: document.querySelector("#resultBox"),
  badge: document.querySelector("#connectionBadge"),
  modelLine: document.querySelector("#modelLine"),
  runMeta: document.querySelector("#runMeta"),
  runsList: document.querySelector("#runsList"),
  columnResizers: document.querySelectorAll(".col-resizer"),
};

const LAYOUT_KEY = "mymanus-web-layout-v1";
const SKILLS_KEY = "mymanus-selected-skills-v1";
const DEFAULT_LAYOUT = {
  prompt: 330,
  run: 460,
};

const MIN_WIDTHS = {
  prompt: 240,
  summary: 320,
  run: 340,
};

const MAX_WIDTHS = {
  prompt: 430,
  run: 680,
};

const TERMINAL_STATUSES = new Set(["completed", "error", "cancelled", "step_limit"]);

function setBadge(text, kind = "neutral") {
  els.badge.textContent = text;
  els.badge.className = `badge ${kind}`;
}

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function compactText(value, size = 80) {
  if (!value) return "-";
  return value.length > size ? `${value.slice(0, size - 1)}…` : value;
}

function attachmentLabel(file) {
  const extension = String(file.extension || "").toUpperCase();
  const size = file.size_label || "";
  return [extension, size].filter(Boolean).join(" · ");
}

function renderAttachmentLinks(attachments = []) {
  if (!Array.isArray(attachments) || !attachments.length) return "";
  const items = attachments
    .map((file) => {
      const url = safeLinkTarget(file.url || "#");
      return `<li><a href="${url}" target="_blank" rel="noopener noreferrer">${escapeHtml(file.name || file.original_name || file.id)}</a><span>${escapeHtml(attachmentLabel(file))}</span></li>`;
    })
    .join("");
  return `<ul class="chat-attachments">${items}</ul>`;
}

function renderAttachments() {
  els.attachmentsList.innerHTML = "";
  const count = state.attachments.length;
  els.attachmentsSummary.textContent = count ? `${count} 个附件` : "未上传";

  if (!count) {
    const empty = document.createElement("div");
    empty.className = "attachment-empty";
    empty.textContent = "支持 docx、pdf、xlsx、png、jpg/jpeg";
    els.attachmentsList.append(empty);
    return;
  }

  for (const file of state.attachments) {
    const item = document.createElement("div");
    item.className = "attachment-item";

    const main = document.createElement("a");
    main.href = file.url || "#";
    main.target = "_blank";
    main.rel = "noopener noreferrer";
    main.className = "attachment-main";

    const name = document.createElement("strong");
    name.textContent = file.name || file.original_name || file.id;

    const meta = document.createElement("span");
    meta.textContent = attachmentLabel(file);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "attachment-remove";
    remove.textContent = "移除";
    remove.addEventListener("click", () => {
      state.attachments = state.attachments.filter((item) => item.id !== file.id);
      renderAttachments();
    });

    main.append(name, meta);
    item.append(main, remove);
    els.attachmentsList.append(item);
  }
}

async function uploadSelectedAttachments(event) {
  const files = [...(event.target.files || [])];
  event.target.value = "";
  if (!files.length) return;

  const allowed = new Set(["docx", "pdf", "xlsx", "png", "jpg", "jpeg"]);
  const existingIds = new Set(state.attachments.map((item) => item.id));
  setBadge("上传中", "running");

  for (const file of files) {
    const extension = (file.name.split(".").pop() || "").toLowerCase();
    if (!allowed.has(extension)) {
      setBadge("格式不支持", "error");
      setResult(`不支持的附件格式：${file.name}`);
      continue;
    }

    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch("/api/uploads", {
        method: "POST",
        body,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const record = await response.json();
      if (!existingIds.has(record.id)) {
        state.attachments.push(record);
        existingIds.add(record.id);
      }
    } catch (error) {
      setBadge("上传失败", "error");
      setResult(String(error));
    }
  }

  renderAttachments();
  if (!state.running) setBadge("待命", "neutral");
}

function loadSelectedSkillIds() {
  try {
    const raw = JSON.parse(localStorage.getItem(SKILLS_KEY) || "[]");
    state.selectedSkillIds = new Set(Array.isArray(raw) ? raw : []);
  } catch {
    state.selectedSkillIds = new Set();
  }
}

function saveSelectedSkillIds() {
  localStorage.setItem(SKILLS_KEY, JSON.stringify([...state.selectedSkillIds]));
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function safeLinkTarget(value) {
  const raw = String(value || "").trim().replace(/&amp;/g, "&");
  if (!raw) return "#";
  if (raw.startsWith("#") || raw.startsWith("/")) return escapeHtml(raw);

  try {
    const parsed = new URL(raw, window.location.origin);
    if (["http:", "https:", "mailto:"].includes(parsed.protocol)) {
      return escapeHtml(raw);
    }
  } catch {
    return "#";
  }

  return "#";
}

function renderInlineMarkdown(value) {
  const token = "\uE000";
  const codeSpans = [];
  let text = String(value || "").replace(/`([^`]+)`/g, (_, code) => {
    codeSpans.push(`<code>${escapeHtml(code)}</code>`);
    return `${token}${codeSpans.length - 1}${token}`;
  });

  text = escapeHtml(text);
  text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, url) => {
    const href = safeLinkTarget(url);
    return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  text = text.replace(/(^|[\s(>])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  text = text.replace(new RegExp(`${token}(\\d+)${token}`, "g"), (_, index) => {
    return codeSpans[Number(index)] || "";
  });

  return text;
}

function tableCells(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isTableSeparator(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function renderMarkdown(value) {
  const source = String(value || "-").replace(/\r\n?/g, "\n");
  const lines = source.split("\n");
  const html = [];

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) continue;

    const fence = trimmed.match(/^```([\w-]*)\s*$/);
    if (fence) {
      const language = fence[1] ? ` class="language-${escapeHtml(fence[1])}"` : "";
      const code = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i]);
        i += 1;
      }
      html.push(`<pre><code${language}>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    if (trimmed === "---" || trimmed === "***") {
      html.push("<hr />");
      continue;
    }

    if (trimmed.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const headers = tableCells(trimmed);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].trim().includes("|")) {
        rows.push(tableCells(lines[i]));
        i += 1;
      }
      i -= 1;
      html.push(
        `<div class="markdown-table"><table><thead><tr>${headers
          .map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`)
          .join("")}</tr></thead><tbody>${rows
          .map(
            (row) =>
              `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`,
          )
          .join("")}</tbody></table></div>`,
      );
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quote = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) {
        quote.push(lines[i].replace(/^>\s?/, ""));
        i += 1;
      }
      i -= 1;
      html.push(`<blockquote>${renderMarkdown(quote.join("\n"))}</blockquote>`);
      continue;
    }

    if (/^[-*+]\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^[-*+]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*+]\s+/, ""));
        i += 1;
      }
      i -= 1;
      html.push(`<ul>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i += 1;
      }
      i -= 1;
      html.push(`<ol>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ol>`);
      continue;
    }

    const paragraph = [trimmed];
    while (
      i + 1 < lines.length &&
      lines[i + 1].trim() &&
      !/^(#{1,4})\s+/.test(lines[i + 1].trim()) &&
      !/^```/.test(lines[i + 1].trim()) &&
      !/^[-*+]\s+/.test(lines[i + 1].trim()) &&
      !/^\d+\.\s+/.test(lines[i + 1].trim()) &&
      !/^>\s?/.test(lines[i + 1])
    ) {
      i += 1;
      paragraph.push(lines[i].trim());
    }
    html.push(`<p>${paragraph.map(renderInlineMarkdown).join("<br />")}</p>`);
  }

  return html.join("");
}

function setResult(value) {
  els.result.innerHTML = renderMarkdown(value || "-");
}

function clearProcessStreams() {
  els.stream.textContent = "";
}

function lastConversationRun(conversation = state.conversation) {
  return conversation.length ? conversation[conversation.length - 1] : null;
}

function updateConversationPanel(run = lastConversationRun()) {
  if (!run) {
    els.continueLabel.textContent = "";
    els.continuePanel.hidden = true;
    return;
  }

  els.continueLabel.textContent = `#${run.id} · ${compactText(run.prompt, 48)}`;
  els.continuePanel.hidden = false;
}

function defaultSkillContent(name = "自定义 Skill") {
  return `# ${name}

## 适用场景
描述这个 skill 适合处理什么任务。

## 执行规则
- 先确认用户目标和已有上下文。
- 选择合适工具执行，并在关键步骤后验证结果。
- 完成后给出清晰的最终答案和必要的文件链接。
`;
}

function updateSkillsSummary() {
  const selected = state.skills.filter((skill) => state.selectedSkillIds.has(skill.id));
  const selectedCount = selected.length;
  const totalCount = state.skills.length;
  els.skillsSummary.textContent = totalCount ? `已选 ${selectedCount} / ${totalCount}` : "暂无 Skills";
  els.skillsDialogMeta.textContent = totalCount
    ? `选择本轮任务要加载的 Skills · 已选 ${selectedCount} 个`
    : "暂无 Skills，可以先新增一个";
  els.openSkillsButton.disabled = false;
  els.skillsSelectedLine.textContent = selectedCount
    ? selected.map((skill) => skill.name || skill.id).slice(0, 3).join("、")
    : "当前未启用";
  if (selectedCount > 3) {
    els.skillsSelectedLine.textContent += ` 等 ${selectedCount} 个`;
  }
}

function openSkillsDialog() {
  els.skillsDialog.hidden = false;
  document.body.classList.add("modal-open");
  updateSkillsSummary();
  window.requestAnimationFrame(() => {
    const checked = els.skillsList.querySelector("input[type='checkbox']:checked");
    const first = checked || els.skillsList.querySelector("input, button") || els.newSkillButton;
    first?.focus();
  });
}

function closeSkillsDialog() {
  closeSkillEditor();
  els.skillsDialog.hidden = true;
  document.body.classList.remove("modal-open");
  els.openSkillsButton.focus();
}

function renderSkills(skills = state.skills) {
  state.skills = Array.isArray(skills) ? skills : [];
  const validIds = new Set(state.skills.map((skill) => skill.id));
  for (const id of [...state.selectedSkillIds]) {
    if (!validIds.has(id)) state.selectedSkillIds.delete(id);
  }
  saveSelectedSkillIds();
  updateSkillsSummary();

  els.skillsList.textContent = "";
  if (!state.skills.length) {
    const empty = document.createElement("div");
    empty.className = "skill-empty";
    empty.textContent = "暂无 Skills";
    els.skillsList.append(empty);
    return;
  }

  for (const skill of state.skills) {
    const item = document.createElement("article");
    item.className = "skill-item";

    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = state.selectedSkillIds.has(skill.id);
    check.setAttribute("aria-label", `启用 ${skill.name}`);

    const body = document.createElement("button");
    body.className = "skill-main";
    body.type = "button";

    const name = document.createElement("strong");
    name.textContent = compactText(skill.name || skill.id, 34);

    const summary = document.createElement("span");
    summary.textContent = compactText(skill.summary || skill.path || "-", 72);

    const edit = document.createElement("button");
    edit.className = "skill-edit-button";
    edit.type = "button";
    edit.textContent = "编辑";

    const remove = document.createElement("button");
    remove.className = "skill-delete-button";
    remove.type = "button";
    remove.textContent = "删除";

    check.addEventListener("change", () => {
      if (check.checked) state.selectedSkillIds.add(skill.id);
      else state.selectedSkillIds.delete(skill.id);
      saveSelectedSkillIds();
      updateSkillsSummary();
    });
    body.addEventListener("click", () => {
      check.checked = !check.checked;
      check.dispatchEvent(new Event("change"));
    });
    edit.addEventListener("click", () => editSkill(skill.id));
    remove.addEventListener("click", () => deleteSkill(skill.id));

    body.append(name, summary);
    item.append(check, body, edit, remove);
    els.skillsList.append(item);
  }
}

async function loadSkills() {
  try {
    const response = await fetch("/api/skills");
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    renderSkills(payload.skills || []);
  } catch {
    els.skillsList.textContent = "";
    const error = document.createElement("div");
    error.className = "skill-empty";
    error.textContent = "Skills 读取失败";
    els.skillsList.append(error);
    state.skills = [];
    updateSkillsSummary();
  }
}

function openSkillEditor(skill = null) {
  state.editingSkillId = skill?.id || null;
  els.skillName.value = skill?.name || "";
  els.skillContent.value = skill?.content || defaultSkillContent(skill?.name || "自定义 Skill");
  els.deleteSkillButton.hidden = !state.editingSkillId;
  els.skillEditor.hidden = false;
  els.skillName.focus();
}

function closeSkillEditor() {
  state.editingSkillId = null;
  els.skillName.value = "";
  els.skillContent.value = "";
  els.skillEditor.hidden = true;
}

async function editSkill(skillId) {
  const response = await fetch(`/api/skills/${encodeURIComponent(skillId)}`);
  if (!response.ok) {
    setBadge("Skill 错误", "error");
    return;
  }
  openSkillEditor(await response.json());
}

async function saveSkill() {
  const name = els.skillName.value.trim();
  const content = els.skillContent.value.trim();
  if (!name || !content) {
    setBadge("Skill 为空", "error");
    return;
  }

  els.saveSkillButton.disabled = true;
  const editingId = state.editingSkillId;
  const response = await fetch(
    editingId ? `/api/skills/${encodeURIComponent(editingId)}` : "/api/skills",
    {
      method: editingId ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, content }),
    },
  );
  els.saveSkillButton.disabled = false;

  if (!response.ok) {
    setBadge("保存失败", "error");
    return;
  }

  const saved = await response.json();
  state.selectedSkillIds.add(saved.id);
  saveSelectedSkillIds();
  closeSkillEditor();
  await loadSkills();
  setBadge("Skill 已保存", "neutral");
}

async function deleteSkill(skillId = state.editingSkillId) {
  if (!skillId) return;
  const skill = state.skills.find((item) => item.id === skillId);
  const label = skill?.name || skillId;
  const ok = window.confirm(`删除这个 Skill？\n${label}`);
  if (!ok) return;

  const response = await fetch(`/api/skills/${encodeURIComponent(skillId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    setBadge("删除失败", "error");
    return;
  }

  state.selectedSkillIds.delete(skillId);
  saveSelectedSkillIds();
  if (state.editingSkillId === skillId) {
    closeSkillEditor();
  }
  await loadSkills();
  setBadge("Skill 已删除", "neutral");
}

function answerForRun(run) {
  if (run.answer) return run.answer;
  if (run.error) return run.error;
  if (run.status === "step_limit") {
    return "任务达到最大步数上限，尚未生成最终答案。请提高步数后重新运行，或基于当前页面继续任务。";
  }
  if (!TERMINAL_STATUSES.has(run.status)) return "运行中...";
  return run.result || "-";
}

function statusLabel(status) {
  const labels = {
    queued: "排队",
    waiting: "等待",
    running: "运行中",
    cancelling: "停止中",
    completed: "完成",
    cancelled: "已停止",
    step_limit: "步数耗尽",
    error: "错误",
  };
  return labels[status] || status || "-";
}

function renderConversation(conversation) {
  if (!conversation.length) {
    setResult("-");
    return;
  }

  els.result.innerHTML = conversation
    .map((run, index) => {
      return `<article class="chat-turn">
        <div class="chat-meta">第 ${index + 1} 轮 · #${escapeHtml(run.id)} · ${escapeHtml(statusLabel(run.status))}</div>
        <section class="chat-message user-message">
          <div class="chat-role">你</div>
          <div class="chat-content">${renderMarkdown(run.prompt || "-")}${renderAttachmentLinks(run.attachments)}</div>
        </section>
        <section class="chat-message assistant-message">
          <div class="chat-role">MyManus</div>
          <div class="chat-content">${renderMarkdown(answerForRun(run))}</div>
        </section>
      </article>`;
    })
    .join("");
  els.result.scrollTop = els.result.scrollHeight;
  updateConversationPanel(lastConversationRun(conversation));
}

function setConversation(conversation) {
  state.conversation = Array.isArray(conversation) ? conversation : [];
  renderConversation(state.conversation);
}

function updateConversationRun(runId, patch) {
  if (!runId) return;
  const index = state.conversation.findIndex((item) => item.id === runId);
  if (index >= 0) {
    state.conversation[index] = { ...state.conversation[index], ...patch };
  } else {
    state.conversation.push({
      id: runId,
      prompt: els.prompt.value.trim(),
      status: state.status,
      ...patch,
    });
  }
  renderConversation(state.conversation);
}

async function showRun(runId, options = {}) {
  const { includeEvents = true, connectActive = true } = options;
  const response = await fetch(`/api/runs/${runId}`);
  if (!response.ok) return null;

  const detail = await response.json();
  state.forceNewConversation = false;
  state.currentRunId = detail.id;
  state.status = detail.status || state.status;
  state.parentRunId = TERMINAL_STATUSES.has(detail.status) ? detail.id : null;
  clearProcessStreams();
  els.runMeta.textContent = `#${detail.id}`;
  setConversation(detail.conversation && detail.conversation.length ? detail.conversation : [detail]);

  if (includeEvents) {
    for (const event of detail.events || []) {
      appendEvent(event);
    }
  }

  if (TERMINAL_STATUSES.has(detail.status)) {
    state.running = false;
    els.runButton.disabled = false;
    els.cancelButton.disabled = true;
    disconnectRunStream();
  } else {
    state.running = true;
    els.runButton.disabled = true;
    els.cancelButton.disabled = false;
    setBadge(detail.status === "running" ? "运行中" : "排队", "running");
    if (connectActive) {
      connectRun(detail.id);
    }
  }

  return detail;
}

async function startNewConversation(options = {}) {
  const { clearPrompt = true } = options;
  disconnectRunStream();
  state.currentRunId = null;
  state.parentRunId = null;
  state.conversation = [];
  state.forceNewConversation = true;
  state.running = false;
  state.status = "idle";
  state.attachments = [];
  els.runButton.disabled = false;
  els.cancelButton.disabled = true;
  clearProcessStreams();
  els.runMeta.textContent = "新对话";
  setResult("-");
  updateConversationPanel(null);
  if (clearPrompt) {
    els.prompt.value = "";
  }
  setBadge("新对话", "neutral");
  await refreshStatus();
  els.prompt.focus();
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), Math.max(min, max));
}

function readPixels(value, fallback) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function getStoredLayout() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(LAYOUT_KEY) || "{}");
    return { ...DEFAULT_LAYOUT, ...saved };
  } catch {
    return { ...DEFAULT_LAYOUT };
  }
}

function storeLayout(layout) {
  try {
    window.localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
  } catch {
    // Browsers can disable local storage; resizing still works for this session.
  }
}

function getLayout() {
  const styles = getComputedStyle(document.documentElement);
  return {
    prompt: readPixels(styles.getPropertyValue("--prompt-width"), DEFAULT_LAYOUT.prompt),
    run: readPixels(styles.getPropertyValue("--run-width"), DEFAULT_LAYOUT.run),
  };
}

function layoutBudget() {
  const rect = els.workspace.getBoundingClientRect();
  const styles = getComputedStyle(els.workspace);
  const gap = readPixels(styles.columnGap, 10);
  const root = getComputedStyle(document.documentElement);
  const handle = readPixels(root.getPropertyValue("--resizer-size"), 8);
  const compact = window.matchMedia("(max-width: 1180px)").matches;
  const mobile = window.matchMedia("(max-width: 760px)").matches;

  if (mobile) {
    return { available: rect.width, compact, mobile };
  }

  const fixed = compact ? handle + gap * 2 : handle * 2 + gap * 4;
  return { available: Math.max(0, rect.width - fixed), compact, mobile };
}

function clampLayout(layout) {
  const next = { ...DEFAULT_LAYOUT, ...layout };
  const budget = layoutBudget();

  if (!budget.mobile) {
    if (budget.compact) {
      const promptMax = Math.min(MAX_WIDTHS.prompt, budget.available - MIN_WIDTHS.summary);
      next.prompt = clamp(next.prompt, MIN_WIDTHS.prompt, promptMax);
    } else {
      const runMax = Math.min(MAX_WIDTHS.run, budget.available - next.prompt - MIN_WIDTHS.summary);
      next.run = clamp(next.run, MIN_WIDTHS.run, runMax);
      const promptMax = Math.min(MAX_WIDTHS.prompt, budget.available - next.run - MIN_WIDTHS.summary);
      next.prompt = clamp(next.prompt, MIN_WIDTHS.prompt, promptMax);
      const runMaxAfterPrompt = Math.min(
        MAX_WIDTHS.run,
        budget.available - next.prompt - MIN_WIDTHS.summary,
      );
      next.run = clamp(next.run, MIN_WIDTHS.run, runMaxAfterPrompt);
    }
  }

  return next;
}

function applyLayout(layout, persist = false) {
  const next = clampLayout(layout);
  document.documentElement.style.setProperty("--prompt-width", `${Math.round(next.prompt)}px`);
  document.documentElement.style.setProperty("--run-width", `${Math.round(next.run)}px`);
  if (persist) {
    storeLayout(next);
  }
  return next;
}

function resizeColumn(handle, delta) {
  const start = getLayout();
  if (handle.dataset.resize === "prompt") {
    return applyLayout({ ...start, prompt: start.prompt + delta });
  }
  return applyLayout({ ...start, run: start.run - delta });
}

function startPointerResize(event) {
  const handle = event.currentTarget;
  const startX = event.clientX;
  const startLayout = getLayout();

  event.preventDefault();
  document.body.classList.add("is-resizing");

  const onMove = (moveEvent) => {
    const delta = moveEvent.clientX - startX;
    if (handle.dataset.resize === "prompt") {
      applyLayout({ ...startLayout, prompt: startLayout.prompt + delta });
    } else {
      applyLayout({ ...startLayout, run: startLayout.run - delta });
    }
  };

  const onUp = () => {
    document.body.classList.remove("is-resizing");
    storeLayout(getLayout());
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    window.removeEventListener("pointercancel", onUp);
  };

  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp, { once: true });
  window.addEventListener("pointercancel", onUp, { once: true });
}

function handleResizeKey(event) {
  const step = event.shiftKey ? 40 : 16;
  let handled = true;

  if (event.key === "ArrowLeft") {
    resizeColumn(event.currentTarget, -step);
  } else if (event.key === "ArrowRight") {
    resizeColumn(event.currentTarget, step);
  } else {
    handled = false;
  }

  if (handled) {
    event.preventDefault();
    storeLayout(getLayout());
  }
}

function initResizableLayout() {
  applyLayout(getStoredLayout());
  for (const handle of els.columnResizers) {
    handle.addEventListener("pointerdown", startPointerResize);
    handle.addEventListener("keydown", handleResizeKey);
  }
  window.addEventListener("resize", () => applyLayout(getLayout()));
}

function eventTitle(event) {
  if (event.type === "status") return `状态 · ${event.status}`;
  if (event.type === "log") return `日志 · ${event.level || "INFO"}`;
  if (event.type === "tools") return "工具";
  if (event.type === "agent") return `${event.agent || "Agent"} · ${event.status || ""}`;
  if (event.type === "result") return "执行摘要";
  if (event.type === "error") return "错误";
  if (event.type === "context") return "上下文";
  if (event.type === "user") return "任务";
  return event.type;
}

function eventBody(event) {
  if (event.type === "log") return event.message || "";
  if (event.type === "status") return event.message || event.status || "";
  if (event.type === "tools") {
    return `已加载 ${event.count} 个工具，其中 Playwright MCP ${event.browser_tools} 个`;
  }
  if (event.type === "agent") return event.content || "";
  if (event.type === "result") return event.content || "";
  if (event.type === "error") return event.message || "";
  if (event.type === "context") return `基于上一轮 #${event.parent_run_id} 继续`;
  if (event.type === "user") return event.content || "";
  return JSON.stringify(event, null, 2);
}

function appendEvent(event) {
  if (event.type === "answer") {
    updateConversationRun(state.currentRunId, {
      answer: event.content || "-",
      status: state.status,
    });
    return;
  }
  if (event.type === "thought") {
    return;
  }

  const node = document.createElement("article");
  const logClass = event.type === "log" ? `log-${String(event.level || "info").toLowerCase()}` : "";
  node.className = `event ${event.type} ${logClass}`.trim();

  const head = document.createElement("div");
  head.className = "event-head";

  const kind = document.createElement("span");
  kind.className = "event-kind";
  kind.textContent = eventTitle(event);

  const time = document.createElement("span");
  time.className = "event-time";
  time.textContent = formatTime(event.time);

  const body = document.createElement("div");
  body.className = "event-body";
  body.textContent = eventBody(event);

  head.append(kind, time);
  node.append(head, body);
  els.stream.append(node);
  els.stream.scrollTop = els.stream.scrollHeight;

  if (event.type === "error") {
    updateConversationRun(state.currentRunId, {
      error: event.message || "-",
      status: "error",
    });
  }
  if (event.type === "status") {
    updateRunStatus(event.status);
    updateConversationRun(state.currentRunId, { status: event.status });
  }
}

function updateRunStatus(status) {
  state.status = status;
  if (TERMINAL_STATUSES.has(status) && state.currentRunId) {
    state.parentRunId = state.currentRunId;
    updateConversationPanel(lastConversationRun());
  }
  if (status === "queued" || status === "waiting" || status === "running") {
    state.running = true;
    els.runButton.disabled = true;
    els.cancelButton.disabled = false;
    setBadge(status === "running" ? "运行中" : "排队", "running");
  } else if (status === "cancelling") {
    state.running = true;
    els.runButton.disabled = true;
    els.cancelButton.disabled = true;
    setBadge("停止中", "running");
  } else if (status === "completed") {
    state.running = false;
    els.runButton.disabled = false;
    els.cancelButton.disabled = true;
    setBadge("完成", "done");
    refreshStatus();
  } else if (status === "cancelled") {
    state.running = false;
    els.runButton.disabled = false;
    els.cancelButton.disabled = true;
    setBadge("已停止", "neutral");
    refreshStatus();
  } else if (status === "step_limit") {
    state.running = false;
    els.runButton.disabled = false;
    els.cancelButton.disabled = true;
    setBadge("步数耗尽", "error");
    refreshStatus();
  } else if (status === "error") {
    state.running = false;
    els.runButton.disabled = false;
    els.cancelButton.disabled = true;
    setBadge("错误", "error");
    refreshStatus();
  }
}

function connectRun(runId) {
  disconnectRunStream();
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws/runs/${runId}`);
  state.socket = socket;

  socket.onopen = () => {
    if (state.currentRunId !== runId) return;
    els.runMeta.textContent = `#${runId}`;
  };
  socket.onmessage = (message) => {
    if (state.currentRunId !== runId) return;
    appendEvent(JSON.parse(message.data));
  };
  socket.onclose = () => {
    if (state.currentRunId === runId && state.running) {
      setBadge("断开", "error");
    }
  };
}

function disconnectRunStream() {
  if (!state.socket) return;
  state.socket.onopen = null;
  state.socket.onmessage = null;
  state.socket.onclose = null;
  state.socket.close();
  state.socket = null;
}

async function startRun(event) {
  event.preventDefault();
  const prompt = els.prompt.value.trim();
  if (!prompt) {
    setBadge("空任务", "error");
    return;
  }

  clearProcessStreams();
  setBadge("启动中", "running");
  els.runButton.disabled = true;

  const requestBody = {
    prompt,
    skill_ids: [...state.selectedSkillIds],
    attachment_ids: state.attachments.map((item) => item.id),
  };
  if (!state.forceNewConversation && state.parentRunId) {
    requestBody.parent_run_id = state.parentRunId;
  }

  const response = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const text = await response.text();
    els.runButton.disabled = false;
    setBadge("错误", "error");
    setResult(text);
    return;
  }

  const run = await response.json();
  state.attachments = [];
  renderAttachments();
  state.forceNewConversation = false;
  state.currentRunId = run.id;
  state.parentRunId = null;
  state.running = true;
  state.status = "queued";
  els.runButton.disabled = true;
  els.cancelButton.disabled = false;
  await showRun(run.id, { includeEvents: false, connectActive: false });
  setBadge("排队", "running");
  connectRun(run.id);
  await refreshStatus();
}

async function cancelRun() {
  if (!state.currentRunId) return;
  state.status = "cancelling";
  setBadge("停止中", "running");
  els.cancelButton.disabled = true;
  try {
    const response = await fetch(`/api/runs/${state.currentRunId}/cancel`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
  } catch (error) {
    els.cancelButton.disabled = false;
    setBadge("停止失败", "error");
    updateConversationRun(state.currentRunId, { error: String(error), status: "error" });
  }
}

async function deleteRun(run) {
  if (!TERMINAL_STATUSES.has(run.status)) {
    setBadge("运行中", "error");
    updateConversationRun(run.id, {
      error: "运行中的任务不能删除，请先停止或等待结束。",
      status: run.status,
    });
    return;
  }

  const ok = window.confirm(`删除这个对话及其继续记录？\n#${run.id}`);
  if (!ok) return;

  const response = await fetch(`/api/runs/${run.id}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const text = await response.text();
    setBadge("删除失败", "error");
    updateConversationRun(run.id, { error: text || "删除失败。", status: "error" });
    return;
  }

  const payload = await response.json();
  const deletedIds = Array.isArray(payload.deleted) ? payload.deleted : [payload.deleted];
  clearCurrentConversationIfDeleted(deletedIds);

  setBadge("已删除", "neutral");
  await refreshStatus();
}

function clearCurrentConversationIfDeleted(deletedIds) {
  const visibleConversationDeleted = state.conversation.some((item) => deletedIds.includes(item.id));

  if (deletedIds.includes(state.currentRunId) || visibleConversationDeleted) {
    state.currentRunId = null;
    state.parentRunId = null;
    state.conversation = [];
    clearProcessStreams();
    setResult("-");
    els.runMeta.textContent = "尚未开始";
    updateConversationPanel(null);
  }
  if (deletedIds.includes(state.parentRunId)) {
    state.parentRunId = null;
    updateConversationPanel(null);
  }
}

async function deleteAllRuns() {
  const ok = window.confirm("删除所有已结束的最近任务？运行中的任务不会被删除。");
  if (!ok) return;

  els.clearRunsButton.disabled = true;
  const response = await fetch("/api/runs", {
    method: "DELETE",
  });
  if (!response.ok) {
    const text = await response.text();
    setBadge("清空失败", "error");
    if (state.currentRunId) {
      updateConversationRun(state.currentRunId, { error: text || "清空失败。", status: state.status });
    }
    await refreshStatus();
    return;
  }

  const payload = await response.json();
  const deletedIds = Array.isArray(payload.deleted) ? payload.deleted : [];
  clearCurrentConversationIfDeleted(deletedIds);

  setBadge(deletedIds.length ? "已清空" : "无可删除", "neutral");
  await refreshStatus();
}

function renderRuns(runs) {
  els.runsList.textContent = "";
  els.clearRunsButton.disabled = !runs.some((run) => TERMINAL_STATUSES.has(run.status));

  if (!runs.length) {
    const empty = document.createElement("div");
    empty.className = "run-item";
    empty.textContent = "-";
    els.runsList.append(empty);
    return;
  }

  for (const run of runs.slice(0, 8)) {
    const item = document.createElement("article");
    item.className = "run-item";

    const openButton = document.createElement("button");
    openButton.className = "run-item-main";
    openButton.type = "button";

    const title = document.createElement("strong");
    title.textContent = compactText(run.prompt, 64);

    const meta = document.createElement("span");
    meta.textContent = `${run.status} · ${formatTime(run.updated_at)}`;

    const deleteButton = document.createElement("button");
    deleteButton.className = "run-item-delete";
    deleteButton.type = "button";
    deleteButton.textContent = "删除";
    deleteButton.disabled = !TERMINAL_STATUSES.has(run.status);

    openButton.append(title, meta);
    openButton.addEventListener("click", () => showRun(run.id));
    deleteButton.addEventListener("click", () => deleteRun(run));

    item.append(openButton, deleteButton);
    els.runsList.append(item);
  }
}

async function refreshStatus() {
  const response = await fetch("/api/status");
  if (!response.ok) {
    setBadge("离线", "error");
    return;
  }
  const info = await response.json();

  els.modelLine.textContent = `${info.model || "-"} · ${info.reasoning_effort || "-"}`;
  if (Array.isArray(info.skills)) {
    renderSkills(info.skills);
  }
  const recentRuns = info.runs || [];
  renderRuns(recentRuns);

  if (!state.running && !state.currentRunId && !state.forceNewConversation) {
    const latestRun = recentRuns.find((run) => TERMINAL_STATUSES.has(run.status));
    if (latestRun) {
      await showRun(latestRun.id, { includeEvents: false });
    }
  }

  if (!state.running && state.status === "idle") {
    setBadge("待命", "neutral");
  }
}

els.form.addEventListener("submit", startRun);
els.cancelButton.addEventListener("click", cancelRun);
els.newConversationButton.addEventListener("click", () => startNewConversation());
els.clearRunsButton.addEventListener("click", deleteAllRuns);
els.openSkillsButton.addEventListener("click", openSkillsDialog);
els.attachmentInput.addEventListener("change", uploadSelectedAttachments);
els.closeSkillsDialogButton.addEventListener("click", closeSkillsDialog);
els.skillsDoneButton.addEventListener("click", closeSkillsDialog);
els.skillsDialog.addEventListener("click", (event) => {
  if (event.target === els.skillsDialog) closeSkillsDialog();
});
els.newSkillButton.addEventListener("click", () => openSkillEditor());
els.saveSkillButton.addEventListener("click", saveSkill);
els.deleteSkillButton.addEventListener("click", deleteSkill);
els.cancelSkillButton.addEventListener("click", closeSkillEditor);
els.clearButton.addEventListener("click", () => {
  clearProcessStreams();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !els.skillsDialog.hidden) {
    closeSkillsDialog();
  }
});

loadSelectedSkillIds();
initResizableLayout();
updateSkillsSummary();
renderAttachments();
loadSkills();
refreshStatus();
