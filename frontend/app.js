const chatEl = document.querySelector("#chat");
const statusEl = document.querySelector("#status");
const formEl = document.querySelector("#chatForm");
const messageEl = document.querySelector("#message");
const fileInput = document.querySelector("#fileInput");
const pickFileBtn = document.querySelector("#pickFileBtn");
const webSearchToggleBtn = document.querySelector("#webSearchToggleBtn");
const sendBtn = document.querySelector("#sendBtn");
const attachmentsEl = document.querySelector("#attachments");
const dropZone = document.querySelector("#dropZone");
const newChatBtn = document.querySelector("#newChatBtn");
const agentOfflineMessage = document.querySelector("#agentOfflineMessage");

const history = [];
let pendingFiles = [];
let lastImageFiles = [];
let activeController = null;
let activeRequestId = 0;
let webSearchEnabled = localStorage.getItem("templeAgentWebSearch") === "1";
let backendOnline = true;
let consecutiveHealthFailures = 0;
let healthCheckInFlight = false;
let readyMessageShown = false;

const ui = {
  ready: "\u6211\u5df2\u51c6\u5907\u597d\u3002\u4f60\u53ef\u4ee5\u50cf ChatGPT \u4e00\u6837\u628a\u56fe\u7247\u62d6\u5230\u7a97\u53e3\uff0c\u7136\u540e\u76f4\u63a5\u63d0\u51fa\u8bc6\u522b\u3001\u7b5b\u9009\u3001\u6574\u7406\u6216\u590d\u6838\u9700\u6c42\u3002",
  thinking: "\u6b63\u5728\u8bc6\u522b\u548c\u7ec4\u7ec7\u56de\u7b54...",
  empty: "\u8bf7\u8f93\u5165\u95ee\u9898\uff0c\u6216\u6dfb\u52a0\u56fe\u7247\u3002",
  lmOk: "\u6a21\u578b API \u5df2\u8fde\u63a5",
  lmBad: "\u6a21\u578b API \u672a\u8fde\u63a5",
  backendBad: "\u540e\u7aef\u672a\u8fde\u63a5",
  newChat: "\u65b0\u5bf9\u8bdd\u5df2\u5f00\u59cb\u3002",
  stopped: "\u5df2\u7ec8\u6b62\u672c\u6b21\u56de\u7b54\u3002",
  closedDuringRequest: "Agent \u5df2\u5173\u95ed\uff0c\u672c\u6b21\u4efb\u52a1\u5df2\u4e2d\u65ad\u3002",
};

function setAgentAvailability(isOnline) {
  if (backendOnline === isOnline && agentOfflineMessage.hidden === isOnline) return;
  backendOnline = isOnline;
  agentOfflineMessage.hidden = isOnline;
  formEl.classList.toggle("agent-offline", !isOnline);
  messageEl.disabled = !isOnline;
  pickFileBtn.disabled = !isOnline;
  webSearchToggleBtn.disabled = !isOnline;
  sendBtn.disabled = !isOnline;

  if (!isOnline && activeController) {
    activeController.abort();
  }
  if (isOnline) messageEl.focus();
}

async function checkBackendHealth({ immediate = false } = {}) {
  if (healthCheckInFlight) return backendOnline;
  healthCheckInFlight = true;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 2500);
  try {
    const res = await fetch(`/api/health?heartbeat=${Date.now()}`, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await res.json();
    consecutiveHealthFailures = 0;
    setAgentAvailability(true);
    return true;
  } catch (err) {
    consecutiveHealthFailures += 1;
    if (immediate || consecutiveHealthFailures >= 2) {
      setAgentAvailability(false);
    }
    return false;
  } finally {
    clearTimeout(timeoutId);
    healthCheckInFlight = false;
  }
}

function isSupportedUpload(file) {
  const lower = (file.name || "").toLowerCase();
  return (file.type || "").startsWith("image/") || lower.endsWith(".zip") || lower.endsWith(".rar");
}

function ensureClipboardFileName(file, index) {
  if (file.name) return file;
  const type = file.type || "application/octet-stream";
  const imageExt = type.startsWith("image/") ? type.split("/")[1].split(";")[0] || "png" : "bin";
  const ext = imageExt === "jpeg" ? "jpg" : imageExt;
  return new File([file], `pasted-${Date.now()}-${index}.${ext}`, { type });
}

function getClipboardFiles(event) {
  const clipboard = event.clipboardData;
  if (!clipboard) return [];

  const pastedFiles = [];
  Array.from(clipboard.files || []).forEach((file, index) => {
    pastedFiles.push(ensureClipboardFileName(file, index));
  });

  Array.from(clipboard.items || []).forEach((item, index) => {
    if (item.kind !== "file") return;
    const file = item.getAsFile();
    if (file) pastedFiles.push(ensureClipboardFileName(file, index));
  });

  const seen = new Set();
  return pastedFiles.filter((file) => {
    if (!isSupportedUpload(file)) return false;
    const key = `${file.name}|${file.size}|${file.type}|${file.lastModified}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function setSendingState(isSending) {
  sendBtn.textContent = isSending ? "\u505c\u6b62" : "\u53d1\u9001";
  sendBtn.classList.toggle("stop", isSending);
  sendBtn.title = isSending ? "\u7ec8\u6b62\u5f53\u524d\u56de\u7b54" : "\u53d1\u9001";
}

function renderWebSearchToggle() {
  webSearchToggleBtn.classList.toggle("is-active", webSearchEnabled);
  webSearchToggleBtn.setAttribute("aria-pressed", webSearchEnabled ? "true" : "false");
  webSearchToggleBtn.title = webSearchEnabled ? "\u5df2\u5f00\u542f\u8054\u7f51\u641c\u7d22" : "\u5df2\u5173\u95ed\u8054\u7f51\u641c\u7d22";
}

function setWebSearchEnabled(enabled) {
  webSearchEnabled = enabled;
  localStorage.setItem("templeAgentWebSearch", enabled ? "1" : "0");
  renderWebSearchToggle();
}

function abortCurrentResponse() {
  if (activeController) {
    activeController.abort();
  }
}

function addMessage(role, content, options = {}) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "A";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${options.pending ? "pending" : ""}`;
  bubble.textContent = content;

  if (options.files && options.files.length) {
    bubble.appendChild(renderUserFiles(options.files));
  }

  if (options.images && options.images.length) {
    bubble.appendChild(renderGallery(options.images));
  }

  if (options.links) {
    const links = document.createElement("div");
    links.className = "link-row";
    Object.entries(options.links).forEach(([name, href]) => {
      const a = document.createElement("a");
      a.href = href;
      a.target = "_blank";
      a.textContent = name.toUpperCase();
      links.appendChild(a);
    });
    bubble.appendChild(links);
  }

  if (options.sources && options.sources.length) {
    bubble.appendChild(renderSources(options.sources));
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;

  if (!options.pending) {
    history.push({ role: role === "assistant" ? "assistant" : "user", content });
  }
  return row;
}

function renderGallery(images) {
  const gallery = document.createElement("div");
  gallery.className = "gallery";
  images.slice(0, 12).forEach((item) => {
    const card = document.createElement("div");
    card.className = "result-card";
    const img = document.createElement("img");
    img.src = item.url;
    img.alt = item.title || "result";
    const title = document.createElement("div");
    title.textContent = item.title || "";
    card.appendChild(img);
    card.appendChild(title);
    gallery.appendChild(card);
  });
  return gallery;
}

function renderUserFiles(files) {
  const wrap = document.createElement("div");
  wrap.className = "user-files";
  files.forEach((file) => {
    if (file.type.startsWith("image/")) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      img.alt = "uploaded image";
      wrap.appendChild(img);
      return;
    }
    const chip = document.createElement("div");
    chip.className = "file-chip";
    chip.textContent = file.name;
    wrap.appendChild(chip);
  });
  return wrap;
}

function renderSources(sources) {
  const wrap = document.createElement("div");
  wrap.className = "sources";
  const title = document.createElement("div");
  title.className = "sources-title";
  title.textContent = "\u6765\u6e90";
  wrap.appendChild(title);
  sources.slice(0, 5).forEach((source, index) => {
    const a = document.createElement("a");
    a.href = source.url;
    a.target = "_blank";
    a.rel = "noreferrer";
    a.textContent = `${index + 1}. ${source.title || source.url}`;
    wrap.appendChild(a);
  });
  return wrap;
}

function setPendingFiles(files) {
  pendingFiles = files.filter(isSupportedUpload);
  renderAttachments();
}

function addPendingFiles(files) {
  setPendingFiles([...pendingFiles, ...Array.from(files)]);
}

function renderAttachments() {
  attachmentsEl.innerHTML = "";
  pendingFiles.forEach((file, index) => {
    const item = document.createElement("div");
    item.className = "attachment";
    if (file.type.startsWith("image/")) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      item.appendChild(img);
    }
    const name = document.createElement("span");
    name.textContent = file.name;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "x";
    remove.addEventListener("click", () => {
      pendingFiles.splice(index, 1);
      renderAttachments();
    });
    item.appendChild(name);
    item.appendChild(remove);
    attachmentsEl.appendChild(item);
  });
}

async function loadHealth() {
  const online = await checkBackendHealth({ immediate: true });
  statusEl.textContent = "";
  if (online && !readyMessageShown) {
    readyMessageShown = true;
    addMessage("assistant", ui.ready);
  }
}

async function sendMessage() {
  if (!backendOnline) {
    setAgentAvailability(false);
    return;
  }

  if (activeController) {
    abortCurrentResponse();
    return;
  }

  const message = messageEl.value.trim();
  if (!message && pendingFiles.length === 0) {
    alert(ui.empty);
    return;
  }

  const filesToSend = [...pendingFiles];
  const hasImageContext = filesToSend.length === 0 && lastImageFiles.length > 0;
  const requestFiles = hasImageContext ? [...lastImageFiles] : filesToSend;
  addMessage("user", message || "\u8bf7\u8bc6\u522b\u8fd9\u4e9b\u56fe\u7247", { files: filesToSend });
  messageEl.value = "";
  setPendingFiles([]);
  if (filesToSend.some((file) => file.type.startsWith("image/"))) {
    lastImageFiles = filesToSend.filter((file) => file.type.startsWith("image/"));
  }

  const pendingRow = addMessage("assistant", ui.thinking, { pending: true });
  const controller = new AbortController();
  const requestId = activeRequestId + 1;
  activeRequestId = requestId;
  activeController = controller;
  setSendingState(true);

  try {
    const form = new FormData();
    form.append("message", message);
    form.append("history", JSON.stringify(history.slice(-8)));
    form.append("use_image_context", hasImageContext ? "1" : "0");
    form.append("web_search_enabled", webSearchEnabled ? "1" : "0");
    requestFiles.forEach((file) => form.append("files", file));
    const res = await fetch("/api/agent_message", { method: "POST", body: form, signal: controller.signal });
    const data = await readJsonResponse(res);
    if (requestId !== activeRequestId) return;
    if (!res.ok) throw new Error(data.detail || data.message || "request failed");
    pendingRow.remove();
    addMessage("assistant", data.reply || "\u6211\u8fd9\u6b21\u6ca1\u6709\u751f\u6210\u6709\u6548\u56de\u7b54\uff0c\u8bf7\u6362\u4e00\u79cd\u95ee\u6cd5\u3002", {
      images: data.images || [],
      links: data.files || {},
      sources: data.sources || [],
    });
  } catch (err) {
    if (requestId !== activeRequestId) return;
    pendingRow.remove();
    if (err.name === "AbortError") {
      addMessage("assistant", backendOnline ? ui.stopped : ui.closedDuringRequest);
    } else {
      addMessage("assistant", `Error: ${err.message}`);
      checkBackendHealth({ immediate: true });
    }
  } finally {
    if (requestId === activeRequestId) {
      activeController = null;
      setSendingState(false);
      messageEl.focus();
    }
  }
}

async function readJsonResponse(res) {
  const body = await res.text();
  try {
    return JSON.parse(body);
  } catch (err) {
    return { detail: body || `${res.status} ${res.statusText}` };
  }
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});

sendBtn.addEventListener("click", (event) => {
  if (activeController) {
    event.preventDefault();
    abortCurrentResponse();
  }
});

messageEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

messageEl.addEventListener("input", () => {
  messageEl.style.height = "auto";
  messageEl.style.height = `${Math.min(messageEl.scrollHeight, 180)}px`;
});

pickFileBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => addPendingFiles(fileInput.files));
webSearchToggleBtn.addEventListener("click", () => setWebSearchEnabled(!webSearchEnabled));

document.addEventListener("paste", (event) => {
  const files = getClipboardFiles(event);
  if (!files.length) return;
  event.preventDefault();
  addPendingFiles(files);
});

["dragenter", "dragover"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    if (name === "drop") addPendingFiles(event.dataTransfer.files);
    dropZone.classList.remove("dragging");
  });
});

newChatBtn.addEventListener("click", () => {
  if (activeController) {
    activeController.abort();
    activeController = null;
    activeRequestId += 1;
    setSendingState(false);
  }
  chatEl.innerHTML = "";
  history.splice(0, history.length);
  setPendingFiles([]);
  lastImageFiles = [];
  addMessage("assistant", ui.newChat);
});

renderWebSearchToggle();
loadHealth();
setInterval(() => checkBackendHealth(), 3000);
