const socket = io();
const paneMap = new Map();
const statusCache = new Map();
const syncState = new Map();
const liveOutput = new Map();
const liveMetrics = new Map();
const runtimeMetrics = new Map();
const liveState = new Map();
const RUN_LIST_LIMIT = 20;
const BASELINE_KEY = "bench-race-baseline-run";
const MODEL_POLICY_ENDPOINT = "/api/settings/model_policy";
const COMFY_SETTINGS_ENDPOINT = "/api/settings/comfy";
const CHECKPOINT_CATALOG_ENDPOINT = "/api/image/checkpoints";
const OPTIONS_STORAGE_KEY = "bench-race-options-expanded";
const POLLING_SETTINGS_KEY = "bench-race-polling-settings";

// Adaptive polling configuration
const DEFAULT_POLLING_CONFIG = {
  idlePollIntervalMs: 30000,   // 30s when no runs active
  activePollIntervalMs: 2000,  // 2s when runs active
  uiUpdateThrottleMs: 500,     // Throttle UI updates to max 2/sec
};
let pollingConfig = { ...DEFAULT_POLLING_CONFIG };

// Global run state tracking
let isRunActive = false;
const activeRunIds = new Set();
let statusPollTimer = null;
let lastUiUpdate = 0;

let baselineRunId = localStorage.getItem(BASELINE_KEY);
let baselineRun = null;
let viewingMode = "live";
let viewingRunId = null;
let currentRunData = null;
let liveRunId = null;
let recentRuns = [];
let activeOverlay = null;
let lastOverlayFocus = null;
let checkpointCatalog = [];

// Fallback reason display strings
const FALLBACK_REASONS = {
  ollama_unreachable: "Ollama was unreachable",
  missing_model: "Model not installed on Ollama",
  stream_error: "Streaming error occurred",
  unknown: "Unknown reason",
};

// Load saved polling settings from localStorage
const loadPollingSettings = () => {
  try {
    const saved = localStorage.getItem(POLLING_SETTINGS_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      pollingConfig = { ...DEFAULT_POLLING_CONFIG, ...parsed };
    }
  } catch (e) {
    console.warn("Failed to load polling settings:", e);
  }
};

const savePollingSettings = () => {
  try {
    localStorage.setItem(POLLING_SETTINGS_KEY, JSON.stringify(pollingConfig));
  } catch (e) {
    console.warn("Failed to save polling settings:", e);
  }
};

// Adaptive polling: switch between idle and active intervals
const getCurrentPollInterval = () => {
  return isRunActive ? pollingConfig.activePollIntervalMs : pollingConfig.idlePollIntervalMs;
};

const scheduleStatusPoll = () => {
  if (statusPollTimer) {
    clearTimeout(statusPollTimer);
  }
  const interval = getCurrentPollInterval();
  statusPollTimer = setTimeout(async () => {
    await fetchStatus();
    scheduleStatusPoll();
  }, interval);
};

const restartPolling = () => {
  scheduleStatusPoll();
};

// Run state management
const setRunActive = (active, runId = null) => {
  const wasActive = isRunActive;
  if (active && runId) {
    activeRunIds.add(runId);
  } else if (!active && runId) {
    activeRunIds.delete(runId);
  }
  isRunActive = activeRunIds.size > 0;

  // Update UI indicator if present
  const indicator = document.getElementById("run-active-indicator");
  if (indicator) {
    indicator.classList.toggle("active", isRunActive);
    indicator.title = isRunActive
      ? `${activeRunIds.size} active run(s)`
      : "No active runs";
  }

  // Restart polling if state changed
  if (wasActive !== isRunActive) {
    console.log(`Run state changed: ${wasActive} -> ${isRunActive}`);
    restartPolling();
  }
};

// Fetch active runs from server to initialize state
const fetchActiveRuns = async () => {
  try {
    const response = await fetch("/api/runs/active");
    if (!response.ok) return;
    const data = await response.json();
    activeRunIds.clear();
    (data.run_ids || []).forEach((id) => activeRunIds.add(id));
    isRunActive = data.is_active || false;
    restartPolling();
  } catch (error) {
    console.warn("Failed to fetch active runs:", error);
  }
};

// Throttle function for UI updates
const throttle = (fn, delay) => {
  let lastCall = 0;
  let timeoutId = null;
  return (...args) => {
    const now = Date.now();
    const remaining = delay - (now - lastCall);
    if (remaining <= 0) {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      lastCall = now;
      fn(...args);
    } else if (!timeoutId) {
      timeoutId = setTimeout(() => {
        lastCall = Date.now();
        timeoutId = null;
        fn(...args);
      }, remaining);
    }
  };
};

const showToast = (message, type = "info") => {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
};

const handleJobResults = (results) => {
  if (!Array.isArray(results)) return;
  results.forEach((r) => {
    const pane = paneMap.get(r.machine_id);
    if (!pane) return;
    const { out, metrics } = pane;

    out.textContent = "";
    if (r.error) {
      out.textContent = `Error starting job: ${r.error}`;
      metrics.innerHTML = `<span class="muted">Failed to start</span>`;
      updateLivePane(r.machine_id, {
        outText: out.textContent,
        metricsHtml: metrics.innerHTML,
        isMock: false,
      });
    } else if (r.skipped) {
      out.textContent = "Skipped (not ready)";
      metrics.innerHTML = `<span class="muted">Machine not ready</span>`;
      updateLivePane(r.machine_id, {
        outText: out.textContent,
        metricsHtml: metrics.innerHTML,
        isMock: false,
      });
    } else {
      metrics.innerHTML = `<span class="muted">Job started…</span>`;
      updateLivePane(r.machine_id, {
        outText: out.textContent,
        metricsHtml: metrics.innerHTML,
        isMock: false,
      });
    }
  });
};

const formatGb = (value) => (value == null ? "n/a" : `${value.toFixed(1)}GB`);
const formatBytes = (value) => {
  if (value == null) return "n/a";
  const gib = value / (1024 ** 3);
  return `${gib.toFixed(1)}GiB`;
};

const formatTimestamp = (isoString) => {
  if (!isoString) return "";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return isoString;
  return date.toLocaleString();
};

const formatMetric = (value, unit, decimals = 1) => {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${value.toFixed(decimals)}${unit}`;
};

const formatBytes = (bytes) => {
  if (bytes == null || Number.isNaN(bytes)) return "n/a";
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb.toFixed(1)} MB`;
  const gb = mb / 1024;
  return `${gb.toFixed(2)} GB`;
};

const renderCheckpointValidation = (items) => {
  const list = document.getElementById("checkpoint-validation-list");
  if (!list) return;
  list.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "helper";
    empty.textContent = "No checkpoints configured.";
    list.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "checkpoint-validation-item";
    const dot = document.createElement("span");
    dot.className = `status-dot ${item.valid ? "ok" : "error"}`;
    const content = document.createElement("div");
    const title = document.createElement("div");
    title.textContent = item.name || item.url;
    const meta = document.createElement("div");
    meta.className = "checkpoint-validation-meta";
    const statusText = item.valid ? "Valid" : item.error || "Invalid";
    const resolved = item.resolved_url ? `Resolved: ${item.resolved_url}` : "";
    const size = item.size_bytes != null ? `Size: ${formatBytes(item.size_bytes)}` : "Size: n/a";
    const etag = item.etag ? `ETag: ${item.etag}` : "";
    const modified = item.last_modified ? `Last-Modified: ${item.last_modified}` : "";
    meta.textContent = [statusText, resolved, size, etag, modified].filter(Boolean).join(" · ");
    content.appendChild(title);
    content.appendChild(meta);
    row.appendChild(dot);
    row.appendChild(content);
    list.appendChild(row);
  });
};

const loadCheckpointCatalog = async (force = false) => {
  try {
    const response = await fetch(
      `${CHECKPOINT_CATALOG_ENDPOINT}${force ? "?refresh=1" : ""}`,
    );
    if (!response.ok) return;
    const data = await response.json();
    checkpointCatalog = data.items || [];
    renderCheckpointValidation(checkpointCatalog);
  } catch (error) {
    console.warn("Failed to load checkpoint catalog", error);
  }
};

const formatDelta = (value, baselineValue, higherIsBetter, unit, decimals = 1) => {
  if (value == null || baselineValue == null) return "";
  const delta = value - baselineValue;
  if (Number.isNaN(delta) || delta === 0) {
    return `<span class="delta neutral">Δ 0${unit}</span>`;
  }
  const isGood = higherIsBetter ? delta > 0 : delta < 0;
  const sign = delta > 0 ? "+" : "-";
  return `<span class="delta ${isGood ? "good" : "bad"}">Δ ${sign}${Math.abs(delta).toFixed(decimals)}${unit}</span>`;
};

const isEditableTarget = (target) =>
  target instanceof HTMLInputElement ||
  target instanceof HTMLTextAreaElement ||
  target?.isContentEditable;

const getFocusableElements = (container) => {
  if (!container) return [];
  return Array.from(
    container.querySelectorAll(
      'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  );
};

const trapFocus = (event, container) => {
  const focusable = getFocusableElements(container);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  const active = document.activeElement;
  if (event.shiftKey && active === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && active === last) {
    event.preventDefault();
    first.focus();
  }
};

const openOverlay = (overlayId) => {
  const overlay = document.getElementById(`${overlayId}-overlay`);
  if (!overlay) return;
  if (activeOverlay && activeOverlay !== overlayId) {
    closeOverlay(activeOverlay);
  }
  overlay.classList.remove("hidden");
  overlay.classList.add("open");
  overlay.setAttribute("aria-hidden", "false");
  activeOverlay = overlayId;
  lastOverlayFocus = document.activeElement;
  const drawer = overlay.querySelector(".drawer");
  const focusable = getFocusableElements(drawer);
  if (focusable.length) {
    focusable[0].focus();
  }
  const keyHandler = (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      closeOverlay(overlayId);
      return;
    }
    if (event.key === "Tab") {
      trapFocus(event, drawer);
    }
  };
  overlay.dataset.keyHandler = "true";
  overlay._keyHandler = keyHandler;
  document.addEventListener("keydown", keyHandler);
};

const closeOverlay = (overlayId) => {
  const overlay = document.getElementById(`${overlayId}-overlay`);
  if (!overlay) return;
  overlay.classList.remove("open");
  overlay.classList.add("hidden");
  overlay.setAttribute("aria-hidden", "true");
  activeOverlay = null;
  const keyHandler = overlay._keyHandler;
  if (keyHandler) {
    document.removeEventListener("keydown", keyHandler);
  }
  overlay._keyHandler = null;
  if (lastOverlayFocus && typeof lastOverlayFocus.focus === "function") {
    lastOverlayFocus.focus();
  }
};

const toggleOverlay = (overlayId) => {
  if (activeOverlay === overlayId) {
    closeOverlay(overlayId);
  } else {
    openOverlay(overlayId);
  }
};

const renderEngineBadge = (engine, fallbackReason) => {
  const engineValue = engine ?? "n/a";
  const isMock = engineValue === "mock";
  let badge = `<span class="engine-badge ${isMock ? "mock" : "ollama"}">${engineValue}</span>`;
  if (isMock && fallbackReason) {
    const reasonText = FALLBACK_REASONS[fallbackReason] || fallbackReason;
    badge += `<div class="fallback-reason">Fallback: ${reasonText}</div>`;
  }
  return badge;
};

const buildMetricsHtml = (metrics, baselineMetrics) => {
  if (!metrics) return `<span class="muted">No data</span>`;
  const ttft = formatMetric(metrics.ttft_ms, " ms");
  const tokS = formatMetric(metrics.tok_s, " tok/s");
  const total = formatMetric(metrics.total_ms, " ms");
  const tokens = metrics.tokens ?? "n/a";
  const ttftDelta = formatDelta(metrics.ttft_ms, baselineMetrics?.ttft_ms, false, " ms");
  const tokSDelta = formatDelta(metrics.tok_s, baselineMetrics?.tok_s, true, " tok/s");
  const totalDelta = formatDelta(metrics.total_ms, baselineMetrics?.total_ms, false, " ms");
  const engine = renderEngineBadge(metrics.engine, metrics.fallback_reason);
  return `
    <div><strong>Model:</strong> ${metrics.model ?? "n/a"}</div>
    <div><strong>Engine:</strong> ${engine}</div>
    <div><strong>TTFT:</strong> ${ttft} ${ttftDelta}</div>
    <div><strong>Gen tokens:</strong> ${tokens}</div>
    <div><strong>Tokens/s:</strong> ${tokS} ${tokSDelta}</div>
    <div><strong>Total:</strong> ${total} ${totalDelta}</div>
  `;
};

function updateMachineStatus(machine) {
  const dot = document.getElementById(`status-dot-${machine.machine_id}`);
  const text = document.getElementById(`status-text-${machine.machine_id}`);
  if (!dot || !text) return;

  // Compute reachability with robust fallback chain
  // Priority: agent_reachable (top-level) > capabilities.agent_reachable > reachable (legacy)
  let agentReachable;
  if (machine.agent_reachable !== undefined && machine.agent_reachable !== null) {
    agentReachable = machine.agent_reachable;
  } else if (machine.capabilities?.agent_reachable !== undefined && machine.capabilities?.agent_reachable !== null) {
    agentReachable = machine.capabilities.agent_reachable;
  } else if (machine.reachable !== undefined && machine.reachable !== null) {
    agentReachable = machine.reachable;
  } else {
    agentReachable = null; // Unknown state
  }

  dot.classList.remove("ready", "missing", "offline", "checking");

  // If reachability is unknown, show "Checking..."
  if (agentReachable === null) {
    dot.classList.add("checking");
    dot.title = "Checking agent status...";
    text.textContent = "Checking...";
  } else if (!agentReachable) {
    dot.classList.add("offline");
    dot.title = machine.error || "Agent offline";
    text.textContent = "Offline";
  } else if (!machine.has_selected_model) {
    dot.classList.add("missing");
    dot.title = `Missing model: ${machine.selected_model || "unknown"}`;
    text.textContent = "Missing model";
  } else {
    dot.classList.add("ready");
    dot.title = "Ready";
    text.textContent = "Ready";
  }
}

function updateModelFit(machine) {
  const fitEl = document.getElementById(`model-fit-${machine.machine_id}`);
  if (!fitEl) return;
  const fit = machine.model_fit || {};
  const label = fit.label || "unknown";
  fitEl.classList.remove("good", "marginal", "risk", "fail", "unknown");
  fitEl.classList.add(label);
  const badge = fitEl.querySelector(".fit-badge");
  const scoreEl = fitEl.querySelector(".fit-score");
  if (!badge || !scoreEl) return;
  if (label === "unknown") {
    badge.textContent = "Fit: --";
    scoreEl.textContent = "--";
    fitEl.title = "Model fit unavailable";
    return;
  }
  const score = fit.fit_score != null ? fit.fit_score.toFixed(1) : "--";
  badge.textContent = `Fit: ${label}`;
  scoreEl.textContent = score;
  const tooltip = [
    `Model size: ${formatBytes(fit.model_bytes)}`,
    `Usable ${fit.memory_label || "VRAM"}: ${formatBytes(fit.usable_vram_bytes)}`,
    `Estimated peak: ${formatBytes(fit.estimated_peak_bytes)}`,
    `Fit ratio: ${fit.fit_ratio ? fit.fit_ratio.toFixed(2) : "n/a"}`,
    "Suggested actions: reduce batch, use fp16, smaller model, or choose more VRAM.",
  ];
  fitEl.title = tooltip.join("\n");
}

const SPARKLINE_WIDTH = 120;
const SPARKLINE_HEIGHT = 24;

const buildSparklinePath = (values, width, height, maxValue) => {
  if (!values || values.length === 0) return "";
  const max = maxValue || Math.max(...values.filter((v) => v != null), 1);
  const step = values.length > 1 ? width / (values.length - 1) : width;
  let d = "";
  let started = false;
  values.forEach((value, index) => {
    if (value == null || Number.isNaN(value)) {
      started = false;
      return;
    }
    const x = index * step;
    const y = height - (Math.min(value, max) / max) * height;
    if (!started) {
      d += `M ${x.toFixed(2)} ${y.toFixed(2)} `;
      started = true;
    } else {
      d += `L ${x.toFixed(2)} ${y.toFixed(2)} `;
    }
  });
  return d.trim();
};

const renderSparkline = (svg, series, options = {}) => {
  if (!svg) return;
  const width = options.width || SPARKLINE_WIDTH;
  const height = options.height || SPARKLINE_HEIGHT;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  series.forEach((item) => {
    const pathData = buildSparklinePath(item.values, width, height, item.maxValue);
    if (!pathData) return;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", pathData);
    path.setAttribute("class", item.className);
    svg.appendChild(path);
  });
  if (options.title) svg.setAttribute("title", options.title);
};

const updateRuntimeMetrics = (machineId, metrics) => {
  const utilSvg = document.getElementById(`sparkline-util-${machineId}`);
  const memSvg = document.getElementById(`sparkline-mem-${machineId}`);
  const utilPlaceholder = document.getElementById(`sparkline-util-placeholder-${machineId}`);
  const memPlaceholder = document.getElementById(`sparkline-mem-placeholder-${machineId}`);
  const utilBlock = utilSvg?.closest(".sparkline-block");
  const memBlock = memSvg?.closest(".sparkline-block");

  if (!metrics) {
    utilPlaceholder?.classList.remove("hidden");
    memPlaceholder?.classList.remove("hidden");
    utilBlock?.classList.add("is-empty");
    memBlock?.classList.add("is-empty");
    return;
  }

  const cpu = metrics.cpu_pct || [];
  const gpu = metrics.gpu_pct || [];
  const vramUsed = metrics.vram_used_mib || [];
  const vramTotal = metrics.vram_total_mib || [];
  const systemMem = metrics.system_mem_used_mib || [];

  const cpuMax = 100;
  const gpuMax = 100;
  const memMax = Math.max(...vramTotal.filter((v) => v != null), ...systemMem.filter((v) => v != null), 1);

  utilPlaceholder?.classList.toggle("hidden", cpu.length > 0);
  utilBlock?.classList.toggle("is-empty", cpu.length === 0);
  renderSparkline(
    utilSvg,
    [
      { values: cpu, className: "cpu-line", maxValue: cpuMax },
      { values: gpu, className: "gpu-line", maxValue: gpuMax },
    ],
    { title: "CPU (solid) + GPU (dashed) utilization" },
  );

  const memValues = vramUsed.some((v) => v != null) ? vramUsed : systemMem;
  const memTitle = metrics.gpu_metrics_available
    ? "VRAM usage"
    : "Unified/system memory (GPU metrics unavailable)";
  memPlaceholder?.classList.toggle("hidden", memValues.length > 0);
  memBlock?.classList.toggle("is-empty", memValues.length === 0);
  renderSparkline(
    memSvg,
    [{ values: memValues, className: "mem-line", maxValue: memMax }],
    { title: memTitle },
  );

  const lastCpu = cpu.filter((v) => v != null).slice(-1)[0];
  const lastGpu = gpu.filter((v) => v != null).slice(-1)[0];
  const lastMem = memValues.filter((v) => v != null).slice(-1)[0];
  if (utilSvg) utilSvg.dataset.sparklineMeta = `CPU ${lastCpu ?? "n/a"}% | GPU ${lastGpu ?? "n/a"}%`;
  if (memSvg) memSvg.dataset.sparklineMeta = `Mem ${lastMem ?? "n/a"} MiB`;
};

const openSparklineModal = (machineId, type) => {
  const metrics = runtimeMetrics.get(machineId) || statusCache.get(machineId)?.runtime_metrics;
  if (!metrics) return;
  const modalLabel = document.getElementById("sparkline-modal-label");
  const modalMeta = document.getElementById("sparkline-modal-meta");
  const modalChart = document.getElementById("sparkline-modal-chart");
  const machineLabel = statusCache.get(machineId)?.label || machineId;
  const cpu = metrics.cpu_pct || [];
  const gpu = metrics.gpu_pct || [];
  const vramUsed = metrics.vram_used_mib || [];
  const vramTotal = metrics.vram_total_mib || [];
  const systemMem = metrics.system_mem_used_mib || [];
  const lastCpu = cpu.filter((v) => v != null).slice(-1)[0];
  const lastGpu = gpu.filter((v) => v != null).slice(-1)[0];
  const lastMem = (vramUsed.some((v) => v != null) ? vramUsed : systemMem).filter((v) => v != null).slice(-1)[0];
  if (type === "util") {
    const cpuMax = 100;
    const gpuMax = 100;
    if (modalLabel) modalLabel.textContent = `${machineLabel} • Utilization`;
    if (modalMeta) modalMeta.textContent = `CPU ${lastCpu ?? "n/a"}% | GPU ${lastGpu ?? "n/a"}%`;
    renderSparkline(
      modalChart,
      [
        { values: cpu, className: "cpu-line", maxValue: cpuMax },
        { values: gpu, className: "gpu-line", maxValue: gpuMax },
      ],
      { width: 480, height: 120 },
    );
  } else {
    const memValues = vramUsed.some((v) => v != null) ? vramUsed : systemMem;
    const memMax = Math.max(...vramTotal.filter((v) => v != null), ...systemMem.filter((v) => v != null), 1);
    if (modalLabel) modalLabel.textContent = `${machineLabel} • Memory`;
    if (modalMeta) modalMeta.textContent = `Mem ${lastMem ?? "n/a"} MiB`;
    renderSparkline(modalChart, [{ values: memValues, className: "mem-line", maxValue: memMax }], {
      width: 480,
      height: 120,
    });
  }
  openOverlay("sparkline");
};

function updateSyncButton(machine) {
  const btn = document.getElementById(`sync-${machine.machine_id}`);
  if (!btn) return;
  const missingRequired = machine.missing_required || {};
  const missingCount = Object.values(missingRequired)
    .flat()
    .filter(Boolean).length;
  const syncing = syncState.get(machine.machine_id)?.active;
  if (missingCount > 0) {
    btn.classList.remove("hidden");
    btn.disabled = Boolean(syncing);
    btn.title = `Missing required models: ${Object.values(missingRequired).flat().join(", ")}`;
  } else {
    btn.classList.add("hidden");
  }
}

function applyStatusResponse(data) {
  statusCache.clear();
  (data.machines || []).forEach((machine) => {
    statusCache.set(machine.machine_id, machine);
    updateMachineStatus(machine);
    updateModelFit(machine);
    if (machine.runtime_metrics) {
      runtimeMetrics.set(machine.machine_id, machine.runtime_metrics);
    }
    updateRuntimeMetrics(machine.machine_id, machine.runtime_metrics);
    updateSyncButton(machine);
  });
  updatePreflightBanner();
}

async function fetchStatus() {
  const selectedModel = document.getElementById("model")?.value;
  const numCtx = document.getElementById("num_ctx")?.value;
  const params = new URLSearchParams();
  if (selectedModel) params.set("model", selectedModel);
  if (numCtx) params.set("num_ctx", numCtx);
  try {
    const response = await fetch(`/api/status?${params.toString()}`);
    const data = await response.json();
    applyStatusResponse(data);
    return data;
  } catch (error) {
    console.error("Failed to fetch status", error);
    return null;
  }
}

function getPreflightStatus() {
  const blocked = [];
  const ready = [];

  statusCache.forEach((machine, machineId) => {
    if (!machine.reachable) {
      blocked.push({ machine_id: machineId, label: machine.label, reason: "agent offline" });
    } else if (!machine.has_selected_model) {
      blocked.push({
        machine_id: machineId,
        label: machine.label,
        reason: `missing model ${machine.selected_model}`,
      });
    } else {
      ready.push({ machine_id: machineId, label: machine.label });
    }
  });

  return { blocked, ready };
}

function updatePreflightBanner() {
  const banner = document.getElementById("preflight-banner");
  if (!banner) return;

  const { blocked, ready } = getPreflightStatus();

  if (blocked.length === 0) {
    banner.classList.add("hidden");
    banner.textContent = "";
  } else {
    banner.classList.remove("hidden");
    const reasons = blocked.map((b) => `${b.reason} (${b.label || b.machine_id})`).join(", ");
    banner.textContent = `${blocked.length} machine(s) blocked: ${reasons}`;
    banner.classList.toggle("error", ready.length === 0);
  }
}

function updateSyncUI(machineId, payload) {
  const progress = document.getElementById(`sync-progress-${machineId}`);
  const fill = document.getElementById(`sync-progress-fill-${machineId}`);
  const text = document.getElementById(`sync-progress-text-${machineId}`);
  const log = document.getElementById(`sync-progress-log-${machineId}`);
  if (!progress || !fill || !text || !log) return;

  progress.classList.remove("hidden");
  const state = syncState.get(machineId) || { logs: [], active: true };
  state.active = true;
  if (payload?.percent != null) {
    progress.querySelector(".progress-bar")?.classList.remove("indeterminate");
    fill.style.width = `${payload.percent}%`;
  } else {
    progress.querySelector(".progress-bar")?.classList.add("indeterminate");
    fill.style.width = "40%";
  }

  const message = payload?.message || payload?.phase || "Syncing";
  const model = payload?.model ? ` ${payload.model}` : "";
  text.textContent = `${message}${model}`;

  if (payload?.message) {
    state.logs.unshift(`${payload.message}${model}`);
    state.logs = state.logs.slice(0, 3);
    log.innerHTML = state.logs.map((entry) => `<li>${entry}</li>`).join("");
  }
  syncState.set(machineId, state);
}

function completeSyncUI(machineId, message, isError = false) {
  const progress = document.getElementById(`sync-progress-${machineId}`);
  const fill = document.getElementById(`sync-progress-fill-${machineId}`);
  const text = document.getElementById(`sync-progress-text-${machineId}`);
  const log = document.getElementById(`sync-progress-log-${machineId}`);
  if (!progress || !fill || !text || !log) return;

  progress.classList.remove("hidden");
  progress.querySelector(".progress-bar")?.classList.remove("indeterminate");
  fill.style.width = "100%";
  text.textContent = message || (isError ? "Sync failed" : "Up to date");
  const state = syncState.get(machineId) || { logs: [], active: false };
  state.active = false;
  syncState.set(machineId, state);
  if (isError) {
    log.innerHTML = `<li>${message || "Sync failed"}</li>`;
  }
}

function initSyncButtons(machines) {
  machines.forEach((machine) => {
    const btn = document.getElementById(`sync-${machine.machine_id}`);
    if (!btn) return;
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      updateSyncUI(machine.machine_id, { message: "Starting sync..." });
      try {
        const response = await fetch(`/api/machines/${machine.machine_id}/sync`, { method: "POST" });
        const data = await response.json();
        if (!response.ok || data.error) {
          throw new Error(data.error || "Sync request failed");
        }
        if (!data.sync_id) {
          completeSyncUI(machine.machine_id, data.message || "Up to date");
          await fetchStatus();
        }
      } catch (error) {
        completeSyncUI(machine.machine_id, error.message, true);
      } finally {
        btn.disabled = false;
      }
    });
  });
}

const setPaneContent = (machineId, { outText, metricsHtml, isMock }) => {
  const pane = paneMap.get(machineId);
  if (!pane) return;
  pane.out.textContent = outText;
  pane.metrics.innerHTML = metricsHtml;
  const paneEl = document.getElementById(`pane-${machineId}`);
  if (paneEl) {
    paneEl.classList.toggle("mock-warning", Boolean(isMock));
  }
};

const updateLivePane = (machineId, { outText, metricsHtml, isMock }) => {
  liveState.set(machineId, { outText, metricsHtml, isMock });
  if (viewingMode !== "live") return;
  setPaneContent(machineId, { outText, metricsHtml, isMock });
};

const refreshLiveDisplay = () => {
  paneMap.forEach((_, machineId) => {
    const state = liveState.get(machineId);
    if (!state) return;
    updateLivePane(machineId, state);
  });
};

const setBaselineRunId = async (runId) => {
  baselineRunId = runId;
  if (runId) {
    localStorage.setItem(BASELINE_KEY, runId);
  } else {
    localStorage.removeItem(BASELINE_KEY);
  }
  await loadBaselineRun();
  renderRecentRuns();
  if (viewingMode === "history" && currentRunData) {
    renderRunToPanes(currentRunData);
  } else {
    refreshLiveDisplay();
  }
};

const loadBaselineRun = async () => {
  if (!baselineRunId) {
    baselineRun = null;
    return;
  }
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(baselineRunId)}`);
    if (!response.ok) {
      baselineRun = null;
      return;
    }
    baselineRun = await response.json();
  } catch (error) {
    console.error("Failed to load baseline run", error);
    baselineRun = null;
  }
};

const getBaselineMachine = (machineId) => {
  if (!baselineRun || !Array.isArray(baselineRun.machines)) return null;
  return baselineRun.machines.find((machine) => machine.machine_id === machineId) || null;
};

const renderRunBanner = (run) => {
  const banner = document.getElementById("run-view-banner");
  const title = document.getElementById("run-view-title");
  const subtitle = document.getElementById("run-view-subtitle");
  const warning = document.getElementById("run-warning-banner");
  if (!banner || !title || !subtitle || !warning) return;

  if (!run) {
    banner.classList.add("hidden");
    return;
  }

  banner.classList.remove("hidden");
  title.textContent = `Viewing past run ${run.run_id}`;
  subtitle.textContent = `${formatTimestamp(run.timestamp)} · ${run.model ?? "n/a"}`;
  const hasMock = (run.machines || []).some((machine) => machine.engine === "mock");
  warning.classList.toggle("hidden", !hasMock);
};

const renderRunToPanes = (run) => {
  if (!run) return;
  currentRunData = run;
  renderRunBanner(run);
  viewingMode = "history";
  viewingRunId = run.run_id;

  paneMap.forEach(({ out, metrics }, machineId) => {
    const entry = (run.machines || []).find((machine) => machine.machine_id === machineId);
    if (!entry) {
      out.textContent = "No output stored for past runs.";
      metrics.innerHTML = `<span class="muted">No data</span>`;
      setPaneContent(machineId, {
        outText: out.textContent,
        metricsHtml: metrics.innerHTML,
        isMock: false,
      });
      return;
    }

    if (entry.status === "skipped") {
      out.textContent = "Skipped for this run.";
      metrics.innerHTML = `<span class="muted">Skipped</span>`;
      setPaneContent(machineId, {
        outText: out.textContent,
        metricsHtml: metrics.innerHTML,
        isMock: false,
      });
      return;
    }

    if (entry.status === "error") {
      out.textContent = entry.error || "Failed to start.";
      metrics.innerHTML = `<span class="muted">Failed to start</span>`;
      setPaneContent(machineId, {
        outText: out.textContent,
        metricsHtml: metrics.innerHTML,
        isMock: false,
      });
      return;
    }

    const metricsData = {
      model: entry.model ?? run.model,
      engine: entry.engine,
      fallback_reason: entry.fallback_reason,
      ttft_ms: entry.ttft_ms,
      tok_s: entry.tok_s,
      total_ms: entry.total_ms,
      tokens: entry.tokens,
    };
    const baselineMetrics = getBaselineMachine(machineId);
    const metricsHtml = buildMetricsHtml(metricsData, baselineMetrics);
    out.textContent = "Output not stored for past runs.";
    setPaneContent(machineId, {
      outText: out.textContent,
      metricsHtml,
      isMock: entry.engine === "mock",
    });
  });
};

const returnToLive = () => {
  viewingMode = "live";
  viewingRunId = null;
  currentRunData = null;
  const banner = document.getElementById("run-view-banner");
  if (banner) banner.classList.add("hidden");
  refreshLiveDisplay();
};

const fetchRecentRuns = async () => {
  try {
    const response = await fetch(`/api/runs?limit=${RUN_LIST_LIMIT}`);
    if (!response.ok) return;
    recentRuns = await response.json();
    renderRecentRuns();
    updateHistoryBadge();
  } catch (error) {
    console.error("Failed to fetch runs", error);
  }
};

const updateHistoryBadge = () => {
  const badge = document.getElementById("history-count");
  if (!badge) return;
  const count = recentRuns.length;
  badge.textContent = String(count);
  if (count > 0) {
    badge.classList.remove("hidden");
  } else {
    badge.classList.add("hidden");
  }
};

const deleteRun = async (runId) => {
  const confirmed = window.confirm("Delete this run permanently? This cannot be undone.");
  if (!confirmed) return;
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
    if (!response.ok) {
      showToast("Failed to delete run.", "error");
      return;
    }
    recentRuns = recentRuns.filter((run) => run.run_id !== runId);
    if (baselineRunId && baselineRunId === runId) {
      await setBaselineRunId(null);
      showToast("Baseline cleared because the run was deleted.", "info");
    }
    renderRecentRuns();
    updateHistoryBadge();
  } catch (error) {
    console.error("Failed to delete run", error);
    showToast("Failed to delete run.", "error");
  }
};

const renderRecentRuns = () => {
  const list = document.getElementById("recent-runs-list");
  if (!list) return;
  list.innerHTML = "";

  if (!recentRuns.length) {
    const empty = document.createElement("div");
    empty.className = "recent-empty";
    empty.textContent = "No runs yet.";
    list.appendChild(empty);
    return;
  }

  recentRuns.forEach((run) => {
    const item = document.createElement("div");
    item.className = "recent-run-item";
    item.dataset.runId = run.run_id;

    const badges = [];
    if (run.has_mock) {
      badges.push('<span class="run-badge warning">Mock engine</span>');
    }
    if (run.type === "image") {
      badges.push('<span class="run-badge image">Image</span>');
    } else {
      badges.push('<span class="run-badge inference">Inference</span>');
    }
    if (baselineRunId && baselineRunId === run.run_id) {
      badges.push('<span class="run-badge baseline">Baseline</span>');
    }
    const warningText = run.has_mock
      ? '<div class="run-warning-text">Mock engine used on at least one machine.</div>'
      : "";

    item.innerHTML = `
      <button class="run-delete-button" type="button" aria-label="Delete run" title="Delete run">🗑️</button>
      <div class="recent-run-header">
        <div>
          <div class="recent-run-title">${run.model ?? "n/a"}</div>
          <div class="recent-run-meta">${formatTimestamp(run.timestamp)}</div>
        </div>
        <div class="recent-run-badges">${badges.join(" ")}</div>
      </div>
      <div class="recent-run-prompt">${run.prompt_preview ?? ""}</div>
      ${warningText}
      <div class="recent-run-actions">
        <button class="btn-secondary btn-small" data-action="view">View</button>
        <button class="btn-secondary btn-small" data-action="pin" title="Pin baseline">Baseline</button>
        <button class="btn-secondary btn-small" data-action="csv">CSV</button>
        <button class="btn-secondary btn-small" data-action="json">JSON</button>
      </div>
    `;

    const deleteButton = item.querySelector(".run-delete-button");
    deleteButton?.addEventListener("click", async (event) => {
      event.stopPropagation();
      await deleteRun(run.run_id);
    });

    item.querySelectorAll("button").forEach((button) => {
      if (button.classList.contains("run-delete-button")) return;
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        const action = button.dataset.action;
        if (action === "view") {
          await loadRun(run.run_id);
        } else if (action === "pin") {
          await setBaselineRunId(run.run_id);
        } else if (action === "csv") {
          window.location.href = `/api/runs/${encodeURIComponent(run.run_id)}/export.csv`;
        } else if (action === "json") {
          window.location.href = `/api/runs/${encodeURIComponent(run.run_id)}/export.json`;
        }
      });
    });

    item.addEventListener("click", async () => {
      await loadRun(run.run_id);
    });

    list.appendChild(item);
  });
};

const loadRun = async (runId) => {
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    if (!response.ok) return;
    const run = await response.json();
    if (run.type === "image") {
      window.location.href = `/image?run_id=${encodeURIComponent(runId)}`;
      return;
    }
    renderRunToPanes(run);
  } catch (error) {
    console.error("Failed to load run", error);
  }
};

const loadModelPolicy = async () => {
  const response = await fetch(MODEL_POLICY_ENDPOINT);
  if (!response.ok) {
    throw new Error("Failed to load model policy");
  }
  return response.json();
};

const updateModelOptions = (models) => {
  const select = document.getElementById("model");
  if (!select) return;
  const currentValue = select.value;
  select.innerHTML = "";
  (models || []).forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    select.appendChild(option);
  });
  if (models?.includes(currentValue)) {
    select.value = currentValue;
  } else if (models?.length) {
    select.value = models[0];
  }
};

const setPolicyFeedback = (summary, error) => {
  const errorEl = document.getElementById("model-policy-error");
  const summaryEl = document.getElementById("model-policy-summary");
  if (errorEl) {
    errorEl.textContent = error || "";
    errorEl.classList.toggle("hidden", !error);
  }
  if (summaryEl) {
    summaryEl.textContent = summary || "";
    summaryEl.classList.toggle("hidden", !summary);
  }
};

socket.on("status", (msg) => {
  document.getElementById("status").innerText = msg.ok ? "Connected" : "Not connected";
});

// Handle run lifecycle events for adaptive polling
socket.on("run_lifecycle", (event) => {
  if (!event || !event.type) return;
  if (event.type === "run_start") {
    setRunActive(true, event.run_id);
    console.log("Run started:", event.run_id);
  } else if (event.type === "run_end") {
    setRunActive(false, event.run_id);
    console.log("Run ended:", event.run_id);
  }
});

socket.on("connect", async () => {
  try {
    const response = await fetch("/api/machines");
    const machines = await response.json();
    paneMap.clear();
    machines.forEach((machine) => {
      const out = document.getElementById(`out-${machine.machine_id}`);
      const metrics = document.getElementById(`metrics-${machine.machine_id}`);
      if (out && metrics) {
        paneMap.set(machine.machine_id, { out, metrics });
      }
      const utilSvg = document.getElementById(`sparkline-util-${machine.machine_id}`);
      const memSvg = document.getElementById(`sparkline-mem-${machine.machine_id}`);
      if (utilSvg) {
        utilSvg.addEventListener("click", () => openSparklineModal(machine.machine_id, "util"));
      }
      if (memSvg) {
        memSvg.addEventListener("click", () => openSparklineModal(machine.machine_id, "mem"));
      }
    });
    initSyncButtons(machines);
    await fetchStatus();
    await loadBaselineRun();
    await fetchRecentRuns();
    // Initialize run state and start adaptive polling
    await fetchActiveRuns();
  } catch (error) {
    console.error("Failed to load machines", error);
  }
});

socket.on("agent_event", (evt) => {
  if (!evt || !evt.machine_id) return;
  const pane = paneMap.get(evt.machine_id);
  if (!pane) return;

  if (evt.type === "llm_token") {
    const text = evt.payload?.text ?? "";
    const current = liveOutput.get(evt.machine_id) || "";
    const updated = current + text;
    liveOutput.set(evt.machine_id, updated);
    updateLivePane(evt.machine_id, {
      outText: updated,
      metricsHtml: pane.metrics.innerHTML,
      isMock: false,
    });
  }

  if (evt.type === "job_done") {
    const payload = evt.payload || {};
    const engine = payload.engine ?? "n/a";
    const isMock = engine === "mock";

    const metricsData = {
      model: payload.model,
      engine,
      fallback_reason: payload.fallback_reason,
      ttft_ms: payload.ttft_ms,
      tok_s: payload.gen_tokens_per_s,
      total_ms: payload.total_ms,
      tokens: payload.gen_tokens,
    };
    liveMetrics.set(evt.machine_id, metricsData);

    const baselineMetrics = getBaselineMachine(evt.machine_id);
    const metricsHtml = buildMetricsHtml(metricsData, baselineMetrics);
    updateLivePane(evt.machine_id, {
      outText: liveOutput.get(evt.machine_id) || pane.out.textContent,
      metricsHtml,
      isMock,
    });

    fetchRecentRuns();
  }

  if (evt.type === "sync_started") {
    updateSyncUI(evt.machine_id, { message: "Sync started" });
  }
  if (evt.type === "sync_progress") {
    updateSyncUI(evt.machine_id, evt.payload || {});
  }
  if (evt.type === "sync_done") {
    completeSyncUI(evt.machine_id, evt.payload?.message || "Up to date");
    fetchStatus();
  }
  if (evt.type === "sync_error") {
    completeSyncUI(evt.machine_id, evt.payload?.message || "Sync failed", true);
  }

  if (evt.type === "runtime_metrics_update") {
    const metrics = evt.payload || {};
    runtimeMetrics.set(evt.machine_id, metrics);
    updateRuntimeMetrics(evt.machine_id, metrics);
  }
});

socket.on("llm_jobs_started", (payload) => {
  const results = Array.isArray(payload) ? payload : payload?.results;
  handleJobResults(results);
});

const runButton = document.getElementById("run");
const generateSampleButton = document.getElementById("generate-sample");
const historyButton = document.getElementById("btn-history");
const settingsButton = document.getElementById("btn-settings");
const optionsToggle = document.getElementById("options-toggle");
const SAMPLE_PROMPT_COOLDOWN_MS = 10000;
let samplePromptCooldownUntil = 0;

const setOptionsExpanded = (expanded) => {
  document.body.classList.toggle("options-expanded", expanded);
  document.body.classList.toggle("options-collapsed", !expanded);
  localStorage.setItem(OPTIONS_STORAGE_KEY, expanded ? "true" : "false");
};

const initOptionsState = () => {
  const stored = localStorage.getItem(OPTIONS_STORAGE_KEY);
  const expanded = stored === "true";
  setOptionsExpanded(expanded);
};

initOptionsState();
optionsToggle?.addEventListener("click", () => {
  const isExpanded = document.body.classList.contains("options-expanded");
  setOptionsExpanded(!isExpanded);
});

const startRun = async () => {
  const { blocked, ready } = getPreflightStatus();

  if (ready.length === 0) {
    showToast("No ready machines available.", "info");
    return;
  }

  const payload = {
    model: document.getElementById("model").value,
    prompt: document.getElementById("prompt").value,
    max_tokens: parseInt(document.getElementById("max_tokens").value, 10),
    num_ctx: parseInt(document.getElementById("num_ctx").value, 10),
    temperature: parseFloat(document.getElementById("temperature").value),
    repeat: parseInt(document.getElementById("repeat").value, 10),
    machine_ids: ready.map((m) => m.machine_id),
  };

  paneMap.forEach(({ out, metrics }, machineId) => {
    // Check if this machine is blocked
    const isBlocked = blocked.some((b) => b.machine_id === machineId);
    if (isBlocked) {
      out.textContent = "Skipped (not ready)";
      metrics.innerHTML = `<span class="muted">Machine not ready</span>`;
      liveOutput.set(machineId, out.textContent);
    } else {
      out.textContent = "Starting job…";
      metrics.innerHTML = `<span class="muted">Queued…</span>`;
      liveOutput.set(machineId, "");
    }
    updateLivePane(machineId, {
      outText: out.textContent,
      metricsHtml: metrics.innerHTML,
      isMock: false,
    });
  });

  try {
    const response = await fetch("/api/start_llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    liveRunId = data.run_id;
    handleJobResults(data.results || data);
    fetchRecentRuns();
    returnToLive();
  } catch (error) {
    console.error("Failed to start jobs", error);
  }
};

// Run button with preflight validation
runButton?.addEventListener("click", async () => {
  await startRun();
});

historyButton?.addEventListener("click", async () => {
  await fetchRecentRuns();
  toggleOverlay("history");
});

settingsButton?.addEventListener("click", async () => {
  try {
    const data = await loadModelPolicy();
    const editor = document.getElementById("model-policy-editor");
    if (editor) {
      editor.value = (data.models || []).join("\n");
    }
    const comfyResponse = await fetch(COMFY_SETTINGS_ENDPOINT);
    if (comfyResponse.ok) {
      const comfy = await comfyResponse.json();
      const baseInput = document.getElementById("comfy-base-url");
      const modelsInput = document.getElementById("comfy-models-path");
      const cacheInput = document.getElementById("comfy-cache-path");
      const checkpointsInput = document.getElementById("comfy-checkpoints");
      if (baseInput) baseInput.value = comfy.base_url || "";
      if (modelsInput) modelsInput.value = comfy.models_path || "";
      if (cacheInput) cacheInput.value = comfy.central_cache_path || "";
      if (checkpointsInput) {
        checkpointsInput.value = (comfy.comfyui_checkpoints || comfy.checkpoint_urls || []).join("\n");
      }
    }
    await loadCheckpointCatalog();
    // Load polling settings into form
    const idlePollInput = document.getElementById("idle-poll-interval");
    const activePollInput = document.getElementById("active-poll-interval");
    if (idlePollInput) idlePollInput.value = Math.round(pollingConfig.idlePollIntervalMs / 1000);
    if (activePollInput) activePollInput.value = pollingConfig.activePollIntervalMs;
    setPolicyFeedback("", "");
    toggleOverlay("settings");
  } catch (error) {
    alert("Could not load settings. Try again.");
  }
});

document.getElementById("refresh-runs")?.addEventListener("click", async () => {
  await fetchRecentRuns();
});

document.getElementById("recheck-checkpoints")?.addEventListener("click", async () => {
  await loadCheckpointCatalog(true);
});

document.querySelectorAll("[data-overlay-close]").forEach((button) => {
  button.addEventListener("click", () => {
    const overlayId = button.getAttribute("data-overlay-close");
    if (overlayId) closeOverlay(overlayId);
  });
});

document.getElementById("settings-cancel")?.addEventListener("click", () => {
  closeOverlay("settings");
});

document.getElementById("settings-save")?.addEventListener("click", async () => {
  const editor = document.getElementById("model-policy-editor");
  if (!(editor instanceof HTMLTextAreaElement)) return;
  const models = editor.value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const baseInput = document.getElementById("comfy-base-url");
  const modelsInput = document.getElementById("comfy-models-path");
  const cacheInput = document.getElementById("comfy-cache-path");
  const checkpointsInput = document.getElementById("comfy-checkpoints");
  const checkpointUrls =
    checkpointsInput instanceof HTMLTextAreaElement
      ? checkpointsInput.value
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean)
      : [];
  const button = document.getElementById("settings-save");
  if (button) button.disabled = true;
  setPolicyFeedback("", "");
  try {
    const response = await fetch(MODEL_POLICY_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ models }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.error || "Failed to save policy");
    }
    updateModelOptions(data.models || models);
    await fetch(COMFY_SETTINGS_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_url: baseInput?.value || "",
        models_path: modelsInput?.value || "",
        central_cache_path: cacheInput?.value || "",
        comfyui_checkpoints: checkpointUrls,
      }),
    });
    // Save polling settings from form
    const idlePollInput = document.getElementById("idle-poll-interval");
    const activePollInput = document.getElementById("active-poll-interval");
    if (idlePollInput) {
      pollingConfig.idlePollIntervalMs = parseInt(idlePollInput.value, 10) * 1000;
    }
    if (activePollInput) {
      pollingConfig.activePollIntervalMs = parseInt(activePollInput.value, 10);
    }
    savePollingSettings();
    restartPolling();
    if (data.missing && Object.keys(data.missing).length > 0) {
      const lines = Object.entries(data.missing).map(
        ([model, machines]) => `${model}: missing on ${machines.join(", ")}`,
      );
      setPolicyFeedback(lines.join(" • "), "");
    } else {
      setPolicyFeedback("Policy saved.", "");
    }
    await fetchStatus();
    closeOverlay("settings");
  } catch (error) {
    setPolicyFeedback("", error.message || "Failed to save policy");
  } finally {
    if (button) button.disabled = false;
  }
});

generateSampleButton?.addEventListener("click", async (event) => {
  const btn = event.currentTarget;
  if (!(btn instanceof HTMLButtonElement)) return;
  const now = Date.now();
  if (now < samplePromptCooldownUntil || btn.disabled) {
    return;
  }
  samplePromptCooldownUntil = now + SAMPLE_PROMPT_COOLDOWN_MS;
  btn.disabled = true;
  btn.innerText = "Generating…";
  try {
    const modelValue = document.getElementById("model")?.value;
    const res = await fetch("/api/generate_sample_prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelValue }),
    });
    let data = null;
    if (res.ok) {
      data = await res.json();
    } else {
      try {
        data = await res.json();
      } catch (error) {
        data = null;
      }
      if (!data?.prompt) {
        throw new Error("failed");
      }
    }
    const promptBox = document.querySelector(".prompt");
    if (promptBox) {
      promptBox.value = data.prompt;
      promptBox.focus();
      promptBox.selectionStart = promptBox.selectionEnd = promptBox.value.length;
    }
  } catch (error) {
    alert("Could not generate a sample prompt. Try again.");
  } finally {
    btn.disabled = false;
    btn.innerText = "Generate Sample";
  }
});

document.addEventListener("keydown", (event) => {
  if (isEditableTarget(event.target)) return;
  if (event.key === "r" || event.key === "R") {
    event.preventDefault();
    fetchRecentRuns();
    toggleOverlay("history");
  }
  if (event.key === "s" || event.key === "S") {
    event.preventDefault();
    settingsButton?.click();
  }
});

const promptEl = document.getElementById("prompt");
promptEl?.addEventListener("keydown", async (event) => {
  if (event.isComposing) return;
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (runButton?.disabled) return;
    await startRun();
  }
});

// Refresh status button
document.getElementById("refresh-caps")?.addEventListener("click", async () => {
  const btn = document.getElementById("refresh-caps");
  if (btn) btn.disabled = true;
  await fetchStatus();
  if (btn) btn.disabled = false;
});

// Update status when model selection changes
document.getElementById("model")?.addEventListener("change", () => {
  fetchStatus();
});

// Run view banner actions
document.getElementById("run-view-return")?.addEventListener("click", () => {
  returnToLive();
});

document.getElementById("run-view-pin")?.addEventListener("click", async () => {
  if (!viewingRunId) return;
  await setBaselineRunId(viewingRunId);
});

document.getElementById("run-view-export-csv")?.addEventListener("click", () => {
  if (!viewingRunId) return;
  window.location.href = `/api/runs/${encodeURIComponent(viewingRunId)}/export.csv`;
});

document.getElementById("run-view-export-json")?.addEventListener("click", () => {
  if (!viewingRunId) return;
  window.location.href = `/api/runs/${encodeURIComponent(viewingRunId)}/export.json`;
});

// Initialize adaptive polling (replaces fixed setInterval)
// Load polling settings and start adaptive polling
loadPollingSettings();
scheduleStatusPoll();

const params = new URLSearchParams(window.location.search);
const runId = params.get("run_id");
if (runId) {
  loadRun(runId);
}
