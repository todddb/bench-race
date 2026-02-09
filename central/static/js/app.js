const socket = io();
const paneMap = new Map();
const statusCache = new Map();
const syncState = new Map();
const liveOutput = new Map();
const liveMetrics = new Map();
const runtimeMetrics = new Map();
const liveState = new Map();
const outputScrollState = new Map();
const RUN_LIST_LIMIT = 20;
const BASELINE_KEY = "bench-race-baseline-run";
const MODEL_POLICY_ENDPOINT = "/api/settings/model_policy";
const COMFY_SETTINGS_ENDPOINT = "/api/settings/comfy";
const CHECKPOINT_CATALOG_ENDPOINT = "/api/image/checkpoints";
const OPTIONS_STORAGE_KEY = "bench-race-options-expanded";
const POLLING_SETTINGS_KEY = "bench-race-polling-settings";
const COMPUTE_SETTINGS_KEY = "bench-race-compute-settings";
const DEBUG_UI = false;
const MODE = document.body?.dataset?.mode || "inference";

// Adaptive polling configuration
const DEFAULT_POLLING_CONFIG = {
  idlePollIntervalMs: 30000,   // 30s when no runs active
  activePollIntervalMs: 2000,  // 2s when runs active
  uiUpdateThrottleMs: 500,     // Throttle UI updates to max 2/sec
};
let pollingConfig = { ...DEFAULT_POLLING_CONFIG };

// Auto N presets are persisted per browser to tune compute pacing.
const DEFAULT_COMPUTE_SETTINGS = {
  autoN: {
    segmented_sieve: 100000000,
    simple_sieve: 40000000,
    trial_division: 4000000,
  },
  progressIntervalS: 1.0,
};
let computeSettings = { ...DEFAULT_COMPUTE_SETTINGS };

// Global run state tracking
let isRunActive = false;
const activeRunIds = new Set();
const activeRunMachinesById = new Map();
const activeRunMachineIds = new Set();
let statusPollTimer = null;
let lastUiUpdate = 0;
let runTrackingActive = false;
let runStartTs = null;
let runEndTs = null;
let runSamples = new Map();
let runDone = new Set();
let runMachines = new Set();

let baselineRunId = localStorage.getItem(BASELINE_KEY);
let baselineRun = null;
let viewingMode = "live";
let viewingRunId = null;
let currentRunData = null;
let liveRunId = null;
let recentRuns = [];
let selectMode = false;
let selectedRunIds = new Set();
let activeOverlay = null;
let lastOverlayFocus = null;
let checkpointCatalog = [];
let computeNManuallyEdited = false;
let currentComputeRepeats = 1;
const computeOutputState = new Map();

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

const clampNumber = (value, min, max) => Math.min(Math.max(value, min), max);

const loadComputeSettings = () => {
  try {
    const saved = localStorage.getItem(COMPUTE_SETTINGS_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        computeSettings = {
          ...DEFAULT_COMPUTE_SETTINGS,
          ...parsed,
          autoN: { ...DEFAULT_COMPUTE_SETTINGS.autoN, ...(parsed.autoN || {}) },
        };
      } else {
        console.warn("Compute settings value is not an object, using defaults");
      }
    }
    computeSettings.progressIntervalS = clampNumber(
      Number(computeSettings.progressIntervalS) || DEFAULT_COMPUTE_SETTINGS.progressIntervalS,
      0.1,
      60,
    );
    computeSettings.autoN = {
      segmented_sieve: Math.max(10, Number(computeSettings.autoN?.segmented_sieve) || DEFAULT_COMPUTE_SETTINGS.autoN.segmented_sieve),
      simple_sieve: Math.max(10, Number(computeSettings.autoN?.simple_sieve) || DEFAULT_COMPUTE_SETTINGS.autoN.simple_sieve),
      trial_division: Math.max(10, Number(computeSettings.autoN?.trial_division) || DEFAULT_COMPUTE_SETTINGS.autoN.trial_division),
    };
  } catch (e) {
    console.warn("Failed to load compute settings, using defaults:", e);
    computeSettings = { ...DEFAULT_COMPUTE_SETTINGS, autoN: { ...DEFAULT_COMPUTE_SETTINGS.autoN } };
  }
};

const saveComputeSettings = () => {
  try {
    localStorage.setItem(COMPUTE_SETTINGS_KEY, JSON.stringify(computeSettings));
  } catch (e) {
    console.warn("Failed to save compute settings:", e);
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

const rebuildActiveRunMachineIds = () => {
  activeRunMachineIds.clear();
  activeRunMachinesById.forEach((machineIds) => {
    machineIds.forEach((machineId) => activeRunMachineIds.add(machineId));
  });
};

const updateAllMachineStatuses = () => {
  statusCache.forEach((machine) => updateMachineStatus(machine));
};

const setActiveRunMachines = (runId, machineIds = []) => {
  if (!runId) return;
  activeRunMachinesById.set(runId, new Set(machineIds));
  rebuildActiveRunMachineIds();
  updateAllMachineStatuses();
};

const clearActiveRunMachines = (runId) => {
  if (!runId) return;
  activeRunMachinesById.delete(runId);
  rebuildActiveRunMachineIds();
  updateAllMachineStatuses();
};

const beginRunTracking = (machineIds = []) => {
  runSamples = new Map();
  runDone = new Set();
  runMachines = new Set(machineIds);
  runStartTs = performance.now();
  runEndTs = null;
  runTrackingActive = true;
  renderAllSparklines();
};

const endRunTracking = () => {
  if (!runStartTs || runEndTs) return;
  runEndTs = performance.now();
  runTrackingActive = false;
  renderAllSparklines();
};

const markRunMachineDone = (machineId) => {
  if (!runTrackingActive || runDone.has(machineId)) return;
  runDone.add(machineId);
  if (runMachines.size > 0 && runDone.size >= runMachines.size) {
    endRunTracking();
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
    activeRunMachinesById.clear();
    if (data.runs) {
      Object.entries(data.runs).forEach(([runId, info]) => {
        const machineIds = info?.machine_ids || [];
        activeRunMachinesById.set(runId, new Set(machineIds));
      });
    }
    rebuildActiveRunMachineIds();
    updateAllMachineStatuses();
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

const showToast = (message, type = "info", onClick = null) => {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  if (onClick) {
    toast.classList.add("clickable");
    toast.title = "Click for details";
    toast.addEventListener("click", () => {
      onClick();
      toast.remove();
    });
  }
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

const formatCount = (value) => {
  if (value == null || Number.isNaN(value)) return "n/a";
  return Number(value).toLocaleString();
};

const formatSeconds = (ms) => {
  if (ms == null || Number.isNaN(ms)) return "n/a";
  return `${(Number(ms) / 1000).toFixed(2)} s`;
};

const buildProgressBar = (percent) => {
  const clamped = Math.max(0, Math.min(100, Number(percent) || 0));
  const width = 24;
  const filled = Math.round((clamped / 100) * width);
  const bar = `${"=".repeat(filled)}${"-".repeat(Math.max(0, width - filled))}`;
  return `Progress: [${bar}] ${clamped.toFixed(0)}%`;
};

const buildComputeMetricsHtml = (metrics) => {
  if (!metrics) return `<span class="muted">No data</span>`;
  return `
    <div><strong>Algorithm:</strong> ${metrics.algorithm ?? "n/a"}</div>
    <div><strong>N:</strong> ${formatCount(metrics.n)}</div>
    <div><strong>Primes:</strong> ${formatCount(metrics.primes_found)}</div>
    <div><strong>Rate:</strong> ${formatMetric(metrics.primes_per_sec, " primes/s", 1)}</div>
    <div><strong>Elapsed:</strong> ${formatSeconds(metrics.elapsed_ms)}</div>
  `;
};

function detectVendor(machine) {
  // Detect vendor from machine/GPU metadata
  const label = (machine.label || "").toLowerCase();
  const gpuName = (machine.capabilities?.gpu_name || machine.gpu_name || "").toLowerCase();
  const acceleratorType = machine.capabilities?.accelerator_type || machine.accelerator_type || "";

  // Apple detection
  if (
    label.includes("macbook") ||
    label.includes("mac mini") ||
    label.includes("mac studio") ||
    label.includes("mac pro") ||
    label.includes("apple") ||
    gpuName.includes("apple") ||
    gpuName.includes("m1") ||
    gpuName.includes("m2") ||
    gpuName.includes("m3") ||
    gpuName.includes("m4") ||
    acceleratorType === "metal"
  ) {
    return "apple";
  }

  // NVIDIA detection
  if (
    gpuName.includes("nvidia") ||
    gpuName.includes("rtx") ||
    gpuName.includes("gtx") ||
    gpuName.includes("tesla") ||
    gpuName.includes("quadro") ||
    acceleratorType === "cuda"
  ) {
    return "nvidia";
  }

  return null;
}

function resolveLogoKey(machine) {
  const explicit = typeof machine.logo === "string" ? machine.logo : machine.vendor;
  const normalized = typeof explicit === "string" ? explicit.trim().toLowerCase() : "";
  return normalized || detectVendor(machine);
}

function updateVendorLogo(machine) {
  const logo = document.getElementById(`vendor-logo-${machine.machine_id}`);
  if (!logo) return;

  const logoKey = resolveLogoKey(machine);
  if (!logoKey) {
    logo.classList.add("hidden");
    return;
  }

  const supportedLogos = new Set(["apple", "nvidia"]);
  if (!supportedLogos.has(logoKey)) {
    logo.classList.add("hidden");
    return;
  }

  const logoUrl = `/static/assets/vendor/${logoKey}.png`;
  if (typeof process !== "undefined" && process.env?.NODE_ENV === "development") {
    console.log(`[logo] ${machine.machine_id}: ${logoUrl}`);
  }

  if (logoKey === "apple") {
    logo.alt = "Apple";
    logo.title = "Apple Silicon";
  } else if (logoKey === "nvidia") {
    logo.alt = "NVIDIA";
    logo.title = "NVIDIA GPU";
  } else {
    logo.alt = logoKey;
    logo.title = logoKey;
  }

  logo.src = logoUrl;
  logo.classList.remove("hidden");
}

function updateMachineStatus(machine) {
  const statusBadge = document.getElementById(`status-badge-${machine.machine_id}`);
  if (!statusBadge) return;

  const dot = statusBadge.querySelector('.status-dot');
  const text = statusBadge.querySelector('.status-text');
  if (!dot || !text) return;

  // Update vendor logo
  updateVendorLogo(machine);

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

  const previousStatus = text.textContent;

  const isRunning = activeRunMachineIds.has(machine.machine_id);

  // If machine is excluded, show "Standby" regardless of connection
  if (machine.excluded) {
    statusBadge.className = 'status-badge standby';
    text.textContent = "Standby";
  } else if (agentReachable === null) {
    // If reachability is unknown, show "Checking..."
    statusBadge.className = 'status-badge checking';
    text.textContent = "Checking";
  } else if (!agentReachable) {
    statusBadge.className = 'status-badge offline';
    text.textContent = "Offline";
  } else if (isRunning) {
    statusBadge.className = 'status-badge running';
    text.textContent = "Running";
  } else {
    statusBadge.className = 'status-badge ready';
    text.textContent = "Ready";
  }

  if (DEBUG_UI && previousStatus !== text.textContent) {
    console.log(`[${machine.machine_id}] Status changed: ${previousStatus} -> ${text.textContent}`);
  }

  // Update reset button state based on agent reachability
  updateResetButtonState(machine.machine_id, agentReachable);
}

function updateModelFit(machine) {
  const fitEl = document.getElementById(`model-fit-${machine.machine_id}`);
  if (!fitEl) return;
  const fit = machine.model_fit || {};

  // Determine label based on fit ratio (available / required) thresholds
  const fitRatio = fit.fit_ratio;
  let label = "unknown";
  if (fitRatio != null) {
    if (fitRatio >= 1.2) {
      label = "good";
    } else if (fitRatio >= 1.0) {
      label = "risk";
    } else {
      label = "fail";
    }
  }

  fitEl.classList.remove("good", "marginal", "risk", "fail", "unknown");
  fitEl.classList.add(label);
  const badge = fitEl.querySelector(".fit-badge");
  const scoreEl = fitEl.querySelector(".fit-score");
  if (!badge || !scoreEl) return;

  if (label === "unknown") {
    badge.textContent = "FIT: --";
    scoreEl.textContent = "";
    fitEl.title = "Model fit unavailable";
    return;
  }

  // Display FIT as multiplier with label badge
  const fitDisplay = fitRatio.toFixed(1) + "×";
  const labelText = label.toUpperCase();
  badge.textContent = `FIT: ${labelText}`;
  scoreEl.textContent = fitDisplay;

  // Updated tooltip with clearer explanation
  const tooltip = [
    "Fit = available memory / estimated model memory",
    "Target: ≥ 1.0× (higher = more headroom)",
    "",
    `Current: ${fitDisplay}`,
    `Available ${fit.memory_label || "VRAM"}: ${formatBytes(fit.usable_vram_bytes)}`,
    `Estimated peak: ${formatBytes(fit.estimated_peak_bytes)}`,
    "",
    "Ranges:",
    "  < 1.0× = FAIL (insufficient memory)",
    "  1.0×–1.2× = RISK (tight fit)",
    "  ≥ 1.2× = GOOD (safe headroom)",
  ];
  fitEl.title = tooltip.join("\n");
}

const SPARKLINE_WIDTH = 120;
const SPARKLINE_HEIGHT = 24;

const { ensureSeriesMinimumPoints: ensureSparklineSeriesMinimumPoints, resolveEmptySparklineMessage } =
  window.SparklineUtils || {};
const ensureSeriesMinimumPoints = ensureSparklineSeriesMinimumPoints ||
  (({ values, times }) => ({ values: values || [], times: times || [] }));
const resolveEmptyMessage = resolveEmptySparklineMessage || (() => "Metrics unavailable");

const buildSparklinePath = (values, width, height, maxValue, times = null, startTime = null, endTime = null) => {
  if (!values || values.length === 0) return "";
  const filteredValues = values.filter((v) => v != null && !Number.isNaN(v));
  if (filteredValues.length === 0) return "";
  const max = maxValue || Math.max(...filteredValues, 1);
  const hasTimes = Array.isArray(times) && times.length === values.length && startTime != null && endTime != null;
  const range = hasTimes ? Math.max(endTime - startTime, 1) : null;
  const step = values.length > 1 ? width / (values.length - 1) : width;
  let d = "";
  let started = false;
  values.forEach((value, index) => {
    if (value == null || Number.isNaN(value)) {
      started = false;
      return;
    }
    let x = index * step;
    if (hasTimes) {
      x = ((times[index] - startTime) / range) * width;
    }
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
  const startTime = options.startTime ?? null;
  const endTime = options.endTime ?? null;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  series.forEach((item) => {
    const pathData = buildSparklinePath(item.values, width, height, item.maxValue, item.times, startTime, endTime);
    if (!pathData) return;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", pathData);
    path.setAttribute("class", item.className);
    svg.appendChild(path);
  });
  if (options.title) svg.setAttribute("title", options.title);
};

const asScalarOrLast = (value) => {
  if (Array.isArray(value)) {
    for (let i = value.length - 1; i >= 0; i -= 1) {
      const entry = value[i];
      if (entry != null && !Number.isNaN(entry)) {
        return entry;
      }
    }
    return value.length ? value[value.length - 1] : null;
  }
  return value;
};

const getLastNumeric = (values) => {
  if (!Array.isArray(values)) {
    if (values != null && !Number.isNaN(values)) {
      return values;
    }
    return null;
  }
  for (let i = values.length - 1; i >= 0; i -= 1) {
    const value = values[i];
    if (value != null && !Number.isNaN(value)) {
      return value;
    }
  }
  return null;
};

const extractSeries = (samples, key) => {
  const values = [];
  const times = [];
  samples.forEach((sample) => {
    const value = sample[key];
    if (value == null || Number.isNaN(value)) return;
    values.push(value);
    times.push(sample.t);
  });
  return { values, times };
};

const recordRunSample = (machineId, metrics) => {
  if (!runTrackingActive || runDone.has(machineId) || !metrics) return;
  const cpu = getLastNumeric(metrics.cpu_pct ?? []);
  const gpu = getLastNumeric(metrics.gpu_pct ?? []);
  const vramUsed = getLastNumeric(metrics.vram_used_mib ?? []);
  const vramTotal = getLastNumeric(metrics.vram_total_mib ?? []);
  const ramUsedBytes = getLastNumeric(metrics.ram_used_bytes ?? []);
  const systemMemMib = getLastNumeric(metrics.system_mem_used_mib ?? []);
  const machine = statusCache.get(machineId);
  const totalRamBytes = machine?.capabilities?.total_system_ram_bytes;

  let memPct = null;
  if (vramUsed != null && vramTotal != null && vramTotal > 0) {
    memPct = (vramUsed / vramTotal) * 100;
  } else if (ramUsedBytes != null && totalRamBytes) {
    memPct = (ramUsedBytes / totalRamBytes) * 100;
  } else if (systemMemMib != null && totalRamBytes) {
    memPct = ((systemMemMib * 1024 * 1024) / totalRamBytes) * 100;
  }

  if (cpu == null && gpu == null && memPct == null) return;

  const samples = runSamples.get(machineId) || [];
  samples.push({ t: performance.now(), cpu, gpu, mem: memPct });
  runSamples.set(machineId, samples);
};

const renderRunSparklines = (machineId, metrics = null) => {
  const utilSvg = document.getElementById(`sparkline-util-${machineId}`);
  const memSvg = document.getElementById(`sparkline-mem-${machineId}`);
  const utilPlaceholder = document.getElementById(`sparkline-util-placeholder-${machineId}`);
  const memPlaceholder = document.getElementById(`sparkline-mem-placeholder-${machineId}`);
  const utilBlock = utilSvg?.closest(".sparkline-block");
  const memBlock = memSvg?.closest(".sparkline-block");
  const resolvedMetrics = metrics || runtimeMetrics.get(machineId) || statusCache.get(machineId)?.runtime_metrics;

  let samples = runSamples.get(machineId) || [];
  if (samples.length === 1) {
    samples = [samples[0], samples[0]];
  }
  if (samples.length === 0) {
    const emptyMessage = resolveEmptyMessage(samples.length);
    if (utilPlaceholder) {
      utilPlaceholder.textContent = emptyMessage;
      utilPlaceholder.title = emptyMessage;
    }
    if (memPlaceholder) {
      memPlaceholder.textContent = emptyMessage;
      memPlaceholder.title = emptyMessage;
    }
    utilPlaceholder?.classList.remove("hidden");
    memPlaceholder?.classList.remove("hidden");
    utilBlock?.classList.add("is-empty");
    memBlock?.classList.add("is-empty");
    return;
  }

  const { values: cpuValues, times: cpuTimes } = ensureSeriesMinimumPoints(extractSeries(samples, "cpu"));
  const { values: gpuValues, times: gpuTimes } = ensureSeriesMinimumPoints(extractSeries(samples, "gpu"));
  const { values: memValues, times: memTimes } = ensureSeriesMinimumPoints(extractSeries(samples, "mem"));
  const startTime = runStartTs ?? samples[0]?.t ?? performance.now();
  const endTime = runEndTs ?? performance.now();
  const gpuAvailability = asScalarOrLast(resolvedMetrics?.gpu_metrics_available);

  const hasGpuUtil = gpuValues.length > 0;
  const hasCpuUtil = cpuValues.length > 0;
  const hasUtil = hasGpuUtil || hasCpuUtil;

  const utilSeries = [];
  let utilTitle = "";
  let utilMeta = "";

  if (hasGpuUtil) {
    utilSeries.push({ values: cpuValues, times: cpuTimes, className: "cpu-line", maxValue: 100 });
    utilSeries.push({ values: gpuValues, times: gpuTimes, className: "gpu-line", maxValue: 100 });
    utilTitle = "GPU utilization (%)";
    const lastCpu = cpuValues.slice(-1)[0];
    const lastGpu = gpuValues.slice(-1)[0];
    utilMeta = `CPU ${lastCpu ?? "n/a"}% | GPU ${lastGpu ?? "n/a"}%`;
  } else if (hasCpuUtil) {
    utilSeries.push({ values: cpuValues, times: cpuTimes, className: "cpu-line", maxValue: 100 });
    utilTitle = "CPU utilization (%)";
    const lastCpu = cpuValues.slice(-1)[0];
    utilMeta = gpuAvailability === false ? `CPU ${lastCpu ?? "n/a"}% (GPU unavailable)` : `CPU ${lastCpu ?? "n/a"}%`;
  } else {
    utilTitle = "Metrics unavailable";
    utilMeta = "Metrics unavailable";
  }

  utilPlaceholder?.classList.toggle("hidden", hasUtil);
  utilBlock?.classList.toggle("is-empty", !hasUtil);
  if (hasUtil) {
    renderSparkline(utilSvg, utilSeries, {
      title: utilTitle,
      startTime,
      endTime,
    });
  }
  if (utilSvg) utilSvg.dataset.sparklineMeta = utilMeta;

  const vramUsed = resolvedMetrics?.vram_used_mib || [];
  const vramTotal = resolvedMetrics?.vram_total_mib || [];
  const ramUsedBytes = resolvedMetrics?.ram_used_bytes || [];
  const systemMemMib = resolvedMetrics?.system_mem_used_mib || [];
  const machine = statusCache.get(machineId);
  const totalRamBytes = machine?.capabilities?.total_system_ram_bytes;

  const hasVramTelemetry =
    getLastNumeric(vramUsed) != null && getLastNumeric(vramTotal) != null;
  const hasSystemMemTelemetry = getLastNumeric(ramUsedBytes) != null || getLastNumeric(systemMemMib) != null;
  const canComputeMem = (hasVramTelemetry && getLastNumeric(vramTotal) > 0) ||
    (hasSystemMemTelemetry && totalRamBytes);

  let memTitle = "";
  let memMeta = "";
  if (memValues.length > 0) {
    memTitle = hasVramTelemetry ? "VRAM used (%)" : "System RAM used (%)";
    const lastPct = memValues.slice(-1)[0];
    memMeta = lastPct != null ? `MEM ${lastPct.toFixed(1)}%` : "MEM n/a";
  } else if (!canComputeMem) {
    memTitle = "Memory telemetry unavailable";
    memMeta = "Memory telemetry unavailable";
  } else {
    memTitle = "Metrics unavailable";
    memMeta = "Metrics unavailable";
  }

  const hasMemData = memValues.length > 0;
  memPlaceholder?.classList.toggle("hidden", hasMemData);
  memBlock?.classList.toggle("is-empty", !hasMemData);
  if (memPlaceholder) {
    if (!hasMemData && !canComputeMem) {
      memPlaceholder.textContent = "N/A";
      memPlaceholder.title =
        "Memory telemetry unavailable (no VRAM metrics and no system RAM usage metric).";
    } else {
      memPlaceholder.textContent = "Metrics unavailable";
      memPlaceholder.title = "";
    }
  }
  if (hasMemData) {
    renderSparkline(memSvg, [{ values: memValues, times: memTimes, className: "mem-line", maxValue: 100 }], {
      title: memTitle,
      startTime,
      endTime,
    });
  }
  if (memSvg) memSvg.dataset.sparklineMeta = memMeta;
};

const renderAllSparklines = () => {
  paneMap.forEach((_, machineId) => {
    renderRunSparklines(machineId);
  });
};

const updateRuntimeMetrics = (machineId, metrics) => {
  recordRunSample(machineId, metrics);
  renderRunSparklines(machineId, metrics);
};

const openSparklineModal = (machineId, type) => {
  const metrics = runtimeMetrics.get(machineId) || statusCache.get(machineId)?.runtime_metrics;
  let samples = runSamples.get(machineId) || [];
  if (samples.length === 1) {
    samples = [samples[0], samples[0]];
  }
  if (!metrics && samples.length === 0) return;
  const modalLabel = document.getElementById("sparkline-modal-label");
  const modalMeta = document.getElementById("sparkline-modal-meta");
  const modalChart = document.getElementById("sparkline-modal-chart");
  const machineLabel = statusCache.get(machineId)?.label || machineId;
  const cpu = metrics?.cpu_pct || [];
  const gpu = metrics?.gpu_pct || [];
  const vramUsed = metrics?.vram_used_mib || [];
  const vramTotal = metrics?.vram_total_mib || [];
  const systemMem = metrics?.system_mem_used_mib || [];
  const ramUsedBytes = metrics?.ram_used_bytes || [];
  const gpuAvailability = asScalarOrLast(metrics?.gpu_metrics_available);

  if (type === "util") {
    if (modalLabel) modalLabel.textContent = `${machineLabel} • Utilization`;

    if (samples.length > 0) {
      const { values: cpuValues, times: cpuTimes } = ensureSeriesMinimumPoints(extractSeries(samples, "cpu"));
      const { values: gpuValues, times: gpuTimes } = ensureSeriesMinimumPoints(extractSeries(samples, "gpu"));
      const hasGpuUtil = gpuValues.length > 0;
      const hasCpuUtil = cpuValues.length > 0;
      const utilSeries = [];
      if (hasGpuUtil) {
        utilSeries.push({ values: cpuValues, times: cpuTimes, className: "cpu-line", maxValue: 100 });
        utilSeries.push({ values: gpuValues, times: gpuTimes, className: "gpu-line", maxValue: 100 });
        if (modalMeta) {
          modalMeta.textContent = `CPU ${cpuValues.slice(-1)[0] ?? "n/a"}% | GPU ${gpuValues.slice(-1)[0] ?? "n/a"}%`;
        }
      } else if (hasCpuUtil) {
        utilSeries.push({ values: cpuValues, times: cpuTimes, className: "cpu-line", maxValue: 100 });
        if (modalMeta) {
          const suffix = gpuAvailability === false ? " (GPU unavailable)" : "";
          modalMeta.textContent = `CPU ${cpuValues.slice(-1)[0] ?? "n/a"}%${suffix}`;
        }
      } else if (modalMeta) {
        modalMeta.textContent = "Metrics unavailable";
      }
      if (utilSeries.length > 0) {
        const startTime = runStartTs ?? samples[0]?.t ?? performance.now();
        renderSparkline(modalChart, utilSeries, {
          width: 480,
          height: 120,
          startTime,
          endTime: runEndTs ?? performance.now(),
        });
      }
    } else {
      const cpuMax = 100;
      const gpuMax = 100;
      const { values: cpuValues } = ensureSeriesMinimumPoints({ values: cpu, times: [] });
      const { values: gpuValues } = ensureSeriesMinimumPoints({ values: gpu, times: [] });
      const hasGpuUtil = gpuValues.some((v) => v != null && !Number.isNaN(v));
      const hasCpuUtil = cpuValues.some((v) => v != null && !Number.isNaN(v));
      const lastCpu = cpu.filter((v) => v != null).slice(-1)[0];
      const lastGpu = gpu.filter((v) => v != null).slice(-1)[0];

      const utilSeries = [];
      if (hasGpuUtil) {
        utilSeries.push({ values: cpuValues, className: "cpu-line", maxValue: cpuMax });
        utilSeries.push({ values: gpuValues, className: "gpu-line", maxValue: gpuMax });
        if (modalMeta) modalMeta.textContent = `CPU ${lastCpu ?? "n/a"}% | GPU ${lastGpu ?? "n/a"}%`;
      } else if (hasCpuUtil) {
        utilSeries.push({ values: cpuValues, className: "cpu-line", maxValue: cpuMax });
        if (modalMeta) {
          const suffix = gpuAvailability === false ? " (GPU unavailable)" : "";
          modalMeta.textContent = `CPU ${lastCpu ?? "n/a"}%${suffix}`;
        }
      } else if (modalMeta) {
        modalMeta.textContent = "Metrics unavailable";
      }

      if (utilSeries.length > 0) {
        renderSparkline(modalChart, utilSeries, { width: 480, height: 120 });
      }
    }
  } else {
    if (modalLabel) modalLabel.textContent = `${machineLabel} • Memory`;

    if (samples.length > 0) {
      const { values: memValues, times: memTimes } = ensureSeriesMinimumPoints(extractSeries(samples, "mem"));
      if (memValues.length > 0) {
        const lastPct = memValues.slice(-1)[0];
        if (modalMeta) modalMeta.textContent = lastPct != null ? `MEM ${lastPct.toFixed(1)}%` : "MEM n/a";
        const startTime = runStartTs ?? samples[0]?.t ?? performance.now();
        renderSparkline(modalChart, [{ values: memValues, times: memTimes, className: "mem-line", maxValue: 100 }], {
          width: 480,
          height: 120,
          startTime,
          endTime: runEndTs ?? performance.now(),
        });
      } else if (modalMeta) {
        modalMeta.textContent = "Metrics unavailable";
      }
    } else {
      const hasVram = vramUsed.some((v) => v != null && !Number.isNaN(v)) &&
                      vramTotal.some((v) => v != null && !Number.isNaN(v));
      const hasSystemMem = systemMem.some((v) => v != null && !Number.isNaN(v)) ||
        ramUsedBytes.some((v) => v != null && !Number.isNaN(v));

      if (hasVram) {
        // Convert VRAM to percentage
        const { values: memValues } = ensureSeriesMinimumPoints({
          values: vramUsed.map((used, i) => {
            const total = vramTotal[i];
            if (used == null || total == null || total === 0) return null;
            return (used / total) * 100;
          }),
          times: [],
        });
        const lastPct = memValues.filter((v) => v != null).slice(-1)[0];
        if (modalMeta) modalMeta.textContent = lastPct != null ? `VRAM ${lastPct.toFixed(1)}%` : "VRAM n/a";
        renderSparkline(modalChart, [{ values: memValues, className: "mem-line", maxValue: 100 }], {
          width: 480,
          height: 120,
        });
      } else if (hasSystemMem) {
        const machine = statusCache.get(machineId);
        const totalRamBytes = machine?.capabilities?.total_system_ram_bytes;

        if (totalRamBytes) {
          let memValues = [];
          if (ramUsedBytes.some((v) => v != null && !Number.isNaN(v))) {
            memValues = ramUsedBytes.map((used) => {
              if (used == null || totalRamBytes === 0) return null;
              return (used / totalRamBytes) * 100;
            });
          } else {
            const totalRamMib = totalRamBytes / (1024 * 1024);
            memValues = systemMem.map((used) => {
              if (used == null || totalRamMib === 0) return null;
              return (used / totalRamMib) * 100;
            });
          }
          const { values: adjustedMemValues } = ensureSeriesMinimumPoints({ values: memValues, times: [] });
          const lastPct = adjustedMemValues.filter((v) => v != null).slice(-1)[0];
          if (modalMeta) modalMeta.textContent = lastPct != null ? `RAM ${lastPct.toFixed(1)}%` : "RAM n/a";
          renderSparkline(modalChart, [{ values: adjustedMemValues, className: "mem-line", maxValue: 100 }], {
            width: 480,
            height: 120,
          });
        } else if (modalMeta) {
          modalMeta.textContent = "Memory telemetry unavailable";
        }
      } else if (modalMeta) {
        modalMeta.textContent = "Metrics unavailable";
      }
    }
  }
  openOverlay("sparkline");
};

function updateSyncButton(machine) {
  const btn = document.getElementById(`sync-${machine.machine_id}`);
  if (!btn) return;
  if (MODE === "compute") {
    btn.classList.add("hidden");
    return;
  }
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
  const previousCache = new Map(statusCache);
  statusCache.clear();
  (data.machines || []).forEach((machine) => {
    const previous = previousCache.get(machine.machine_id) || {};
    const merged = { ...previous, ...machine };
    if (machine.excluded === undefined && previous.excluded !== undefined) {
      merged.excluded = previous.excluded;
    }
    if (machine.logo === undefined && previous.logo !== undefined) {
      merged.logo = previous.logo;
    }
    statusCache.set(merged.machine_id, merged);
    applyMachineExcludedState(merged.machine_id, merged.excluded);
    updateMachineStatus(merged);
    updateModelFit(merged);
    if (merged.runtime_metrics) {
      runtimeMetrics.set(merged.machine_id, merged.runtime_metrics);
    }
    updateRuntimeMetrics(merged.machine_id, merged.runtime_metrics);
    updateSyncButton(merged);
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
    if (machine.excluded) {
      blocked.push({ machine_id: machineId, label: machine.label, reason: "excluded" });
    } else if (!machine.reachable) {
      blocked.push({ machine_id: machineId, label: machine.label, reason: "agent offline" });
    } else if (MODE !== "compute" && !machine.has_selected_model) {
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

const isScrolledToBottom = (el) => el.scrollTop + el.clientHeight >= el.scrollHeight - 4;

const maybeAutoScrollOutput = (machineId) => {
  if (viewingMode !== "live") return;
  const pane = paneMap.get(machineId);
  if (!pane) return;
  if (outputScrollState.get(machineId) === false) return;
  pane.out.scrollTop = pane.out.scrollHeight;
};

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
  maybeAutoScrollOutput(machineId);
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

    let metricsHtml = `<span class="muted">No data</span>`;
    let isMock = false;
    if (run.type === "compute") {
      metricsHtml = buildComputeMetricsHtml({
        algorithm: entry.algorithm ?? run.settings?.algorithm,
        n: entry.n ?? run.settings?.n,
        threads_requested: entry.threads_requested ?? run.settings?.threads,
        threads_used: entry.threads_used,
        primes_found: entry.primes_found,
        elapsed_ms: entry.elapsed_ms,
        primes_per_sec: entry.primes_per_sec,
        repeat_index: entry.repeat_index,
        repeat_total: run.settings?.repeats,
      });
      out.textContent = "Output not stored for past runs.";
    } else {
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
      metricsHtml = buildMetricsHtml(metricsData, baselineMetrics);
      out.textContent = "Output not stored for past runs.";
      isMock = entry.engine === "mock";
    }
    setPaneContent(machineId, {
      outText: out.textContent,
      metricsHtml,
      isMock,
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

const toggleSelectMode = () => {
  selectMode = !selectMode;
  selectedRunIds.clear();
  updateSelectModeUI();
  renderRecentRuns();
};

const updateSelectModeUI = () => {
  const toggleBtn = document.getElementById("toggle-select-mode");
  const selectAllBtn = document.getElementById("select-all-runs");
  const deleteSelectedBtn = document.getElementById("delete-selected-runs");
  if (toggleBtn) toggleBtn.textContent = selectMode ? "Cancel" : "Select";
  if (selectAllBtn) selectAllBtn.classList.toggle("hidden", !selectMode);
  if (deleteSelectedBtn) {
    deleteSelectedBtn.classList.toggle("hidden", !selectMode);
    deleteSelectedBtn.textContent = `Delete Selected (${selectedRunIds.size})`;
    deleteSelectedBtn.disabled = selectedRunIds.size === 0;
  }
};

const toggleRunSelection = (runId) => {
  if (selectedRunIds.has(runId)) {
    selectedRunIds.delete(runId);
  } else {
    selectedRunIds.add(runId);
  }
  updateSelectModeUI();
  const item = document.querySelector(`.recent-run-item[data-run-id="${runId}"]`);
  if (item) {
    item.classList.toggle("selected", selectedRunIds.has(runId));
    const checkbox = item.querySelector(".run-select-checkbox");
    if (checkbox) checkbox.checked = selectedRunIds.has(runId);
  }
};

const selectAllRuns = () => {
  const allSelected = selectedRunIds.size === recentRuns.length;
  selectedRunIds.clear();
  if (!allSelected) {
    recentRuns.forEach((run) => selectedRunIds.add(run.run_id));
  }
  updateSelectModeUI();
  document.querySelectorAll(".recent-run-item").forEach((item) => {
    const runId = item.dataset.runId;
    item.classList.toggle("selected", selectedRunIds.has(runId));
    const checkbox = item.querySelector(".run-select-checkbox");
    if (checkbox) checkbox.checked = selectedRunIds.has(runId);
  });
  const selectAllBtn = document.getElementById("select-all-runs");
  if (selectAllBtn) selectAllBtn.textContent = allSelected ? "Select All" : "Deselect All";
};

const bulkDeleteRuns = async (runIds) => {
  if (!runIds.length) return;
  const confirmed = window.confirm(`Delete ${runIds.length} run(s) permanently? This cannot be undone.`);
  if (!confirmed) return;
  try {
    const response = await fetch("/api/runs/bulk-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_ids: runIds }),
    });
    if (!response.ok) {
      showToast("Failed to delete runs.", "error");
      return;
    }
    const result = await response.json();
    const deletedSet = new Set(result.deleted);
    recentRuns = recentRuns.filter((run) => !deletedSet.has(run.run_id));
    if (baselineRunId && deletedSet.has(baselineRunId)) {
      await setBaselineRunId(null);
      showToast("Baseline cleared because the run was deleted.", "info");
    }
    selectedRunIds.clear();
    selectMode = false;
    updateSelectModeUI();
    renderRecentRuns();
    updateHistoryBadge();
    const msg = result.errors.length
      ? `Deleted ${result.deleted.length} run(s). ${result.errors.length} failed.`
      : `Deleted ${result.deleted.length} run(s).`;
    showToast(msg, result.errors.length ? "error" : "success");
  } catch (error) {
    console.error("Failed to bulk delete runs", error);
    showToast("Failed to delete runs.", "error");
  }
};

const deleteAllRuns = async () => {
  const ids = recentRuns.map((run) => run.run_id);
  await bulkDeleteRuns(ids);
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
    if (selectMode && selectedRunIds.has(run.run_id)) item.classList.add("selected");
    item.dataset.runId = run.run_id;

    const badges = [];
    if (run.has_mock) {
      badges.push('<span class="run-badge warning">Mock engine</span>');
    }
    if (run.type === "image") {
      badges.push('<span class="run-badge image">Image</span>');
    } else if (run.type === "compute") {
      badges.push('<span class="run-badge compute">Compute</span>');
    } else {
      badges.push('<span class="run-badge inference">Inference</span>');
    }
    if (baselineRunId && baselineRunId === run.run_id) {
      badges.push('<span class="run-badge baseline">Baseline</span>');
    }
    const warningText = run.has_mock
      ? '<div class="run-warning-text">Mock engine used on at least one machine.</div>'
      : "";

    const checkboxHtml = selectMode
      ? `<label class="run-select-label" title="Select run"><input type="checkbox" class="run-select-checkbox" ${selectedRunIds.has(run.run_id) ? "checked" : ""} /></label>`
      : "";

    item.innerHTML = `
      ${checkboxHtml}
      <button class="run-delete-button${selectMode ? " hidden" : ""}" type="button" aria-label="Delete run" title="Delete run">🗑️</button>
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

    if (selectMode) {
      const checkbox = item.querySelector(".run-select-checkbox");
      checkbox?.addEventListener("change", (event) => {
        event.stopPropagation();
        toggleRunSelection(run.run_id);
      });
      item.addEventListener("click", (event) => {
        if (event.target.tagName === "INPUT" || event.target.tagName === "BUTTON") return;
        toggleRunSelection(run.run_id);
      });
    } else {
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
    }

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
    if (run.type === "compute" && MODE !== "compute") {
      window.location.href = `/compute?run_id=${encodeURIComponent(runId)}`;
      return;
    }
    if (run.type !== "compute" && MODE === "compute") {
      window.location.href = `/inference?run_id=${encodeURIComponent(runId)}`;
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
    if (Array.isArray(event.machine_ids)) {
      setActiveRunMachines(event.run_id, event.machine_ids);
    }
    console.log("Run started:", event.run_id);
    if (!runTrackingActive) {
      beginRunTracking(Array.from(statusCache.keys()));
    }
  } else if (event.type === "run_end") {
    setRunActive(false, event.run_id);
    clearActiveRunMachines(event.run_id);
    console.log("Run ended:", event.run_id);
    endRunTracking();
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
        outputScrollState.set(machine.machine_id, true);
        out.addEventListener("scroll", () => {
          outputScrollState.set(machine.machine_id, isScrolledToBottom(out));
        });
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

  if (evt.type === "compute_line") {
    const line = evt.payload?.line ?? "";
    const state = computeOutputState.get(evt.machine_id) || {
      progressBarLine: buildProgressBar(0),
      logLines: [],
    };
    const progressMatch = line.match(/^Progress:\s*([0-9]+(?:\.[0-9]+)?)%/);
    if (progressMatch) {
      state.progressBarLine = buildProgressBar(progressMatch[1]);
    }
    if (line) {
      state.logLines.push(line);
    }
    computeOutputState.set(evt.machine_id, state);
    const updated = [state.progressBarLine, ...state.logLines].join("\n");
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
    markRunMachineDone(evt.machine_id);

    fetchRecentRuns();
  }

  if (evt.type === "compute_done") {
    const payload = evt.payload || {};
    const metricsData = {
      algorithm: payload.algorithm,
      n: payload.n,
      threads_requested: payload.threads_requested,
      threads_used: payload.threads_used,
      primes_found: payload.primes_found,
      elapsed_ms: payload.elapsed_ms,
      primes_per_sec: payload.primes_per_sec,
      repeat_index: payload.repeat_index,
      repeat_total: currentComputeRepeats,
    };
    liveMetrics.set(evt.machine_id, metricsData);
    const metricsHtml = buildComputeMetricsHtml(metricsData);
    const outText = payload.ok === false
      ? `Compute failed: ${payload.error || "Unknown error"}`
      : (liveOutput.get(evt.machine_id) || pane.out.textContent);
    updateLivePane(evt.machine_id, {
      outText,
      metricsHtml,
      isMock: false,
    });
    markRunMachineDone(evt.machine_id);
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

  beginRunTracking(ready.map((m) => m.machine_id));

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

const resolveRecommendedN = (algorithm) => computeSettings.autoN?.[algorithm]
  || DEFAULT_COMPUTE_SETTINGS.autoN.segmented_sieve;

const applyRecommendedN = (force = false) => {
  const nInput = document.getElementById("compute-n");
  const algorithmInput = document.getElementById("compute-algorithm");
  if (!(nInput instanceof HTMLInputElement) || !(algorithmInput instanceof HTMLSelectElement)) return;
  if (!force && computeNManuallyEdited) return;
  const recommended = resolveRecommendedN(algorithmInput.value);
  nInput.value = String(recommended);
  computeNManuallyEdited = false;
};

const computeAlgorithmInput = document.getElementById("compute-algorithm");
const computeNInput = document.getElementById("compute-n");
const computeRecommendedButton = document.getElementById("compute-n-recommended");

computeAlgorithmInput?.addEventListener("change", () => applyRecommendedN(false));
computeNInput?.addEventListener("input", () => {
  computeNManuallyEdited = true;
});
computeRecommendedButton?.addEventListener("click", () => applyRecommendedN(true));
applyRecommendedN(false);

const startCompute = async () => {
  const { blocked, ready } = getPreflightStatus();

  if (ready.length === 0) {
    showToast("No ready machines available.", "info");
    return;
  }

  const algorithmEl = document.getElementById("compute-algorithm");
  const nEl = document.getElementById("compute-n");
  const threadsEl = document.getElementById("compute-threads");
  const repeatEl = document.getElementById("compute-repeat");
  if (
    !(algorithmEl instanceof HTMLSelectElement) ||
    !(nEl instanceof HTMLInputElement) ||
    !(threadsEl instanceof HTMLInputElement) ||
    !(repeatEl instanceof HTMLInputElement)
  ) {
    showToast("Compute inputs not ready.", "error");
    return;
  }

  const nValue = parseInt(nEl.value, 10);
  if (Number.isNaN(nValue) || nValue < 10) {
    showToast("N must be an integer ≥ 10.", "error");
    return;
  }

  const threadsValue = Math.max(1, parseInt(threadsEl.value, 10) || 1);
  const repeatsValue = Math.max(1, parseInt(repeatEl.value, 10) || 1);
  currentComputeRepeats = repeatsValue;

  beginRunTracking(ready.map((m) => m.machine_id));

  const payload = {
    algorithm: algorithmEl.value,
    n: nValue,
    threads: threadsValue,
    repeats: repeatsValue,
    progress_interval_s: computeSettings.progressIntervalS,
    machine_ids: ready.map((m) => m.machine_id),
  };

  paneMap.forEach(({ out, metrics }, machineId) => {
    const isBlocked = blocked.some((b) => b.machine_id === machineId);
    if (isBlocked) {
      out.textContent = "Skipped (not ready)";
      metrics.innerHTML = `<span class="muted">Machine not ready</span>`;
      liveOutput.set(machineId, out.textContent);
    } else {
      const progressBar = buildProgressBar(0);
      computeOutputState.set(machineId, {
        progressBarLine: progressBar,
        logLines: ["Starting compute…"],
      });
      out.textContent = `${progressBar}\nStarting compute…`;
      metrics.innerHTML = `<span class="muted">Queued…</span>`;
      liveOutput.set(machineId, out.textContent);
    }
    updateLivePane(machineId, {
      outText: out.textContent,
      metricsHtml: metrics.innerHTML,
      isMock: false,
    });
  });

  try {
    const response = await fetch("/api/compute/run", {
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
    console.error("Failed to start compute jobs", error);
  }
};

// Run button with preflight validation
runButton?.addEventListener("click", async () => {
  if (MODE === "compute") {
    await startCompute();
  } else {
    await startRun();
  }
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
    const computeAutoSegmentedInput = document.getElementById("compute-auto-n-segmented");
    const computeAutoSimpleInput = document.getElementById("compute-auto-n-simple");
    const computeAutoTrialInput = document.getElementById("compute-auto-n-trial");
    const computeIntervalInput = document.getElementById("compute-progress-interval");
    if (computeAutoSegmentedInput) computeAutoSegmentedInput.value = computeSettings.autoN.segmented_sieve;
    if (computeAutoSimpleInput) computeAutoSimpleInput.value = computeSettings.autoN.simple_sieve;
    if (computeAutoTrialInput) computeAutoTrialInput.value = computeSettings.autoN.trial_division;
    if (computeIntervalInput) computeIntervalInput.value = computeSettings.progressIntervalS;
    setPolicyFeedback("", "");
    toggleOverlay("settings");
  } catch (error) {
    alert("Could not load settings. Try again.");
  }
});

document.getElementById("refresh-runs")?.addEventListener("click", async () => {
  await fetchRecentRuns();
});

document.getElementById("toggle-select-mode")?.addEventListener("click", () => {
  toggleSelectMode();
});

document.getElementById("select-all-runs")?.addEventListener("click", () => {
  selectAllRuns();
});

document.getElementById("delete-selected-runs")?.addEventListener("click", async () => {
  await bulkDeleteRuns([...selectedRunIds]);
});

document.getElementById("delete-all-runs")?.addEventListener("click", async () => {
  await deleteAllRuns();
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
    const computeAutoSegmentedInput = document.getElementById("compute-auto-n-segmented");
    const computeAutoSimpleInput = document.getElementById("compute-auto-n-simple");
    const computeAutoTrialInput = document.getElementById("compute-auto-n-trial");
    const computeIntervalInput = document.getElementById("compute-progress-interval");
    if (computeAutoSegmentedInput) {
      computeSettings.autoN.segmented_sieve = Math.max(10, parseInt(computeAutoSegmentedInput.value, 10) || 10);
      computeAutoSegmentedInput.value = computeSettings.autoN.segmented_sieve;
    }
    if (computeAutoSimpleInput) {
      computeSettings.autoN.simple_sieve = Math.max(10, parseInt(computeAutoSimpleInput.value, 10) || 10);
      computeAutoSimpleInput.value = computeSettings.autoN.simple_sieve;
    }
    if (computeAutoTrialInput) {
      computeSettings.autoN.trial_division = Math.max(10, parseInt(computeAutoTrialInput.value, 10) || 10);
      computeAutoTrialInput.value = computeSettings.autoN.trial_division;
    }
    if (computeIntervalInput) {
      computeSettings.progressIntervalS = clampNumber(parseFloat(computeIntervalInput.value) || 1, 0.1, 60);
      computeIntervalInput.value = computeSettings.progressIntervalS;
    }
    saveComputeSettings();
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
  if (MODE === "compute") return;
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (runButton?.disabled) return;
    await startRun();
  }
});

// Toggle machine excluded status
async function toggleMachineExcluded(machineId) {
  const machine = statusCache.get(machineId);
  if (!machine) return;

  const newExcludedState = !machine.excluded;

  try {
    const response = await fetch(`/api/machines/${encodeURIComponent(machineId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ excluded: newExcludedState }),
    });

    if (!response.ok) {
      throw new Error(`Failed to update machine: ${response.statusText}`);
    }

    const data = await response.json();

    // Update local state
    machine.excluded = data.excluded;
    statusCache.set(machineId, machine);

    // Update UI
    applyMachineExcludedState(machineId, data.excluded);
    updateMachineStatus(machine);
    updatePreflightBanner();

    console.log(`Machine ${machineId} excluded status updated to: ${data.excluded}`);
  } catch (error) {
    console.error(`Failed to toggle excluded status for ${machineId}:`, error);
    showToast(`Failed to update machine status: ${error.message}`, "error");
  }
}

function applyMachineExcludedState(machineId, excluded) {
  const pane = document.getElementById(`pane-${machineId}`);
  const toggleCheckbox = document.getElementById(`toggle-exclude-${machineId}`);

  if (pane) {
    pane.setAttribute("data-excluded", excluded.toString());
    if (excluded) {
      pane.classList.add("pane-excluded");
    } else {
      pane.classList.remove("pane-excluded");
    }
  }

  // Update checkbox state (checked = enabled = NOT excluded)
  if (toggleCheckbox) {
    toggleCheckbox.checked = !excluded;
  }
}

// Initialize toggle checkboxes for all machines
function initToggleButtons() {
  const toggleCheckboxes = document.querySelectorAll(".toggle-exclude-input");
  toggleCheckboxes.forEach((checkbox) => {
    const machineId = checkbox.getAttribute("data-machine-id");
    if (machineId) {
      checkbox.addEventListener("change", () => toggleMachineExcluded(machineId));
    }
  });
}

// Reset agent with detailed diagnostics
async function resetAgent(machineId) {
  const btn = document.getElementById(`reset-${machineId}`);
  const statusBadge = document.getElementById(`status-badge-${machineId}`);
  if (!btn) return;

  // Check if agent is online
  const machine = statusCache.get(machineId);
  if (!machine || !machine.reachable) {
    showToast("Cannot reset: agent is offline", "error");
    return;
  }

  // Disable button and show loading state
  btn.disabled = true;
  btn.classList.add("loading");
  const originalText = btn.textContent;
  btn.textContent = "Resetting...";

  // Update status to "Standby" during reset
  if (statusBadge) {
    const statusText = statusBadge.querySelector('.status-text');
    statusBadge.className = "status-badge standby";
    if (statusText) statusText.textContent = "Standby";
  }

  try {
    const response = await fetch(`/api/agents/${encodeURIComponent(machineId)}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const result = await response.json();

    if (response.ok && result.ok) {
      const durationText = result.duration_ms
        ? ` in ${(result.duration_ms / 1000).toFixed(1)}s`
        : "";
      if (result.warnings) {
        showToast(
          `Reset complete (warnings) for ${machine.label || machineId}${durationText}`,
          "warning",
          () => showResetDiagnosticsModal(machineId, result)
        );
      } else {
        showToast(`Reset successful for ${machine.label || machineId}${durationText}`, "success");
      }
      // Refresh status after reset (will update to "Ready")
      await fetchStatus();
    } else {
      // Show diagnostics modal for failed resets
      const errorMsg = result.error || "Reset failed";
      showToast(
        `Reset failed for ${machine.label || machineId} — click to view diagnostics`,
        "error",
        () => showResetDiagnosticsModal(machineId, result)
      );
      console.error("Reset failed:", result);
      // Also show modal immediately
      showResetDiagnosticsModal(machineId, result);
      // Restore status on error
      await fetchStatus();
    }
  } catch (error) {
    showToast(`Reset error: ${error.message}`, "error");
    console.error(`Failed to reset agent ${machineId}:`, error);
    // Restore status on error
    await fetchStatus();
  } finally {
    // Restore button state
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = originalText;
  }
}

// Initialize reset buttons for all machines
function initResetButtons() {
  const resetButtons = document.querySelectorAll(".btn-reset");
  resetButtons.forEach((btn) => {
    const machineId = btn.getAttribute("data-machine-id");
    if (machineId) {
      btn.addEventListener("click", () => resetAgent(machineId));
    }
  });
}

// Update reset button disabled state based on agent status
function updateResetButtonState(machineId, reachable) {
  const btn = document.getElementById(`reset-${machineId}`);
  if (btn) {
    btn.disabled = !reachable;
    btn.title = reachable
      ? "Reset Agent (restart Ollama + ComfyUI)"
      : "Agent offline";
  }
}

// Call initialization after DOM is ready
initToggleButtons();
initResetButtons();

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
loadComputeSettings();
scheduleStatusPoll();

const params = new URLSearchParams(window.location.search);
const runId = params.get("run_id");
if (runId) {
  loadRun(runId);
}

// ======================================
// Reset Diagnostics Modal
// ======================================

let lastResetDiagnostics = new Map(); // Store diagnostics per machine

function showResetDiagnosticsModal(machineId, diagnostics) {
  lastResetDiagnostics.set(machineId, diagnostics);

  const modal = document.getElementById("reset-diagnostics-modal");
  if (!modal) return;

  const machine = statusCache.get(machineId);
  const machineName = machine ? (machine.label || machineId) : machineId;

  // Fill in summary
  document.getElementById("diag-agent-name").textContent = machineName;

  const statusBadge = document.getElementById("diag-status");
  statusBadge.textContent = diagnostics.ok ? "Success" : "Failed";
  statusBadge.className = diagnostics.ok ? "status-badge ready" : "status-badge offline";

  document.getElementById("diag-duration").textContent =
    diagnostics.duration_ms ? `${(diagnostics.duration_ms / 1000).toFixed(1)}s` : "N/A";

  // Fill in notes
  const notesSection = document.getElementById("diag-notes-section");
  const notesList = document.getElementById("diag-notes");
  if (diagnostics.notes && diagnostics.notes.length > 0) {
    notesList.innerHTML = "";
    diagnostics.notes.forEach(note => {
      const li = document.createElement("li");
      li.textContent = note;
      notesList.appendChild(li);
    });
    notesSection.style.display = "block";
  } else {
    notesSection.style.display = "none";
  }

  // Fill in Ollama details
  const ollama = diagnostics.ollama || {};
  document.getElementById("diag-ollama-healthy").textContent = ollama.healthy ? "✓ Yes" : "✗ No";
  document.getElementById("diag-ollama-time").textContent =
    ollama.time_to_ready_ms ? `${(ollama.time_to_ready_ms / 1000).toFixed(1)}s` : "N/A";
  document.getElementById("diag-ollama-log").textContent = ollama.start_log_file || "N/A";
  document.getElementById("diag-ollama-stdout").textContent = ollama.start_stdout_tail || "(empty)";
  document.getElementById("diag-ollama-stderr").textContent = ollama.start_stderr_tail || "(empty)";

  // Fill in ComfyUI details
  const comfyui = diagnostics.comfyui || {};
  document.getElementById("diag-comfyui-healthy").textContent = comfyui.healthy ? "✓ Yes" : "✗ No";
  document.getElementById("diag-comfyui-time").textContent =
    comfyui.time_to_ready_ms ? `${(comfyui.time_to_ready_ms / 1000).toFixed(1)}s` : "N/A";
  document.getElementById("diag-comfyui-log").textContent = comfyui.start_log_file || "N/A";
  document.getElementById("diag-comfyui-stdout").textContent = comfyui.start_stdout_tail || "(empty)";
  document.getElementById("diag-comfyui-stderr").textContent = comfyui.start_stderr_tail || "(empty)";

  // Fill in full JSON
  document.getElementById("diag-full-json").textContent = JSON.stringify(diagnostics, null, 2);

  // Show modal
  modal.classList.remove("hidden");
}

function hideResetDiagnosticsModal() {
  const modal = document.getElementById("reset-diagnostics-modal");
  if (modal) {
    modal.classList.add("hidden");
  }
}

// Initialize modal close handlers
document.getElementById("reset-modal-close")?.addEventListener("click", hideResetDiagnosticsModal);
document.getElementById("reset-modal-close-btn")?.addEventListener("click", hideResetDiagnosticsModal);
document.querySelector("#reset-diagnostics-modal .modal-overlay")?.addEventListener("click", hideResetDiagnosticsModal);

// Copy to clipboard functionality
document.querySelectorAll(".btn-copy").forEach(btn => {
  btn.addEventListener("click", () => {
    const targetId = btn.getAttribute("data-copy-target");
    const targetEl = document.getElementById(targetId);
    if (targetEl) {
      const text = targetEl.textContent;
      navigator.clipboard.writeText(text).then(() => {
        const originalText = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => {
          btn.textContent = originalText;
        }, 1500);
      }).catch(err => {
        console.error("Failed to copy:", err);
        showToast("Failed to copy to clipboard", "error");
      });
    }
  });
});
