const socket = io();
const paneMap = new Map();
const statusCache = new Map();
const RUN_LIST_LIMIT = 20;
const BASELINE_KEY = "bench-race-baseline-run";
const OPTIONS_STORAGE_KEY = "bench-race-options-expanded";
const MODEL_POLICY_ENDPOINT = "/api/settings/model_policy";
const COMFY_SETTINGS_ENDPOINT = "/api/settings/comfy";
const CHECKPOINT_CATALOG_ENDPOINT = "/api/image/checkpoints";
const POLLING_SETTINGS_KEY = "bench-race-polling-settings";
const PREVIEW_SETTINGS_KEY = "bench-race-preview-settings";
const COMPUTE_SETTINGS_KEY = "bench-race-compute-settings";

// Adaptive polling configuration
const DEFAULT_POLLING_CONFIG = {
  idlePollIntervalMs: 30000,   // 30s when no runs active
  activePollIntervalMs: 2000,  // 2s when runs active
  uiUpdateThrottleMs: 500,     // Throttle UI updates to max 2/sec
};
let pollingConfig = { ...DEFAULT_POLLING_CONFIG };

const DEFAULT_COMPUTE_SETTINGS = {
  streamFirstK: 100,
  progressIntervalS: 1.0,
};
let computeSettings = { ...DEFAULT_COMPUTE_SETTINGS };

// Preview settings
const DEFAULT_PREVIEW_CONFIG = {
  enableLivePreview: true,     // Whether to show live preview images
  previewThrottleMs: 1000,     // Throttle preview updates (1 per second)
  previewResolution: 256,       // Preview image size
};
let previewConfig = { ...DEFAULT_PREVIEW_CONFIG };

// Global run state tracking
let isRunActive = false;
const activeRunIds = new Set();
const activeRunMachinesById = new Map();
const activeRunMachineIds = new Set();
let statusPollTimer = null;
let lastPreviewUpdate = new Map(); // machineId -> timestamp

let viewingMode = "live";
let viewingRunId = null;
let liveRunId = null;
let recentRuns = [];
let activeOverlay = null;
let lastOverlayFocus = null;
let checkpointCatalog = [];
const checkpointSyncState = new Map();

// Load saved settings from localStorage
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

const loadPreviewSettings = () => {
  try {
    const saved = localStorage.getItem(PREVIEW_SETTINGS_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      previewConfig = { ...DEFAULT_PREVIEW_CONFIG, ...parsed };
    }
    // Default live preview off on macOS to reduce WindowServer load
    if (navigator.platform?.toLowerCase().includes("mac") && !saved) {
      previewConfig.enableLivePreview = false;
    }
  } catch (e) {
    console.warn("Failed to load preview settings:", e);
  }
};

const savePreviewSettings = () => {
  try {
    localStorage.setItem(PREVIEW_SETTINGS_KEY, JSON.stringify(previewConfig));
  } catch (e) {
    console.warn("Failed to save preview settings:", e);
  }
};

const clampNumber = (value, min, max) => Math.min(Math.max(value, min), max);

const loadComputeSettings = () => {
  try {
    const saved = localStorage.getItem(COMPUTE_SETTINGS_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      computeSettings = { ...DEFAULT_COMPUTE_SETTINGS, ...parsed };
    }
  } catch (e) {
    console.warn("Failed to load compute settings:", e);
  }
  computeSettings.streamFirstK = clampNumber(Number(computeSettings.streamFirstK) || 0, 0, 5000);
  computeSettings.progressIntervalS = clampNumber(
    Number(computeSettings.progressIntervalS) || DEFAULT_COMPUTE_SETTINGS.progressIntervalS,
    0.1,
    60,
  );
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

  // Update preview visibility when run state changes
  updatePreviewVisibility();

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

const updateAllPaneStatuses = () => {
  paneMap.forEach((_pane, machineId) => {
    const machine = statusCache.get(machineId);
    if (machine) {
      updateMachinePaneStatus(machine);
    }
  });
};

const setActiveRunMachines = (runId, machineIds = []) => {
  if (!runId) return;
  activeRunMachinesById.set(runId, new Set(machineIds));
  rebuildActiveRunMachineIds();
  updateAllPaneStatuses();
};

const clearActiveRunMachines = (runId) => {
  if (!runId) return;
  activeRunMachinesById.delete(runId);
  rebuildActiveRunMachineIds();
  updateAllPaneStatuses();
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
    updateAllPaneStatuses();
    updatePreviewVisibility();
    restartPolling();
  } catch (error) {
    console.warn("Failed to fetch active runs:", error);
  }
};

// Preview visibility management - hide when idle, show when active
const updatePreviewVisibility = () => {
  const showPreview = isRunActive && previewConfig.enableLivePreview;
  paneMap.forEach((pane, machineId) => {
    const previewEl = document.getElementById(`preview-${machineId}`);
    const placeholderEl = document.getElementById(`preview-placeholder-${machineId}`);
    if (previewEl) {
      if (showPreview) {
        previewEl.classList.remove("preview-hidden");
      } else {
        previewEl.classList.add("preview-hidden");
      }
    }
    if (placeholderEl) {
      placeholderEl.style.display = showPreview ? "none" : "block";
    }
  });
};

// Check if preview update is allowed (throttling)
const canUpdatePreview = (machineId) => {
  if (!previewConfig.enableLivePreview) return false;
  const now = Date.now();
  const lastUpdate = lastPreviewUpdate.get(machineId) || 0;
  if (now - lastUpdate < previewConfig.previewThrottleMs) {
    return false;
  }
  lastPreviewUpdate.set(machineId, now);
  return true;
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
    // Display label with filename in secondary text
    title.textContent = item.label || item.name || item.url;
    if (item.label && item.filename && item.label !== item.filename) {
      const filenameSub = document.createElement("span");
      filenameSub.style.fontSize = "0.85em";
      filenameSub.style.color = "var(--text-secondary, #666)";
      filenameSub.textContent = ` (${item.filename})`;
      title.appendChild(filenameSub);
    }
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

const renderCheckpointOptions = (items) => {
  const select = document.getElementById("checkpoint");
  if (!(select instanceof HTMLSelectElement)) return;
  const current = select.value;
  select.innerHTML = "";
  items.forEach((item) => {
    const option = document.createElement("option");
    // Store by ID (stable across configs), display label
    option.value = item.id || item.name; // Fallback to name for backwards compat
    option.textContent = item.label || item.name;
    // Add filename as data attribute for tooltip/reference
    if (item.filename) {
      option.setAttribute("data-filename", item.filename);
      option.setAttribute("title", `File: ${item.filename}`);
    }
    select.appendChild(option);
  });
  if (current && items.some((item) => (item.id || item.name) === current)) {
    select.value = current;
  }
};

const loadCheckpointCatalog = async (force = false) => {
  try {
    const response = await fetch(
      `${CHECKPOINT_CATALOG_ENDPOINT}${force ? "?refresh=1" : ""}`,
    );
    if (!response.ok) return;
    const data = await response.json();
    checkpointCatalog = data.items || [];
    const validItems = checkpointCatalog.filter((item) => item.valid);
    renderCheckpointOptions(validItems);
    renderCheckpointValidation(checkpointCatalog);
    updateCheckpointLabel(document.getElementById("checkpoint")?.value || "");
  } catch (error) {
    console.warn("Failed to load checkpoint catalog", error);
  }
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

const setPaneStatus = (machineId, statusClass, statusText) => {
  const badge = document.getElementById(`status-badge-${machineId}`);
  const dot = badge?.querySelector(".status-dot");
  const text = badge?.querySelector(".status-text");
  if (badge) {
    badge.classList.remove("ready", "offline", "standby", "checking", "running");
    if (["ready", "offline", "standby", "checking", "running"].includes(statusClass)) {
      badge.classList.add(statusClass);
    }
  }
  if (dot) {
    dot.className = `status-dot ${statusClass}`;
  }
  if (text) {
    text.textContent = statusText;
  }
  // Update reset button state based on status
  const isOnline = statusClass === "ready" || statusClass === "running" || statusClass === "syncing";
  updateResetButtonState(machineId, isOnline);
};

const setPaneSyncStatus = (machineId, message, percent = null) => {
  const percentText = percent != null ? ` ${percent.toFixed(1)}%` : "";
  setPaneStatus(machineId, "syncing", `Syncing${percentText}`);
  setPaneMetrics(machineId, `<span class="muted">${message}</span>`);
};

const setPaneMetrics = (machineId, html) => {
  const metrics = document.getElementById(`metrics-${machineId}`);
  if (metrics) metrics.innerHTML = html;
};

const updateMachinePaneStatus = (machine) => {
  updateVendorLogo(machine);

  const isRunning = activeRunMachineIds.has(machine.machine_id);

  if (machine.excluded) {
    setPaneStatus(machine.machine_id, "standby", "Standby");
    return { blocked: true, excluded: true };
  }
  if (!machine.reachable) {
    setPaneStatus(machine.machine_id, "offline", "Agent offline");
    return { blocked: true, excluded: false };
  }
  if (!machine.comfy_running) {
    setPaneStatus(machine.machine_id, "offline", "ComfyUI down");
    return { blocked: true, excluded: false };
  }
  if (isRunning) {
    setPaneStatus(machine.machine_id, "running", "Running");
    return { blocked: false, excluded: false };
  }
  if (machine.missing_checkpoint) {
    setPaneStatus(machine.machine_id, "missing", "Missing checkpoint");
    return { blocked: true, excluded: false };
  }

  setPaneStatus(machine.machine_id, "ready", "Ready");
  return { blocked: false, excluded: false };
};

const setPreviewImage = (machineId, src, force = false) => {
  const img = document.getElementById(`preview-${machineId}`);
  if (!img) return;
  // Force is used for final images (not live previews)
  if (!force && !previewConfig.enableLivePreview) {
    return;
  }
  img.src = src || "";
  // Use CSS class instead of inline style for proper preview hiding
  if (src) {
    img.classList.remove("preview-hidden");
  } else {
    img.classList.add("preview-hidden");
  }
};

const updateCheckpointLabel = (checkpointId) => {
  // Find checkpoint in catalog by ID to get label
  let label = checkpointId || "--";
  if (checkpointId && checkpointCatalog.length > 0) {
    const found = checkpointCatalog.find((item) => (item.id || item.name) === checkpointId);
    if (found) {
      label = found.label || found.name || checkpointId;
    }
  }
  document.querySelectorAll(".model-fit").forEach((el) => {
    el.textContent = `Checkpoint: ${label}`;
  });
};

const fetchStatus = async () => {
  const checkpoint = document.getElementById("checkpoint")?.value || "";
  updateCheckpointLabel(checkpoint);
  try {
    const response = await fetch(`/api/image/status?checkpoint=${encodeURIComponent(checkpoint)}`);
    if (!response.ok) return;
    const data = await response.json();
    const machines = data.machines || [];
    let blockedCount = 0;
    let excludedCount = 0;
    statusCache.clear();
    machines.forEach((m) => {
      statusCache.set(m.machine_id, m);
      const result = updateMachinePaneStatus(m);
      if (result.blocked) {
        blockedCount += 1;
      }
      if (result.excluded) {
        excludedCount += 1;
      }
      const syncButton = document.getElementById(`sync-${m.machine_id}`);
      if (syncButton) {
        if (m.missing_checkpoint) {
          syncButton.classList.remove("hidden");
        } else {
          syncButton.classList.add("hidden");
        }
      }
    });
    const banner = document.getElementById("preflight-banner");
    if (banner) {
      if (blockedCount > 0) {
        let message = `${blockedCount} machine(s) blocked`;
        if (excludedCount > 0) {
          message += ` (${excludedCount} excluded`;
          if (blockedCount > excludedCount) {
            message += ", others missing checkpoint or ComfyUI offline";
          }
          message += ")";
        } else {
          message += " (missing checkpoint or ComfyUI offline)";
        }
        banner.textContent = message + ".";
        banner.classList.remove("hidden");
      } else {
        banner.classList.add("hidden");
      }
    }
  } catch (error) {
    console.error("Failed to fetch status", error);
  }
};

const syncCheckpointsAll = async (checkpointNames) => {
  const payload = { checkpoint_names: checkpointNames };
  const response = await fetch("/api/image/sync_checkpoints", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || "Sync failed");
  }
  return response.json();
};

const handleJobResults = (results) => {
  if (!Array.isArray(results)) return;
  results.forEach((r) => {
    if (r.error) {
      setPaneMetrics(r.machine_id, `<span class="muted">Error: ${r.error}</span>`);
      return;
    }
    if (r.skipped) {
      setPaneMetrics(r.machine_id, `<span class="muted">Skipped</span>`);
      return;
    }
    setPaneMetrics(r.machine_id, `<span class="muted">Job started…</span>`);
  });
};

const startRun = async () => {
  const prompt = document.getElementById("prompt")?.value || "";
  const checkpoint = document.getElementById("checkpoint")?.value || "";
  const seedMode = document.getElementById("seed-mode")?.value || "fixed";
  const seedInput = document.getElementById("seed");
  const seed = seedInput ? parseInt(seedInput.value, 10) : null;
  const steps = parseInt(document.getElementById("steps")?.value || "30", 10);
  const resolution = document.getElementById("resolution")?.value || "1024x1024";
  const numImages = parseInt(document.getElementById("num-images")?.value || "1", 10);
  const repeat = parseInt(document.getElementById("repeat")?.value || "1", 10);
  const [width, height] = resolution.split("x").map((value) => parseInt(value, 10));

  const payload = {
    prompt,
    checkpoint,
    seed_mode: seedMode,
    seed: Number.isNaN(seed) ? null : seed,
    steps,
    width,
    height,
    num_images: numImages,
    repeat,
  };

  paneMap.forEach((_, machineId) => {
    setPaneMetrics(machineId, `<span class="muted">Queued…</span>`);
  });

  try {
    const response = await fetch("/api/start_image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      showToast(data.error || "Failed to start run", "error");
      return;
    }
    liveRunId = data.run_id;
    if (seedMode === "random" && seedInput) {
      seedInput.value = data.seed;
    }
    handleJobResults(data.results || []);
    fetchRecentRuns();
    returnToLive();
  } catch (error) {
    console.error("Failed to start image run", error);
    showToast("Failed to start image run.", "error");
  }
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
    if (run.type === "image") {
      badges.push('<span class="run-badge image">Image</span>');
    } else if (run.type === "compute") {
      badges.push('<span class="run-badge compute">Compute</span>');
    } else {
      badges.push('<span class="run-badge inference">Inference</span>');
    }

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
      <div class="recent-run-actions">
        <button class="btn-secondary btn-small" data-action="view">View</button>
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

const renderRunToPanes = (run) => {
  if (!run) return;
  viewingMode = "history";
  viewingRunId = run.run_id;
  const banner = document.getElementById("run-view-banner");
  if (banner) banner.classList.remove("hidden");
  const title = document.getElementById("run-view-title");
  if (title) title.textContent = run.model || "Image run";
  const subtitle = document.getElementById("run-view-subtitle");
  if (subtitle) subtitle.textContent = formatTimestamp(run.timestamp);

  (run.machines || []).forEach((entry) => {
    const machineId = entry.machine_id;
    if (!machineId) return;
    const images = entry.images || [];
    if (images.length > 0) {
      setPreviewImage(machineId, `/api/runs/${encodeURIComponent(run.run_id)}/images/${images[0]}`);
    } else if (entry.preview_path) {
      setPreviewImage(machineId, `/api/runs/${encodeURIComponent(run.run_id)}/images/${entry.preview_path}`);
    }
    const metricsHtml = `
      <div><strong>Queue:</strong> ${formatMetric(entry.queue_latency_ms, " ms")}</div>
      <div><strong>Gen:</strong> ${formatMetric(entry.gen_time_ms, " ms")}</div>
      <div><strong>Total:</strong> ${formatMetric(entry.total_ms, " ms")}</div>
      <div><strong>Steps:</strong> ${entry.steps ?? run.settings?.steps ?? "n/a"}</div>
      <div><strong>Resolution:</strong> ${entry.resolution ?? `${run.settings?.width}x${run.settings?.height}`}</div>
      <div><strong>Seed:</strong> ${entry.seed ?? run.settings?.seed ?? "n/a"}</div>
      <div><strong>Checkpoint:</strong> ${entry.checkpoint ?? run.model ?? "n/a"}</div>
    `;
    setPaneMetrics(machineId, metricsHtml);
    if (entry.status === "error") {
      setPaneStatus(machineId, "offline", "Error");
    } else if (entry.status === "complete") {
      setPaneStatus(machineId, "ready", "Complete");
    }
  });
};

const returnToLive = () => {
  viewingMode = "live";
  viewingRunId = null;
  const banner = document.getElementById("run-view-banner");
  if (banner) banner.classList.add("hidden");
};

const loadRun = async (runId) => {
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    if (!response.ok) return;
    const run = await response.json();
    if (run.type && run.type !== "image") {
      const target = run.type === "compute" ? "compute" : "inference";
      window.location.href = `/${target}?run_id=${encodeURIComponent(runId)}`;
      return;
    }
    renderRunToPanes(run);
  } catch (error) {
    console.error("Failed to load run", error);
  }
};

const loadModelPolicy = async () => {
  const response = await fetch(MODEL_POLICY_ENDPOINT);
  if (!response.ok) throw new Error("Failed to load policy");
  return response.json();
};

const setPolicyFeedback = (message, error) => {
  const errorEl = document.getElementById("model-policy-error");
  const summaryEl = document.getElementById("model-policy-summary");
  if (errorEl) {
    if (error) {
      errorEl.textContent = error;
      errorEl.classList.remove("hidden");
    } else {
      errorEl.classList.add("hidden");
    }
  }
  if (summaryEl) {
    if (message) {
      summaryEl.textContent = message;
      summaryEl.classList.remove("hidden");
    } else {
      summaryEl.classList.add("hidden");
    }
  }
};

document.querySelectorAll(".pane").forEach((pane) => {
  const machineId = pane.id.replace("pane-", "");
  paneMap.set(machineId, pane);
});

const runButton = document.getElementById("run");
const refreshButton = document.getElementById("refresh-caps");
const historyButton = document.getElementById("btn-history");
const settingsButton = document.getElementById("btn-settings");
const seedModeSelect = document.getElementById("seed-mode");
const seedInput = document.getElementById("seed");

seedModeSelect?.addEventListener("change", () => {
  if (!(seedInput instanceof HTMLInputElement)) return;
  if (seedModeSelect.value === "random") {
    seedInput.disabled = true;
  } else {
    seedInput.disabled = false;
  }
});

runButton?.addEventListener("click", async () => {
  await startRun();
});

refreshButton?.addEventListener("click", async () => {
  await fetchStatus();
});

const syncAllButton = document.getElementById("sync-checkpoints");
syncAllButton?.addEventListener("click", async () => {
  if (!(syncAllButton instanceof HTMLButtonElement)) return;
  const selected = document.getElementById("checkpoint")?.value || "";
  let checkpointNames = [];
  if (selected) {
    checkpointNames = [selected];
  } else {
    checkpointNames = checkpointCatalog.filter((item) => item.valid).map((item) => item.name);
  }
  if (!checkpointNames.length) {
    showToast("Select a checkpoint first.", "error");
    return;
  }
  syncAllButton.disabled = true;
  try {
    await syncCheckpointsAll(checkpointNames);
    showToast("Checkpoint sync started.", "info");
  } catch (error) {
    showToast(error.message || "Sync failed.", "error");
  } finally {
    syncAllButton.disabled = false;
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
    // Load polling and preview settings into form
    const enableLivePreviewInput = document.getElementById("enable-live-preview");
    const idlePollInput = document.getElementById("idle-poll-interval");
    const activePollInput = document.getElementById("active-poll-interval");
    const previewThrottleInput = document.getElementById("preview-throttle");
    const computeStreamInput = document.getElementById("compute-stream-first-k");
    const computeIntervalInput = document.getElementById("compute-progress-interval");
    if (enableLivePreviewInput) enableLivePreviewInput.checked = previewConfig.enableLivePreview;
    if (idlePollInput) idlePollInput.value = Math.round(pollingConfig.idlePollIntervalMs / 1000);
    if (activePollInput) activePollInput.value = pollingConfig.activePollIntervalMs;
    if (previewThrottleInput) previewThrottleInput.value = previewConfig.previewThrottleMs;
    if (computeStreamInput) computeStreamInput.value = computeSettings.streamFirstK;
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
    // Save polling and preview settings from form
    const enableLivePreviewInput = document.getElementById("enable-live-preview");
    const idlePollInput = document.getElementById("idle-poll-interval");
    const activePollInput = document.getElementById("active-poll-interval");
    const previewThrottleInput = document.getElementById("preview-throttle");
    if (enableLivePreviewInput) {
      previewConfig.enableLivePreview = enableLivePreviewInput.checked;
    }
    if (idlePollInput) {
      pollingConfig.idlePollIntervalMs = parseInt(idlePollInput.value, 10) * 1000;
    }
    if (activePollInput) {
      pollingConfig.activePollIntervalMs = parseInt(activePollInput.value, 10);
    }
    if (previewThrottleInput) {
      previewConfig.previewThrottleMs = parseInt(previewThrottleInput.value, 10);
    }
    const computeStreamInput = document.getElementById("compute-stream-first-k");
    const computeIntervalInput = document.getElementById("compute-progress-interval");
    if (computeStreamInput) {
      computeSettings.streamFirstK = clampNumber(parseInt(computeStreamInput.value, 10) || 0, 0, 5000);
      computeStreamInput.value = computeSettings.streamFirstK;
    }
    if (computeIntervalInput) {
      computeSettings.progressIntervalS = clampNumber(parseFloat(computeIntervalInput.value) || 1, 0.1, 60);
      computeIntervalInput.value = computeSettings.progressIntervalS;
    }
    savePreviewSettings();
    saveComputeSettings();
    try {
      localStorage.setItem(POLLING_SETTINGS_KEY, JSON.stringify(pollingConfig));
    } catch (e) {
      console.warn("Failed to save polling settings:", e);
    }
    updatePreviewVisibility();
    restartPolling();
    setPolicyFeedback("Settings saved.", "");
    closeOverlay("settings");
  } catch (error) {
    setPolicyFeedback("", error.message || "Failed to save settings");
  } finally {
    if (button) button.disabled = false;
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

document.getElementById("run-view-return")?.addEventListener("click", () => {
  returnToLive();
});

document.querySelectorAll(".btn-sync").forEach((button) => {
  button.addEventListener("click", async (event) => {
    const target = event.currentTarget;
    if (!(target instanceof HTMLButtonElement)) return;
    const machineId = target.id.replace("sync-", "");
    target.disabled = true;
    const checkpoint = document.getElementById("checkpoint")?.value || "";
    if (!checkpoint) {
      showToast("Select a checkpoint first.", "error");
      target.disabled = false;
      return;
    }
    try {
      const response = await fetch(
        `/api/machines/${encodeURIComponent(machineId)}/sync_image_checkpoints`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ checkpoint_names: [checkpoint] }),
        },
      );
      if (!response.ok) throw new Error("Sync failed");
      showToast("Sync started.", "info");
      await fetchStatus();
    } catch (error) {
      showToast("Sync failed.", "error");
    } finally {
      target.disabled = false;
    }
  });
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
  } else if (event.type === "run_end") {
    setRunActive(false, event.run_id);
    clearActiveRunMachines(event.run_id);
    console.log("Run ended:", event.run_id);
  }
});

socket.on("agent_event", (event) => {
  if (!event || !event.type) return;
  if (!event.type.startsWith("image_")) return;
  const payload = event.payload || {};
  const runId = payload.run_id;
  if (viewingMode === "history" && viewingRunId !== runId) return;
  if (viewingMode === "live" && liveRunId && liveRunId !== runId) return;
  const machineId = event.machine_id;
  if (!machineId) return;

  if (event.type === "image_checkpoint_sync_start") {
    checkpointSyncState.set(machineId, { active: true, name: payload.items?.[0] });
    setPaneSyncStatus(machineId, "Starting checkpoint sync...");
    return;
  }
  if (event.type === "image_checkpoint_sync_progress") {
    const name = payload.name || payload.model;
    if (name) {
      checkpointSyncState.set(machineId, { active: true, name, percent: payload.percent });
    }
    const message = name ? `${name}: ${payload.message || "Syncing"}` : payload.message || "Syncing";
    setPaneSyncStatus(machineId, message, payload.percent);
    if (payload.error) {
      setPaneMetrics(machineId, `<span class="muted">Error: ${payload.error}</span>`);
    }
    return;
  }
  if (event.type === "image_checkpoint_sync_done") {
    const results = payload.results || [];
    if (results.length) {
      const lines = results.map((result) => {
        if (result.status === "error") {
          return `${result.name}: error`;
        }
        return `${result.name}: ${result.status}`;
      });
      setPaneMetrics(machineId, `<span class="muted">${lines.join(" · ")}</span>`);
      const hasError = results.some((result) => result.status === "error");
      setPaneStatus(machineId, hasError ? "missing" : "ready", hasError ? "Sync error" : "Sync complete");
    }
    fetchStatus();
    return;
  }

  if (event.type === "image_progress") {
    setPaneMetrics(
      machineId,
      `<div><strong>Progress:</strong> ${payload.step ?? 0}/${payload.total_steps ?? 0}</div>`,
    );
  }
  if (event.type === "image_preview") {
    // Apply throttling to preview updates to reduce CPU/GPU load
    if (payload.image_b64 && canUpdatePreview(machineId)) {
      setPreviewImage(machineId, `data:image/jpeg;base64,${payload.image_b64}`);
    }
  }
  if (event.type === "image_complete") {
    const images = payload.images || [];
    if (images.length > 0 && images[0].image_b64) {
      // Force show final image even if live preview is disabled
      setPreviewImage(machineId, `data:image/png;base64,${images[0].image_b64}`, true);
    }
    const metricsHtml = `
      <div><strong>Queue:</strong> ${formatMetric(payload.queue_latency_ms, " ms")}</div>
      <div><strong>Gen:</strong> ${formatMetric(payload.gen_time_ms, " ms")}</div>
      <div><strong>Total:</strong> ${formatMetric(payload.total_ms, " ms")}</div>
      <div><strong>Steps:</strong> ${payload.steps ?? "n/a"}</div>
      <div><strong>Resolution:</strong> ${payload.resolution ?? "n/a"}</div>
      <div><strong>Seed:</strong> ${payload.seed ?? "n/a"}</div>
      <div><strong>Checkpoint:</strong> ${payload.checkpoint ?? "n/a"}</div>
    `;
    setPaneMetrics(machineId, metricsHtml);
    setPaneStatus(machineId, "ready", "Complete");
  }
  if (event.type === "image_error") {
    const remediation = Array.isArray(payload.remediation) && payload.remediation.length
      ? `<ul class="remediation">${payload.remediation.map((item) => `<li>${item}</li>`).join("")}</ul>`
      : "";
    setPaneMetrics(
      machineId,
      `<div class="muted">${payload.message || "Error"}</div>${remediation}`,
    );
    setPaneStatus(machineId, "offline", "Error");
    document.getElementById(`pane-${machineId}`)?.classList.add("error");
  }
});

const applyOptionsLayout = () => {
  const expanded = localStorage.getItem(OPTIONS_STORAGE_KEY);
  if (expanded === "true") {
    document.body.classList.add("options-expanded");
    document.body.classList.remove("options-collapsed");
  } else {
    document.body.classList.add("options-collapsed");
    document.body.classList.remove("options-expanded");
  }
};

document.getElementById("options-toggle")?.addEventListener("click", () => {
  const expanded = document.body.classList.contains("options-expanded");
  if (expanded) {
    document.body.classList.remove("options-expanded");
    document.body.classList.add("options-collapsed");
    localStorage.setItem(OPTIONS_STORAGE_KEY, "false");
  } else {
    document.body.classList.add("options-expanded");
    document.body.classList.remove("options-collapsed");
    localStorage.setItem(OPTIONS_STORAGE_KEY, "true");
  }
});

applyOptionsLayout();

// Toggle machine excluded status
async function toggleMachineExcluded(machineId) {
  try {
    // Get current state from pane data attribute
    const pane = document.getElementById(`pane-${machineId}`);
    const currentExcluded = pane?.getAttribute("data-excluded") === "true";
    const newExcludedState = !currentExcluded;

    const response = await fetch(`/api/machines/${encodeURIComponent(machineId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ excluded: newExcludedState }),
    });

    if (!response.ok) {
      throw new Error(`Failed to update machine: ${response.statusText}`);
    }

    const data = await response.json();

    // Update UI
    updateMachineExcludedUI(machineId, data.excluded);
    const machine = statusCache.get(machineId);
    if (machine) {
      machine.excluded = data.excluded;
      updateMachinePaneStatus(machine);
    }

    // Refresh status to update banner
    await fetchStatus();

    console.log(`Machine ${machineId} excluded status updated to: ${data.excluded}`);
  } catch (error) {
    console.error(`Failed to toggle excluded status for ${machineId}:`, error);
    showToast(`Failed to update machine status: ${error.message}`, "error");
  }
}

function updateMachineExcludedUI(machineId, excluded) {
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

// Initialize toggle buttons for all machines
function initToggleButtons() {
  const toggleCheckboxes = document.querySelectorAll(".toggle-exclude-input");
  toggleCheckboxes.forEach((checkbox) => {
    const machineId = checkbox.getAttribute("data-machine-id");
    if (machineId) {
      checkbox.addEventListener("change", () => toggleMachineExcluded(machineId));
    }
  });
}

// Reset agent
async function resetAgent(machineId) {
  const btn = document.getElementById(`reset-${machineId}`);
  const statusBadge = document.getElementById(`status-badge-${machineId}`);
  if (!btn) return;

  // Check if agent is online by looking at pane status
  const pane = document.getElementById(`pane-${machineId}`);
  const isOffline = statusBadge?.classList.contains("offline");

  if (isOffline) {
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
      const label = pane?.querySelector(".pane-title")?.textContent || machineId;
      if (result.warnings) {
        showToast(`Reset complete (warnings) for ${label}`, "warning");
      } else {
        showToast(`Reset successful for ${label}`, "success");
      }
      // Refresh status after reset (will update to "Ready")
      await fetchStatus();
    } else {
      const errorMsg = result.error || "Reset failed";
      showToast(`Reset failed: ${errorMsg}`, "error");
      console.error("Reset failed:", result);
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
function updateResetButtonState(machineId, isOnline) {
  const btn = document.getElementById(`reset-${machineId}`);
  if (btn) {
    btn.disabled = !isOnline;
    btn.title = isOnline
      ? "Reset Agent (restart Ollama + ComfyUI)"
      : "Agent offline";
  }
}

// Call initialization
initToggleButtons();
initResetButtons();

// Initialize settings and adaptive polling
loadPollingSettings();
loadPreviewSettings();
loadComputeSettings();
loadCheckpointCatalog();
fetchStatus();
scheduleStatusPoll();
fetchRecentRuns();
// Fetch initial run state
fetchActiveRuns();

const params = new URLSearchParams(window.location.search);
const runId = params.get("run_id");
if (runId) {
  loadRun(runId);
}
