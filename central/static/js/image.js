const socket = io();
const paneMap = new Map();
const STATUS_POLL_MS = 12000;
const RUN_LIST_LIMIT = 20;
const BASELINE_KEY = "bench-race-baseline-run";
const OPTIONS_STORAGE_KEY = "bench-race-options-expanded";
const MODEL_POLICY_ENDPOINT = "/api/settings/model_policy";
const COMFY_SETTINGS_ENDPOINT = "/api/settings/comfy";

let viewingMode = "live";
let viewingRunId = null;
let liveRunId = null;
let recentRuns = [];
let activeOverlay = null;
let lastOverlayFocus = null;

const showToast = (message, type = "info") => {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
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

const setPaneStatus = (machineId, statusClass, statusText) => {
  const dot = document.getElementById(`status-dot-${machineId}`);
  const text = document.getElementById(`status-text-${machineId}`);
  if (dot) {
    dot.className = `status-dot ${statusClass}`;
  }
  if (text) {
    text.textContent = statusText;
  }
};

const setPaneMetrics = (machineId, html) => {
  const metrics = document.getElementById(`metrics-${machineId}`);
  if (metrics) metrics.innerHTML = html;
};

const setPreviewImage = (machineId, src) => {
  const img = document.getElementById(`preview-${machineId}`);
  if (!img) return;
  img.src = src || "";
  img.style.display = src ? "block" : "none";
};

const updateCheckpointLabel = (checkpoint) => {
  document.querySelectorAll(".model-fit").forEach((el) => {
    el.textContent = checkpoint ? `Checkpoint: ${checkpoint}` : "Checkpoint: --";
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
    machines.forEach((m) => {
      if (!m.reachable) {
        setPaneStatus(m.machine_id, "offline", "Agent offline");
        blockedCount += 1;
        return;
      }
      if (!m.comfy_running) {
        setPaneStatus(m.machine_id, "offline", "ComfyUI down");
        blockedCount += 1;
        return;
      }
      if (m.missing_checkpoint) {
        setPaneStatus(m.machine_id, "missing", "Missing checkpoint");
        blockedCount += 1;
      } else {
        setPaneStatus(m.machine_id, "ready", "Ready");
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
        banner.textContent = `${blockedCount} machine(s) blocked (missing checkpoint or ComfyUI offline).`;
        banner.classList.remove("hidden");
      } else {
        banner.classList.add("hidden");
      }
    }
  } catch (error) {
    console.error("Failed to fetch status", error);
  }
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
        checkpointsInput.value = (comfy.checkpoint_urls || []).join("\n");
      }
    }
    setPolicyFeedback("", "");
    toggleOverlay("settings");
  } catch (error) {
    alert("Could not load settings. Try again.");
  }
});

document.getElementById("refresh-runs")?.addEventListener("click", async () => {
  await fetchRecentRuns();
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
        checkpoint_urls: checkpointUrls,
      }),
    });
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
    try {
      const response = await fetch(`/api/machines/${encodeURIComponent(machineId)}/sync_image`, {
        method: "POST",
      });
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

socket.on("agent_event", (event) => {
  if (!event || !event.type) return;
  if (!event.type.startsWith("image_")) return;
  const payload = event.payload || {};
  const runId = payload.run_id;
  if (viewingMode === "history" && viewingRunId !== runId) return;
  if (viewingMode === "live" && liveRunId && liveRunId !== runId) return;
  const machineId = event.machine_id;
  if (!machineId) return;

  if (event.type === "image_progress") {
    setPaneMetrics(
      machineId,
      `<div><strong>Progress:</strong> ${payload.step ?? 0}/${payload.total_steps ?? 0}</div>`,
    );
  }
  if (event.type === "image_preview") {
    if (payload.image_b64) {
      setPreviewImage(machineId, `data:image/jpeg;base64,${payload.image_b64}`);
    }
  }
  if (event.type === "image_complete") {
    const images = payload.images || [];
    if (images.length > 0 && images[0].image_b64) {
      setPreviewImage(machineId, `data:image/png;base64,${images[0].image_b64}`);
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
    setPaneMetrics(machineId, `<span class="muted">${payload.message || "Error"}</span>`);
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
fetchStatus();
setInterval(fetchStatus, STATUS_POLL_MS);
fetchRecentRuns();

const params = new URLSearchParams(window.location.search);
const runId = params.get("run_id");
if (runId) {
  loadRun(runId);
}
