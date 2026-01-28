const socket = io();
const paneMap = new Map();
const statusCache = new Map();
const syncState = new Map();
const liveOutput = new Map();
const liveMetrics = new Map();
const liveState = new Map();
const STATUS_POLL_MS = 12000;
const RUN_LIST_LIMIT = 20;
const BASELINE_KEY = "bench-race-baseline-run";

let baselineRunId = localStorage.getItem(BASELINE_KEY);
let baselineRun = null;
let viewingMode = "live";
let viewingRunId = null;
let currentRunData = null;
let liveRunId = null;
let recentRuns = [];

// Fallback reason display strings
const FALLBACK_REASONS = {
  ollama_unreachable: "Ollama was unreachable",
  missing_model: "Model not installed on Ollama",
  stream_error: "Streaming error occurred",
  unknown: "Unknown reason",
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

  dot.classList.remove("ready", "missing", "offline", "checking");
  if (!machine.reachable) {
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
  const status = fit.status || "unknown";
  fitEl.classList.remove("good", "average", "bad", "unknown");
  fitEl.classList.add(status);
  if (status === "unknown") {
    fitEl.textContent = "Model Fit: --";
    return;
  }
  const memoryLabel = fit.memory_label ? ` ${fit.memory_label}` : "";
  fitEl.textContent = `Model Fit: ${status.charAt(0).toUpperCase() + status.slice(1)} (est ${formatGb(
    fit.needed_gb,
  )} / ${formatGb(fit.available_gb)}${memoryLabel})`;
}

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
  } catch (error) {
    console.error("Failed to fetch runs", error);
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
    if (baselineRunId && baselineRunId === run.run_id) {
      badges.push('<span class="run-badge baseline">Baseline</span>');
    }
    const warningText = run.has_mock
      ? '<div class="run-warning-text">Mock engine used on at least one machine.</div>'
      : "";

    item.innerHTML = `
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
        <button class="btn-secondary btn-small" data-action="pin">Pin baseline</button>
        <button class="btn-secondary btn-small" data-action="csv">CSV</button>
        <button class="btn-secondary btn-small" data-action="json">JSON</button>
      </div>
    `;

    item.querySelectorAll("button").forEach((button) => {
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
    renderRunToPanes(run);
  } catch (error) {
    console.error("Failed to load run", error);
  }
};

socket.on("status", (msg) => {
  document.getElementById("status").innerText = msg.ok ? "Connected" : "Not connected";
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
    });
    initSyncButtons(machines);
    await fetchStatus();
    await loadBaselineRun();
    await fetchRecentRuns();
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
});

socket.on("llm_jobs_started", (payload) => {
  const results = Array.isArray(payload) ? payload : payload?.results;
  handleJobResults(results);
});

const runButton = document.getElementById("run");

const startRun = async () => {
  const runOnlyReady = document.getElementById("run-only-ready")?.checked ?? true;
  const { blocked, ready } = getPreflightStatus();

  // If "run only ready" is ON, skip blocked machines
  // If "run only ready" is OFF, block entire run if any machine fails
  if (!runOnlyReady && blocked.length > 0) {
    const reasons = blocked.map((b) => `${b.label || b.machine_id}: ${b.reason}`).join("\n");
    alert(`Cannot run: some machines are not ready.\n\n${reasons}\n\nEnable "Run only ready machines" to skip blocked machines.`);
    return;
  }

  if (ready.length === 0) {
    alert("No machines are ready to run. Check agent status or model availability.");
    return;
  }

  const payload = {
    model: document.getElementById("model").value,
    prompt: document.getElementById("prompt").value,
    max_tokens: parseInt(document.getElementById("max_tokens").value, 10),
    num_ctx: parseInt(document.getElementById("num_ctx").value, 10),
    temperature: parseFloat(document.getElementById("temperature").value),
    repeat: parseInt(document.getElementById("repeat").value, 10),
  };

  // If running only ready machines, specify which ones
  if (runOnlyReady && blocked.length > 0) {
    payload.machine_ids = ready.map((m) => m.machine_id);
  }

  paneMap.forEach(({ out, metrics }, machineId) => {
    // Check if this machine is blocked
    const isBlocked = blocked.some((b) => b.machine_id === machineId);
    if (runOnlyReady && isBlocked) {
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

setInterval(() => {
  fetchStatus();
}, STATUS_POLL_MS);
