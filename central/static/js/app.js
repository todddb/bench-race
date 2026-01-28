const socket = io();
const paneMap = new Map();
const statusCache = new Map();
const syncState = new Map();
const STATUS_POLL_MS = 12000;

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
    } else {
      metrics.innerHTML = `<span class="muted">Job started…</span>`;
    }
  });
};

const formatGb = (value) => (value == null ? "n/a" : `${value.toFixed(1)}GB`);

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
  } catch (error) {
    console.error("Failed to load machines", error);
  }
});

socket.on("agent_event", (evt) => {
  if (!evt || !evt.machine_id) return;
  const pane = paneMap.get(evt.machine_id);
  if (!pane) return;
  const { out, metrics } = pane;

  if (evt.type === "llm_token") {
    const text = evt.payload?.text ?? "";
    out.textContent += text;
    out.scrollTop = out.scrollHeight;
  }

  if (evt.type === "job_done") {
    const payload = evt.payload || {};
    const engine = payload.engine ?? "n/a";
    const isMock = engine === "mock";

    // Build engine display with badge
    let engineDisplay = `<span class="engine-badge ${isMock ? "mock" : "ollama"}">${engine}</span>`;
    if (isMock && payload.fallback_reason) {
      const reasonText = FALLBACK_REASONS[payload.fallback_reason] || payload.fallback_reason;
      engineDisplay += `<div class="fallback-reason">Fallback: ${reasonText}</div>`;
    }

    metrics.innerHTML = `
      <div><strong>Model:</strong> ${payload.model ?? "n/a"}</div>
      <div><strong>Engine:</strong> ${engineDisplay}</div>
      <div><strong>TTFT:</strong> ${payload.ttft_ms != null ? payload.ttft_ms.toFixed(1) : "n/a"} ms</div>
      <div><strong>Gen tokens:</strong> ${payload.gen_tokens ?? "n/a"}</div>
      <div><strong>Tokens/s:</strong> ${payload.gen_tokens_per_s != null ? payload.gen_tokens_per_s.toFixed(1) : "n/a"}</div>
      <div><strong>Total:</strong> ${payload.total_ms != null ? payload.total_ms.toFixed(1) : "n/a"} ms</div>
    `;

    // Highlight the pane if mock was used
    const paneEl = document.getElementById(`pane-${evt.machine_id}`);
    if (paneEl) {
      paneEl.classList.toggle("mock-warning", isMock);
    }
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

socket.on("llm_jobs_started", (results) => {
  console.log("Jobs started", results);
  handleJobResults(results);
});

// Run button with preflight validation
document.getElementById("run").addEventListener("click", async () => {
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
    } else {
      out.textContent = "Starting job…";
      metrics.innerHTML = `<span class="muted">Queued…</span>`;
    }
  });

  try {
    const response = await fetch("/api/start_llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const results = await response.json();
    handleJobResults(results);
  } catch (error) {
    console.error("Failed to start jobs", error);
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

setInterval(() => {
  fetchStatus();
}, STATUS_POLL_MS);
