const socket = io();
const paneMap = new Map();
let capabilitiesCache = new Map();

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

// Fetch and update capabilities for all machines
async function fetchCapabilities() {
  try {
    const response = await fetch("/api/capabilities");
    const caps = await response.json();
    capabilitiesCache.clear();
    caps.forEach((cap) => {
      if (cap.machine_id) {
        capabilitiesCache.set(cap.machine_id, cap);
        updateStatusPill(cap);
      }
    });
    return caps;
  } catch (error) {
    console.error("Failed to fetch capabilities", error);
    return [];
  }
}

// Update status pill for a machine based on capabilities
function updateStatusPill(cap) {
  const pill = document.getElementById(`status-pill-${cap.machine_id}`);
  if (!pill) return;

  // Remove all status classes
  pill.classList.remove("ready", "missing-model", "agent-unreachable", "ollama-unreachable", "checking");

  if (cap.agent_reachable === false) {
    pill.textContent = "Agent unreachable";
    pill.title = cap.error || "Cannot connect to agent";
    pill.classList.add("agent-unreachable");
  } else if (cap.ollama_reachable === false) {
    pill.textContent = "Ollama unreachable";
    pill.title = "Ollama API is not responding";
    pill.classList.add("ollama-unreachable");
  } else {
    // Check if selected model is available
    const selectedModel = document.getElementById("model")?.value;
    const ollamaModels = cap.ollama_models || [];
    if (selectedModel && !ollamaModels.includes(selectedModel)) {
      pill.textContent = "Missing model";
      pill.title = `Model "${selectedModel}" not installed. Available: ${ollamaModels.join(", ") || "none"}`;
      pill.classList.add("missing-model");
    } else {
      pill.textContent = "Ready";
      pill.title = "Agent and Ollama ready";
      pill.classList.add("ready");
    }
  }
}

// Update all status pills (e.g., when model selection changes)
function updateAllStatusPills() {
  capabilitiesCache.forEach((cap) => updateStatusPill(cap));
  updatePreflightBanner();
}

// Check preflight status and return blocked machines
function getPreflightStatus() {
  const selectedModel = document.getElementById("model")?.value;
  const blocked = [];
  const ready = [];

  capabilitiesCache.forEach((cap, machineId) => {
    if (cap.agent_reachable === false) {
      blocked.push({ machine_id: machineId, label: cap.label, reason: "agent unreachable" });
    } else if (cap.ollama_reachable === false) {
      blocked.push({ machine_id: machineId, label: cap.label, reason: "Ollama unreachable" });
    } else {
      const ollamaModels = cap.ollama_models || [];
      if (selectedModel && !ollamaModels.includes(selectedModel)) {
        blocked.push({ machine_id: machineId, label: cap.label, reason: `missing model ${selectedModel}` });
      } else {
        ready.push({ machine_id: machineId, label: cap.label });
      }
    }
  });

  return { blocked, ready };
}

// Update preflight warning banner
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
    // Fetch capabilities after loading machines
    await fetchCapabilities();
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
    alert("No machines are ready to run. Check agent and Ollama status.");
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

// Refresh capabilities button
document.getElementById("refresh-caps")?.addEventListener("click", async () => {
  const btn = document.getElementById("refresh-caps");
  if (btn) btn.disabled = true;
  await fetchCapabilities();
  if (btn) btn.disabled = false;
});

// Update status pills when model selection changes
document.getElementById("model")?.addEventListener("change", () => {
  updateAllStatusPills();
});
