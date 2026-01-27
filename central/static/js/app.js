const socket = io();
const paneMap = new Map();

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
    metrics.innerHTML = `
      <div><strong>Model:</strong> ${payload.model ?? "n/a"}</div>
      <div><strong>Engine:</strong> ${payload.engine ?? "n/a"}</div>
      <div><strong>TTFT:</strong> ${payload.ttft_ms ?? "n/a"} ms</div>
      <div><strong>Gen tokens:</strong> ${payload.gen_tokens ?? "n/a"}</div>
      <div><strong>Tokens/s:</strong> ${payload.gen_tokens_per_s ?? "n/a"}</div>
      <div><strong>Total:</strong> ${payload.total_ms ?? "n/a"} ms</div>
    `;
  }
});

socket.on("llm_jobs_started", (results) => {
  console.log("Jobs started", results);
  handleJobResults(results);
});

document.getElementById("run").addEventListener("click", () => {
  const payload = {
    model: document.getElementById("model").value,
    prompt: document.getElementById("prompt").value,
    max_tokens: parseInt(document.getElementById("max_tokens").value, 10),
    num_ctx: parseInt(document.getElementById("num_ctx").value, 10),
    temperature: parseFloat(document.getElementById("temperature").value),
    repeat: parseInt(document.getElementById("repeat").value, 10),
  };

  paneMap.forEach(({ out, metrics }) => {
    out.textContent = "Starting job…";
    metrics.innerHTML = `<span class="muted">Queued…</span>`;
  });

  fetch("/api/start_llm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((response) => response.json())
    .then((results) => handleJobResults(results))
    .catch((error) => {
      console.error("Failed to start jobs", error);
    });
});
