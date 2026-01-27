const socket = io();

socket.on("status", (msg) => {
  document.getElementById("status").innerText = msg.ok ? "Connected" : "Not connected";
});

socket.on("llm_jobs_started", (results) => {
  console.log("Jobs started", results);
  // Stub: show job IDs for now
  results.forEach((r) => {
    const out = document.getElementById(`out-${r.machine_id}`);
    const metrics = document.getElementById(`metrics-${r.machine_id}`);
    if (!out || !metrics) return;

    out.textContent = "";
    if (r.error) {
      out.textContent = `Error starting job: ${r.error}`;
      metrics.innerHTML = `<span class="muted">Failed to start</span>`;
    } else {
      out.textContent = `Job started: ${JSON.stringify(r.job)}`;
      metrics.innerHTML = `<span class="muted">Streaming not wired yet</span>`;
    }
  });
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

  socket.emit("llm_run", payload);
});
