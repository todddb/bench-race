const assert = require("node:assert/strict");
const { createComfyStartupController } = require("../../central/static/js/comfy_startup.js");

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const testReadyPath = async () => {
  let statusMessage = "";
  let ready = false;
  let checks = 0;
  const controller = createComfyStartupController({
    startEngine: async () => {},
    checkHealth: async () => {
      checks += 1;
      return checks >= 2;
    },
    onStatus: (msg) => { statusMessage = msg; },
    onReady: () => { ready = true; },
    intervalMs: 5,
    maxAttempts: 5,
  });

  await controller.start();
  await sleep(20);

  assert.equal(statusMessage, "Starting ComfyUI…");
  assert.equal(ready, true);
};

const testTimeoutPath = async () => {
  let error = null;
  const controller = createComfyStartupController({
    startEngine: async () => {},
    checkHealth: async () => false,
    onError: (err) => { error = err; },
    intervalMs: 5,
    maxAttempts: 2,
  });

  await controller.start();
  await sleep(25);

  assert.ok(error instanceof Error);
};

(async () => {
  await testReadyPath();
  await testTimeoutPath();
  console.log("comfy startup controller tests passed");
})();
