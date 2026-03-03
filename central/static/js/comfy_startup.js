(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.createComfyStartupController = factory().createComfyStartupController;
  }
}(typeof self !== "undefined" ? self : this, function () {
  const createComfyStartupController = ({
    startEngine,
    checkHealth,
    onStatus,
    onReady,
    onError,
    intervalMs = 2000,
    maxAttempts = 30,
  }) => {
    let attempts = 0;
    let timer = null;
    let stopped = false;
    let startInFlight = null;

    const clearTimer = () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    };

    const stop = () => {
      stopped = true;
      clearTimer();
    };

    const start = async () => {
      if (startInFlight) return startInFlight;
      stopped = false;
      attempts = 0;
      onStatus?.("Starting ComfyUI…");
      startInFlight = (async () => {
        try {
          await startEngine();
        } catch (error) {
          onError?.(error);
          clearTimer();
          return;
        }

        timer = setInterval(async () => {
          if (stopped) {
            clearTimer();
            return;
          }
          attempts += 1;
          try {
            const healthy = await checkHealth();
            if (healthy) {
              clearTimer();
              onReady?.();
              return;
            }
          } catch (_error) {
            // Continue polling until timeout.
          }

          if (attempts >= maxAttempts) {
            clearTimer();
            onError?.(new Error("ComfyUI failed to become healthy before timeout."));
          }
        }, intervalMs);
      })();

      await startInFlight;
      startInFlight = null;
    };

    return { start, stop };
  };

  return { createComfyStartupController };
}));
