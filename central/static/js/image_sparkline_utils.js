const DEFAULT_SAMPLER_INTERVAL_S = 1.0;

const resolveSamplerIntervalS = (samplerIntervalS) => {
  const resolved = Number(samplerIntervalS);
  if (Number.isFinite(resolved) && resolved > 0) return resolved;
  return DEFAULT_SAMPLER_INTERVAL_S;
};

const selectWindowedSampleIndices = (timestamps, window, samplerIntervalS, fallbackPoints = 12) => {
  if (!Array.isArray(timestamps) || timestamps.length === 0) return [];
  const intervalS = resolveSamplerIntervalS(samplerIntervalS);
  const padS = Math.max(2 * intervalS, 1.0);
  const startS = window?.startMs != null ? window.startMs / 1000 : null;
  const endS = window?.endMs != null ? window.endMs / 1000 : null;
  const startBound = startS != null ? startS - padS : null;
  const endBound = endS != null ? endS + padS : null;
  let indices = [];

  if (startBound != null || endBound != null) {
    timestamps.forEach((ts, index) => {
      if ((startBound == null || ts >= startBound) && (endBound == null || ts <= endBound)) {
        indices.push(index);
      }
    });
  }

  if (indices.length === 0) {
    if (endBound != null) {
      const candidates = [];
      timestamps.forEach((ts, index) => {
        if (ts <= endBound) candidates.push(index);
      });
      if (candidates.length > 0) {
        indices = candidates.slice(-fallbackPoints);
      }
    }

    if (indices.length === 0) {
      const startIndex = Math.max(0, timestamps.length - fallbackPoints);
      indices = Array.from({ length: timestamps.length - startIndex }, (_, offset) => startIndex + offset);
    }
  }

  if (indices.length === 1) {
    indices = [indices[0], indices[0]];
  }

  return indices;
};

const sparklineUtils = { selectWindowedSampleIndices };

if (typeof module !== "undefined" && module.exports) {
  module.exports = sparklineUtils;
}

if (typeof window !== "undefined") {
  window.ImageSparklineUtils = {
    ...(window.ImageSparklineUtils || {}),
    ...sparklineUtils,
  };
}
