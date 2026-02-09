(() => {
  const ensureSeriesMinimumPoints = ({ values, times }) => {
    const safeValues = Array.isArray(values) ? values : [];
    const safeTimes = Array.isArray(times) ? times : [];

    if (safeValues.length === 1) {
      const duplicatedTimes = safeTimes.length === 1 ? [safeTimes[0], safeTimes[0]] : safeTimes;
      return {
        values: [safeValues[0], safeValues[0]],
        times: duplicatedTimes,
      };
    }

    return { values: safeValues, times: safeTimes };
  };

  const resolveEmptySparklineMessage = (sampleCount) => {
    if (sampleCount === 0) {
      return "No samples captured (run < sampler interval)";
    }
    return "Metrics unavailable";
  };

  const sparklineUtils = { ensureSeriesMinimumPoints, resolveEmptySparklineMessage };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = sparklineUtils;
  }

  if (typeof window !== "undefined") {
    window.SparklineUtils = {
      ...(window.SparklineUtils || {}),
      ...sparklineUtils,
    };
  }
})();
