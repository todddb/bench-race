const assert = require("node:assert/strict");
const {
  ensureSeriesMinimumPoints,
  resolveEmptySparklineMessage,
} = require("../../central/static/js/sparkline_utils.js");

const runSingleSampleTest = () => {
  const result = ensureSeriesMinimumPoints({ values: [42], times: [10] });
  assert.deepStrictEqual(result.values, [42, 42]);
  assert.deepStrictEqual(result.times, [10, 10]);
};

const runEmptySeriesTest = () => {
  const result = ensureSeriesMinimumPoints({ values: [], times: [] });
  assert.deepStrictEqual(result.values, []);
  assert.deepStrictEqual(result.times, []);
};

const runEmptyMessageTest = () => {
  assert.equal(
    resolveEmptySparklineMessage(0),
    "No samples captured (run < sampler interval)",
  );
};

runSingleSampleTest();
runEmptySeriesTest();
runEmptyMessageTest();

console.log("sparkline utils tests passed");
