const assert = require("node:assert/strict");
const { selectWindowedSampleIndices } = require("../../central/static/js/image_sparkline_utils.js");

const runZeroSampleWindowTest = () => {
  const timestamps = [50, 51, 52];
  const window = { startMs: 200000, endMs: 201000 };
  const indices = selectWindowedSampleIndices(timestamps, window, 1.0, 12);
  assert.ok(indices.length >= 2, "Expected at least 2 indices after fallback");
  assert.ok(indices.every((index) => index >= 0 && index < timestamps.length));
};

const runOneSampleWindowTest = () => {
  const timestamps = [100];
  const window = { startMs: 100000, endMs: 100000 };
  const indices = selectWindowedSampleIndices(timestamps, window, 1.0, 12);
  assert.equal(indices.length, 2, "Expected duplicated index for single sample");
  assert.equal(indices[0], indices[1], "Expected duplicated index values");
};

const runMultiSampleWindowTest = () => {
  const timestamps = [10, 11, 12, 13];
  const window = { startMs: 10000, endMs: 13000 };
  const indices = selectWindowedSampleIndices(timestamps, window, 1.0, 12);
  assert.ok(indices.length >= 2, "Expected at least 2 indices for multi-sample window");
  assert.ok(indices.every((index) => index >= 0 && index < timestamps.length));
};

runZeroSampleWindowTest();
runOneSampleWindowTest();
runMultiSampleWindowTest();

console.log("image sparkline utils tests passed");
