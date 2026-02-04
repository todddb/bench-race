# Runtime Metrics & Model Fit

## Model fit score

The central service computes a memory-first fit score for each agent and selected model:

```
usable_vram = vram_bytes * usable_vram_fraction
estimated_peak = model_bytes * activation_multiplier
fit_ratio = estimated_peak / usable_vram
fit_score = 1 / fit_ratio
```

Labels are derived from `fit_ratio`:

| Fit ratio | Label |
| --- | --- |
| <= 0.6 | good |
| 0.6–0.8 | marginal |
| 0.8–1.0 | risk |
| > 1.0 | fail |

The UI badge shows the label, fit score, and a tooltip with a breakdown and suggested actions.

## Runtime metrics

Agents sample metrics on a rolling buffer:

```yaml
runtime_sampler:
  enabled: true
  interval_s: 1
  buffer_len: 120
```

Metrics include CPU utilization, GPU utilization (when available), VRAM usage, and system memory
usage. Agents expose the buffer at `/api/agent/runtime_metrics` and push updates to the central
websocket as `runtime_metrics_update`.

If GPU metrics are unavailable, GPU fields are `null` and the UI shows placeholders while
using system memory as a proxy for unified memory.
