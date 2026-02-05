# Manual Testing Guide: Agent Offline Detection Hysteresis

This guide describes how to manually test the new hysteresis-based agent offline detection feature.

## Overview

The new system prevents agents from flickering to "Offline" status due to brief health check misses during CPU-heavy workloads. It implements:

1. **Hysteresis logic**: Requires 3 consecutive failures OR 12s without success before marking offline
2. **Degraded state**: Shows yellow "Degraded" status after missed checks but before offline
3. **Run-aware behavior**: Agents running jobs stay in "Degraded" longer before going offline
4. **Diagnostic tooltips**: Hover over status badges to see detailed health information

## Test Scenarios

### Test 1: Single Missed Health Check (Should NOT Go Offline)

**Expected behavior**: Agent should show "Degraded" briefly, then recover to "Ready" without showing "Offline".

**Steps**:
1. Start central: `cd central && python app.py`
2. Start an agent: `cd agent && python agent_app.py`
3. Observe agent shows "Ready" in UI
4. Temporarily block agent port (e.g., firewall rule for 2-3 seconds)
5. Observe agent shows "Degraded" (yellow) instead of "Offline" (red)
6. Unblock agent port
7. Observe agent recovers to "Ready" within 2 seconds

**Success criteria**: Agent never shows "Offline" status.

### Test 2: Three Consecutive Missed Checks (Should Go Offline)

**Expected behavior**: After 3 consecutive failures, agent should show "Offline".

**Steps**:
1. Start central and agent
2. Block agent port for ~8 seconds (enough for 3+ health checks at 2s interval)
3. Observe progression: Ready → Degraded (after 1st miss) → Degraded (after 2nd) → Offline (after 3rd)
4. Check central logs for status transition messages
5. Unblock agent
6. Observe agent recovers: Offline → Ready

**Success criteria**:
- Agent shows "Degraded" after 1-2 failures
- Agent shows "Offline" after 3 failures
- Logs show detailed transition info (consecutive_failures, last_success_age_s, last_error)

### Test 3: CPU-Heavy Compute Load (Primary Use Case)

**Expected behavior**: Agent should NOT go offline during heavy compute, even if it misses 1-2 health checks.

**Steps**:
1. Start central and agent
2. Navigate to Compute benchmark page
3. Run compute with N=100,000,000 (segmented_sieve algorithm)
4. During compute, watch agent status in UI
5. Status should remain "Running" (green) even if compute briefly blocks
6. If a health check is missed, status might briefly show "Running" with degraded indicator (stays green)
7. After compute completes, status returns to "Ready"

**Success criteria**:
- Agent never shows "Offline" during compute
- UI shows "Running" status throughout
- Hover tooltip may show "missed N checks" but agent stays reachable
- Central logs show any degraded transitions but not offline transitions

### Test 4: Extended Downtime (Should Go Offline)

**Expected behavior**: If agent is truly down for >12 seconds, it should eventually show "Offline".

**Steps**:
1. Start central and agent
2. Stop the agent process completely
3. Observe status transitions over 15 seconds:
   - 0s: Ready
   - 2s: Degraded (1 failure)
   - 4s: Degraded (2 failures)
   - 6s: Offline (3 failures)
   - Stays Offline
4. Restart agent
5. Observe recovery to Ready within 2 seconds

**Success criteria**:
- Agent goes Offline after ~6s (3 failures at 2s interval)
- Recovers quickly when restarted
- Logs show clear transition reasoning

### Test 5: Diagnostic Tooltips

**Expected behavior**: Status badges should show helpful diagnostic information on hover.

**Steps**:
1. Start central and agent
2. Let agent run normally (Ready state) - tooltip should be empty or minimal
3. Simulate 1 failure (Degraded state) - hover to see:
   - "Missed 1 health check; last seen Xs ago. May be busy with compute."
4. Simulate 3+ failures (Offline state) - hover to see:
   - "Missed 3 health checks; last seen Xs ago. Error: [error message]"

**Success criteria**: Tooltips provide actionable diagnostic info.

### Test 6: Multiple Agents with Mixed States

**Expected behavior**: Each agent tracks its own health state independently.

**Steps**:
1. Start central with 2+ agents configured
2. Start agent 1 only
3. Observe:
   - Agent 1: Ready
   - Agent 2: Checking → Degraded → Offline (never started)
4. Start agent 2
5. Observe agent 2 recovers to Ready
6. Stop agent 1
7. Observe only agent 1 goes Offline, agent 2 stays Ready

**Success criteria**: Each agent's status is independent and correct.

## Verification Checklist

After testing, verify:

- [ ] Single missed check does not cause "Offline" status
- [ ] Three consecutive failures do cause "Offline" status
- [ ] Compute workloads (N=100M) do not cause "Offline" flicker
- [ ] Time-based threshold (12s) works correctly
- [ ] Run-aware behavior keeps running agents from going offline prematurely
- [ ] Status tooltips show useful diagnostic information
- [ ] Central logs include detailed transition info
- [ ] Agent recovers quickly (within 2s) when health checks resume
- [ ] UI properly displays Ready (green), Degraded (yellow), Offline (red)
- [ ] WebSocket metrics continue to work during degraded state

## Log Messages to Check

Look for these log messages in central logs during transitions:

**Recovery (info level)**:
```
Agent {machine_id} status recovered: degraded → ready
```

**Degradation/Offline (warning level)**:
```
Agent {machine_id} status changed: ready → degraded (consecutive_failures=1, last_success_age_s=2.1s, last_error='timeout')
Agent {machine_id} status changed: degraded → offline (consecutive_failures=3, last_success_age_s=6.3s, last_error='timeout')
```

## Configuration

The following constants in `central/app.py` control hysteresis behavior:

- `OFFLINE_THRESHOLD_S = 12`: Time threshold for marking offline (seconds)
- `CONSECUTIVE_FAILURE_THRESHOLD = 3`: Number of consecutive failures before offline

Default polling interval in UI: 2s during active runs, 30s when idle.

## Troubleshooting

**Agent still flickers Offline during compute:**
- Check that compute functions yield to event loop (`await asyncio.sleep(0)`)
- Verify CONSECUTIVE_FAILURE_THRESHOLD is set to 3 or higher
- Confirm polling timeout is reasonable (2s default)

**Agent takes too long to show Offline when truly down:**
- Reduce CONSECUTIVE_FAILURE_THRESHOLD or OFFLINE_THRESHOLD_S
- Current settings mark offline after ~6s (3 failures × 2s poll interval)

**Degraded state not showing in UI:**
- Check browser console for JavaScript errors
- Verify CSS includes `.status-badge.degraded` class
- Check that API response includes `agent_status` field

**Tooltips not showing diagnostic info:**
- Verify `health_diagnostics` is in API response
- Check browser console for JavaScript errors in `updateMachineStatus`
- Ensure tooltip is set with `statusBadge.title = tooltip`
