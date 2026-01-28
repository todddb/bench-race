/**
 * Service Control Card - Frontend logic for managing agent/central services
 */

(function() {
  'use strict';

  // Configuration
  const POLL_INTERVAL_MS = 3000;
  const BACKOFF_MULTIPLIER = 1.5;
  const MAX_BACKOFF_MS = 30000;

  // State
  const serviceState = {
    agent: { running: false, pid: null, info: null },
    central: { running: false, pid: null, info: null }
  };
  let currentBackoff = POLL_INTERVAL_MS;
  let pollTimeoutId = null;

  // Toast notifications
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    // Auto-remove after 4 seconds
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // Update UI for a specific service
  function updateServiceUI(component) {
    const state = serviceState[component];
    const statusDot = document.getElementById(`${component}-status-dot`);
    const statusText = document.getElementById(`${component}-status-text`);
    const startBtn = document.getElementById(`${component}-start`);
    const stopBtn = document.getElementById(`${component}-stop`);
    const pidSpan = document.getElementById(`${component}-pid`);
    const infoSpan = document.getElementById(`${component}-info`);

    if (!statusDot || !statusText) return;

    if (state.running) {
      statusDot.className = 'status-dot running';
      statusText.textContent = 'Running';
      startBtn.disabled = true;
      stopBtn.disabled = false;
    } else {
      statusDot.className = 'status-dot stopped';
      statusText.textContent = 'Stopped';
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }

    if (pidSpan) pidSpan.textContent = state.pid || '-';
    if (infoSpan) infoSpan.textContent = state.info || '-';
  }

  // Fetch status for a component
  async function fetchStatus(component) {
    try {
      const response = await fetch(`/api/service/${component}/status`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      serviceState[component] = {
        running: data.running || false,
        pid: data.pid || null,
        info: data.info || null
      };
      updateServiceUI(component);

      // Reset backoff on success
      currentBackoff = POLL_INTERVAL_MS;
      return true;
    } catch (error) {
      console.error(`Failed to fetch ${component} status:`, error);
      return false;
    }
  }

  // Fetch all service statuses
  async function fetchAllStatuses() {
    const agentOk = await fetchStatus('agent');
    const centralOk = await fetchStatus('central');

    if (!agentOk || !centralOk) {
      // Apply exponential backoff
      currentBackoff = Math.min(currentBackoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_MS);
    }

    // Schedule next poll
    pollTimeoutId = setTimeout(fetchAllStatuses, currentBackoff);
  }

  // Perform action (start/stop) on a service
  async function performAction(component, action) {
    const btn = document.getElementById(`${component}-${action}`);
    if (btn) {
      btn.disabled = true;
      btn.classList.add('loading');
    }

    try {
      const response = await fetch(`/api/service/${component}/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || data.message || `HTTP ${response.status}`);
      }

      // Show success message
      const actionVerb = action === 'start' ? 'started' : 'stopped';
      showToast(`${component.charAt(0).toUpperCase() + component.slice(1)} ${actionVerb}`, 'success');

      // Immediately refresh status
      await fetchStatus(component);
    } catch (error) {
      console.error(`Failed to ${action} ${component}:`, error);
      showToast(`Failed to ${action} ${component}: ${error.message}`, 'error');

      // Re-enable button on error
      if (btn) {
        btn.disabled = false;
      }
    } finally {
      if (btn) {
        btn.classList.remove('loading');
      }
    }
  }

  // Toggle details visibility
  function toggleDetails(component) {
    const details = document.getElementById(`${component}-details`);
    if (details) {
      details.classList.toggle('show');
    }
  }

  // Initialize event listeners
  function initEventListeners() {
    // Agent buttons
    document.getElementById('agent-start')?.addEventListener('click', () => performAction('agent', 'start'));
    document.getElementById('agent-stop')?.addEventListener('click', () => performAction('agent', 'stop'));
    document.getElementById('agent-details-btn')?.addEventListener('click', () => toggleDetails('agent'));

    // Central buttons
    document.getElementById('central-start')?.addEventListener('click', () => performAction('central', 'start'));
    document.getElementById('central-stop')?.addEventListener('click', () => performAction('central', 'stop'));
    document.getElementById('central-details-btn')?.addEventListener('click', () => toggleDetails('central'));
  }

  // Initialize on DOM ready
  function init() {
    initEventListeners();
    fetchAllStatuses();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    if (pollTimeoutId) {
      clearTimeout(pollTimeoutId);
    }
  });
})();
