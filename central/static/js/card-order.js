/**
 * card-order.js
 * Drag-and-drop card reordering with SortableJS + localStorage persistence.
 * Loaded after app.js / image.js and Sortable.min.js on every page mode.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "bench-race:cardOrder:v1";

  /* ---- helpers ---- */

  function getMode() {
    return document.body.getAttribute("data-mode") || "inference";
  }

  function storageKey() {
    return STORAGE_KEY + ":" + getMode();
  }

  function readOrder() {
    try {
      var raw = localStorage.getItem(storageKey());
      if (raw) return JSON.parse(raw);
    } catch (_) {}
    return null;
  }

  function saveOrder(ids) {
    try {
      localStorage.setItem(storageKey(), JSON.stringify(ids));
    } catch (_) {}
  }

  /** Extract machine_id from a pane element's id attribute (e.g. "pane-foo" -> "foo"). */
  function machineId(el) {
    return (el.id || "").replace(/^pane-/, "");
  }

  /* ---- restore saved order by reordering DOM ---- */

  function restoreOrder(grid) {
    var saved = readOrder();
    if (!saved || !saved.length) return;

    var panes = Array.from(grid.children).filter(function (el) {
      return el.classList.contains("pane");
    });
    var paneMap = {};
    panes.forEach(function (p) {
      paneMap[machineId(p)] = p;
    });

    // Build ordered list: saved IDs first, then any new (unknown) IDs appended
    var seen = {};
    var ordered = [];
    saved.forEach(function (id) {
      if (paneMap[id]) {
        ordered.push(paneMap[id]);
        seen[id] = true;
      }
    });
    panes.forEach(function (p) {
      var id = machineId(p);
      if (!seen[id]) ordered.push(p);
    });

    // Re-append in order (moves existing DOM nodes, no re-creation)
    ordered.forEach(function (p) {
      grid.appendChild(p);
    });
  }

  /* ---- current order snapshot ---- */

  function currentOrder(grid) {
    return Array.from(grid.children)
      .filter(function (el) {
        return el.classList.contains("pane");
      })
      .map(function (el) {
        return machineId(el);
      });
  }

  /* ---- keyboard reorder support ---- */

  function setupKeyboard(grid) {
    grid.addEventListener("keydown", function (e) {
      var handle = e.target.closest(".drag-handle");
      if (!handle) return;
      var pane = handle.closest(".pane");
      if (!pane) return;

      var panes = Array.from(grid.children).filter(function (el) {
        return el.classList.contains("pane");
      });
      var idx = panes.indexOf(pane);

      var moved = false;
      if ((e.key === "ArrowUp" || e.key === "ArrowLeft") && idx > 0) {
        grid.insertBefore(pane, panes[idx - 1]);
        moved = true;
      } else if (
        (e.key === "ArrowDown" || e.key === "ArrowRight") &&
        idx < panes.length - 1
      ) {
        // insertBefore the element *after* the next sibling to move down
        var next = panes[idx + 1];
        if (next.nextSibling) {
          grid.insertBefore(pane, next.nextSibling);
        } else {
          grid.appendChild(pane);
        }
        moved = true;
      }

      if (moved) {
        e.preventDefault();
        saveOrder(currentOrder(grid));
        handle.focus();
      }
    });
  }

  /* ---- init ---- */

  function init() {
    if (typeof Sortable === "undefined") {
      console.warn("card-order.js: SortableJS not loaded, skipping drag-and-drop init.");
      return;
    }

    var grid = document.querySelector(".grid");
    if (!grid) return;

    // Restore saved order before SortableJS binds
    restoreOrder(grid);

    Sortable.create(grid, {
      handle: ".drag-handle",
      animation: 200,
      easing: "cubic-bezier(0.25, 1, 0.5, 1)",
      ghostClass: "sortable-ghost",
      dragClass: "sortable-drag",
      chosenClass: "sortable-chosen",
      forceFallback: false,
      fallbackTolerance: 3,
      onEnd: function () {
        saveOrder(currentOrder(grid));
      },
    });

    // Keyboard support
    setupKeyboard(grid);
  }

  // Run on DOMContentLoaded or immediately if already loaded
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
