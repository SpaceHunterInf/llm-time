"use strict";

/* Progressive Enhancement Only:
   100% of prose, walkthroughs, tables, MathML equations, and SVG charts are pre-rendered
   server-side in Python so the portal works identically even with JavaScript disabled.
   Security compliance: zero innerHTML/outerHTML/localStorage/alert/inline .style mutations. */

(function () {
  function makeEl(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function initThemeToggle() {
    var btn = document.getElementById("theme-toggle-btn");
    var lbl = document.getElementById("theme-toggle-label");
    if (!btn) {
      return;
    }
    btn.addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme") || "light";
      var next = cur === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      btn.setAttribute("aria-pressed", String(next === "dark"));
      if (lbl) {
        lbl.textContent = next === "dark" ? "Light" : "Dark";
      }
    });
  }

  function initPresentation() {
    var slides = Array.from(document.querySelectorAll("[data-slide]"));
    var start = document.getElementById("present-brief");
    var controls = document.querySelector(".slide-controls");
    var previous = document.getElementById("prev-slide");
    var next = document.getElementById("next-slide");
    var exit = document.getElementById("exit-slides");
    var index = 0;
    var active = false;

    function show(target) {
      index = Math.max(0, Math.min(slides.length - 1, target));
      slides.forEach(function (slide, i) { slide.classList.toggle("current-slide", i === index); });
      document.getElementById("slide-count").textContent = (index + 1) + " / " + slides.length;
      document.getElementById("slide-name").textContent = slides[index].dataset.slide;
      previous.disabled = index === 0;
      next.disabled = index === slides.length - 1;
      var heading = slides[index].querySelector("h1, h2");
      heading.tabIndex = -1;
      heading.focus({preventScroll: true});
      window.scrollTo(0, 0);
    }

    function closePresentation() {
      active = false;
      document.documentElement.classList.remove("presenting");
      controls.hidden = true;
      slides[index].scrollIntoView();
      start.focus({preventScroll: true});
    }

    function followFragment() {
      if (!location.hash) { return; }
      var target = document.getElementById(location.hash.slice(1));
      if (!target) { return; }
      if (active) { closePresentation(); }
      var chapter = target.closest(".notebook-chapter");
      if (chapter) {
        chapter.open = true;
        target.scrollIntoView();
      }
    }

    start.addEventListener("click", function () {
      var top = document.querySelector(".brief-mast").getBoundingClientRect().bottom + 32;
      var visible = slides.findIndex(function (slide) { return slide.getBoundingClientRect().bottom > top; });
      active = true;
      controls.hidden = false;
      document.documentElement.classList.add("presenting");
      show(Math.max(0, visible));
    });
    previous.addEventListener("click", function () { show(index - 1); });
    next.addEventListener("click", function () { show(index + 1); });
    exit.addEventListener("click", closePresentation);
    document.querySelector(".brief-brand").addEventListener("click", function () {
      if (active) { closePresentation(); }
    });
    document.addEventListener("keydown", function (event) {
      if (!active || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) { return; }
      if (event.key === "Escape") { closePresentation(); }
      else if (event.key === "ArrowRight" || event.key === "PageDown") { show(index + 1); }
      else if (event.key === "ArrowLeft" || event.key === "PageUp") { show(index - 1); }
      else { return; }
      event.preventDefault();
    });
    document.addEventListener("click", function (event) {
      var link = event.target.closest("a");
      if (link && link.hash && link.hash === location.hash) { followFragment(); }
    });
    window.addEventListener("hashchange", followFragment);
    followFragment();
  }

  function renderSelectedTrace(sample) {
    var box = document.getElementById("trace-viewer-box");
    if (!box || !sample) {
      return;
    }
    box.replaceChildren();

    var bannerText =
      "Run: " +
      String(sample.run_id || "") +
      " · Env: " +
      String(sample.env_slug || "") +
      " · Arm: " +
      String(sample.harness || "") +
      " · Seed: " +
      String(sample.seed || "") +
      " · Episode #" +
      String(sample.episode || "") +
      " · Public Score: " +
      String(sample.public_score || 0) +
      " (" +
      String(sample.normalized_utility_pct || 0) +
      "% Utility)";
    box.appendChild(makeEl("div", "trace-sample-banner mono", bannerText));

    var steps = Array.isArray(sample.steps) ? sample.steps : [];
    steps.forEach(function (st) {
      var card = makeEl("div", "trace-turn-box");
      card.appendChild(
        makeEl(
          "div",
          "trace-turn-label mono",
          "Turn " + String(st.turn || "?") + " · " + String(st.tool || "")
        )
      );
      if (st.rationale) {
        card.appendChild(makeEl("div", "trace-rationale", String(st.rationale)));
      }
      var ioText =
        "args: " +
        JSON.stringify(st.args || {}) +
        "\nobs:  " +
        JSON.stringify(st.obs || {});
      card.appendChild(makeEl("pre", "trace-json-line", ioText));
      box.appendChild(card);
    });
  }

  function initTraceExplorer() {
    var sel = document.getElementById("trace-sample-select");
    var samples = Array.isArray(window.__TRACE_SAMPLES__) ? window.__TRACE_SAMPLES__ : [];
    if (!sel || !samples.length) {
      return;
    }
    sel.addEventListener("change", function () {
      var idx = parseInt(sel.value, 10);
      if (!isNaN(idx) && idx >= 0 && idx < samples.length) {
        renderSelectedTrace(samples[idx]);
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.documentElement.classList.add("enhanced");
    initThemeToggle();
    initPresentation();
    initTraceExplorer();
  });
})();
