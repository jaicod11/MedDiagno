/* MedDiagno - Main JS */

// ── Result display ──────────────────────────────────────────────────────────
function showResult(data, cardId = "resultCard") {
  const card = document.getElementById(cardId);
  if (!card) return;

  const isHigh   = data.risk === "high";
  card.className = `result-card show ${isHigh ? "high" : "low"}`;

  const icons = { high: "⚠️", low: "✅" };
  card.querySelector(".result-icon").textContent = icons[data.risk] || "🩺";
  card.querySelector(".result-title").textContent = data.result;

  const confVal = card.querySelector(".conf-val");
  const confFill = card.querySelector(".confidence-fill");
  if (confVal)  confVal.textContent = data.confidence.toFixed(1) + "%";
  if (confFill) {
    confFill.style.width = "0%";
    setTimeout(() => confFill.style.width = data.confidence + "%", 80);
  }

  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showError(msg, cardId = "resultCard") {
  const card = document.getElementById(cardId);
  if (!card) return;
  card.className = "result-card show high";
  card.querySelector(".result-icon").textContent = "❌";
  card.querySelector(".result-title").textContent = msg;
  const confWrap = card.querySelector(".confidence-bar-wrap");
  if (confWrap) confWrap.style.display = "none";
}

// ── Submit form helper ──────────────────────────────────────────────────────
function setupPredictForm(formId, endpoint) {
  const form = document.getElementById(formId);
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn  = form.querySelector("[type=submit]");
    const spin = form.querySelector(".loading-spinner");
    const card = document.getElementById("resultCard");
    if (card) card.className = "result-card";

    btn.disabled = true;
    btn.textContent = "Analyzing…";
    if (spin) spin.style.display = "block";

    try {
      const resp = await fetch(endpoint, { method: "POST", body: new FormData(form) });
      const data = await resp.json();

      if (data.success) {
        showResult(data);
        if (spin) spin.style.display = "none";
        const confWrap = document.querySelector(".confidence-bar-wrap");
        if (confWrap) confWrap.style.display = "";
      } else {
        showError(data.error || "Unexpected error. Please try again.");
      }
    } catch (err) {
      showError("Network error. Please check your connection.");
    } finally {
      btn.disabled = false;
      btn.textContent = btn.dataset.label || "Analyze";
      if (spin) spin.style.display = "none";
    }
  });
}

// ── Range slider live display ───────────────────────────────────────────────
function initRangeSliders() {
  document.querySelectorAll('input[type="range"]').forEach(input => {
    const display = document.getElementById(input.id + "_val");
    if (display) {
      display.textContent = input.value;
      input.addEventListener("input", () => display.textContent = input.value);
    }
  });
}

// ── Image upload preview ────────────────────────────────────────────────────
function initImageUpload() {
  const zone    = document.getElementById("uploadZone");
  const input   = document.getElementById("fileInput");
  const preview = document.getElementById("imgPreview");
  if (!zone || !input) return;

  zone.addEventListener("click", () => input.click());

  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault(); zone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });

  input.addEventListener("change", () => {
    if (input.files[0]) setFile(input.files[0]);
  });

  function setFile(file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;

    const reader = new FileReader();
    reader.onload = (e) => {
      if (preview) { preview.src = e.target.result; preview.style.display = "block"; }
      zone.querySelector(".upload-text").textContent = file.name;
    };
    reader.readAsDataURL(file);
  }
}

// ── Dashboard charts ────────────────────────────────────────────────────────
function initCharts() {
  // Doughnut — prediction type distribution
  const donutEl = document.getElementById("typeChart");
  if (donutEl && window.Chart) {
    const counts = JSON.parse(donutEl.dataset.counts || "{}");
    new Chart(donutEl, {
      type: "doughnut",
      data: {
        labels: ["Diabetes", "Heart", "Skin"],
        datasets: [{
          data: [counts.diabetes || 0, counts.heart || 0, counts.skin || 0],
          backgroundColor: ["#f59e0b", "#ff5c72", "#60a5fa"],
          borderWidth: 0, hoverOffset: 6
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        cutout: "72%",
        plugins: {
          legend: { position: "bottom", labels: { font: { size: 12 }, padding: 14, boxWidth: 10 } },
          tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw}` } }
        }
      }
    });
  }

  // Bar — weekly activity
  const barEl = document.getElementById("activityChart");
  if (barEl && window.Chart) {
    const activity = JSON.parse(barEl.dataset.activity || "{}");
    const labels   = Object.keys(activity);
    const values   = Object.values(activity);
    new Chart(barEl, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Predictions",
          data: values,
          backgroundColor: "rgba(0,212,170,0.6)",
          borderColor: "#00d4aa",
          borderWidth: 1.5,
          borderRadius: 6,
          hoverBackgroundColor: "#00d4aa"
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, ticks: { stepSize: 1, font: { size: 11 } }, grid: { color: "rgba(0,0,0,0.05)" } },
          x: { grid: { display: false }, ticks: { font: { size: 11 } } }
        }
      }
    });
  }
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initRangeSliders();
  initImageUpload();
  initCharts();

  // Auto-dismiss flash messages
  setTimeout(() => {
    document.querySelectorAll(".flash").forEach(el => {
      el.style.transition = "opacity 0.5s";
      el.style.opacity = "0";
      setTimeout(() => el.remove(), 500);
    });
  }, 5000);
});
