/**
 * OVERTAKE — Formula 1 AI Race Strategy & Replay Engine
 * Client-side Controller & Visualization
 */

// State
const state = {
  currentRaceId: "2023_bahrain",
  targetDriver: "VER",
  currentLap: 1,
  totalLaps: 57,
  activeView: "race",
  isPlaying: false,
  playSpeed: 1,
  playTimer: null,
  cachedReplayState: null,
  selectedEngine: "search",
  backtestEngine: "search",
};

// DOM Elements
const raceSelect = document.getElementById("race-select");
const driverSelect = document.getElementById("driver-select");
const lapSlider = document.getElementById("lap-slider");
const currentLapNum = document.getElementById("current-lap-num");
const totalLapsNum = document.getElementById("total-laps-num");
const btnPlayPause = document.getElementById("btn-play-pause");
const btnPrevLap = document.getElementById("btn-prev-lap");
const btnNextLap = document.getElementById("btn-next-lap");
const trackStatusPill = document.getElementById("track-status-pill");
const trackStatusText = document.getElementById("track-status-text");
const trackTempText = document.getElementById("track-temp-text");
const weatherCondText = document.getElementById("weather-cond-text");

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  loadRaces();
  updateViewData();
});

// Event Listeners
function initEventListeners() {
  // Tab Switching
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".view-panel").forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      state.activeView = btn.dataset.view;
      const targetPanel = document.getElementById(`view-${state.activeView}`);
      if (targetPanel) targetPanel.classList.add("active");

      updateViewData();
    });
  });

  // Race Selection Change
  raceSelect.addEventListener("change", (e) => {
    state.currentRaceId = e.target.value;
    state.currentLap = 1;
    lapSlider.value = 1;
    stopReplay();
    updateViewData();
  });

  // Driver Selection Change
  driverSelect.addEventListener("change", (e) => {
    state.targetDriver = e.target.value;
    updateViewData();
  });

  // Scrubber Slider
  lapSlider.addEventListener("input", (e) => {
    state.currentLap = parseInt(e.target.value, 10);
    currentLapNum.textContent = state.currentLap;
    updateViewData();
  });

  // Controls
  btnPrevLap.addEventListener("click", () => {
    if (state.currentLap > 1) {
      state.currentLap--;
      lapSlider.value = state.currentLap;
      currentLapNum.textContent = state.currentLap;
      updateViewData();
    }
  });

  btnNextLap.addEventListener("click", () => {
    if (state.currentLap < state.totalLaps) {
      state.currentLap++;
      lapSlider.value = state.currentLap;
      currentLapNum.textContent = state.currentLap;
      updateViewData();
    }
  });

  btnPlayPause.addEventListener("click", togglePlay);

  // Speed controls
  document.querySelectorAll(".speed-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".speed-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.playSpeed = parseInt(btn.dataset.speed, 10);
      if (state.isPlaying) {
        stopReplay();
        startReplay();
      }
    });
  });

  // Sandbox simulation trigger
  const btnRunCustom = document.getElementById("btn-run-custom-sim");
  if (btnRunCustom) {
    btnRunCustom.addEventListener("click", runCustomSimulation);
  }

  // Strategy Engine Selector Toggle (FR-7 Engines 1-3)
  document.querySelectorAll(".engine-toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".engine-toggle-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.selectedEngine = btn.dataset.engine;
      loadStrategyView();
    });
  });

  // Backtest Engine Selector Toggle
  document.querySelectorAll(".bt-engine-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".bt-engine-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.backtestEngine = btn.dataset.engine;
      loadBacktestView();
    });
  });

  // Keyboard navigation shortcuts
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.code === "ArrowLeft") {
      e.preventDefault();
      btnPrevLap.click();
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      btnNextLap.click();
    } else if (e.code === "Space") {
      e.preventDefault();
      togglePlay();
    }
  });
}

// Playback Loop
function togglePlay() {
  if (state.isPlaying) {
    stopReplay();
  } else {
    startReplay();
  }
}

function startReplay() {
  state.isPlaying = true;
  btnPlayPause.textContent = "⏸";
  const intervalMs = Math.max(250, 1000 / state.playSpeed);

  state.playTimer = setInterval(() => {
    if (state.currentLap >= state.totalLaps) {
      stopReplay();
      return;
    }
    state.currentLap++;
    lapSlider.value = state.currentLap;
    currentLapNum.textContent = state.currentLap;
    updateViewData();
  }, intervalMs);
}

function stopReplay() {
  state.isPlaying = false;
  btnPlayPause.textContent = "▶";
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
}

// API Loaders & Data Fetching
async function loadRaces() {
  try {
    const res = await fetch("/api/races");
    if (res.ok) {
      const races = await res.json();
      if (races && races.length > 0) {
        raceSelect.innerHTML = "";
        races.forEach((r) => {
          const opt = document.createElement("option");
          opt.value = r.race_id;
          opt.textContent = `${r.season || 2023} ${r.event_name || r.race_id} (${r.circuit || "GP"})`;
          raceSelect.appendChild(opt);
        });
        raceSelect.value = state.currentRaceId;
      }
    }
  } catch (err) {
    console.warn("Using default races list", err);
  }
}

async function updateViewData() {
  currentLapNum.textContent = state.currentLap;

  // Always fetch current replay state
  try {
    const res = await fetch(`/api/replay/${state.currentRaceId}/${state.currentLap}`);
    if (res.ok) {
      const raceState = await res.json();
      state.cachedReplayState = raceState;
      state.totalLaps = raceState.total_laps || 57;
      totalLapsNum.textContent = state.totalLaps;
      lapSlider.max = state.totalLaps;

      updateStatusHeader(raceState);

      if (state.activeView === "race") renderRaceView(raceState);
    }
  } catch (err) {
    console.warn("Replay fetch error:", err);
  }

  // Load view-specific active panel
  if (state.activeView === "strategy") loadStrategyView();
  else if (state.activeView === "simulation") loadSimulationView();
  else if (state.activeView === "tyre") loadTyreView();
  else if (state.activeView === "backtest") loadBacktestView();
}

function updateStatusHeader(raceState) {
  // Update status badge
  trackStatusPill.className = "status-pill";
  if (raceState.safety_car || raceState.status === "SAFETY_CAR") {
    trackStatusPill.classList.add("yellow");
    trackStatusText.textContent = "SAFETY CAR";
  } else if (raceState.status === "VIRTUAL_SAFETY_CAR") {
    trackStatusPill.classList.add("yellow");
    trackStatusText.textContent = "VSC ACTIVE";
  } else if (raceState.status === "RED_FLAG") {
    trackStatusPill.classList.add("red");
    trackStatusText.textContent = "RED FLAG";
  } else {
    trackStatusPill.classList.add("green");
    trackStatusText.textContent = "TRACK CLEAR";
  }

  const w = raceState.weather || {};
  trackTempText.textContent = `${Math.round(w.track_temp || 32)}°C`;
  weatherCondText.textContent = w.is_wet ? "WET" : "DRY";
}

// ----------------------------------------------------
// VIEW 1: RACE REPLAY RENDERING
// ----------------------------------------------------
function renderRaceView(raceState) {
  const tbody = document.getElementById("leaderboard-body");
  tbody.innerHTML = "";

  const sortedDrivers = Object.entries(raceState.positions || {}).sort((a, b) => a[1] - b[1]);

  // Synchronize driver select options with actual drivers in the session
  if (sortedDrivers.length > 0) {
    const existingOptions = Array.from(driverSelect.options).map((o) => o.value);
    const gridDrivers = sortedDrivers.map(([d]) => d);
    const matchesGrid = gridDrivers.length === existingOptions.length && gridDrivers.every((d) => existingOptions.includes(d));
    if (!matchesGrid) {
      driverSelect.innerHTML = "";
      sortedDrivers.forEach(([d, p]) => {
        const opt = document.createElement("option");
        opt.value = d;
        opt.textContent = `${d} (P${p})`;
        driverSelect.appendChild(opt);
      });
      if (gridDrivers.includes(state.targetDriver)) {
        driverSelect.value = state.targetDriver;
      } else {
        state.targetDriver = gridDrivers[0];
        driverSelect.value = state.targetDriver;
      }
    }
  }

  const driversList = [];
  const gapsList = [];

  sortedDrivers.forEach(([driver, pos]) => {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    if (driver === state.targetDriver) tr.classList.add("highlighted");
    tr.title = `Click to focus on ${driver}`;
    tr.addEventListener("click", () => {
      if (state.targetDriver !== driver) {
        state.targetDriver = driver;
        driverSelect.value = driver;
        updateViewData();
      }
    });

    const tyreInfo = (raceState.tyres && raceState.tyres[driver]) || ["MEDIUM", 1];
    const comp = tyreInfo[0];
    const age = tyreInfo[1];
    const gap = (raceState.gaps && raceState.gaps[driver] != null) ? raceState.gaps[driver] : 0.0;
    const lastLap = (raceState.last_lap_times && raceState.last_lap_times[driver]) ? `${raceState.last_lap_times[driver].toFixed(2)}s` : "--";
    const stops = (raceState.pit_stops_count && raceState.pit_stops_count[driver] != null) ? raceState.pit_stops_count[driver] : 0;

    tr.innerHTML = `
      <td><strong>P${pos}</strong></td>
      <td><strong>${driver}</strong></td>
      <td><span class="tyre-badge ${comp}">${comp[0]}</span></td>
      <td>${age} L</td>
      <td>${pos === 1 ? "LEADER" : `+${gap.toFixed(1)}s`}</td>
      <td>${lastLap}</td>
      <td>${stops}</td>
    `;
    tbody.appendChild(tr);

    driversList.push(driver);
    gapsList.push(gap);
  });

  // Render Gap to Leader Bar Chart
  const top10Drivers = driversList.slice(0, 10).reverse();
  const top10Gaps = gapsList.slice(0, 10).reverse();
  const colors = top10Drivers.map(d => d === state.targetDriver ? "#00f2fe" : "#ff1801");

  const trace = {
    x: top10Gaps,
    y: top10Drivers,
    type: "bar",
    orientation: "h",
    marker: { color: colors, opacity: 0.85 },
  };

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "Outfit, sans-serif", color: "#9ca3af" },
    margin: { t: 10, r: 20, l: 60, b: 40 },
    xaxis: { title: "Gap to Leader (Seconds)", gridcolor: "rgba(255,255,255,0.05)" },
    yaxis: { gridcolor: "transparent" },
  };

  Plotly.react("gap-chart", [trace], layout, { responsive: true, displayModeBar: false });
}

// ----------------------------------------------------
// VIEW 2: STRATEGY OPTIMIZER RENDERING
// ----------------------------------------------------
async function loadStrategyView() {
  try {
    const res = await fetch(`/api/strategy/${state.currentRaceId}/${state.currentLap}?driver=${state.targetDriver}&sims=150&engine=${state.selectedEngine}`);
    if (res.ok) {
      const data = await res.json();
      renderStrategyView(data);
    }
  } catch (err) {
    console.error("Strategy fetch error:", err);
  }
}

function renderStrategyView(data) {
  // Sync engine toggle button active state
  const activeEngine = data.engine || state.selectedEngine;
  document.querySelectorAll(".engine-toggle-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.engine === activeEngine);
  });

  document.getElementById("strategy-confidence").textContent = `CONFIDENCE: ${((data.confidence || 0.8) * 100).toFixed(0)}%`;
  document.getElementById("rec-action-badge").textContent = data.action || "BOX THIS LAP";

  const engineLabels = {
    search: "ENGINE: MC SEARCH",
    gametheory: "ENGINE: GAME THEORY",
    rl: "ENGINE: RL POLICY (PPO)",
  };
  const engineBadge = document.getElementById("rec-engine-badge");
  if (engineBadge) {
    engineBadge.textContent = engineLabels[activeEngine] || `ENGINE: ${activeEngine.toUpperCase()}`;
  }

  const tyreBadge = document.getElementById("rec-tyre-badge");
  const recTyre = data.tyre || "MEDIUM";
  tyreBadge.textContent = `FOR ${recTyre}`;
  tyreBadge.className = `hero-compound-badge ${recTyre.toLowerCase()}`;

  const gainVal = document.getElementById("rec-gain-val");
  const gain = data.expected_gain != null ? data.expected_gain : 0.0;
  gainVal.textContent = `${gain >= 0 ? "+" : ""}${gain.toFixed(1)} POS`;
  gainVal.className = gain >= 0 ? "val positive" : "val";

  const expPos = data.expected_position != null ? data.expected_position : 1.0;
  document.getElementById("rec-exp-pos").textContent = `P ${expPos.toFixed(1)}`;

  const podProb = data.podium_prob != null ? data.podium_prob : 0.0;
  document.getElementById("rec-podium-prob").textContent = `${(podProb * 100).toFixed(1)}%`;
  document.getElementById("rec-reasoning-text").textContent = data.reasoning || "Optimized pit strategy evaluated.";

  // Render Candidates Table
  const tbody = document.getElementById("candidates-body");
  tbody.innerHTML = "";

  (data.candidates || []).forEach((c) => {
    const compList = Array.isArray(c.compounds) ? c.compounds : [];
    const firstComp = compList.length > 0 ? compList[0] : recTyre;
    const isOptimal = c.action === data.action && (firstComp === recTyre || !compList.length);
    const tr = document.createElement("tr");
    if (isOptimal) tr.classList.add("highlighted");

    const pitLap = c.pit_laps && c.pit_laps.length ? `Lap ${c.pit_laps.join(", ")}` : "None";
    const comp = compList.length > 0 ? compList.join(" → ") : "Stay Out";
    const cPos = c.expected_position != null ? c.expected_position.toFixed(1) : "--";
    const cWin = c.win_prob != null ? (c.win_prob * 100).toFixed(1) : "0.0";
    const cPodium = c.podium_prob != null ? (c.podium_prob * 100).toFixed(1) : "0.0";

    tr.innerHTML = `
      <td><strong>${c.description || c.name || "Strategy"}</strong></td>
      <td>${pitLap}</td>
      <td><span class="tyre-badge ${firstComp}">${comp}</span></td>
      <td>P ${cPos}</td>
      <td>${cWin}%</td>
      <td>${cPodium}%</td>
      <td>${isOptimal ? '<span class="badge live">RECOMMENDED</span>' : '<span class="badge">ALTERNATIVE</span>'}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ----------------------------------------------------
// VIEW 3: MONTE CARLO SIMULATION RENDERING
// ----------------------------------------------------
async function loadSimulationView() {
  try {
    const res = await fetch(`/api/simulation/${state.currentRaceId}/${state.currentLap}?driver=${state.targetDriver}&sims=300`);
    if (res.ok) {
      const data = await res.json();
      renderSimulationView(data);
    }
  } catch (err) {
    console.error("Simulation fetch error:", err);
  }
}

async function runCustomSimulation() {
  const pitLap = parseInt(document.getElementById("custom-pit-lap").value, 10);
  const compound = document.getElementById("custom-compound").value;
  const sims = parseInt(document.getElementById("custom-sims").value, 10);

  try {
    const res = await fetch(`/api/simulation/${state.currentRaceId}/${state.currentLap}?driver=${state.targetDriver}&pit_lap=${pitLap}&compound=${compound}&sims=${sims}`);
    if (res.ok) {
      const data = await res.json();
      renderSimulationView(data);
    }
  } catch (err) {
    console.error("Custom simulation error:", err);
  }
}

function renderSimulationView(data) {
  const dist = data.finish_prob_by_position || {};
  const positions = Object.keys(dist).map(Number).sort((a, b) => a - b);
  const probs = positions.map((p) => (dist[p] * 100));

  const trace = {
    x: positions.map((p) => `P${p}`),
    y: probs,
    type: "bar",
    marker: {
      color: positions.map((p) => (p <= 3 ? "#00f2fe" : "#ff1801")),
      opacity: 0.85,
    },
  };

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "Outfit, sans-serif", color: "#9ca3af" },
    margin: { t: 20, r: 20, l: 50, b: 40 },
    xaxis: { title: "Finishing Position", gridcolor: "transparent" },
    yaxis: { title: "Probability (%)", gridcolor: "rgba(255,255,255,0.05)" },
  };

  Plotly.react("sim-distribution-chart", [trace], layout, { responsive: true, displayModeBar: false });

  // Update percentiles
  const p = data.percentiles || { p10: 1, p50: 2, p90: 4 };
  document.getElementById("p10-val").textContent = `P${p.p10}`;
  document.getElementById("p50-val").textContent = `P${p.p50}`;
  document.getElementById("p90-val").textContent = `P${p.p90}`;
}

// ----------------------------------------------------
// VIEW 4: TYRE DEGRADATION RENDERING
// ----------------------------------------------------
async function loadTyreView() {
  try {
    const res = await fetch(`/api/tyre/${state.currentRaceId}/${state.targetDriver}/${state.currentLap}?include_shap=true`);
    if (res.ok) {
      const data = await res.json();
      renderTyreView(data);
    }
  } catch (err) {
    console.error("Tyre fetch error:", err);
  }
}

function renderTyreView(data) {
  document.getElementById("tyre-circuit-badge").textContent = `${(data.circuit || "Circuit").toUpperCase()} · TRACK ${Math.round(data.track_temp || 30)}°C`;
  
  const compPill = document.getElementById("current-compound-pill");
  const compName = data.compound || "MEDIUM";
  compPill.textContent = compName;
  compPill.className = `compound-icon-large ${compName.toLowerCase()}`;

  document.getElementById("tyre-age-display").textContent = `${data.current_age} Laps`;
  document.getElementById("tyre-loss-display").textContent = `+${(data.current_pace_loss_seconds || 0).toFixed(2)} s/lap`;

  const maxLife = compName === "SOFT" ? 22 : (compName === "MEDIUM" ? 32 : 44);
  const remainingLaps = Math.max(0, maxLife - data.current_age);
  document.getElementById("tyre-cliff-display").textContent = `Lap ${state.currentLap + remainingLaps} (${remainingLaps} Laps remaining)`;

  const healthPct = Math.max(5, Math.min(100, (remainingLaps / maxLife) * 100));
  document.getElementById("tyre-health-fill").style.width = `${healthPct}%`;

  // Render SHAP Explainability Breakdown
  const shapContainer = document.getElementById("tyre-shap-container");
  const shapList = document.getElementById("tyre-shap-list");
  if (shapContainer && shapList) {
    if (data.shap && Object.keys(data.shap).length > 0) {
      shapContainer.style.display = "block";
      shapList.innerHTML = "";
      for (const [feature, val] of Object.entries(data.shap)) {
        const div = document.createElement("div");
        div.className = "shap-item";
        const formattedVal = val >= 0 ? `+${val.toFixed(3)}s` : `${val.toFixed(3)}s`;
        div.innerHTML = `
          <span class="shap-name">${feature.replace(/_/g, " ")}:</span>
          <span class="shap-val ${val >= 0 ? "loss" : "gain"}">${formattedVal}</span>
        `;
        shapList.appendChild(div);
      }
    } else {
      shapContainer.style.display = "none";
    }
  }

  // Plot Degradation Curves
  const curves = data.degradation_curves || {};
  const traces = [];

  const compStyles = {
    SOFT: { color: "#ff1801", name: "Soft (C3/C4/C5)" },
    MEDIUM: { color: "#ffd700", name: "Medium (C2/C3)" },
    HARD: { color: "#ffffff", name: "Hard (C1/C2)" },
  };

  for (const [comp, curve] of Object.entries(curves)) {
    const style = compStyles[comp] || { color: "#00f2fe", name: comp };
    traces.push({
      x: curve.map((c) => c.age),
      y: curve.map((c) => c.pace_loss_seconds),
      mode: "lines",
      name: style.name,
      line: { color: style.color, width: comp === data.compound ? 3.5 : 2 },
    });
  }

  // Add marker for current age
  traces.push({
    x: [data.current_age],
    y: [data.current_pace_loss_seconds],
    mode: "markers",
    name: "Current State",
    marker: { color: "#00f2fe", size: 12, symbol: "diamond" },
  });

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "Outfit, sans-serif", color: "#9ca3af" },
    margin: { t: 20, r: 20, l: 50, b: 40 },
    xaxis: { title: "Tyre Age (Laps Completed)", gridcolor: "rgba(255,255,255,0.05)" },
    yaxis: { title: "Pace Loss vs Fresh Tyre (Seconds/Lap)", gridcolor: "rgba(255,255,255,0.05)" },
    legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0.4)" },
  };

  Plotly.react("tyre-deg-chart", traces, layout, { responsive: true, displayModeBar: false });
}

// ----------------------------------------------------
// VIEW 6: BACKTESTING & EVALUATION RENDERING
// ----------------------------------------------------
async function loadBacktestView() {
  try {
    const res = await fetch(`/api/backtest?engine=${state.backtestEngine}`);
    if (res.ok) {
      const data = await res.json();
      renderBacktestView(data);
    }
  } catch (err) {
    console.error("Backtest fetch error:", err);
  }
}

function renderBacktestView(data) {
  // Sync backtest engine selector buttons
  document.querySelectorAll(".bt-engine-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.engine === state.backtestEngine);
  });

  const engineLabels = {
    search: "CALIBRATED · MC SEARCH",
    gametheory: "CALIBRATED · GAME THEORY",
    rl: "CALIBRATED · RL POLICY",
  };
  const badge = document.getElementById("bt-engine-badge");
  if (badge) {
    badge.textContent = engineLabels[state.backtestEngine] || `CALIBRATED · ${state.backtestEngine.toUpperCase()}`;
  }

  document.getElementById("bt-improved-count").textContent = `${data.strategies_improved || 0} / ${data.total_evaluations || 0}`;
  document.getElementById("bt-success-rate").textContent = `${(data.success_rate_pct || 0).toFixed(1)}%`;
  document.getElementById("bt-avg-delta").textContent = `${(data.average_position_gain || 0) >= 0 ? "+" : ""}${(data.average_position_gain || 0).toFixed(1)} POS`;

  const tbody = document.getElementById("backtest-body");
  tbody.innerHTML = "";

  (data.details || []).forEach((row) => {
    const tr = document.createElement("tr");
    const verdict = row.verdict || "MATCHED";
    const isAdvantage = verdict.includes("BEAT") || verdict.includes("ADVANTAGE") || verdict.includes("IMPROVED");
    if (isAdvantage) tr.classList.add("highlighted");

    const aiTyre = row.ai_tyre || "MEDIUM";
    const regretInfo = row.regret_vs_hindsight != null ? ` (${row.regret_vs_hindsight.toFixed(1)} regret)` : "";

    tr.innerHTML = `
      <td><strong>${row.circuit || row.race_id || "Race"}</strong></td>
      <td><strong>${row.driver || "VER"}</strong></td>
      <td>Lap ${row.decision_lap != null ? row.decision_lap : "--"}</td>
      <td>${row.actual_action || "BOX"}</td>
      <td>P ${row.actual_finish != null ? row.actual_finish : "--"}</td>
      <td><span class="tyre-badge ${aiTyre}">${row.ai_action || "BOX"}</span></td>
      <td>P ${row.ai_expected_position != null ? row.ai_expected_position.toFixed(1) : "--"}</td>
      <td><span class="badge ${isAdvantage ? 'live' : ''}">${verdict}${regretInfo}</span></td>
    `;
    tbody.appendChild(tr);
  });
}
