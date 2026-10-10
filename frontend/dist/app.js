/* Overtake pit wall: client controller.
 *
 * One lap state drives everything. The timing tower and race view refresh on
 * every lap; the heavy views (strategy call, simulation, corner analysis)
 * only compute when playback is paused, and keep their last render on screen
 * (dimmed) while a new one is on the way.
 */

const tok = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const T = {
  text: tok("--text"), text2: tok("--text-2"), text3: tok("--text-3"),
  bg: tok("--bg"), panel: tok("--panel"), divider: tok("--divider"), raise: tok("--raise"),
  gain: tok("--gain"), loss: tok("--loss"), muted: "#4A4F5E",
  tyre: { SOFT: tok("--soft"), MEDIUM: tok("--medium"), HARD: tok("--hard"), INTERMEDIATE: tok("--inter"), WET: tok("--wet") },
  engine: { search: tok("--eng-search"), gametheory: tok("--eng-gametheory"), rl: tok("--eng-rl"), real: tok("--eng-real") },
  font: "Archivo, system-ui, sans-serif", mono: "JetBrains Mono, monospace",
};
const ENGINE_NAMES = { search: "Search", gametheory: "Game theory", rl: "RL policy", real: "What the team did" };

const state = {
  raceId: null, driver: null, lap: 1, totalLaps: 99,
  view: "race", playing: false, speed: 1, timer: null,
  replay: null, races: [],
  engine: "search", call: null,
  backtest: null, btSubset: "all", models: null,
  cornersLoadedFor: null,
};
const tokens = {};  // latest request id per channel, to drop stale responses
const $ = (id) => document.getElementById(id);

// ---------- helpers ----------

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not json */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function claim(channel) { tokens[channel] = (tokens[channel] || 0) + 1; return tokens[channel]; }
function isCurrent(channel, id) { return tokens[channel] === id; }

function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (p, d = 0) => (p == null ? "–" : `${(p * 100).toFixed(d)}%`);
const fx = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const sentence = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase() : "");

function tyreDot(compound, age) {
  const c = String(compound || "").toUpperCase();
  const letter = c ? c[0] : "?";
  return `<span class="tyre"><span class="tyre-dot ${esc(c)}" title="${esc(sentence(c))}">${esc(letter)}</span>${age != null ? `<span class="tyre-age">${esc(age)}</span>` : ""}</span>`;
}

function planText(c, lap) {
  const pits = c.pit_laps || [];
  const comps = c.compounds || [];
  if (!pits.length) return "Stay out";
  if (pits.length > 1) return `Two stops: lap ${pits[0]} ${sentence(comps[0])}, lap ${pits[1]} ${sentence(comps[1])}`;
  const d = pits[0] - lap;
  return d <= 1 ? `Box now for ${sentence(comps[0])}` : `Box lap ${pits[0]} for ${sentence(comps[0])}`;
}

function callHeadline(rec, lap) {
  const pits = (rec.strategy && rec.strategy.pit_laps) || (rec.pit_lap ? [rec.pit_lap] : []);
  if (!pits.length) return { action: "Stay out", tyre: null };
  const comps = (rec.strategy && rec.strategy.compounds) || [rec.tyre];
  const d = pits[0] - lap;
  const action = d <= 1 ? "Box, box" : `Box in ${d} laps`;
  const then = pits.length > 1 ? `then lap ${pits[1]} for ${sentence(comps[1])}` : null;
  return { action, tyre: comps[0], then };
}

function raceLabel(id) {
  const r = state.races.find((x) => x.race_id === id);
  return r ? `${r.season} ${r.circuit}` : id;
}

function setBusy(view, busy) { $(`view-${view}`).classList.toggle("busy", busy); }

// Plotly: quiet chrome, hairline solid grid, text tokens for all text
function layout(extra = {}) {
  const axis = (a = {}) => ({
    gridcolor: T.divider, linecolor: T.divider, zerolinecolor: T.divider, zerolinewidth: 1,
    tickfont: { family: T.mono, size: 11, color: T.text2 },
    title: { font: { size: 12, color: T.text2 }, ...(a.title || {}) },
    automargin: true, ...a,
  });
  const { xaxis, yaxis, ...rest } = extra;
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: T.font, color: T.text2, size: 12 },
    margin: { t: 8, r: 16, l: 8, b: 8 },
    hoverlabel: { bgcolor: T.panel, bordercolor: T.divider, font: { family: T.font, color: T.text, size: 12 } },
    showlegend: false,
    xaxis: axis(xaxis), yaxis: axis(yaxis),
    ...rest,
  };
}
const PLOT_CFG = { responsive: true, displayModeBar: false };
const plot = (id, data, lay) => Plotly.react(id, data, layout(lay), PLOT_CFG);

// ---------- races, drivers, laps ----------

async function loadRaces() {
  const races = await api("/api/races");
  state.races = races.sort((a, b) => (a.season - b.season) || (a.round - b.round));
  const sel = $("race-select");
  const bySeason = {};
  state.races.forEach((r) => (bySeason[r.season] = bySeason[r.season] || []).push(r));
  sel.innerHTML = Object.keys(bySeason).sort().reverse().map((season) =>
    `<optgroup label="${season}">${bySeason[season].map((r) =>
      `<option value="${esc(r.race_id)}">${esc(r.event_name || r.race_id)} (${esc(r.circuit)})</option>`).join("")}</optgroup>`
  ).join("");
  state.raceId = state.races.some((r) => r.race_id === "2023_bahrain") ? "2023_bahrain" : state.races[0].race_id;
  sel.value = state.raceId;
}

function setLap(lap) {
  state.lap = Math.max(1, Math.min(state.totalLaps, lap));
  $("lap-slider").value = state.lap;
  $("lap-now").textContent = state.lap;
  refresh();
}

async function refresh() {
  const id = claim("replay");
  try {
    const rs = await api(`/api/replay/${state.raceId}/${state.lap}`);
    if (!isCurrent("replay", id)) return;
    state.replay = rs;
    state.totalLaps = rs.total_laps || state.totalLaps;
    $("lap-slider").max = state.totalLaps;
    $("lap-total").textContent = state.totalLaps;
    syncDrivers(rs);
    renderConditions(rs);
    renderTower(rs);
    renderCallStaleness();
  } catch (err) {
    console.error(err);
  }
  renderActiveView();
}

function syncDrivers(rs) {
  const order = Object.entries(rs.positions || {}).sort((a, b) => a[1] - b[1]).map(([d]) => d);
  if (!order.length) return;
  if (!state.driver || !order.includes(state.driver)) state.driver = order.includes("VER") ? "VER" : order[0];
  const sel = $("driver-select");
  const have = Array.from(sel.options).map((o) => o.value).join(",");
  if (have !== order.join(",")) {
    sel.innerHTML = order.map((d) => `<option value="${d}">${d}</option>`).join("");
  }
  sel.value = state.driver;
}

function renderConditions(rs) {
  const chip = $("flag-chip");
  let cls = "flag-green", text = "Green flag";
  if (rs.status === "RED_FLAG") { cls = "flag-red"; text = "Red flag"; }
  else if (rs.status === "VIRTUAL_SAFETY_CAR") { cls = "flag-yellow"; text = "Virtual safety car"; }
  else if (rs.safety_car || rs.status === "SAFETY_CAR") { cls = "flag-yellow"; text = "Safety car"; }
  chip.className = `flag-chip ${cls}`;
  $("flag-text").textContent = text;
  const w = rs.weather || {};
  $("track-temp").textContent = w.track_temp != null ? `${Math.round(w.track_temp)}°C` : "–";
  $("air-temp").textContent = w.air_temp != null ? `${Math.round(w.air_temp)}°C` : "–";
  $("wet-chip").textContent = w.is_wet ? "Wet" : "Dry";
}

function renderTower(rs) {
  const order = Object.entries(rs.positions || {}).sort((a, b) => a[1] - b[1]);
  const leader = order.length ? order[0][0] : null;
  const leaderLap = (rs.last_lap_times || {})[leader] || 90;
  const fl = (rs.fastest_lap || {}).driver;
  $("tower-rows").innerHTML = order.map(([d, pos]) => {
    const gap = (rs.gaps || {})[d];
    const intv = (rs.intervals || {})[d];
    const [comp, age] = (rs.tyres || {})[d] || ["", null];
    let gapTxt = "Leader";
    if (pos !== 1 && gap != null) gapTxt = gap > leaderLap ? `+${Math.floor(gap / leaderLap)} lap` : `+${gap.toFixed(3)}`;
    const intTxt = pos === 1 || intv == null ? "" : `+${intv.toFixed(1)}`;
    return `<li class="tower-row${d === state.driver ? " focused" : ""}" data-driver="${d}" title="Focus ${d}">
      <span class="pos">${pos}</span>
      <span class="drv">${d}${d === fl ? '<span class="fl" title="Fastest lap so far">FL</span>' : ""}</span>
      <span class="r gap">${gapTxt}</span>
      <span class="r int">${intTxt}</span>
      <span>${tyreDot(comp, age)}</span>
    </li>`;
  }).join("");
  const retired = Object.entries(rs.retired || {});
  $("tower-retired").textContent = retired.length ? `Out: ${retired.map(([d, why]) => `${d} (${why.replace(/^Retired \(|\)$/g, "").toLowerCase()})`).join(", ")}` : "";
}

function focusDriver(d) {
  if (!d || d === state.driver) return;
  state.driver = d;
  $("driver-select").value = d;
  if (state.replay) renderTower(state.replay);
  renderCallStaleness();
  renderActiveView();
}

// ---------- views ----------

const heavyViews = new Set(["strategy", "simulation"]);

function renderActiveView() {
  const v = state.view;
  if (v === "race") renderRace();
  else if (v === "tyre") loadTyre();
  else if (v === "model") loadModels();
  else if (v === "backtest") loadBacktest();
  else if (v === "driver") maybeAutoCorners();
  else if (heavyViews.has(v) && !state.playing) heavyRefresh();
  if (v === "strategy" && !state.playing) autoCall();
}

const heavyRefresh = debounce(() => {
  if (state.view === "simulation") loadSimulation();
}, 500);

// Race view
async function renderRace() {
  const rs = state.replay;
  if (!rs) return;
  const order = Object.entries(rs.positions || {}).sort((a, b) => a[1] - b[1]).map(([d]) => d);
  const gaps = order.map((d) => (rs.gaps || {})[d] ?? 0);
  plot("chart-gaps", [{
    type: "bar", orientation: "h", x: gaps, y: order,
    marker: { color: order.map((d) => (d === state.driver ? T.text : T.muted)), cornerradius: 4 },
    hovertemplate: "%{y}  +%{x:.3f} s<extra></extra>",
  }], {
    bargap: 0.35,
    xaxis: { title: { text: "Seconds behind the leader" }, rangemode: "tozero" },
    yaxis: { autorange: "reversed", tickfont: { family: T.font, size: 12, color: T.text }, gridcolor: "rgba(0,0,0,0)" },
  });

  const id = claim("events");
  try {
    const ev = await api(`/api/replay/${state.raceId}/events/${state.lap}`);
    if (!isCurrent("events", id)) return;
    const stops = (ev.pit_stops || []).slice().reverse().slice(0, 40);
    $("pit-rows").innerHTML = stops.length ? stops.map((s) => `<tr>
        <td class="r">${s.lap}</td><td>${esc(s.driver)}</td>
        <td>${tyreDot(s.compound_before)} <span class="muted">to</span> ${tyreDot(s.compound_after)}</td>
        <td class="r">${s.duration != null ? `${s.duration.toFixed(1)} s` : "–"}</td></tr>`).join("")
      : `<tr><td colspan="4" class="muted">No stops yet.</td></tr>`;
    const rc = (ev.race_control || []).filter((m) => m.event_type !== "OTHER").slice().reverse().slice(0, 30);
    $("rc-rows").innerHTML = rc.length ? rc.map((m) =>
      `<li class="${esc(m.event_type)}"><span class="num">${m.lap ?? ""}</span><span>${esc(sentence(m.message))}</span></li>`).join("")
      : `<li><span></span><span class="muted">Nothing yet.</span></li>`;
  } catch (err) { console.error(err); }
}

// Strategy call (right panel) + strategy view
async function requestCall() {
  const lap = state.lap, driver = state.driver, engine = state.engine;
  const id = claim("call");
  $("call").classList.add("loading");
  $("btn-call").disabled = true;
  $("btn-call").textContent = "Calling…";
  setBusy("strategy", true);
  try {
    const rec = await api(`/api/strategy/${state.raceId}/${lap}?driver=${driver}&engine=${engine}&sims=200`);
    if (!isCurrent("call", id)) return;
    state.call = { rec, lap, driver, engine, raceId: state.raceId, at: new Date() };
    renderCall();
    renderStrategyView();
  } catch (err) {
    if (isCurrent("call", id)) $("call-why").innerHTML = `<span class="error">${esc(err.message)}</span>`;
  } finally {
    if (isCurrent("call", id)) {
      $("call").classList.remove("loading");
      $("btn-call").disabled = false;
      $("btn-call").textContent = "Ask for a call";
      setBusy("strategy", false);
    }
  }
}

const autoCall = debounce(() => {
  const c = state.call;
  if (state.playing) return;
  if (c && c.lap === state.lap && c.driver === state.driver && c.engine === state.engine && c.raceId === state.raceId) return;
  requestCall();
}, 700);

function renderCall() {
  const c = state.call;
  if (!c) return;
  const { rec, lap, driver } = c;
  const head = callHeadline(rec, lap);
  const box = $("call");
  box.classList.remove("idle");
  $("call-time").textContent = `Lap ${lap}  ${c.at.toTimeString().slice(0, 8)}`;
  const pos = state.replay && state.replay.positions ? state.replay.positions[driver] : null;
  $("call-to").textContent = `To ${driver}${pos && lap === state.lap ? `, running P${pos}` : ""}. ${ENGINE_NAMES[c.engine]}.`;
  $("call-action").textContent = head.action;
  $("call-tyre").innerHTML = head.tyre
    ? `${tyreDot(head.tyre)} <span>${esc(sentence(head.tyre))}${head.then ? `, ${esc(head.then)}` : ""}</span>`
    : `<span class="muted">Current set to the flag</span>`;

  const gain = rec.expected_gain ?? 0;
  const facts = [
    ["Expected finish", `P${fx(rec.expected_position, 1)}`],
    ["Against staying out", `${gain > 0 ? "+" : ""}${fx(gain, 1)} pos`],
    ["Podium chance", pct(rec.podium_prob)],
    c.engine === "rl" ? ["Policy certainty", pct(rec.confidence)] : ["Win chance", pct(rec.win_prob)],
  ];
  $("call-facts").innerHTML = facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join("");
  $("call-why").textContent = rec.reasoning || "";
  renderCallStaleness();
}

function renderCallStaleness() {
  const c = state.call, el = $("call-stale");
  if (!c) { el.hidden = true; return; }
  const parts = [];
  if (c.raceId !== state.raceId) parts.push("This call is from another race.");
  else {
    if (c.lap !== state.lap) parts.push(`Called on lap ${c.lap}; you are on lap ${state.lap}.`);
    if (c.driver !== state.driver) parts.push(`Called for ${c.driver}, not ${state.driver}.`);
  }
  el.hidden = !parts.length;
  el.textContent = parts.join(" ");
}

function renderStrategyView() {
  const c = state.call;
  if (!c || c.raceId !== state.raceId) {
    $("cand-rows").innerHTML = `<tr><td class="muted">Ask for a call to see the options.</td></tr>`;
    return;
  }
  const { rec, lap, engine } = c;
  const cands = (rec.candidates || []).slice().sort((a, b) => a.expected_position - b.expected_position);
  const chosen = rec.strategy ? rec.strategy.name : null;
  const isPick = (k) => (chosen ? k.name === chosen : false);
  const color = T.engine[engine];

  $("strategy-sub").textContent =
    `${ENGINE_NAMES[engine]} for ${c.driver} on lap ${lap}. Expected finishing position from Monte Carlo, lower is better.`;

  const labels = cands.map((k) => planText(k, lap));
  const traces = [];
  if (engine === "gametheory") {
    // Spread across the rival's possible replies: how much its answer matters
    cands.forEach((k, i) => {
      const vals = Object.values(k.vs_rival_options || {});
      if (vals.length > 1) traces.push({
        type: "scatter", mode: "lines", x: [Math.min(...vals), Math.max(...vals)], y: [labels[i], labels[i]],
        line: { color: T.text3, width: 2 }, hoverinfo: "skip",
      });
    });
  }
  const ci = cands.map((k) => k.ci_95 || null);
  traces.push({
    type: "scatter", mode: "markers", x: cands.map((k) => k.expected_position), y: labels,
    marker: {
      size: cands.map((k) => (isPick(k) ? 14 : 9)),
      color: cands.map((k) => (isPick(k) ? T.text : color)),
      line: { color: T.bg, width: 2 },
    },
    error_x: ci.some(Boolean) ? {
      type: "data", symmetric: false, color: T.text3, thickness: 1.5, width: 0,
      array: cands.map((k, i) => (ci[i] ? ci[i][1] - k.expected_position : 0)),
      arrayminus: cands.map((k, i) => (ci[i] ? k.expected_position - ci[i][0] : 0)),
    } : undefined,
    customdata: cands.map((k) => [pct(k.win_prob, 1), pct(k.podium_prob, 1), k.rival_response || ""]),
    hovertemplate: "%{y}<br>Expected P%{x:.2f}<br>Win %{customdata[0]}  Podium %{customdata[1]}" +
      (engine === "gametheory" ? "<br>Rival answers: %{customdata[2]}" : "") + "<extra></extra>",
  });
  $("chart-candidates").style.height = `${Math.max(180, 40 + cands.length * 34)}px`;
  plot("chart-candidates", traces, {
    xaxis: { title: { text: engine === "gametheory" ? "Expected position after the rival's best reply (grey: range over its replies)" : "Expected finishing position" } },
    yaxis: { autorange: "reversed", tickfont: { family: T.font, size: 12, color: T.text }, gridcolor: "rgba(0,0,0,0)" },
  });

  const extra = engine === "gametheory" ? `<th>Rival (${esc(rec.rival_driver || "")}) answers</th>`
    : engine === "rl" ? `<th class="r">Policy</th>` : `<th class="r">95% interval</th>`;
  $("cand-head").innerHTML = `<tr><th>Option</th><th>Tyres</th><th class="r">Expected</th><th class="r">Win</th><th class="r">Podium</th>${extra}</tr>`;
  $("cand-rows").innerHTML = cands.map((k, i) => {
    const tyres = (k.compounds || []).map((cpd) => tyreDot(cpd)).join(" ") || '<span class="muted">Current</span>';
    const extraCell = engine === "gametheory" ? `<td>${esc(planText(rivalPlan(k.rival_response), lap))}</td>`
      : engine === "rl" ? `<td class="r">${pct(k.policy_prob)}</td>`
      : `<td class="r">${k.ci_95 ? `P${fx(k.ci_95[0], 1)} to P${fx(k.ci_95[1], 1)}` : "–"}</td>`;
    return `<tr class="${isPick(k) ? "pick" : ""}"><td>${esc(labels[i])}</td><td>${tyres}</td>
      <td class="r">P${fx(k.expected_position, 2)}</td><td class="r">${pct(k.win_prob, 1)}</td><td class="r">${pct(k.podium_prob, 1)}</td>${extraCell}</tr>`;
  }).join("");

  const notes = {
    search: "Each option simulated from the same starting state with the same random draws, so differences come from the plan, not luck.",
    gametheory: `For each of our options, ${rec.rival_driver || "the car ahead"} picks whichever reply is best for itself; we keep the option that is best after that reply.`,
    rl: "The trained policy picks the call; all four of its actions are still simulated so you can check it.",
  };
  $("strategy-note").textContent = notes[engine];
}

function rivalPlan(name) {
  // Rebuild pit_laps/compounds from a candidate name for display
  if (!name || name === "STAY_OUT") return {};
  const m = name.match(/^BOX_(NOW|LAP_(\d+))_(\w+)$/);
  if (!m) return {};
  return { pit_laps: [m[2] ? Number(m[2]) : state.call.lap + 1], compounds: [m[3]] };
}

// Simulation view
async function loadSimulation() {
  const plan = $("sim-plan").value;
  const n = $("sim-n").value;
  let q = `driver=${state.driver}&sims=${n}`;
  if (plan === "pit") q += `&pit_lap=${$("sim-pit-lap").value}&compound=${$("sim-compound").value}`;
  const id = claim("sim");
  setBusy("simulation", true);
  try {
    const res = await api(`/api/simulation/${state.raceId}/${state.lap}?${q}`);
    if (!isCurrent("sim", id)) return;
    renderSimulation(res, plan);
  } catch (err) {
    if (isCurrent("sim", id)) $("sim-sub").innerHTML = `<span class="error">${esc(err.message)}</span>`;
  } finally {
    if (isCurrent("sim", id)) setBusy("simulation", false);
  }
}

function renderSimulation(r, plan) {
  const planTxt = plan === "pit" ? `pitting on lap ${$("sim-pit-lap").value} for ${sentence($("sim-compound").value)}` : "no further stop";
  $("sim-sub").textContent = `${r.n_sims} simulations of ${state.driver} from the end of lap ${state.lap}, ${planTxt}.`;
  const p = r.percentiles || {};
  const tiles = [
    ["Expected finish", `P${fx(r.expected_position, 1)}`, r.position_std != null ? `spread ±${fx(r.position_std, 1)}` : ""],
    ["Middle 80%", `P${p.p10}–P${p.p90}`, `median P${p.p50}`],
    ["Win", pct(r.win_prob, 1), ""],
    ["Podium", pct(r.podium_prob, 1), ""],
    ["Points", pct(r.points_prob, 1), ""],
  ];
  $("sim-tiles").innerHTML = tiles.map(([l, v, d]) =>
    `<div class="tile"><div class="tile-label">${l}</div><div class="tile-value">${esc(v)}</div><div class="tile-detail">${esc(d)}</div></div>`).join("");

  const dist = r.finish_prob_by_position || {};
  const nCars = Object.keys(state.replay?.positions || {}).length || 20;
  const pos = Array.from({ length: nCars }, (_, i) => i + 1);
  plot("chart-pmf", [{
    type: "bar", x: pos, y: pos.map((k) => (dist[k] || 0) * 100),
    marker: { color: T.engine.search, cornerradius: 4 },
    hovertemplate: "P%{x}: %{y:.1f}%<extra></extra>",
  }], {
    bargap: 0.25,
    xaxis: { title: { text: "Finishing position" }, dtick: 1, gridcolor: "rgba(0,0,0,0)" },
    yaxis: { title: { text: "Share of simulations (%)" }, rangemode: "tozero" },
  });

  const b = r.trajectory_bands || [];
  const x = b.map((s) => s.lap);
  const band = (key) => b.map((s) => s[key]);
  const edge = { type: "scatter", mode: "lines", x, line: { width: 0, color: T.engine.search, shape: "hv" }, hoverinfo: "skip" };
  const lo = Math.min(...band("p10"), ...band("p50")), hi = Math.max(...band("p90"), ...band("p50"));
  plot("chart-ribbon", [
    { ...edge, y: band("p90") },
    { ...edge, y: band("p10"), fill: "tonexty", fillcolor: "rgba(57,135,229,0.12)" },
    { ...edge, y: band("p75") },
    { ...edge, y: band("p25"), fill: "tonexty", fillcolor: "rgba(57,135,229,0.24)" },
    {
      type: "scatter", mode: "lines", x, y: band("p50"),
      line: { color: T.engine.search, width: 2, shape: "hv" },
      customdata: b.map((s) => [s.p10, s.p90]),
      hovertemplate: "Lap %{x}: median P%{y}, 80% between P%{customdata[0]} and P%{customdata[1]}<extra></extra>",
    },
  ], {
    xaxis: { title: { text: "Lap" } },
    yaxis: { title: { text: "Position" }, range: [hi + 0.5, lo - 0.5], dtick: hi - lo > 10 ? 2 : 1 },
  });
}

// Tyre view
async function loadTyre() {
  const id = claim("tyre");
  setBusy("tyre", true);
  try {
    const t = await api(`/api/tyre/${state.raceId}/${state.driver}/${state.lap}?include_shap=true`);
    if (!isCurrent("tyre", id)) return;
    renderTyre(t);
  } catch (err) {
    if (isCurrent("tyre", id)) $("tyre-sub").innerHTML = `<span class="error">${esc(err.message)}</span>`;
  } finally {
    if (isCurrent("tyre", id)) setBusy("tyre", false);
  }
}

function renderTyre(t) {
  const comp = String(t.compound || "").toUpperCase();
  const curve = (t.degradation_curves || {})[comp] || [];
  const at = (age) => (curve.find((p) => p.age === age) || {}).pace_loss_seconds;
  const dry = ["SOFT", "MEDIUM", "HARD"].includes(comp);
  $("tyre-sub").textContent = `${state.driver} at ${t.circuit}, end of lap ${state.lap}. Seconds per lap lost to wear versus the same compound when fresh.`;
  const tiles = [
    ["On", `${sentence(comp)}, ${t.current_age} laps`, ""],
    ["Losing now", dry ? `${fx(t.current_pace_loss_seconds, 2)} s/lap` : "–", dry ? "" : "Wet tyres are not modelled"],
    ["In 5 laps", dry && at(t.current_age + 5) != null ? `${fx(at(t.current_age + 5), 2)} s/lap` : "–", ""],
    ["In 10 laps", dry && at(t.current_age + 10) != null ? `${fx(at(t.current_age + 10), 2)} s/lap` : "–", ""],
  ];
  $("tyre-tiles").innerHTML = tiles.map(([l, v, d]) =>
    `<div class="tile"><div class="tile-label">${l}</div><div class="tile-value">${esc(v)}</div><div class="tile-detail">${esc(d)}</div></div>`).join("");

  const traces = ["SOFT", "MEDIUM", "HARD"].map((c) => {
    const pts = (t.degradation_curves || {})[c] || [];
    return {
      type: "scatter", mode: "lines", name: sentence(c), x: pts.map((p) => p.age), y: pts.map((p) => p.pace_loss_seconds),
      line: { color: T.tyre[c], width: 2 },
      hovertemplate: `${sentence(c)}, %{x} laps: %{y:.2f} s/lap<extra></extra>`,
    };
  });
  if (dry) traces.push({
    type: "scatter", mode: "markers", name: "Now", x: [t.current_age], y: [t.current_pace_loss_seconds],
    marker: { size: 12, color: T.tyre[comp], line: { color: T.bg, width: 2 } },
    hovertemplate: `Now: ${t.current_age} laps, %{y:.2f} s/lap<extra></extra>`, showlegend: false,
  });
  plot("chart-wear", traces, {
    showlegend: true,
    legend: { orientation: "h", x: 0, y: 1.08, font: { color: T.text, size: 12 } },
    margin: { t: 30, r: 16, l: 8, b: 8 },
    xaxis: { title: { text: "Tyre age (laps)" } },
    yaxis: { title: { text: "Seconds per lap lost" }, rangemode: "tozero" },
  });

  const names = { tyre_age: "Tyre age", compound: "Compound", circuit: "Circuit", track_temp: "Track temperature" };
  const base = (t.shap || {}).bias;
  const shap = Object.entries(t.shap || {}).filter(([k]) => k !== "bias").sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  $("shap-note").textContent = base != null
    ? `Starts from the training average, ${fx(base, 2)} s/lap; each bar moves it up or down. Seconds per lap.`
    : "SHAP contributions to the current prediction, in seconds per lap.";
  if (!shap.length) {
    Plotly.purge("chart-shap");
    $("chart-shap").innerHTML = `<p class="note">No explanation available for this tyre.</p>`;
    return;
  }
  $("chart-shap").innerHTML = "";
  plot("chart-shap", [{
    type: "bar", orientation: "h",
    y: shap.map(([k, v]) => `${names[k] || k}  ${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`), x: shap.map(([, v]) => v),
    marker: { color: shap.map(([, v]) => (v >= 0 ? T.loss : T.gain)), cornerradius: 4 },
    hovertemplate: "%{y} s/lap<extra></extra>",
  }], {
    bargap: 0.45,
    xaxis: { title: { text: "Adds wear (right) or removes it (left), s/lap" }, zerolinecolor: T.text3 },
    yaxis: { autorange: "reversed", tickfont: { family: T.font, size: 12, color: T.text }, gridcolor: "rgba(0,0,0,0)" },
  });
}

// Driver / corner view
function maybeAutoCorners() {
  $("corner-sub").textContent = `${state.driver}, lap ${state.lap}, against the fastest lap set so far.`;
  if (!state.playing) cornersSoon();
}
const cornersSoon = debounce(() => { if (state.view === "driver") loadCorners(); }, 500);

async function loadCorners() {
  const id = claim("corners");
  const btn = $("btn-corners");
  btn.disabled = true;
  setBusy("driver", true);
  $("corner-status").textContent = state.cornersLoadedFor === state.raceId ? "Analysing…" : "Loading this race's telemetry, this can take up to a minute…";
  try {
    const r = await api(`/api/corner-analysis/${state.raceId}/${state.driver}/${state.lap}`);
    if (!isCurrent("corners", id)) return;
    state.cornersLoadedFor = state.raceId;
    renderCorners(r);
    $("corner-status").textContent = "Updates as you move through the race.";
  } catch (err) {
    if (isCurrent("corners", id)) $("corner-status").innerHTML = `<span class="error">${esc(err.message)}</span>`;
  } finally {
    if (isCurrent("corners", id)) { btn.disabled = false; setBusy("driver", false); }
  }
}

const turnName = (c) => (c ? (/^\d+$/.test(String(c.corner_label ?? "")) || c.corner_label == null ? `Turn ${c.corner_label ?? c.corner_number}` : c.corner_label) : "–");

function renderCorners(r) {
  $("corner-sub").textContent = `${r.driver}, lap ${r.lap}, against ${r.benchmark_driver}'s lap ${r.benchmark_lap}, the fastest set so far.`;
  const worst = r.worst_corner, best = r.best_corner;
  const tiles = [
    ["Lost in the corners", `${r.total_time_lost_s >= 0 ? "+" : ""}${fx(r.total_time_lost_s, 2)} s`, `${r.n_corners} corners`],
    ["Worst", turnName(worst), worst ? `+${fx(worst.time_lost_s, 3)} s` : ""],
    ["Best", turnName(best), best ? `${best.time_lost_s >= 0 ? "+" : ""}${fx(best.time_lost_s, 3)} s` : ""],
  ];
  $("corner-tiles").innerHTML = tiles.map(([l, v, d]) =>
    `<div class="tile"><div class="tile-label">${l}</div><div class="tile-value">${esc(v)}</div><div class="tile-detail">${esc(d)}</div></div>`).join("");
  const cs = r.corners || [];
  const labels = cs.map((c) => turnName(c).replace(/^Turn /, "T"));
  plot("chart-corners", [{
    type: "bar", x: labels, y: cs.map((c) => c.time_lost_s),
    marker: { color: cs.map((c) => (c.time_lost_s >= 0 ? T.loss : T.gain)), cornerradius: 4 },
    hovertemplate: "%{x}: %{y:+.3f} s<extra></extra>",
  }], {
    bargap: Math.max(0.35, 1 - 24 / (($("chart-corners").clientWidth || 800) / Math.max(1, cs.length))),
    xaxis: { type: "category", gridcolor: "rgba(0,0,0,0)", tickfont: { family: T.font, size: 11, color: T.text2 } },
    yaxis: { title: { text: "Time lost (+) or gained (−), s" }, zerolinecolor: T.text3 },
  });
  const signed = (v, d, unit) => (v == null ? "–" : `${v > 0 ? "+" : ""}${Number(v).toFixed(d)}${unit}`);
  $("corner-rows").innerHTML = cs.map((c, i) => `<tr>
    <td>${esc(turnName(c))}</td>
    <td class="r">${signed(c.time_lost_s, 3, " s")}</td>
    <td class="r">${signed(c.min_speed_delta_kmh, 1, " km/h")}</td>
    <td class="r">${signed(c.throttle_delta_pct, 1, " pts")}</td>
    <td class="r">${signed(c.brake_point_delta_m, 1, " m")}</td>
    <td class="r">${c.gear_driver ?? "–"} <span class="muted">vs ${c.gear_benchmark ?? "–"}</span></td></tr>`).join("");
}

// Models view
async function loadModels() {
  if (state.models) return;
  try {
    state.models = await api("/api/evaluation/models");
    renderModels(state.models);
  } catch (err) {
    $("tyre-eval").innerHTML = `<tr><td class="error">${esc(err.message)}</td></tr>`;
  }
}

function renderModels(m) {
  // Tyre wear evaluation
  const t = m.tyre;
  if (t) {
    const rows = [
      ["No wear (baseline)", "zero"], ["Average curve (baseline)", "mean_curve"], ["Bayesian state-space", "bayes"],
      ["XGBoost, circuit-aware", "xgb_current"], ["XGBoost, circuit-agnostic", "xgb_no_circuit_no_temp"],
    ].filter(([, k]) => t[k]);
    const cols = [["All test races", "all"], ["Circuits seen in training", "seen_circuit"], ["New circuits", "unseen_circuit"], ["Dry", "dry_races"], ["Wet", "wet_races"]];
    const best = Object.fromEntries(cols.map(([, g]) => [g, Math.min(...rows.map(([, k]) => t[k][g]?.curve_mae ?? Infinity))]));
    $("tyre-eval").innerHTML = `<thead><tr><th>Model</th>${cols.map(([l]) => `<th class="r">${l}</th>`).join("")}</tr></thead><tbody>` +
      rows.map(([label, k]) => `<tr><td>${label}</td>${cols.map(([, g]) => {
        const v = t[k][g]; if (!v) return `<td class="r">–</td>`;
        const ci = v.curve_mae_ci ? ` <span class="muted">${fx(v.curve_mae_ci[0])}–${fx(v.curve_mae_ci[1])}</span>` : "";
        return `<td class="r${v.curve_mae === best[g] ? " best" : ""}">${fx(v.curve_mae)}${ci}</td>`;
      }).join("")}</tr>`).join("") + "</tbody>";
    const served = (g, k) => t[k]?.[g]?.curve_mae;
    const lines = ["Served: circuit-aware XGBoost on circuits it has trained on, circuit-agnostic XGBoost elsewhere. Track temperature is ignored."];
    const mc = (g) => t.mean_curve?.[g]?.curve_mae;
    if (mc("unseen_circuit") < served("unseen_circuit", "xgb_no_circuit_no_temp")) {
      lines.push(`On new circuits the plain average curve (${fx(mc("unseen_circuit"))}) still beats the served circuit-agnostic model (${fx(served("unseen_circuit", "xgb_no_circuit_no_temp"))}).`);
    }
    if (served("seen_circuit", "xgb_current") < mc("seen_circuit")) {
      lines.push(`Where the circuit is known, XGBoost is clearly better (${fx(served("seen_circuit", "xgb_current"))} vs ${fx(mc("seen_circuit"))}).`);
    }
    const cov = t.bayes_interval_coverage_90;
    if (cov) lines.push(`Bayesian 90% intervals cover ${pct(cov.all)} of held-out points (should be 90%).`);
    lines.push("Wet-race numbers use dry-compound laps only and are unreliable.");
    $("tyre-eval-note").textContent = lines.join(" ");
  }

  // Lap time
  const lt = m.lap_time?.metrics || {};
  const held = lt.held_out_test_races || lt.held_out_races;
  if (held) {
    const tiles = [
      ["Next lap, error", `${fx(held.mae_h1)} s`, `repeat-recent-pace baseline ${fx(held.baseline_mae_h1)} s`],
      ["1 to 30 laps ahead", `${fx(held.mae)} s`, `baseline ${fx(held.baseline_mae)} s`],
      ["Next lap, RMSE", `${fx(held.rmse_h1)} s`, "big misses are rain and incidents"],
      ["Trained on", lt.n_train_races ? `${lt.n_train_races} races` : "–", lt.held_out_test_races ? "frozen test races held out" : "leave-one-race-out"],
    ];
    $("lap-tiles").innerHTML = tiles.map(([l, v, d]) =>
      `<div class="tile"><div class="tile-label">${l}</div><div class="tile-value">${esc(v)}</div><div class="tile-detail">${esc(d)}</div></div>`).join("");
  }
  const byRace = Object.entries(lt.held_out_test_races_mae_h1_by_race || lt.held_out_races_mae_h1_by_race || {}).sort((a, b) => b[1] - a[1]);
  $("chart-lap-races").style.height = `${Math.max(200, 30 + byRace.length * 22)}px`;
  plot("chart-lap-races", [{
    type: "bar", orientation: "h", y: byRace.map(([r]) => raceLabel(r)), x: byRace.map(([, v]) => v),
    marker: { color: T.engine.search, cornerradius: 4 }, hovertemplate: "%{y}: %{x:.2f} s<extra></extra>",
  }], {
    bargap: 0.35, xaxis: { title: { text: "Next-lap error (s)" }, rangemode: "tozero" },
    yaxis: { autorange: "reversed", tickfont: { family: T.font, size: 11, color: T.text }, gridcolor: "rgba(0,0,0,0)" },
  });
  const fi = Object.entries(m.lap_time?.feature_importance || {}).slice(0, 8);
  plot("chart-lap-features", [{
    type: "bar", orientation: "h", y: fi.map(([k]) => sentence(k.replace(/_/g, " "))), x: fi.map(([, v]) => v * 100),
    marker: { color: T.engine.search, cornerradius: 4 }, hovertemplate: "%{y}: %{x:.1f}% of gain<extra></extra>",
  }], {
    bargap: 0.35, xaxis: { title: { text: "Share of model gain (%)" }, rangemode: "tozero" },
    yaxis: { autorange: "reversed", tickfont: { family: T.font, size: 11, color: T.text }, gridcolor: "rgba(0,0,0,0)" },
  });

  const tracks = m.lap_time_tracks;
  if (tracks && Object.keys(tracks).length) {
    const names = { persistence: "Repeat recent pace", xgboost: "XGBoost", gru: "GRU", gnn: "Graph neural network" };
    const groups = [["All", "all"], ["New circuits", "unseen_circuit"], ["Dry", "dry_races"], ["Wet", "wet_races"]];
    const metric = (v) => v?.lap_mae ?? v?.mae;
    const best = Object.fromEntries(groups.map(([, g]) => [g, Math.min(...Object.values(tracks).map((r) => metric(r[g]) ?? Infinity))]));
    $("lap-tracks").innerHTML = `<h4>Four ways to predict the next lap, same held-out races</h4>
      <table class="data"><thead><tr><th>Model</th>${groups.map(([l]) => `<th class="r">${l}</th>`).join("")}</tr></thead><tbody>` +
      Object.entries(tracks).map(([k, r]) => `<tr><td>${names[k] || k}</td>${groups.map(([, g]) => {
        const v = metric(r[g]); return `<td class="r${v === best[g] ? " best" : ""}">${v == null ? "–" : `${fx(v)} s`}</td>`;
      }).join("")}</tr>`).join("") + "</tbody></table><p class=\"note\">Mean absolute error of the next lap, seconds.</p>";
  } else {
    $("lap-tracks").innerHTML = `<p class="note">The GRU and graph-network tracks have not been scored on the held-out races yet (scripts/run_lap_time_eval.py).</p>`;
  }

  const rl = m.rl_policy;
  if (!rl) { $("rl-meta").innerHTML = `<p class="note">No trained policy on disk; the RL engine falls back to a greedy Monte Carlo choice.</p>`; return; }
  const ev = rl.held_out || null;
  let html = `<p class="note">Trained ${esc(rl.trained_at || "")} on ${esc(rl.n_train_states ?? "?")} decision states from the training races for ${esc((rl.timesteps || 0).toLocaleString())} steps. Scored below on ${esc(ev?.n_states ?? "?")} states from the ${esc((rl.held_out_races || []).length)} held-out races.</p>`;
  const policies = [["rl_policy", "PPO policy"], ["always_stay_out", "Always stay out"], ["always_pit_soft", "Always pit for Soft"], ["random", "Random"]];
  const rowsEv = ev ? policies.filter(([k]) => ev[k]) : [];
  if (rowsEv.length) {
    const bestRegret = Math.min(...rowsEv.map(([k]) => ev[k].mean_regret));
    const trainRegret = rl.train?.rl_policy?.mean_regret;
    html += `<table class="data"><thead><tr><th>Policy</th><th class="r">Mean regret</th><th class="r">Picked best</th></tr></thead><tbody>` +
      rowsEv.map(([k, label]) => `<tr><td>${label}</td><td class="r${ev[k].mean_regret === bestRegret ? " best" : ""}">${fx(ev[k].mean_regret)}</td><td class="r">${fx(ev[k].picked_best_pct, 0)}%</td></tr>`).join("") +
      "</tbody></table>";
    const notes = [];
    if (ev.rl_policy && ev.always_stay_out && ev.rl_policy.mean_regret > ev.always_stay_out.mean_regret) {
      notes.push(`On held-out races the policy does worse than simply staying out (${fx(ev.rl_policy.mean_regret)} vs ${fx(ev.always_stay_out.mean_regret)}).`);
    }
    if (trainRegret != null) notes.push(`On its own training states it scores ${fx(trainRegret)}, so it is overfitting.`);
    notes.push("Regret is measured inside the simulator: this shows how well the policy learned the simulator, not that it beats a real strategist.");
    html += `<p class="note">${notes.join(" ")}</p>`;
  }
  $("rl-meta").innerHTML = html;
}

// Backtest view
async function loadBacktest() {
  if (!state.backtest) {
    try { state.backtest = await api("/api/evaluation/backtest"); }
    catch (err) { $("bt-sub").innerHTML = `<span class="error">${esc(err.message)}</span>`; return; }
  }
  renderBacktest();
}

function renderBacktest() {
  const { summary, points } = state.backtest;
  const block = summary[state.btSubset] || { n_points: 0 };
  const contenders = ["search", "gametheory", "rl", "real"].filter((k) => block[k]);
  $("bt-sub").textContent = `${summary.all.n_points} decision points across ${summary.n_races} races. One lap before a real pit stop, each engine makes its call; every call and the team's real one are then re-simulated on a shared seed none of them chose with.`;

  if (!block.n_points) {
    Plotly.purge("chart-regret");
    $("chart-regret").innerHTML = `<p class="note">No decision points in this group yet.</p>`;
    $("bt-summary").innerHTML = "";
  } else {
    $("chart-regret").innerHTML = "";
    const rows = contenders.filter((k) => block[k].mean_regret != null);
    plot("chart-regret", [{
      type: "scatter", mode: "markers",
      x: rows.map((k) => block[k].mean_regret), y: rows.map((k) => ENGINE_NAMES[k]),
      marker: { size: 12, color: rows.map((k) => T.engine[k]), line: { color: T.bg, width: 2 } },
      error_x: {
        type: "data", symmetric: false, color: T.text3, thickness: 2, width: 0,
        array: rows.map((k) => (block[k].mean_regret_ci95 ? block[k].mean_regret_ci95[1] - block[k].mean_regret : 0)),
        arrayminus: rows.map((k) => (block[k].mean_regret_ci95 ? block[k].mean_regret - block[k].mean_regret_ci95[0] : 0)),
      },
      hovertemplate: "%{y}: %{x:.2f} positions<extra></extra>",
    }], {
      xaxis: { title: { text: "Mean regret in positions, with 95% interval (0 = always the best option)" }, rangemode: "tozero" },
      yaxis: { autorange: "reversed", tickfont: { family: T.font, size: 12, color: T.text }, gridcolor: "rgba(0,0,0,0)" },
    });
    $("chart-regret").style.height = `${60 + rows.length * 44}px`;
    Plotly.Plots.resize("chart-regret");

    $("bt-summary").innerHTML = `<thead><tr><th>Strategy</th><th class="r">Decisions</th><th class="r">Mean regret</th><th class="r">Picked best</th><th class="r">Better than team</th><th class="r">Same</th><th class="r">Worse</th><th class="r">Seconds per call</th></tr></thead><tbody>` +
      contenders.map((k) => {
        const b = block[k], vr = b.vs_real || {};
        const ci = b.mean_regret_ci95 ? ` <span class="muted">${fx(b.mean_regret_ci95[0])}–${fx(b.mean_regret_ci95[1])}</span>` : "";
        return `<tr><td><span class="key" style="background:${T.engine[k]}"></span>${ENGINE_NAMES[k]}</td>
          <td class="r">${b.n}</td><td class="r">${fx(b.mean_regret)}${ci}</td><td class="r">${b.picked_best_pct != null ? `${fx(b.picked_best_pct, 0)}%` : "–"}</td>
          <td class="r">${vr.better_pct != null ? `${fx(vr.better_pct, 0)}%` : "–"}</td><td class="r">${vr.tie_pct != null ? `${fx(vr.tie_pct, 0)}%` : "–"}</td><td class="r">${vr.worse_pct != null ? `${fx(vr.worse_pct, 0)}%` : "–"}</td>
          <td class="r">${b.mean_seconds != null ? fx(b.mean_seconds, 0) : "–"}</td></tr>`;
      }).join("") + "</tbody>";
  }

  const cal = summary.simulator_calibration;
  $("bt-calibration").textContent = cal
    ? `How far to trust this: re-simulating what each team really did lands ${fx(cal.mae_real_strategy_vs_actual_finish, 1)} positions from the real result on average (correlation ${fx(cal.correlation)}, ${points.length} points). Regret is measured inside the simulator, so it is only as good as that.`
    : "";

  const raceName = raceLabel;
  const eng = (e) => (e && e.strategy ? `${esc(e.strategy.replace(/_/g, " ").toLowerCase())} <span class="num muted">${fx(e.regret)}</span>` : "–");
  $("bt-points").innerHTML = points.map((p) => {
    const real = p.real?.pit_laps ? `Lap ${p.real.pit_laps.join(", ")} ${(p.real.compounds || []).map((c) => tyreDot(c)).join(" ")}` : "–";
    return `<tr><td>${esc(raceName(p.race_id))}</td><td>${esc(p.driver)}</td><td class="r">${p.decision_lap}</td>
      <td>${real}</td><td class="r">${p.actual_finish != null ? `P${p.actual_finish}` : "–"}</td>
      <td>${eng(p.engines.search)}</td><td>${eng(p.engines.gametheory)}</td><td>${eng(p.engines.rl)}</td></tr>`;
  }).join("");
}

// ---------- playback ----------

function setPlaying(on) {
  state.playing = on;
  clearInterval(state.timer);
  $("btn-play").setAttribute("aria-label", on ? "Pause" : "Play");
  $("icon-play").innerHTML = on
    ? '<path d="M4 3h3v10H4zM9 3h3v10H9z" class="fill" />'
    : '<path d="M5 3v10l8-5z" class="fill" />';
  if (on) {
    state.timer = setInterval(() => {
      if (state.lap >= state.totalLaps) { setPlaying(false); return; }
      setLap(state.lap + 1);
    }, Math.max(250, 1000 / state.speed));
  } else {
    renderActiveView();
  }
}

// ---------- wiring ----------

function init() {
  $("race-select").addEventListener("change", (e) => {
    state.raceId = e.target.value;
    setPlaying(false);
    state.cornersLoadedFor = null;
    setLap(1);
  });
  $("driver-select").addEventListener("change", (e) => focusDriver(e.target.value));
  $("tower-rows").addEventListener("click", (e) => {
    const row = e.target.closest(".tower-row");
    if (row) focusDriver(row.dataset.driver);
  });
  $("lap-slider").addEventListener("input", (e) => setLap(parseInt(e.target.value, 10)));
  $("btn-prev").addEventListener("click", () => setLap(state.lap - 1));
  $("btn-next").addEventListener("click", () => setLap(state.lap + 1));
  $("btn-play").addEventListener("click", () => setPlaying(!state.playing));
  document.querySelectorAll(".speed-btn").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".speed-btn").forEach((x) => x.classList.toggle("active", x === b));
    state.speed = parseInt(b.dataset.speed, 10);
    if (state.playing) setPlaying(true);
  }));
  document.querySelectorAll(".view-tab").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".view-tab").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${b.dataset.view}`));
    state.view = b.dataset.view;
    renderActiveView();
    if (state.view === "strategy") renderStrategyView();
  }));
  document.querySelectorAll(".engine-pick .seg").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".engine-pick .seg").forEach((x) => x.classList.toggle("active", x === b));
    state.engine = b.dataset.engine;
    if (state.view === "strategy") autoCall(); else renderCallStaleness();
  }));
  $("btn-call").addEventListener("click", requestCall);
  $("btn-corners").addEventListener("click", loadCorners);
  $("bt-subset").addEventListener("click", (e) => {
    const b = e.target.closest(".seg"); if (!b) return;
    document.querySelectorAll("#bt-subset .seg").forEach((x) => x.classList.toggle("active", x === b));
    state.btSubset = b.dataset.subset;
    if (state.backtest) renderBacktest();
  });
  const syncPlanFields = () => {
    const off = $("sim-plan").value !== "pit";
    ["sim-pit-lap", "sim-compound"].forEach((id) => { $(id).disabled = off; $(id).closest(".field").classList.toggle("off", off); });
    $("sim-pit-lap").min = state.lap + 1;
    if (Number($("sim-pit-lap").value) <= state.lap) $("sim-pit-lap").value = Math.min(state.totalLaps - 1, state.lap + 3);
  };
  $("sim-plan").addEventListener("change", syncPlanFields);
  syncPlanFields();
  $("sim-form").addEventListener("submit", (e) => { e.preventDefault(); syncPlanFields(); loadSimulation(); });

  window.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.code === "ArrowLeft") { e.preventDefault(); setLap(state.lap - 1); }
    else if (e.code === "ArrowRight") { e.preventDefault(); setLap(state.lap + 1); }
    else if (e.code === "Space") { e.preventDefault(); setPlaying(!state.playing); }
  });
}

// Deep links: #race=2023_bahrain&lap=20&driver=VER&view=strategy&engine=search
function readHash() {
  return Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
}
function writeHash() {
  const h = new URLSearchParams({ race: state.raceId, lap: state.lap, driver: state.driver || "", view: state.view, engine: state.engine });
  history.replaceState(null, "", `#${h}`);
}
function selectView(view) {
  const tab = document.querySelector(`.view-tab[data-view="${view}"]`);
  if (tab) tab.click();
}
function selectEngine(engine) {
  document.querySelectorAll(".engine-pick .seg").forEach((x) => x.classList.toggle("active", x.dataset.engine === engine));
  state.engine = engine;
}

document.addEventListener("DOMContentLoaded", async () => {
  init();
  try { await loadRaces(); } catch (err) { console.error(err); return; }
  const h = readHash();
  if (h.race && state.races.some((r) => r.race_id === h.race)) { state.raceId = h.race; $("race-select").value = h.race; }
  if (h.driver) state.driver = h.driver;
  if (h.engine && ENGINE_NAMES[h.engine]) selectEngine(h.engine);
  if (h.view) {
    state.view = h.view;
    document.querySelectorAll(".view-tab").forEach((x) => x.classList.toggle("active", x.dataset.view === h.view));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${h.view}`));
  }
  setLap(h.lap ? parseInt(h.lap, 10) : 10);
  setInterval(writeHash, 1000);
});
