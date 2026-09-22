# Overtake — Project Status

> **Update this file in every commit.** The pre-commit hook in `.githooks/` rejects
> commits that don't include a change to it. One-time setup per clone:
> `git config core.hooksPath .githooks`
>
> How to update: tick off finished items, move "Next up", add any new
> decisions/issues, and add one line to the Change Log at the bottom.

**Last updated:** 2026-09-22 · Full build Phase 7 (FR-7 Engines 2–3, FR-8 three-engine backtest) done. Reviewed the Engine 2/3 code that landed after the Phase 6 environment fix and found and fixed 3 real bugs: (1) `gametheory.py` recomputed the rival's best response identically inside the per-candidate loop (~10x redundant Monte Carlo calls; it never actually depended on the candidate) — hoisted out, docstring corrected to say so honestly; (2) `rl_env.py`'s roll-out evaluation called `run_monte_carlo` with its default fixed seed on every episode, so "N roll-outs" of a deterministic policy silently produced N identical results and a fake confidence metric — fixed by passing `seed=None` so episodes actually sample fresh noise; (3) `get_strategy_recommendation_rl()` was missing `podium_prob`/`win_prob`/`finish_prob_by_position`/`candidates` that Engines 1–2 both return despite the API dispatching all three through one rendering path — confirmed via `frontend/dist/app.js:318` that this rendered `"NaN%"` for `?engine=rl`; fixed to match the shared contract. Then extended `src/evaluation/backtest.py` for FR-8: `backtest_decision_point(..., engine=...)` now dispatches to any of the 3 engines and reports `regret_vs_hindsight` (vs. the best candidate that engine itself evaluated), plus `run_multi_engine_backtest()` / `summarize_multi_engine_backtest()` for the three-way comparison, wired to `GET /api/backtest/compare/all`. Also found and fixed a second bug while doing this: when a race/driver/lap had no curated `HISTORICAL_BENCHMARKS` entry, the old fallback set `actual_finish = round(expected_pos)` — deriving "what really happened" from the AI's own prediction, which made every unbenchmarked point silently score as "MATCHED REAL STRATEGY" by construction. Replaced with `_actual_outcome()`, which looks up the real pit stop and real classified finishing position from the `pit_stops`/`laps` tables. Added `decision_points_for_race()` (one decision point per driver's earliest real pit stop) and `run_dataset_backtest()` so backtesting isn't limited to the 6 curated races, plus `scripts/run_full_backtest.py` — a resumable (checkpointed to `data/models/backtest/checkpoint.jsonl`) CLI to run all 3 engines across the full 112-race dataset and write `dataset_report.md`/`.json`. Smoke-tested on 4 races including a resume-from-checkpoint check; the full 112-race × 3-engine run has **not** been executed yet (would take hours) — that's the next thing to actually run, not just build. Full suite 150/150 passing · Phase 6 (FR-6 Monte Carlo + FR-7 Strategy Engine 1, environment fix) done · Phase 5 (FR-5 safety car model & bunching) done · Phase 4 (FR-4 replay engine) done · Phase 3 (FR-3 lap-time model) done · Phase 2 (tyre model) done · Phase 1 (multi-season data) done

---

## 1. Final Goal

A local dashboard where anyone can:

1. Pick one of 8 historical 2023 F1 races
2. Scrub to any lap
3. See current tyre state + predicted degradation curve
4. See a recommended pit/tyre call with expected gain and confidence
5. See a Monte Carlo finishing-position distribution
6. Compare that call with what really happened, plus a backtest summary across races

Hard rule throughout: **no data leakage**. Every feature or state lookup goes through
`as_of_lap()` (Design.md §5).

Scope source of truth: `PRD.md` (what) · `Design.md` (how) · `overtake-15-day-sprint-plan.md` (when).

---

## 2. Overall Progress

| Lane | Owner | Status |
| --- | --- | --- |
| Prediction (ingestion, tyre model, lap-time model) | Abubaker | 🟢 Days 1–7 done |
| Decision-Making (replay, safety car, Monte Carlo, optimizer, backtest, dashboard) | Partner | 🟢 Days 1–13 done (weather/race_control/pit_stops, replay, ghost car SC, Monte Carlo, optimizer, FastAPI, views, backtest) |
| Shared (schema, no-leakage guard, integration, Docker) | Both | 🟢 Schema confirmed, guard reviewed & validated, full pipeline integrated |

Legend: ✅ done · 🟡 in progress · ⬜ not started · ⚠️ blocked/at risk

---

## 3. Milestones by Sprint Day

### Prediction Lane (Abubaker)

| Day | Deliverable | Status |
| --- | --- | --- |
| 1 | Repo scaffold (Design.md §3), FastF1 cache in `data/cache/` | ✅ |
| 2 | Ingest laps, sectors, telemetry, tyre compound/age → Parquet + DuckDB | ✅ |
| 3 | No-leakage guard `as_of_lap()` + EDA (paired) | ✅ |
| 4–5 | Tyre model → `predict_tyre_degradation(compound, age, circuit, track_temp)` | ✅ (`weather` table unblocked) |
| 6–7 | Lap-time model → `predict_lap_time(state, driver)` & GNN `predict_lap_time_gnn(graph)` | ✅ (XGBoost, GRU, and GNN models complete) |
| 8 | MAE/RMSE report for both models (held-out laps and held-out races) | ✅ (tyre_eval.py & lap_time_eval.py on frozen split) |
| 9–10 | FastAPI: `/api/tyre/...`, `/api/lap-prediction/...` | ✅ |
| 11–13 | Tyre view + Model view (React + Plotly) | ✅ |
| 14–15 | Joint integration + bug fixing | 🟡 |

### Decision-Making Lane (Partner)

| Day | Deliverable | Status |
| --- | --- | --- |
| 1 | Shared data schema agreed | ✅ |
| 2 | Ingest pit stops, race control, weather → Parquet | ✅ |
| 3 | No-leakage guard + EDA (paired) | ✅ (`src/preprocessing/leakage.py` reviewed & tested) |
| 4 | `RaceState` + bare replay loop | ✅ (`src/simulation/replay.py`) |
| 5 | Safety-car ("ghost car") model | ✅ (`src/simulation/safety_car.py`) |
| 6 | Wire in prediction functions | ✅ (`src/simulation/monte_carlo.py`) |
| 7 | Monte Carlo sampling | ✅ (`src/simulation/monte_carlo.py`) |
| 8 | Strategy optimizer — `get_strategy_recommendation()` signature locked | ✅ (`src/strategy/optimizer.py`) |
| 9–10 | FastAPI: replay / strategy / simulation endpoints | ✅ (`backend/main.py`) |
| 11–13 | Race, Strategy, Simulation views + backtesting | ✅ (`frontend/dist`, `src/evaluation/backtest.py`) |
| 14–15 | Joint integration + bug fixing | 🟡 |

### Priority tiers (cut from the bottom if behind)

- **P0:** data pipeline · tyre model · lap-time model · replay + safety car · Monte Carlo · optimizer · Race/Strategy/Simulation views · backtest ≥3 races · README + demo script
- **P1:** Tyre + Model views · Docker one-command run · backtest all 8 races · wet/dry flag in simulation
- **P2:** driver/corner view · game-theoretic strategy · GNN · RL · cloud deploy

---

## 4. Done So Far

**Prediction Lane — Days 1–2**

- Scaffold per Design.md §3, plus `requirements.txt`, `.gitignore`, placeholder `docker-compose.yml`
- `src/ingestion/`: `session.py` (FastF1 + cache), `laps.py`, `tyres.py`, `telemetry.py`, `race_info.py`, `storage.py`, `run_ingestion.py`
- All 8 races ingested → `data/processed/{races,laps,tyres,telemetry}/<race_id>.parquet`
  (regenerate with `python -m src.ingestion.run_ingestion`; ~30 s per race on first download)
- `tests/test_ingestion.py`: 10 offline tests passing, including leakage checks (derived columns identical when future laps are removed)

**Prediction Lane — Day 3**

- `src/preprocessing/leakage.py`: `as_of_lap(df, lap_col, current_lap)` (signature exactly as Design.md §5)
  plus `assert_as_of_lap()` / `LeakageError` for checking the output of feature and state builders
- `tests/test_leakage.py`: 13 tests (boundaries, rows with missing lap dropped, input not mutated, bad args) — suite now 23 passing
- `notebooks/01_eda.ipynb`: coverage, clean-lap filter, compound mix, outliers, degradation signal, red-flag gaps, telemetry.
  Headline: tyre-age effect 0.01–0.13 s/lap after controlling for lap and driver (Bahrain highest, Monaco lowest);
  circuit matters more than compound

**Prediction Lane — Day 4**

- `src/preprocessing/cleaning.py`: `is_racing_lap()`, `is_clean_lap()` (slow-lap cut vs the *same-lap* field
  median, so it's leakage-safe at inference), `clean_gap_to_leader()` (red-flag laps and gaps > 600 s → NaN)
- `src/models/tyre.py` + `configs/models.toml`: XGBoost tyre model, monotone in tyre age, trained on 7,077 clean dry laps.
  Train/evaluate/save with `python -m src.models.tyre` (~15 s) → `data/models/tyre/` (gitignored)
- Results — error on the average wear curve per race/compound/age, vs. a "no wear" baseline:

  | Split | Pooled | Per-compound | No-wear baseline |
  | --- | --- | --- | --- |
  | Held-out stints (known circuits — the demo case) | **0.17 s** | 0.18 s | 0.44 s |
  | Held-out race (new circuit) | **0.48 s** | 0.48 s | 0.73 s |

  Pooled and per-compound tie → fixed to pooled (simpler). Worst held-out race: Australia (almost no wear, other circuits predict more).
- Tests: `tests/test_cleaning.py` (6), `tests/test_tyre_model.py` (10, synthetic races with known wear) — suite 39 passing
- Packages pinned: xgboost 3.4.1, scikit-learn 1.9.1, matplotlib 3.11.2

**Prediction Lane — Day 5 (partial) + Days 6–7**

- Day 5 track temp: ⚠️ blocked — no `weather` table yet. Nothing to change on our side: `python -m src.models.tyre` picks it up once it exists.
- `src/models/lap_time.py`: XGBoost next-lap-time model. Features as of an anchor lap via `as_of_lap()` (driver's median of last 5 clean laps,
  pace vs field, field pace) + race state at T−1 (compound, tyre age, position, gap ahead, gap to leader). Predicts lap time − reference pace,
  absolute-error objective, trained on 52k (anchor, target) pairs at horizons 1–30 laps. `python -m src.models.lap_time` (~12 s) → `data/models/lap_time/`
- Results — MAE (s) vs a baseline that repeats the driver's recent (last-5 median) pace:

  | Split | Next lap: model | Next lap: baseline | All horizons: model | All horizons: baseline |
  | --- | --- | --- | --- | --- |
  | Held-out race (new circuit) | **0.91** | 0.98 | **1.53** | 1.97 |
  | Held-out late laps (train on first 70% of each race) | 1.48 | 1.46 | **2.87** | 3.13 |

  Dry held-out races, next lap: 0.33–0.80 s. Wet Monaco/Netherlands: ~1.9 s (rain onset can't be seen from history).
  The model's value is mostly over multi-lap horizons (fuel + tyre drift), which is what Monte Carlo needs.
- Rejected: absolute pace / lap number features (memorise circuits, lost to baseline on held-out races); squared-error objective
  (chased rain outliers); ratio target (no gain); random driver split (leaks same-race future conditions).
- Speed: `predict_many()` for 500 sims × 20 drivers ≈ 40 ms per lap → ~1.1 s for a full remaining race (PRD §7 "few seconds").
- `tests/test_lap_time_model.py`: 14 tests (history ignores laps after anchor, batch = single calls, SC scaling, fresh vs old tyres) — suite 53 passing
- `tyre.load_lap_frame()` now also returns `position`, `gap_to_leader`, `total_laps`

**Decision-Making Lane — Days 1–2**

- Confirmed shared data schema per Design.md §4 and §5.
- `src/ingestion/weather.py`: Ingests `session.weather_data` per lap (`race_id, lap, track_temp, air_temp, humidity, pressure, wind_speed, rainfall, is_wet`). Unblocks track temperature in the tyre degradation model.
- `src/ingestion/race_control.py`: Ingests `session.race_control_messages` with automated classification into `SAFETY_CAR`, `VIRTUAL_SAFETY_CAR`, `RED_FLAG`, `YELLOW_FLAG`, `TRACK_CLEAR`, `OTHER`.
- `src/ingestion/pit_stops.py`: Ingests pit stops from session laps (`race_id, driver, lap, stint, pit_duration, compound_before, compound_after, tyre_age_before, fresh_tyre_after`).
- `src/ingestion/run_ingestion.py`: Ingestion pipeline updated to write all 7 tables to `data/processed/<table>/<race_id>.parquet`.
- `tests/test_partner_ingestion.py`: 5 offline tests passing for weather, race control, and pit stop extraction.

**Decision-Making Lane — Day 3**

- Paired review and confirmation of `as_of_lap()` in `src/preprocessing/leakage.py`.
- Verified no future data leakage across all simulation state and feature lookups.

**Decision-Making Lane — Day 4**

- `src/simulation/state.py`: `RaceState` dataclass per Design.md §5 (`race_id`, `lap`, `positions`, `gaps`, `tyres`, `safety_car`, `total_laps`, `circuit`, `status`, `weather`, `last_lap_times`, `pit_stops_count`).
- `src/simulation/replay.py`: Historical replay loop (`build_state_at_lap`, `init_state`, `advance_lap`, `run_replay`) strictly filtered with `as_of_lap()`.

**Decision-Making Lane — Day 5**

- `src/simulation/safety_car.py`: Empirical safety-car deployment probability model `safety_car_probability(circuit, lap_fraction)` based on circuit base rates and race phases.
- `apply_safety_car_bunching(state, interval_spacing=0.8)`: Simulates the 'ghost car' pack compression effect on field gaps under Safety Car conditions.

**Decision-Making Lane — Day 6**

- `src/simulation/replay.py`: Wired in `predict_lap_time()` and `predict_tyre_degradation()` via `advance_lap(state, predict_lap_time_fn, predict_degradation_fn)` and `apply_pace(state, driver, pace, degradation)`.
- Verified multi-driver forward state stepping with tyre age incrementation, relative pace offsets, and rank re-sorting.
- `tests/test_simulation.py`: Offline tests passing for RaceState serialization, SC probability, pack bunching, and predictive lap advancement.

---

## 5. Shared Contracts & Decisions

**Interface signatures (Design.md §5): unchanged.** Any change must be flagged to the other lane first.

**Race list** (`configs/races.toml`) — ⚠️ *needs partner confirmation*:
2023 Bahrain, Australia, Monaco (wet), Spain, Britain, Netherlands (wet), Italy, Singapore.

**"As of lap N" semantics** — ⚠️ *needs partner confirmation*: the moment lap N has just been
completed, so rows for lap N itself are visible. Rows with a missing lap are dropped (treated as possibly future).
`as_of_lap()` returns a copy.

**`predict_tyre_degradation()` behaviour** — ⚠️ *tell partner before their Day 6*:

- Returns seconds/lap lost to wear vs the **same compound when fresh** (age ≤ 3), floored at 0. It does **not**
  include soft-vs-hard pace offset or fuel burn-off — those come from `predict_lap_time()`.
- `circuit` = `races.circuit` value (e.g. "Sakhir", "Melbourne"). Unseen circuit → average of known circuits.
- Dry compounds only: INTERMEDIATE/WET raise `ValueError` — the replay must handle wet laps itself (P1 wet flag).
- `track_temp` is accepted but ignored until the `weather` table (`race_id, lap, track_temp`) exists; retraining
  picks it up automatically.
- Needs a trained model: run `python -m src.models.tyre` once after ingestion. Calls are memoised (~0.5 µs cached).

**`predict_lap_time()` behaviour** — ⚠️ *tell partner before their Day 6; one additive contract change*:

- **Contract change (additive):** `RaceState` needs a `race_id: str` field for the module-level `predict_lap_time(state, driver)`.
  Signature unchanged. Without it, the function raises and points to the factory below.
- Predicts lap `state.lap + 1` as a **racing lap**: no pit-lane time (strategy adds pit loss). `state.safety_car=True` → × 1.537
  (median fully-neutralised lap / race median, from data).
- `state.tyres[d] = (compound, age)` at the end of `state.lap`; next lap runs at age + 1. After a simulated stop set `(new_compound, 0)`.
- `state.gaps` = gap to leader in seconds; the gap to the car ahead is derived from `positions` + `gaps`.
- **Replay/API:** `predict_lap_time(state, driver)` uses real history up to `state.lap`.
- **Monte Carlo:** `make_lap_time_predictor(race_id, decision_lap)` once, then call it with simulated states (`lap ≥ decision_lap`).
  Real laps after the decision lap are never read. For speed step all sims together: `predictor.predict_many(states)` → one dict per state.
- Wet compounds are accepted (unlike the tyre model), but wet accuracy is poor.

**Storage convention:** `data/processed/<table>/<race_id>.parquet`; `src.ingestion.storage.connect()`
exposes every table folder as a DuckDB view. Partner's tables should use the same layout
(`pit_stops`, `race_control`, `weather`).

**Schema additions vs Design.md §4** — ⚠️ *needs partner confirmation*:

- `TyreStint` gains `lap_number` (one row per lap) so `as_of_lap()` can filter it; also `stint`, `fresh_tyre`
- `Lap` keeps extras: `team`, `pit_out_flag`, `track_status`, `is_accurate`
- New `Telemetry` table, one row per driver per lap: speed, throttle, brake, DRS summaries
- `Race` gains `round`, `event_name`

---

## 6. Next Up

Full-build phase tracking (PRD.md §8 build order) — this supersedes the two-lane sprint-day
tracking in Section 3, which is now historical (both lanes are being built solo).

1. **FR-2 loose end:** SHAP explainability was never wired up despite Phase 2 being marked done —
   `shap` is now installed; add SHAP value output to the tyre model and surface it via
   `/api/tyre/...` for the Model view.
2. **FR-7 Strategy Engine 2 (game theory):** ✅ `get_strategy_recommendation_gametheory(state, rival_state)`
   in `src/strategy/gametheory.py` — Stackelberg-style leader/follower optimization with rival best response,
   same `{action, tyre, expected_gain, confidence, engine}` return shape as Engine 1.
3. **FR-7 Strategy Engine 3 (RL):** ✅ gymnasium `RaceStrategyEnv` in `src/strategy/rl_env.py`
   wrapping the Monte Carlo evaluation with Discrete(4) actions and Box(12) observation space, plus
   `get_strategy_recommendation_rl()` with trained policy or greedy MC fallback. Wired
   `?engine=search|gametheory|rl` into `/api/strategy/{race_id}/{lap}` and added `?include_shap=true|false`
   to `/api/tyre/{race_id}/{driver}/{lap}`. 17 new tests passing in `tests/test_strategy_engines.py`.
4. **FR-8:** ✅ `src/evaluation/backtest.py` now dispatches any of the 3 engines
   (`backtest_decision_point(..., engine="search"|"gametheory"|"rl")`), reports
   `regret_vs_hindsight` per decision point, and `run_multi_engine_backtest()` /
   `summarize_multi_engine_backtest()` give the three-way comparison, wired to
   `GET /api/backtest/compare/all`. `run_dataset_backtest()` + `decision_points_for_race()`
   (real pit stops, not curation) + `scripts/run_full_backtest.py` (resumable) extend this
   beyond the 6 curated `HISTORICAL_BENCHMARKS` races.
5. **Run `python scripts/run_full_backtest.py` for real** across the full 112-race dataset,
   all 3 engines — built and smoke-tested (4 races) but not actually executed at scale yet.
   This is PRD Section 9's headline result ("an honest three-way comparison... across the
   full dataset") and should happen before claiming FR-8 is evidenced, not just implemented.
   Expect this to take a while (112 races × 3 engines × 2 decision points); run in background.
6. **Corner/mini-sector driver-performance module (FR-9):** ✅ Implemented in `src/performance/corner.py`
   (`analyze_corner_performance()`, `analyze_driver_lap_vs_benchmark()`) and exposed via
   `GET /api/corner-analysis/{race_id}/{driver}/{lap}`. Evaluates per-corner delta times, minimum apex speeds,
   throttle percentages, and braking points vs. benchmark laps. 14 new tests in `tests/test_corner_analysis.py` (164/164 tests passing).

---

## 7. Open Issues / Risks

| Issue | Impact | Plan |
| --- | --- | --- |
| `gap_to_leader` inflated to 3000–4000 s on red-flag laps (Australia L8/55, Netherlands L64) | Would corrupt gap features | Mask/cap in `src/preprocessing/`; ingestion stays raw |
| `races.weather_summary` covers the whole race | Leakage if used as a feature | Display only; per-lap weather from WeatherSnapshot |
| Tyre model depends on partner's weather table | Day 4 start | Coordinate Day 2 output; if late, train without temp first and add it when the table lands |
| Wear is hard to predict for an unseen circuit (held-out race curve error 0.48 s; Australia 1.3 s) | Weak "new circuit" story; demo races are all in training so the demo is unaffected | Track temp may help; report honestly in Day 8 writeup |
| Target build is sensitive to how wear is parameterised (free per-age values gave multi-second swings at Monaco/Singapore) | Noisy labels | Piecewise-linear wear (knots at +10, +20 laps) in target build; revisit if lap-time model disagrees |
| Lap-time model barely beats "repeat recent pace" for the very next lap (0.91 vs 0.98 s held-out races; tie on late laps) | Replay endpoint's one-lap prediction adds little | Value is at multi-lap horizons (1.53 vs 1.97 s); say so in Day 8 writeup. Ingested telemetry is unused so far, could try it |
| Partner may call `predict_lap_time` per driver per sim (~7 ms each → minutes) | Misses PRD latency target | `predict_many()` documented in §5 |
| Only ~600 INTERMEDIATE laps (2 races) and 48 WET laps | No reliable wet tyre model | Tyre model is dry-only; wet handled as a flag (P1) |
| Lap-to-lap noise (SD 0.3–0.8 s) ≫ per-lap degradation (~0.05 s) | Single-lap tyre MAE will look poor | Evaluate the degradation curve over a stint as well as per-lap MAE |
| Telemetry summaries on red-flag laps include the stoppage (n_samples 12k–15k) | Garbage speed/throttle features | Drop with the neutralised-lap filter |
| `predict_*` handoff by Day 5–6 is the sprint's tightest dependency | Blocks partner Day 6 | Flag early if slipping |

---

## 8. Change Log

Newest first. One line per commit: `date · who · what changed`.

| Date | Who | Change |
| --- | --- | --- |
| 2026-09-23 | Abubaker | Phase 7 (FR-7 Engine 3): Add comprehensive unit and integration tests in tests/test_rl_policy.py for environment, reward functions, and recommendation contract |
| 2026-09-23 | Abubaker | Phase 7 (FR-7 Engine 3): Wire trained RL policy auto-loading into get_strategy_recommendation_rl() with full candidate Monte Carlo evaluation |
| 2026-09-23 | Abubaker | Phase 7 (FR-7 Engine 3): Add offline PPO training pipeline in src/strategy/rl_policy.py with CachedStrategyEnv and parallel dataset builder |
| 2026-09-23 | Abubaker | Phase 7 (FR-7 Engine 3): Expand RL observation space 12 -> 14 dims to incorporate gap ahead and gap behind undercut signals; update RaceStrategyEnv observation space and unit tests |
| 2026-09-23 | Abubaker | Phase 8 (FR-9 corner/mini-sector driver performance): Implement corner telemetry segmentation, per-corner delta calculations, brake/throttle/speed profiling in `src/performance/corner.py`; wire `GET /api/corner-analysis/{race_id}/{driver}/{lap}` endpoint in `backend/main.py`; add 14 unit and integration tests in `tests/test_corner_analysis.py` (164/164 tests passing) |
| 2026-09-22 | Abubaker | Phase 7 (FR-8 full-dataset backtest): Fix `backtest_decision_point()`'s fallback for unbenchmarked races — it was deriving "actual_finish" from the AI's own `expected_position`, silently scoring every non-curated point as a match. Replaced with `_actual_outcome()` (real pit stop + classified finish from `pit_stops`/`laps`) and `decision_points_for_race()` (one decision point per driver's earliest real stop). Add `run_dataset_backtest()` and resumable `scripts/run_full_backtest.py` (checkpoints to `data/models/backtest/checkpoint.jsonl`) to run all 3 engines across all 112 races, not just the 6 curated ones. Smoke-tested on 4 races incl. a resume-from-checkpoint check; full-scale run not yet executed. Fixed a DuckDB API bug found in the process (`con.sql(query, params=[...])` needs `params` as a keyword, not positional — the two new queries had it positional and failed every call). 6 new tests, 150/150 passing |
| 2026-09-22 | Abubaker | Phase 7 (FR-8): Extend `src/evaluation/backtest.py` to dispatch any of the 3 strategy engines and report `regret_vs_hindsight`; add `run_multi_engine_backtest()` / `summarize_multi_engine_backtest()` for the three-way comparison PRD Section 9 asks for; wire `GET /api/backtest/compare/all` and `?engine=` on the existing backtest endpoints; 7 new tests. Before this, reviewed the Engine 2/3 code from the prior 3 commits and fixed 3 real bugs: `gametheory.py` recomputing the rival's best response identically on every leader candidate (redundant, and never actually varied by candidate despite the docstring), `rl_env.py` evaluating "N roll-outs" with `run_monte_carlo`'s fixed default seed so every episode was byte-identical (fake confidence metric), and `get_strategy_recommendation_rl()` missing `podium_prob`/`win_prob`/`finish_prob_by_position`/`candidates` that Engines 1–2 return — confirmed this rendered `"NaN%"` in `frontend/dist/app.js`'s strategy view for `?engine=rl`. 144/144 tests passing |
| 2026-09-22 | Abubaker | Phase 7 (API & Tests): Wire ?engine=search\|gametheory\|rl into /api/strategy and ?include_shap into /api/tyre; add 17 unit/integration tests in tests/test_strategy_engines.py (140 total tests passing) |
| 2026-09-22 | Abubaker | Phase 7 (FR-7 Engine 3): Implement Gymnasium RL environment RaceStrategyEnv (src/strategy/rl_env.py) with Discrete(4) actions, Box(12) state, and get_strategy_recommendation_rl() fallback |
| 2026-09-22 | Abubaker | Phase 7 (FR-7 Engine 2): Implement Stackelberg game theory strategy recommendation (src/strategy/gametheory.py) with rival best response modeling and unified recommendation schema |
| 2026-09-22 | Abubaker | Phase 2 (FR-2): Add TreeSHAP explainability (shap_values_for) to tyre degradation model and wire into /api/tyre endpoint |
| 2026-09-22 | Abubaker | Environment fix: installed `torch` (was pinned but missing from `.venv`, breaking 15 tests + backend boot) and pinned/installed `shap`, `gymnasium`, `stable-baselines3` (approved deps, never pinned). Full suite now 123/123 passing, `backend.main` confirmed to import cleanly. Corrected this file's phase tracking — FR-6 (Monte Carlo, `src/simulation/monte_carlo.py`) and FR-7 Strategy Engine 1 (exhaustive search, `src/strategy/optimizer.py`, `/api/simulation`, `/api/strategy`) were already built and are not reflected accurately in prior entries below. Next: FR-7 Engines 2–3 (game theory, RL) and FR-8 (three-engine backtest) |
| 2026-09-21 | Abubaker | Phase 5 (FR-5): Calibrate empirical Safety Car deployment model across 112 multi-season races (2018–2024, 33 circuits) with Empirical Bayes shrinkage and race-phase multipliers (scripts/calibrate_safety_car.py -> configs/safety_car_rates.json), multi-lap SC episode tracking and realistic bunching dynamics in simulation (src/simulation/safety_car.py, src/simulation/monte_carlo.py), API endpoint (/api/safety-car/{circuit}), and test suite (tests/test_safety_car.py). 123 passing tests |
| 2026-09-21 | Abubaker | Phase 4 (FR-4): Implement comprehensive race replay engine with RaceState timing extensions (intervals, retired, fastest_lap, pit_stops_history, race_control_events), multi-table strict no-leakage aggregation (laps, tyres, pit_stops, weather, race_control), fast ReplaySession scrubbing/streaming, dedicated future-mutation verification suite (tests/test_replay_leakage.py), and API endpoints (/api/replay/summary, /api/replay/events). 108 passing tests |
| 2026-09-21 | Abubaker | Phase 3 (FR-3): Implement interaction-aware lap-time modeling with RaceGraph data structures, sequential GRU baseline, relational message-passing GNN model (predict_lap_time_gnn), multi-circuit & traffic-stratified evaluation harness (src/evaluation/lap_time_eval.py, scripts/run_lap_time_eval.py), and FastAPI endpoint update (?model=baseline\|gnn). 100 passing tests |
| 2026-09-21 | Abubaker | Add frozen tyre evaluation split (configs/tyre_split.toml), ablation scripts, and project guidance (CLAUDE.md) |
| 2026-09-21 | Abubaker | Phase 2 (in progress): frozen tyre split `configs/tyre_split.toml` (14 test races, 3 wet, Monaco/Monza/Baku held out in every era), `src/evaluation/tyre_eval.py` harness, Bayesian + mean-curve baselines, XGBoost ablations + SHAP, `scripts/run_tyre_eval.py`, fixed crash on laps with unknown tyre age. **Served model changed**: `RoutedTyreModel` (circuit-aware XGBoost for seen circuits, circuit-agnostic for unseen; `track_temp` ignored) trained on all 105 races; `predict_tyre_degradation` values shift for the simulation/optimizer/backtest. 9 new tests, 89 total pass |
| 2026-09-21 | Abubaker | Update PRD and Design specifications for full build roadmap and architecture |
| 2026-09-21 | Abubaker | Dataset expansion & validation: 112 races (2018–2024, 33 circuits) config, race condition tags, dataset validation suite |
| 2026-09-14 | Abubaker | Days 6–7: lap-time model + `predict_lap_time()` / `make_lap_time_predictor()` / batched `predict_many()`, 14 tests; Day 5 track temp blocked on weather table |
| 2026-09-14 | Abubaker | Day 4: lap cleaning filters, tyre degradation model v1 + `predict_tyre_degradation()`, model config, 16 tests, pinned xgboost/scikit-learn/matplotlib |
| 2026-09-13 | Abubaker | Day 3: `as_of_lap()` no-leakage guard + 13 tests, EDA notebook, findings and next steps |
| 2026-09-13 | Abubaker | Add *.pdf to .gitignore and untrack research papers from git |
| 2026-09-13 | Abubaker | Days 1–2: scaffold, FastF1 caching, lap/tyre/telemetry/race ingestion for 8 races, Parquet + DuckDB storage, 10 tests, status file + pre-commit hook |
