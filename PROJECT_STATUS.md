# Overtake — Project Status

> **Update this file in every commit.** The pre-commit hook in `.githooks/` rejects
> commits that don't include a change to it. One-time setup per clone:
> `git config core.hooksPath .githooks`
>
> How to update: tick off finished items, move "Next up", add any new
> decisions/issues, and add one line to the Change Log at the bottom.

**Last updated:** 2026-09-14 · Prediction Lane · Day 4 tyre model v1 trained, `predict_tyre_degradation()` ready (without track temp)

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
|---|---|---|
| Prediction (ingestion, tyre model, lap-time model) | Abubaker | 🟢 Days 1–4 done, ahead on Day 5 handoff |
| Decision-Making (replay, safety car, Monte Carlo, optimizer, backtest) | Partner | ⚪ Not yet reported here |
| Shared (schema, no-leakage guard, integration, Docker) | Both | 🟡 Schema drafted, guard written — awaiting partner review |

Legend: ✅ done · 🟡 in progress · ⬜ not started · ⚠️ blocked/at risk

---

## 3. Milestones by Sprint Day

### Prediction Lane (Abubaker)
| Day | Deliverable | Status |
|---|---|---|
| 1 | Repo scaffold (Design.md §3), FastF1 cache in `data/cache/` | ✅ |
| 2 | Ingest laps, sectors, telemetry, tyre compound/age → Parquet + DuckDB | ✅ |
| 3 | No-leakage guard `as_of_lap()` + EDA (paired) | ✅ (partner review pending) |
| 4–5 | Tyre model → `predict_tyre_degradation(compound, age, circuit, track_temp)` | 🟡 v1 usable now; track temp pending partner's weather table · **handoff due end of Day 5** |
| 6–7 | Lap-time model → `predict_lap_time(state, driver)` | ⬜ **handoff Day 6–7** |
| 8 | MAE/RMSE report for both models (held-out laps and held-out races) | ⬜ |
| 9–10 | FastAPI: `/api/tyre/...`, `/api/lap-prediction/...` | ⬜ |
| 11–13 | Tyre view + Model view (React + Plotly) | ⬜ |
| 14–15 | Joint integration + bug fixing | ⬜ |

### Decision-Making Lane (Partner)
| Day | Deliverable | Status |
|---|---|---|
| 1 | Shared data schema agreed | 🟡 draft — see §5 |
| 2 | Ingest pit stops, race control, weather → Parquet | ⬜ |
| 3 | No-leakage guard + EDA (paired) | 🟡 guard in `src/preprocessing/leakage.py` — please review |
| 4 | `RaceState` + bare replay loop | ⬜ |
| 5 | Safety-car ("ghost car") model | ⬜ |
| 6 | Wire in prediction functions | ⬜ |
| 7 | Monte Carlo sampling | ⬜ |
| 8 | Strategy optimizer — `get_strategy_recommendation()` signature locked | ⬜ |
| 9–10 | FastAPI: replay / strategy / simulation endpoints | ⬜ |
| 11–13 | Race, Strategy, Simulation views + backtesting | ⬜ |
| 14–15 | Joint integration + bug fixing | ⬜ |

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
  |---|---|---|---|
  | Held-out stints (known circuits — the demo case) | **0.17 s** | 0.18 s | 0.44 s |
  | Held-out race (new circuit) | **0.48 s** | 0.48 s | 0.73 s |

  Pooled and per-compound tie → fixed to pooled (simpler). Worst held-out race: Australia (almost no wear, other circuits predict more).
- Tests: `tests/test_cleaning.py` (6), `tests/test_tyre_model.py` (10, synthetic races with known wear) — suite 39 passing
- Packages pinned: xgboost 3.4.1, scikit-learn 1.9.1, matplotlib 3.11.2

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

1. Confirm with partner: race list, schema additions, "as of lap N" semantics, and `predict_tyre_degradation()` behaviour (§5)
2. Day 5: add track temp once the `weather` table lands and retrain; check whether it helps held-out-race error
   (the only feature that could explain new-circuit wear). Tyre curve plot for the Tyre view.
3. Day 6–7: lap-time model `src/models/lap_time.py` → `predict_lap_time(state, driver)`; features via `as_of_lap()`,
   lagged telemetry, `clean_gap_to_leader()`

---

## 7. Open Issues / Risks

| Issue | Impact | Plan |
|---|---|---|
| `gap_to_leader` inflated to 3000–4000 s on red-flag laps (Australia L8/55, Netherlands L64) | Would corrupt gap features | Mask/cap in `src/preprocessing/`; ingestion stays raw |
| `races.weather_summary` covers the whole race | Leakage if used as a feature | Display only; per-lap weather from WeatherSnapshot |
| Tyre model depends on partner's weather table | Day 4 start | Coordinate Day 2 output; if late, train without temp first and add it when the table lands |
| Wear is hard to predict for an unseen circuit (held-out race curve error 0.48 s; Australia 1.3 s) | Weak "new circuit" story; demo races are all in training so the demo is unaffected | Track temp may help; report honestly in Day 8 writeup |
| Target build is sensitive to how wear is parameterised (free per-age values gave multi-second swings at Monaco/Singapore) | Noisy labels | Piecewise-linear wear (knots at +10, +20 laps) in target build; revisit if lap-time model disagrees |
| Only ~600 INTERMEDIATE laps (2 races) and 48 WET laps | No reliable wet tyre model | Tyre model is dry-only; wet handled as a flag (P1) |
| Lap-to-lap noise (SD 0.3–0.8 s) ≫ per-lap degradation (~0.05 s) | Single-lap tyre MAE will look poor | Evaluate the degradation curve over a stint as well as per-lap MAE |
| Telemetry summaries on red-flag laps include the stoppage (n_samples 12k–15k) | Garbage speed/throttle features | Drop with the neutralised-lap filter |
| `predict_*` handoff by Day 5–6 is the sprint's tightest dependency | Blocks partner Day 6 | Flag early if slipping |

---

## 8. Change Log

Newest first. One line per commit: `date · who · what changed`.

| Date | Who | Change |
|---|---|---|
| 2026-09-14 | Abubaker | Day 4: lap cleaning filters, tyre degradation model v1 + `predict_tyre_degradation()`, model config, 16 tests, pinned xgboost/scikit-learn/matplotlib |
| 2026-09-14 | Abubaker | Day 3: `as_of_lap()` no-leakage guard + 13 tests, EDA notebook, findings and next steps |
| 2026-09-13 | Abubaker | Add *.pdf to .gitignore and untrack research papers from git |
| 2026-09-13 | Abubaker | Days 1–2: scaffold, FastF1 caching, lap/tyre/telemetry/race ingestion for 8 races, Parquet + DuckDB storage, 10 tests, status file + pre-commit hook |
