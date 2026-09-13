# Overtake — Project Status

> **Update this file in every commit.** The pre-commit hook in `.githooks/` rejects
> commits that don't include a change to it. One-time setup per clone:
> `git config core.hooksPath .githooks`
>
> How to update: tick off finished items, move "Next up", add any new
> decisions/issues, and add one line to the Change Log at the bottom.

**Last updated:** 2026-09-13 · Prediction Lane · Days 1–2 ingestion complete

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
| Prediction (ingestion, tyre model, lap-time model) | Abubaker | 🟢 Days 1–2 done |
| Decision-Making (replay, safety car, Monte Carlo, optimizer, backtest) | Partner | ⚪ Not yet reported here |
| Shared (schema, no-leakage guard, integration, Docker) | Both | 🟡 Schema drafted, guard pending (Day 3) |

Legend: ✅ done · 🟡 in progress · ⬜ not started · ⚠️ blocked/at risk

---

## 3. Milestones by Sprint Day

### Prediction Lane (Abubaker)
| Day | Deliverable | Status |
|---|---|---|
| 1 | Repo scaffold (Design.md §3), FastF1 cache in `data/cache/` | ✅ |
| 2 | Ingest laps, sectors, telemetry, tyre compound/age → Parquet + DuckDB | ✅ |
| 3 | No-leakage guard `as_of_lap()` + EDA (paired) | ⬜ |
| 4–5 | Tyre model → `predict_tyre_degradation(compound, age, circuit, track_temp)` | ⬜ **handoff due end of Day 5** |
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
| 3 | No-leakage guard + EDA (paired) | ⬜ |
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

---

## 5. Shared Contracts & Decisions

**Interface signatures (Design.md §5): unchanged.** Any change must be flagged to the other lane first.

**Race list** (`configs/races.toml`) — ⚠️ *needs partner confirmation*:
2023 Bahrain, Australia, Monaco (wet), Spain, Britain, Netherlands (wet), Italy, Singapore.

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

1. Confirm race list + schema additions with partner (§5)
2. Day 3 (paired): write `src/preprocessing/leakage.py::as_of_lap()` + its unit test; EDA notebook
3. Day 4: start tyre model — needs partner's per-lap `track_temp` (WeatherSnapshot table)

---

## 7. Open Issues / Risks

| Issue | Impact | Plan |
|---|---|---|
| `gap_to_leader` inflated to 3000–4000 s on red-flag laps (Australia L8/55, Netherlands L64) | Would corrupt gap features | Mask/cap in `src/preprocessing/`; ingestion stays raw |
| `races.weather_summary` covers the whole race | Leakage if used as a feature | Display only; per-lap weather from WeatherSnapshot |
| Tyre model depends on partner's weather table | Day 4 start | Coordinate Day 2 output |
| `predict_*` handoff by Day 5–6 is the sprint's tightest dependency | Blocks partner Day 6 | Flag early if slipping |

---

## 8. Change Log

Newest first. One line per commit: `date · who · what changed`.

| Date | Who | Change |
|---|---|---|
| 2026-09-13 | Abubaker | Add *.pdf to .gitignore and untrack research papers from git |
| 2026-09-13 | Abubaker | Days 1–2: scaffold, FastF1 caching, lap/tyre/telemetry/race ingestion for 8 races, Parquet + DuckDB storage, 10 tests, status file + pre-commit hook |
