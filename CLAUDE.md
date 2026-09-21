# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this is

Overtake: F1 race-strategy platform. Ingests FastF1 data into Parquet/DuckDB, trains tyre-degradation and lap-time models (XGBoost), replays a race to any lap, runs Monte Carlo simulation, recommends pit calls, and backtests them. Served by FastAPI with a static frontend (`frontend/dist`, no build step). Scope docs: `PRD.md` (what), `Design.md` (how; §5 has the locked cross-lane interfaces), `overtake-15-day-sprint-plan.md` (when), `PROJECT_STATUS.md` (current progress).

## Commands

Windows dev environment; venv at `.venv`.

```bash
pip install -r requirements.txt
git config core.hooksPath .githooks          # once per clone
python -m src.ingestion.run_ingestion                 # all races in configs/races.toml (resumable, skips existing)
python -m src.ingestion.run_ingestion 2023_bahrain    # specific race(s)
python -m src.ingestion.run_ingestion --force         # rebuild existing
python -m src.models.tyre                    # retrain the served RoutedTyreModel (all races)
python scripts/run_tyre_eval.py              # score the frozen tyre test set (do this once, not for tuning)
python -m src.models.lap_time                # retrain lap-time model
python scripts/validate_dataset.py           # writes configs/race_tags.json + data/processed/dataset_report.md
python scripts/build_race_list.py            # regenerates configs/races.toml
python -m pytest tests
python -m pytest tests/test_leakage.py::test_name   # single test
uvicorn backend.main:app --reload            # API + dashboard
```

## Commit rule

The pre-commit hook (`.githooks/pre-commit`) **rejects any commit that does not include `PROJECT_STATUS.md`**. Update it (tick milestones, note decisions, add a Change Log line) and `git add` it every commit.

## Architecture

Pipeline: `src/ingestion` → Parquet (`data/processed/<table>/<race_id>.parquet`, one file per table per race) → DuckDB views via `src/ingestion/storage.connect()` → `src/models` (tyre, lap_time) → `src/simulation` (replay → `RaceState`, safety_car, monte_carlo) → `src/strategy/optimizer` → `src/evaluation/backtest` → `backend/main.py` (FastAPI, `/api/*`, mounts `frontend/dist` at `/`).

- **Two lanes share interfaces.** Prediction lane (ingestion, tyre, lap-time) exposes `predict_tyre_degradation(compound, age, circuit, track_temp)` and `predict_lap_time(state, driver)`; Decision lane (replay, SC, Monte Carlo, optimizer, backtest, dashboard) consumes them. Signatures are contracts in Design.md §5 — don't change them casually.
- **No-leakage rule (hard).** Every feature or state lookup must go through `src/preprocessing/leakage.as_of_lap(df, lap_col, current_lap)`; "as of lap N" means laps ≤ N only. Whole-race attributes (weather summary, race tags in `configs/race_tags.json`) are for stratifying evaluation only, never model features.
- **Race identity.** `race_id = <season>_<short_circuit_name>` is the primary key in every table; never rename an existing id. `configs/races.toml` is the versioned race list (112 races: 2022–2024 full at `train_weight` 1.0, curated 2018–2021 at 0.4); changing it changes all downstream results.
- **Ingestion** (`run_ingestion.py`): tables are `races, laps, tyres, telemetry, weather, race_control, pit_stops`. Bump `SCHEMA_VERSION` when a table's columns/meaning change so stale races re-ingest (tracked in `data/processed/manifest.json`). Handles retries and FastF1's 500 calls/hour limit (sleeps on rate limit); any new download must stay resumable. FastF1 cache lives in `data/cache/` (large).
- **Tyre model evaluation.** `configs/tyre_split.toml` is a frozen split (14 test races incl. 3 wet; Monaco/Monza/Baku held out in every era). Never tune on it or regenerate it; use the inner CV helpers in `src/evaluation/tyre_eval.py`. The served model is `RoutedTyreModel` (circuit-aware for seen circuits, circuit-agnostic for unseen; `track_temp` deliberately ignored). Wet-race numbers are unreliable (dry-compound laps only).
- **Model config** lives in `configs/models.toml`; changing it changes predictions the decision lane consumes, so retrain and note it in `PROJECT_STATUS.md`.
- Planned but not yet built per Design.md: GNN lap-time model, game-theory and RL strategy engines, corner analysis, `?engine=`/`?model=` API params.

## Working agreements (from memory)

Follow PRD.md §8 phases in order; present a short plan before each phase and a check-in (built / measured / failed) after it. Report unflattering results plainly. Extra deps approved beyond Design.md §2: torch, shap, gymnasium (Bayesian tyre model in NumPy/SciPy, no PyMC).
