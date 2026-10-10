# Overtake

An F1 race-strategy platform. It replays any of 112 historical races (2018–2024) lap by lap, predicts tyre wear and lap times, simulates the rest of the race thousands of times, and asks three different strategy engines for a pit call: exhaustive search, a game-theoretic engine that anticipates the car ahead's reply, and a trained RL policy. It then backtests all three against what the real teams did.

Everything a model sees at lap N comes from laps ≤ N only (`src/preprocessing/leakage.as_of_lap`). That rule has tests.

**Current progress:** [`PROJECT_STATUS.md`](PROJECT_STATUS.md). **Scope:** [`PRD.md`](PRD.md) (what) · [`Design.md`](Design.md) (how).

## Run it

With Docker (data and trained models are baked into the image):

```bash
docker compose up --build        # http://localhost:8000
```

Or locally (Python 3.13):

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
git config core.hooksPath .githooks              # pre-commit hook: every commit updates PROJECT_STATUS.md
uvicorn backend.main:app                         # http://localhost:8000
```

Building everything from scratch (hours; FastF1 allows 500 calls/hour and every step is resumable):

```bash
python -m src.ingestion.run_ingestion     # 112 races -> data/processed (Parquet, read through DuckDB)
python scripts/validate_dataset.py        # race tags + dataset report
python scripts/calibrate_safety_car.py    # safety-car rates per circuit
python -m src.models.tyre                 # tyre-wear model
python -m src.models.lap_time             # lap-time model (scores the frozen test races, then trains on all)
python -m src.strategy.rl_policy all      # RL policy: simulate training states, train PPO, evaluate
python scripts/run_full_backtest.py       # three-engine backtest over every race
python -m pytest tests
```

## The dashboard

A pit-wall layout: timing tower on the left, the active view in the middle, the strategy call on the right.

| View | What it shows |
| --- | --- |
| Race | Gap to the leader, pit stops and race-control messages so far |
| Strategy | Every plan the chosen engine simulated, with expected finish and its uncertainty; for game theory, the rival's best reply to each plan |
| Simulation | Finishing-position distribution and position band through the remaining laps, for any plan you type in |
| Tyres | Predicted wear curves and a SHAP breakdown of the current wear prediction |
| Driver | Corner-by-corner time lost against the fastest lap set so far |
| Models | Held-out accuracy of every model, baselines included |
| Backtest | The three engines and the real teams compared on regret |

Any state is a link: `http://localhost:8000/#race=2023_bahrain&lap=20&driver=LEC&view=strategy&engine=gametheory`.

### Demo walkthrough

1. 2023 Bahrain, lap 20, LEC, Strategy view. Ask for a call with **Search**, then switch to **Game theory**. The table now shows what VER would do in reply to each plan.
2. Same lap, **Simulation**: try "Pit on lap 30 for Hard" against "No further stop".
3. **Tyres** for the wear curve and why the model predicts it; **Driver** for the corner where LEC loses most to the fastest lap so far.
4. Pick a wet race (2023 Netherlands) and scrub through the rain: the flag chip and tyre swatches change; the Models view says how weak the wet numbers are.
5. **Backtest**: the three engines against history, and how far the simulator itself can be trusted.

## How it works

```
FastF1 ─► src/ingestion ─► data/processed/<table>/<race_id>.parquet ─► DuckDB views
                                   │
            ┌──────────────────────┴───────────────────────┐
     src/models/tyre                              src/models/lap_time
  (wear vs fresh, XGBoost,                   (next-lap pace, XGBoost; GRU and
   circuit-aware / -agnostic)                 GNN tracks evaluated alongside)
            └──────────────────────┬───────────────────────┘
                     src/simulation (replay as of lap N, safety-car model,
                     batched Monte Carlo of the remaining laps)
                                   │
          src/strategy: search · game theory (Stackelberg) · RL (PPO)
                                   │
                     src/evaluation/backtest ─► backend (FastAPI) ─► frontend/dist
```

The Monte Carlo steps every simulation of every candidate plan together, one lap-time-model call per lap, and gives simulation *i* of every plan the same random draws, so differences between plans come from the plans, not luck.

## Results

See the Models and Backtest views, or [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the numbers and their caveats.
