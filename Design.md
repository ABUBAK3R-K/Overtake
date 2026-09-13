# Design — Overtake
**Technical Architecture & Interface Contracts**

Companion to `PRD.md` (defines *what*/*why*) and `overtake-15-day-sprint-plan.md` (defines *when*). This document defines *how*.

---

## 1. Architecture Overview

```
FastF1 / OpenF1 (raw data)
        │
        ▼
  Ingestion Layer  ──────────────────────────────┐
        │                                        │
        ▼                                        │
  Feature Engineering + No-Leakage Guard          │  (shared foundation, both lanes depend on this)
        │                                        │
        ▼                                        ▼
┌───────────────────────┐            ┌─────────────────────────────┐
│   PREDICTION LANE      │            │   DECISION-MAKING LANE       │
│  Tyre Degradation Model│──predicts─▶│  Race Replay Engine          │
│  Lap-Time Model        │   pace     │  (+ Safety-Car "Ghost Car")  │
└───────────────────────┘            │        │                     │
                                      │        ▼                     │
                                      │  Monte Carlo Simulation      │
                                      │        │                     │
                                      │        ▼                     │
                                      │  Strategy Optimizer          │
                                      │        │                     │
                                      │        ▼                     │
                                      │  Backtesting / Evaluation    │
                                      └─────────────────────────────┘
                                                 │
                                                 ▼
                                    FastAPI backend (both lanes expose endpoints)
                                                 │
                                                 ▼
                                    React + Plotly Dashboard (6 views, split by owner)
```

The Prediction Lane and Decision-Making Lane develop in parallel against the interface contracts in Section 5, so neither person blocks on the other until integration.

---

## 2. Tech Stack

| Layer | Choice |
|---|---|
| Data ingestion | FastF1 (primary), OpenF1 (optional supplement) |
| Storage | Parquet files, queried via DuckDB |
| Modeling | scikit-learn / XGBoost (baseline for both models) |
| Simulation | Pure Python + NumPy (no external simulation framework needed at this scale) |
| Backend | FastAPI |
| Frontend | React + TypeScript + Plotly |
| Packaging | Docker + docker-compose (local demo target) |
| Testing | pytest |

---

## 3. Repo Structure

```
overtake/
├── data/
│   └── cache/                # FastF1 cache
├── notebooks/                 # exploration, EDA
├── src/
│   ├── ingestion/             # FastF1/OpenF1 loaders
│   ├── preprocessing/         # cleaning, feature engineering, no-leakage guard
│   ├── models/                # tyre + lap-time models
│   ├── simulation/             # replay engine, safety-car model, Monte Carlo
│   ├── strategy/               # strategy optimizer
│   └── evaluation/             # backtesting
├── backend/                    # FastAPI app
├── frontend/                   # React dashboard
├── tests/
├── configs/                    # race list, model hyperparameters
├── docker-compose.yml
└── README.md
```

---

## 4. Data Model

| Entity | Key fields |
|---|---|
| **Race** | race_id, circuit, season, total_laps, weather_summary |
| **Lap** | race_id, driver, lap_number, lap_time, sector_1/2/3_time, position, gap_to_leader, pit_flag |
| **TyreStint** | race_id, driver, compound, stint_start_lap, tyre_age_at_lap |
| **PitStop** | race_id, driver, lap, pit_duration, compound_before, compound_after |
| **RaceControlEvent** | race_id, lap, event_type (SC / VSC / RedFlag / YellowFlag) |
| **WeatherSnapshot** | race_id, lap, track_temp, air_temp, is_wet |

All entities are keyed by `race_id` + `lap` or `driver`, enabling the no-leakage guard (Section 5) to filter any of them consistently by "as of lap N."

---

## 5. Core Interfaces (Contracts Between Lanes)

These signatures are the contract the two lanes build against. **Do not change a signature after Day 8 (see sprint plan Section 7) without telling the other person** — the dashboard and backtesting code depend on them being stable.

```python
# Shared foundation — src/preprocessing/leakage.py
def as_of_lap(df: pd.DataFrame, lap_col: str, current_lap: int) -> pd.DataFrame:
    """Returns only rows with lap_col <= current_lap. Every feature or
    state lookup anywhere in the project routes through this."""

# Prediction Lane hands off:
def predict_tyre_degradation(compound: str, age: int, circuit: str,
                              track_temp: float) -> float:
    """Returns predicted pace loss (seconds) for this tyre state."""

def predict_lap_time(state: "RaceState", driver: str) -> float:
    """Returns predicted next-lap time (seconds) for this driver
    given the current race state."""

# Decision-Making Lane hands off:
@dataclass
class RaceState:
    lap: int
    positions: dict[str, int]        # driver -> position
    gaps: dict[str, float]            # driver -> gap_to_leader_seconds
    tyres: dict[str, tuple[str, int]]  # driver -> (compound, age)
    safety_car: bool

def get_strategy_recommendation(state: RaceState) -> dict:
    """Returns {action, tyre, expected_gain, confidence}."""

def run_monte_carlo(state: RaceState, strategy: dict, n_sims: int = 1000) -> dict:
    """Returns {finish_prob_by_position, expected_position, expected_time}."""
```

---

## 6. Module Design

### 6.1 Ingestion (`src/ingestion/`)
Thin wrappers around FastF1's session-loading API, one function per data type (laps, telemetry, pit stops, race control, weather), each writing a Parquet file per race into `data/`. OpenF1 supplement, if used, lives here too as an alternate loader behind the same output schema.

### 6.2 Feature Engineering & No-Leakage Guard (`src/preprocessing/`)
Houses `as_of_lap()` (Section 5) and all derived features (rolling pace, gap trends, stint length). Every feature function takes a `current_lap` argument and calls `as_of_lap()` internally — no feature function should read a full-race dataframe without this filter.

### 6.3 Tyre Degradation Model (`src/models/tyre.py`)
XGBoost regressor. Inputs: compound, tyre age, circuit, track temperature, recent lap pace trend. Output: predicted pace loss. Trained per-compound or with compound as a categorical feature — try both, report which generalizes better across circuits.

### 6.4 Lap-Time Model (`src/models/lap_time.py`)
XGBoost regressor (baseline) predicting next-lap time from driver, circuit, tyre state, previous lap/sector times, gaps, and traffic indicators. Report MAE/RMSE against a held-out set of laps, not held-out races only — both splits reveal different failure modes.

### 6.5 Replay Engine & Safety-Car Model (`src/simulation/replay.py`)
Owns the `RaceState` object and `advance_lap()` step function. The safety-car model is a lookup of historical deployment frequency by circuit and race-phase (built from ingested `RaceControlEvent` data) — deliberately simple (empirical frequency, not a learned model) since the goal is a defensible baseline, not a research contribution on its own.

### 6.6 Monte Carlo Simulation (`src/simulation/monte_carlo.py`)
For a given state and strategy, runs N independent forward simulations sampling noise around `predict_lap_time()` and drawing safety-car events from the Section 6.5 model, aggregating into a finishing-position distribution. N=500–1000 balances stability against the sub-few-second latency target in `PRD.md` Section 7.

### 6.7 Strategy Optimizer (`src/strategy/optimizer.py`)
Exhaustive search over a candidate set (pit lap ± window, 2–3 compound choices), each candidate scored via Section 6.6, ranked by expected outcome. Deliberately not RL or game-theoretic for this iteration — see `PRD.md` Section 4 non-goals and Section 9 future extensions below.

### 6.8 Backtesting (`src/evaluation/backtest.py`)
For each race, re-runs the strategy optimizer at several real decision points and compares against the real outcome and the best-possible hindsight strategy (computed by exhaustively trying real historical strategy variants against the actual pace data). Outputs a comparison table across races — this is the project's central evidence of whether the system works.

### 6.9 API Layer (`backend/`)

| Endpoint | Returns |
|---|---|
| `GET /api/races` | list of ingested races |
| `GET /api/replay/{race_id}/{lap}` | `RaceState` at that lap |
| `GET /api/tyre/{race_id}/{driver}/{lap}` | degradation prediction |
| `GET /api/lap-prediction/{race_id}/{driver}/{lap}` | next-lap prediction |
| `GET /api/strategy/{race_id}/{lap}` | `get_strategy_recommendation()` output |
| `GET /api/simulation/{race_id}/{lap}` | `run_monte_carlo()` output |
| `GET /api/backtest/{race_id}` | backtest comparison table |

### 6.10 Frontend / Dashboard (`frontend/`)
Six views, each a self-contained React component fetching from its own endpoint: **Race**, **Strategy**, **Simulation** (Decision-Making Lane's views); **Tyre**, **Model** (Prediction Lane's views); **Driver/corner** (P2 — not built this iteration, see `PRD.md` Section 4).

---

## 7. Testing Strategy

- **Unit test** on `as_of_lap()` — this is the single highest-value test in the repo given how much depends on it.
- **Model evaluation harness** — a script that reports MAE/RMSE for both models on a fixed held-out split, run on every significant model change.
- **Integration test** — a full `run_replay()` on one cached race completes without exception and produces a plausible final classification.

---

## 8. Deployment Design

Local-only for this iteration: `docker-compose.yml` running backend + frontend + a mounted data volume, started with a single `docker-compose up`. No live external API dependency required at demo time — all race data is pre-cached. Cloud deployment is P2 (see `PRD.md` Section 8) and, if attempted, is additive rather than a redesign.

---

## 9. Future Extensions (Not Built This Iteration)

Documented here so the architecture doesn't accidentally preclude them later:

- **GNN interaction layer:** would replace the single-driver `predict_lap_time()` with a graph-based model where each race timestep is a graph (nodes = cars, edges = gap/DRS-range relationships), producing an interaction-aware pace prediction. Slots in at the same point `predict_lap_time()` is called today — the interface wouldn't need to change, only the implementation behind it.
- **Reinforcement-learning strategy engine:** would replace `get_strategy_recommendation()`'s exhaustive search with a learned policy (DQN/PPO), trained against the same `run_monte_carlo()` simulator already built — the simulator is reusable as the RL environment's reward signal.
- **Competitor-aware / game-theoretic strategy:** would extend the strategy optimizer to jointly consider a rival's likely response (Stackelberg-style), rather than optimizing one car in isolation.

None of these require a rearchitecture — they're designed to slot into the existing interface points, which is the main reason the interfaces in Section 5 are worth keeping stable.
