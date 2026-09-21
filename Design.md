# Design — Overtake (Full Build)
**Technical Architecture & Interface Contracts**

Companion to `PRD.md` (defines *what*/*why*). This document defines *how*, at full scope — see `PRD.md` Section 12 for what changed from the original sprint-scoped version of this document.

---

## 1. Architecture Overview

```
FastF1 / OpenF1 (raw data, ~60-70 races, 15+ circuits)
        │
        ▼
  Ingestion Layer
        │
        ▼
  Feature Engineering + No-Leakage Guard   (shared foundation, everything below depends on this)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
┌────────────────────────────┐          ┌─────────────────────────────────┐
│     PREDICTION LANE         │          │      DECISION-MAKING LANE        │
│  Tyre Degradation Model      │─predicts▶│  Race Replay Engine              │
│   (XGBoost + Bayesian state)│  pace    │   + Safety-Car "Ghost Car"       │
│  Lap-Time Model               │         │        │                         │
│   (XGBoost/GRU baseline      │         │        ▼                         │
│    + GNN interaction model)  │         │  Monte Carlo Simulation           │
│  Corner/Driver Performance    │         │        │                         │
│   Module                      │         │        ▼                         │
└────────────────────────────┘          │  Strategy Engines (3, compared)   │
                                          │   1. Exhaustive Search            │
                                          │   2. Game-Theoretic (Stackelberg) │
                                          │   3. RL Policy (DQN/PPO)          │
                                          │        │                         │
                                          │        ▼                         │
                                          │  Backtesting / Evaluation         │
                                          └─────────────────────────────────┘
                                                       │
                                                       ▼
                                         FastAPI backend (both lanes expose endpoints)
                                                       │
                                                       ▼
                                    React + Plotly Dashboard — see Section 10 for design brief
```

---

## 2. Tech Stack

| Layer | Choice |
|---|---|
| Data ingestion | FastF1 (primary, 2018+), OpenF1 (optional 2023+ supplement) |
| Storage | Parquet, queried via DuckDB |
| Tabular modeling | scikit-learn / XGBoost |
| Graph modeling | PyTorch Geometric (for the GNN interaction model) |
| RL | Stable-Baselines3 (PPO/DQN) against the project's own Monte Carlo simulator as environment |
| Simulation | Python + NumPy |
| Backend | FastAPI |
| Frontend | React + TypeScript + Plotly |
| Packaging | Docker + docker-compose; cloud target TBD per `PRD.md` Section 13 |
| Testing | pytest |

---

## 3. Repo Structure

```
overtake/
├── data/
│   └── cache/                 # FastF1 cache
├── notebooks/
├── src/
│   ├── ingestion/              # FastF1/OpenF1 loaders
│   ├── preprocessing/          # cleaning, features, no-leakage guard
│   ├── models/
│   │   ├── tyre.py             # XGBoost + Bayesian state-space
│   │   ├── lap_time.py         # XGBoost/GRU baseline
│   │   └── lap_time_gnn.py     # graph interaction-aware model
│   ├── simulation/              # replay engine, safety-car model, Monte Carlo
│   ├── strategy/
│   │   ├── search.py            # exhaustive search baseline
│   │   ├── game_theory.py       # competitor-aware extension
│   │   └── rl_policy.py         # trained RL strategy engine
│   ├── performance/              # corner/mini-sector driver analysis
│   └── evaluation/               # backtesting across all three engines
├── backend/
├── frontend/
├── tests/
├── configs/                       # race list, hyperparameters
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
| **CornerTelemetry** | race_id, driver, lap, corner_number, min_speed, throttle_pct, brake_point, gear |

All entities key by `race_id` + `lap`/`driver`, so the no-leakage guard filters any of them consistently.

---

## 5. Core Interfaces (Contracts Between Lanes)

```python
# Shared foundation
def as_of_lap(df: pd.DataFrame, lap_col: str, current_lap: int) -> pd.DataFrame:
    """Only rows with lap_col <= current_lap. Every lookup anywhere routes
    through this — including inside the GNN, RL, and game-theory code."""

# Prediction Lane
def predict_tyre_degradation(compound, age, circuit, track_temp) -> float: ...

def predict_lap_time(state: "RaceState", driver: str) -> float:
    """Baseline (XGBoost/GRU) prediction."""

@dataclass
class RaceGraph:
    lap: int
    nodes: dict[str, dict]                    # driver -> {position, tyre, pace, gap}
    edges: list[tuple[str, str, dict]]         # (driver_a, driver_b, {gap, drs_range, wake_effect})

def predict_lap_time_gnn(graph: RaceGraph) -> dict[str, float]:
    """Interaction-aware prediction for all drivers at once."""

def analyze_corner_performance(driver_telemetry, benchmark_telemetry) -> dict:
    """Per-corner time-loss breakdown vs. a benchmark lap."""

# Decision-Making Lane
@dataclass
class RaceState:
    lap: int
    positions: dict[str, int]
    gaps: dict[str, float]
    tyres: dict[str, tuple[str, int]]
    safety_car: bool

def run_monte_carlo(state: RaceState, strategy: dict, n_sims: int = 1000) -> dict: ...

def get_strategy_recommendation_search(state: RaceState) -> dict:
    """Exhaustive search baseline. All three engines below return this
    same shape — {action, tyre, expected_gain, confidence, engine} —
    so the dashboard can render any of them identically."""

def get_strategy_recommendation_gametheory(state: RaceState, rival_state: RaceState) -> dict:
    """Stackelberg-style, models rival's likely response."""

def get_strategy_recommendation_rl(state: RaceState, policy) -> dict:
    """Trained policy, same output shape as the other two engines."""
```

---

## 6. Module Design

### 6.1 Ingestion
Wrappers around FastF1's session API, one function per data type, covering the full ~60–70 race target across 15+ circuits selected deliberately for diversity (high/low degradation, street/permanent circuits, at least several wet races, at least several safety-car-heavy races).

### 6.2 Feature Engineering & No-Leakage Guard
Houses `as_of_lap()` and all derived features. Every feature function takes `current_lap` and filters through this — including the GNN's graph-construction step and the RL environment's observation function, which are easy places to accidentally leak future state if not careful.

### 6.3 Tyre Degradation Model
XGBoost regressor (compound, age, circuit, track temp, recent pace trend), compared against a Bayesian state-space model treating degradation as a latent process behind lap time — report both, and surface SHAP values in the dashboard's Model view.

### 6.4 Lap-Time Model — Baseline
XGBoost/GRU regressor, driver-independent, as the comparison point for the GNN model below. Report MAE/RMSE across circuits, not one.

### 6.5 Lap-Time Model — GNN Interaction-Aware
Each race timestep becomes a `RaceGraph` (Section 5): nodes are cars with their own state, edges encode gap/DRS-range/dirty-air relationships between nearby cars. A graph-attention or spatio-temporal GNN layer produces a pace prediction per driver that accounts for who's around them — not available to the baseline model. Compare directly against Section 6.4 on identical held-out laps; report both, honestly, even if the improvement is modest.

### 6.6 Replay Engine & Safety-Car Model
Owns `RaceState` and `advance_lap()`. Safety-car probability is an empirical lookup by circuit/race-phase built from `RaceControlEvent` data — kept simple by design, since it's a supporting model, not the project's contribution.

### 6.7 Monte Carlo Simulation
For a given state and strategy, N≥1000 forward simulations sampling noise around the pace model in use (baseline or GNN) and drawing safety-car events, aggregating into a finishing-position distribution.

### 6.8 Strategy Engine 1 — Exhaustive Search
Candidate generation (pit-lap window × compound choices) scored via Section 6.7, ranked by expected outcome. The baseline every other engine is measured against.

### 6.9 Strategy Engine 2 — Competitor-Aware Game Theory
Extends the search to jointly model a rival's likely response (Stackelberg-style: this car's best move given the rival's best response to it), rather than optimizing in isolation.

### 6.10 Strategy Engine 3 — Reinforcement Learning
A DQN or PPO policy trained with the Section 6.7 Monte Carlo simulator as its environment (race state → observation, pit/tyre decision → action, finishing performance → reward). Reuses the simulator already built for Section 6.8 rather than needing a separate one.

### 6.11 Corner/Driver-Performance Module
Segments each lap into corners using `CornerTelemetry`, compares a driver's speed/throttle/brake trace against a benchmark lap (session best or the driver's own reference), and quantifies estimated time lost per corner.

### 6.12 Backtesting
Re-runs all three strategy engines at multiple real decision points across the full dataset, comparing each against the real outcome and the best-possible hindsight strategy. This is the evidence behind the three-way comparison in `PRD.md` FR-7/FR-8 — the central experiment of the whole project.

### 6.13 API Layer

| Endpoint | Returns |
|---|---|
| `GET /api/races` | full ingested race library |
| `GET /api/replay/{race_id}/{lap}` | `RaceState` |
| `GET /api/tyre/{race_id}/{driver}/{lap}` | degradation prediction + SHAP values |
| `GET /api/lap-prediction/{race_id}/{driver}/{lap}?model=baseline\|gnn` | pace prediction from either track |
| `GET /api/strategy/{race_id}/{lap}?engine=search\|gametheory\|rl` | recommendation from the selected engine |
| `GET /api/simulation/{race_id}/{lap}` | Monte Carlo distribution |
| `GET /api/corner-analysis/{race_id}/{driver}/{lap}` | per-corner time-loss breakdown |
| `GET /api/backtest/{race_id}` | all-three-engines comparison table |

### 6.14 Frontend / Dashboard
Six views — **Race, Strategy, Simulation, Tyre, Driver/Corner, Model** — each fetching its own endpoint(s). The Strategy and Simulation views additionally expose the three-engine toggle (FR-7). Built to the design brief in Section 10, not a default component-library look.

---

## 7. Testing Strategy

- Unit test on `as_of_lap()` — still the single highest-value test in the repo.
- Model evaluation harness reporting MAE/RMSE for every model (tyre ×2, lap-time ×2) on a fixed multi-circuit held-out split.
- Integration test: full `run_replay()` plus all three strategy engines on one cached race, no exceptions.
- Backtest smoke test: the Section 6.12 comparison runs end-to-end on a small race subset in CI, full dataset run separately (it's expensive).

---

## 8. Deployment Design

Dockerized locally via `docker-compose up`, no live external dependency at demo time (all data pre-cached). Cloud deployment is a delivered milestone (`PRD.md` Section 8, step 10) — host TBD (`PRD.md` Section 13).

---

## 9. What's No Longer a "Future Extension"

The GNN model, RL engine, and game-theoretic strategy were previously documented here as forward-looking sketches, deliberately deferred for the 15-day sprint. They're now full module designs — Sections 6.5, 6.9, and 6.10 — built against the same interfaces that were originally kept stable specifically so this transition wouldn't require a rearchitecture. It doesn't.

---

## 10. Design Brief — "Pit Wall"

The dashboard should read as a purpose-built motorsport tool, not a generic admin panel. Ground every choice in the actual visual vernacular of F1 timing screens and pit-wall software, which already has a distinctive, information-dense language — use it rather than defaulting to generic SaaS-dashboard conventions.

**Color** — a graphite/asphalt base, not pure black, with color used *functionally*, not decoratively:
- Base surfaces: `#15171C` (background), `#1E212A` (panel), `#2A2E38` (dividers)
- Text: `#E8E9ED` (primary), `#8B90A0` (secondary)
- Tyre compounds (the FIA's actual colors — real information, not decoration): Soft `#DA291C`, Medium `#FFD100`, Hard `#F0F0F0`, Intermediate `#43B02A`, Wet `#0067B1`
- Timing deltas (real timing-screen convention, don't invent a fourth meaning): purple `#9B30FF` fastest overall, green `#3CB878` personal best

**Type** — one grotesk/sans for UI chrome; monospace used *only* for tabular numeric data (lap times, gaps, deltas) where digit alignment genuinely helps scanning — never as ambient label styling.

**Layout** — reject the symmetric card grid. Model it on an actual pit wall: a persistent, dense timing-tower strip (position / driver / gap / tyre swatch) anchoring one edge, a large central focus area for the active view, and a strategy panel styled like a radioed-in call — bordered, timestamped, high-contrast — since that recommendation is the one moment the whole product exists to deliver.

**Spend boldness once** — the strategy panel is the one place to be visually loud. Timing tower, charts, and navigation stay quiet and legible around it.

**Explicitly avoid:** cream background + serif + terracotta; near-black + single neon accent as decoration; identical rounded cards with soft drop-shadows; tracked-out ALL-CAPS eyebrow labels; middle-dot-joined meta strings; monospace used as generic label flavor rather than for real tabular numbers.

**Before calling the dashboard done:** write the color/type tokens as actual CSS variables, check them against the "avoid" list above, build, then screenshot and self-critique against both the tokens and the avoid list — don't ship on first pass.
