# Overtake — 15–20 Day Sprint Plan
*A reference document for building the project fast, as a two-person team.*

---

## 1. What Is Overtake?

Overtake is an AI-powered Formula 1 race strategy and performance system — think of it as an "AI race engineer." You feed it a historical F1 race, and at any lap, it can tell you:

- How much longer the current tyre will realistically last
- What the recommended pit-stop and tyre choice is, right now
- How likely different finishing outcomes are, given that recommendation
- How that recommendation compares to what the real team actually did

The core trick that makes this buildable without live F1 data access: instead of connecting to a real race, **you replay a real historical race lap-by-lap**, only ever showing the system data that would have existed up to that point (no peeking at the future — this rule matters everywhere and comes up again below). At the end, you can compare Overtake's calls against history and say, concretely, "here's where it would have made a better or worse call than the real strategist."

The system is a pipeline:

```
F1 Data → Cleaning/Features → [Tyre Model + Lap-Time Model] → Race Replay Engine
→ Monte Carlo Simulation → Strategy Optimizer → Dashboard
```

Two things predict pace (tyre model, lap-time model). One thing simulates the race forward under uncertainty (Monte Carlo replay engine). One thing decides what to do with those simulations (strategy optimizer). A dashboard shows all of it.

---

## 2. Goals for This Sprint (Definition of Done)

By the end of the sprint, someone should be able to:

1. Open a local dashboard, pick one of 6–8 real historical races
2. Scrub to any lap in that race
3. See the current tyre state and a predicted degradation curve
4. See a recommended pit/tyre call, with an expected time gain and a confidence figure
5. See a Monte Carlo-generated probability spread for finishing position
6. See how that recommendation compares to what actually happened in the real race

That's the whole demo. Everything below is built in service of that one walkthrough.

**Being honest about the compression:** the original project plan is a multi-month research-grade system (production dashboard, reinforcement learning, full corner-by-corner telemetry analysis, live deployment). Doing all of that *properly* in 15 days with two people isn't realistic, and pretending otherwise would just mean shipping something broken. So this plan implements a **lean, working version of every core stage**, and explicitly pushes the research-grade extensions (RL, the GNN interaction layer, full corner analysis, cloud deployment) into a clearly marked stretch tier — see Section 6 for exactly what's in and what's deferred, and why that's the right call for a sprint like this.

---

## 3. How the Work Splits

Two lanes, running in parallel, syncing at fixed checkpoints:

- **Prediction lane (you):** data pipeline, tyre degradation model, lap-time model
- **Decision-making lane (Partner):** race replay engine, Monte Carlo simulation, strategy optimizer, backtesting

Neither lane is "the hard one" — the decision-making lane is just as substantial (it's simulation-under-uncertainty and search, not plumbing). The dashboard at the end is split by *view*, not by frontend/backend, so both of you touch the full stack.

---

## 4. Partner's Track — Decision-Making Engine, Day by Day

This is the detailed build order. Each day has a concrete deliverable — something that either runs, or visibly doesn't, so there's never ambiguity about whether a day succeeded.

### Day 1 — Environment + Data Contract (with you)
Set up the shared repo and environment (see the earlier kickoff doc for the exact commands). The one thing to nail down today specifically: **agree on the shared data schema** — what columns every race's Parquet file will have (driver, lap, sector times, tyre compound/age, gap-to-leader, position, pit flag, track/air temp, safety-car flag). Everything downstream depends on this being settled before either of you writes ingestion code.

### Day 2 — Ingest Race-Control & Weather Data
Your ingestion responsibility: pit-stop records, race-control messages (flags, safety cars, red flags), and weather data, for the 6–8 races you've picked together. (You handles laps/telemetry/tyre data — split this way because your Day 5 safety-car logic needs race-control data first.)

```python
# src/ingestion/race_control.py
def load_race_control(session) -> pd.DataFrame:
    """Returns lap, message_type (SC/VSC/RedFlag/etc), lap_number"""
    ...

def load_pit_stops(session) -> pd.DataFrame:
    """Returns driver, lap, pit_duration, compound_before, compound_after"""
    ...
```

**Deliverable:** three Parquet files per race (race control, pit stops, weather), queryable via DuckDB.

### Day 3 — EDA + No-Leakage Guard (with you)
Pair on this one. Write and agree on the shared no-leakage function both lanes will use everywhere:

```python
def as_of_lap(df, lap_col, current_lap):
    """Only rows with lap_col <= current_lap. Every feature/state
    lookup in the whole project routes through something like this."""
    return df[df[lap_col] <= current_lap]
```

This is the single most important piece of shared code in the entire project — a leak here silently invalidates every later result, so it's worth both of you actually reading each other's usage of it, not just trusting it.

### Day 4 — Race State Object + Bare Replay Loop
Build the skeleton that everything else plugs into. At this stage it just replays *actual* history — no prediction, no uncertainty yet. That's intentional: get the loop mechanically correct first.

```python
@dataclass
class RaceState:
    lap: int
    positions: dict          # driver -> position
    gaps: dict                # driver -> gap_to_leader_seconds
    tyres: dict                # driver -> (compound, age)
    safety_car: bool

def advance_lap(state: RaceState, race_data) -> RaceState:
    """Pulls the ACTUAL next-lap data for now — replaced with
    predictions in Day 6."""
    ...

def run_replay(race_data) -> list[RaceState]:
    state = init_state(race_data)
    history = [state]
    for lap in range(2, race_data.total_laps + 1):
        state = advance_lap(state, race_data)
        history.append(state)
    return history
```

**Deliverable:** running `run_replay()` on any ingested race prints a sensible lap-by-lap state trace.

### Day 5 — Safety Car ("Ghost Car") Logic
Using the race-control data from Day 2, build an empirical safety-car probability model — no need for anything fancy, historical frequency by circuit and race-phase is a legitimate approach used in published race-simulation research:

```python
def safety_car_probability(circuit: str, lap_fraction: float) -> float:
    """Looks up historical SC deployment rate for this circuit at
    this point in the race, from your ingested race-control data."""
    ...

def apply_safety_car_bunching(state: RaceState) -> RaceState:
    """When a safety car triggers, compress gaps toward zero —
    this is the 'ghost car' effect."""
    ...
```

**Deliverable:** a replay run that, when it hits a lap where a real safety car occurred, visibly bunches the field.

### Day 6 — Wire In Predictions (integration point with your models)
This is the day the simulator stops just replaying history and starts predicting the future. **Depends on you having a first working `predict_lap_time()` and `predict_tyre_degradation()` by end of Day 5** — flag early if that's slipping, this is the sprint's tightest dependency.

```python
def advance_lap(state, predict_lap_time_fn, predict_degradation_fn):
    for driver in state.positions:
        pace = predict_lap_time_fn(state, driver)
        degradation = predict_degradation_fn(state, driver)
        state = apply_pace(state, driver, pace, degradation)
    return state
```

### Day 7 — Monte Carlo Sampling
Turn the single deterministic replay into many sampled futures:

```python
def run_monte_carlo(state, strategy, predict_lap_time_fn, n_sims=1000):
    outcomes = []
    for _ in range(n_sims):
        sim_state = deepcopy(state)
        for lap in range(state.lap + 1, total_laps + 1):
            noisy_pace = predict_lap_time_fn(sim_state) + sample_noise()
            sc = random() < safety_car_probability(circuit, lap / total_laps)
            sim_state = advance_lap(sim_state, noisy_pace, sc, strategy)
        outcomes.append(sim_state.finishing_position)
    return summarize(outcomes)  # {finish_prob_by_position, expected_position, ...}
```

Start with `n_sims=500–1000` — enough for a stable distribution without slow iteration while you're debugging.

**Deliverable:** for one race, one lap, one strategy — a probability distribution over finishing positions.

### Day 8 — Strategy Optimizer (Exhaustive Search)
Generate candidate strategies (pit lap ± a window, 2–3 compound choices), run Monte Carlo on each, rank by expected outcome:

```python
def get_strategy_recommendation(state) -> dict:
    candidates = generate_candidates(state)          # e.g. pit windows × compounds
    results = {c: run_monte_carlo(state, c) for c in candidates}
    best = max(results, key=lambda c: results[c].expected_value)
    return {
        "action": best.action, "tyre": best.compound,
        "expected_gain": best.expected_value - baseline,
        "confidence": best.top3_probability,
    }
```

**This is the handoff function the API layer calls — keep this exact signature stable**, since the dashboard's Strategy and Simulation views are built directly on top of it.

### Days 9–10 — Wrap Your Engine in FastAPI
Expose your own logic as endpoints (you'll do the same for the models):

```
GET /api/replay/{race_id}/{lap}         -> RaceState
GET /api/strategy/{race_id}/{lap}       -> recommendation
GET /api/simulation/{race_id}/{lap}     -> Monte Carlo distribution
```

### Days 11–13 — Your Dashboard Views
Build the **Strategy view**, **Simulation view**, and **Race view** end-to-end (API + React component each) — these map directly to what you've already built, so you own their full stack. (You takes Tyre view, Model view; Driver/corner view is a stretch item — see Section 6.)

### Days 12–13 (parallel to above, or right after) — Backtesting
The scientifically interesting part: for each ingested race, run your strategy engine at several real decision points and compare its call against what actually happened *and* against the best possible hindsight strategy.

```python
def backtest_race(race_data):
    results = []
    for lap in decision_points(race_data):
        rec = get_strategy_recommendation(state_at(race_data, lap))
        actual = actual_strategy_at(race_data, lap)
        hindsight = best_hindsight_strategy(race_data, lap)
        results.append(compare(rec, actual, hindsight))
    return results
```

**Deliverable:** a table/plot showing, across your 6–8 races, how often and by how much Overtake's calls beat, matched, or lost to the real strategist — this is your single strongest "is this actually good" evidence for the writeup.

### Days 14–15 — Integration, Bug Fixing, Demo Prep
Full end-to-end run on at least 2–3 races. Fix whatever breaks when your engine and their models actually talk to each other for a full race, not just a single lap.

### Days 16–20 (buffer / stretch)
See Section 6 — pick up P1/P2 items here if you're on schedule, or absorb slippage from Days 1–15 if not.

---

## 5. Your Track — Prediction Models, Day by Day

(Included so the dependency timing above makes sense — Partner's Day 6 needs these.)

- **Day 2:** Ingest laps, sector times, telemetry, tyre compound/age for the 6–8 races.
- **Day 3:** Pair with Partner on the no-leakage guard + EDA.
- **Day 4–5:** Tyre degradation model (XGBoost: compound, age, circuit, track temp → pace loss). Ship `predict_tyre_degradation(compound, age, circuit, temp) -> float` by end of Day 5 — this is the hard dependency for their Day 6.
- **Day 6–7:** Lap-time/next-lap model (XGBoost baseline; note published benchmarks on this exact task land around 50% pit-window classification accuracy, so calibrate your expectations — this is a genuinely hard prediction problem, not a modeling mistake if it's not near-perfect). Ship `predict_lap_time(state, driver) -> float`.
- **Day 8:** Report MAE/RMSE for both models — this is your version of Partner's Day 12–13 backtest, and belongs in the same writeup section.
- **Days 9–10:** Wrap your models in FastAPI (`/api/tyre/{race_id}/{driver}/{lap}`, `/api/lap-prediction/...`).
- **Days 11–13:** Tyre view + Model view (feature importance, MAE display) dashboard, end-to-end.
- **Days 14–15:** Integration + bug fixing (joint with Partner).
- **Days 16–20:** If ahead of schedule, this is where the GNN interaction-layer stretch goal lives (Section 6) — it's your specialization tie-in, worth protecting time for if the core path finishes early.

---

## 6. Priority Tiers — What to Cut First If You're Behind

Ordered so that if the sprint slips, you cut from the bottom up and the Day-15 demo in Section 2 still works.

**P0 — the demo doesn't exist without these:**
Data pipeline (6–8 races) · tyre model · lap-time model · replay engine with ghost-car safety-car logic · exhaustive-search strategy optimizer + Monte Carlo · Race/Strategy/Simulation dashboard views · backtesting on at least 3 races · README + demo script

**P1 — meaningfully strengthens it, cut second:**
Tyre view + Model view · Docker packaging for local one-command run · backtesting across the full 6–8 races · a basic wet/dry weather flag as a Monte Carlo input

**P2 — genuine stretch, cut first, don't start until P0 and P1 are solid:**
Driver/corner mini-sector telemetry view · the game-theoretic competitor-aware strategy extension · GNN interaction-layer model · reinforcement-learning strategy engine · cloud deployment (a local Docker Compose demo is a completely legitimate sprint deliverable — don't burn sprint days on hosting)

---

## 7. Sync Points — Don't Skip These

| Day | What's exchanged |
|---|---|
| 1 | Shared data schema agreed |
| 3 | No-leakage guard written together |
| 5–6 | You hand off `predict_tyre_degradation()` and `predict_lap_time()` |
| 8 | Partner's `get_strategy_recommendation()` signature is locked — don't change it after this without telling you, the dashboard depends on it |
| 14–15 | Full joint integration pass |

---

## 8. Final Demo Script (Day 15+)

1. Open dashboard, pick a race with a real mid-race pit decision
2. Scrub to the lap just before that decision
3. Show the predicted degradation curve, the recommended call, the confidence, the Monte Carlo distribution
4. Reveal what actually happened, and how it compares
5. Show the backtest summary across all races — the headline number for the whole project

That five-step walkthrough is also, almost verbatim, your interview answer to "tell me about a project."
