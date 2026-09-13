# PRD — Overtake
**AI-Powered F1 Race Strategy & Performance Intelligence**

---

## 1. Overview

Overtake is a decision-support system that replays historical Formula 1 races lap by lap and, at any point in the race, predicts tyre degradation and upcoming pace, simulates likely race outcomes under uncertainty, and recommends a pit-stop/tyre strategy — then lets you compare that recommendation against what actually happened.

It is not a live system. It never needs a live F1 data feed to be complete or useful — historical replay is the core product, not a placeholder for one.

**One-line definition:** An AI race engineer that learns tyre and lap-pace behavior from real telemetry, simulates race futures under uncertainty, and recommends pit-stop strategy — with every call checkable against what the real team actually did.

---

## 2. Problem / Motivation

Race strategy is a decision made under heavy uncertainty (tyre wear, safety cars, weather, rival strategy) with real consequences (seconds or positions lost). Teams solve this with proprietary tooling and human judgment. There's no reason a system built entirely on public data can't approximate the same reasoning, be transparent about its confidence, and — crucially for this project — be checked against history to see whether its calls were actually any good.

---

## 3. Goals

- Predict tyre degradation and short-horizon lap pace from real telemetry.
- Simulate many possible race futures per candidate strategy (Monte Carlo), not just one deterministic guess.
- Recommend a strategy with a stated expected gain and confidence.
- Backtest every recommendation against real history and against the best-possible hindsight strategy, and report the results honestly.
- Present all of the above in a dashboard usable by someone who isn't the developer.

## 4. Non-Goals (This Iteration)

Explicitly out of scope for the current 15–20 day build (see Section 8 for the full priority breakdown):
- Live/real-time F1 data ingestion
- Reinforcement-learning strategy engine
- The GNN car-interaction model (tracked as a named future extension in `Design.md`, not built now)
- Full corner-by-corner telemetry/driver-performance analysis
- Production cloud deployment (a local, one-command demo is the target)
- Competitor-aware game-theoretic strategy (exhaustive search is the target for this iteration)

---

## 5. Users & Use Cases

Primary "users" for this iteration are the two of you (as builders and demo presenters) and anyone you show the demo to (interviewers, mentors). The feature set is grounded in a race-analyst persona:

- **As a race analyst**, I want to see the predicted degradation curve for the current tyre stint, so I can judge how much longer it's safe to stay out.
- **As a race analyst**, I want a ranked pit/tyre recommendation with an expected time gain and a confidence figure, so I can decide whether to act now.
- **As a race analyst**, I want to see the probability spread of finishing outcomes for a given strategy, not just one number, so I understand the risk involved.
- **As a viewer/evaluator**, I want to see the recommendation compared against what actually happened in the race, so I can judge whether the system is trustworthy.

---

## 6. Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | System ingests lap times, sector times, tyre compound/age, pit stops, race-control events, and weather for a fixed set of 6–8 historical races. |
| FR-2 | System predicts tyre degradation (pace loss vs. stint age) per compound/circuit/temperature combination. |
| FR-3 | System predicts next-lap pace given current race state. |
| FR-4 | System replays a race lap by lap, exposing only data available at or before the current lap at every step. |
| FR-5 | System models safety-car deployment probabilistically and reflects its bunching effect on the field during simulation. |
| FR-6 | System runs Monte Carlo simulation (≥500 samples) per candidate strategy and reports a distribution over finishing outcomes. |
| FR-7 | System generates candidate pit/tyre strategies and ranks them by expected outcome. |
| FR-8 | System backtests its recommendations against real historical outcomes and the best-possible hindsight strategy, across at least 3 races. |
| FR-9 | Dashboard lets a user select a race, scrub to any lap, and view the current tyre/degradation state, the strategy recommendation, and the simulation distribution. |
| FR-10 | Dashboard shows model accuracy (MAE/RMSE, feature importance) for transparency. |

## 7. Non-Functional Requirements

- **No data leakage:** at no point may a model or the replay engine access data from a lap later than the one currently being evaluated. This is a hard constraint, not a best-effort guideline — see `Design.md` for the enforcement mechanism.
- **Reproducibility:** the race list is fixed and versioned; ingested data is cached locally so results don't change between runs due to an external API being unavailable or updated.
- **Local-first:** the full system must run via a single `docker-compose up` with no live external dependency required at demo time.
- **Reasonable simulation latency:** a single strategy's Monte Carlo evaluation (≥500 sims) should complete in a few seconds on a laptop, not minutes — informs the sampling count and model complexity choices in `Design.md`.

---

## 8. Scope & Priority

**P0 — required for the system to be considered working at all:**
Data pipeline (6–8 races) · tyre model · lap-time model · replay engine with safety-car modeling · Monte Carlo simulation · exhaustive-search strategy optimizer · Race/Strategy/Simulation dashboard views · backtesting on ≥3 races

**P1 — strengthens the result, build if P0 is solid:**
Tyre view + Model view dashboards · Docker packaging for one-command local run · backtesting across all 6–8 races · a basic wet/dry weather flag as a simulation input

**P2 — explicitly deferred, don't start until P0 and P1 are done:**
Driver/corner telemetry view · competitor-aware game-theoretic strategy · GNN interaction layer · reinforcement-learning strategy engine · cloud deployment

## 9. Success Metrics / Acceptance Criteria

- Tyre and lap-time models report MAE/RMSE on a held-out set (not cherry-picked laps).
- The backtest (FR-8) produces a clear, honest answer to "how often did Overtake's call beat, match, or lose to the real strategist" — a negative or mixed result is an acceptable, reportable outcome; a suppressed or cherry-picked one is not.
- The demo walkthrough (pick race → scrub to lap → see recommendation → see comparison to history) runs on at least 2 different races without manual intervention.

---

## 10. Data Sources

- **FastF1** — primary source, telemetry/laps/timing from the 2018 season onward.
- **OpenF1** — optional supplement for dense in-session data on 2023+ races if finer granularity is wanted for the replay feel.
- **jolpica-f1** — season-level results/standings context (the community successor to the now-retired Ergast API).

## 11. Risks & Assumptions

- The 6–8 race selection must deliberately include at least one wet race and one safety-car-heavy race, or the safety-car and weather logic will never be exercised or validated.
- Published research on this exact lap-time prediction task reports accuracy in the 50% range for pit-window classification — treat that as the realistic ceiling, not a sign of a broken model.
- A two-person, 15–20 day timeline is tight; the priority tiers in Section 8 exist specifically so a schedule slip has a predefined, non-negotiated answer for what gets cut.

## 12. Timeline

See `overtake-15-day-sprint-plan.md` for the day-by-day schedule and lane assignments. This document defines *what* is being built; that one defines *when*.

## 13. Open Questions

- Final list of 6–8 races (owner: both, due Day 1)
- Whether P1 cloud deployment is attempted or the demo stays local-only (decide once P0 is confirmed on schedule, around Day 13–14)
