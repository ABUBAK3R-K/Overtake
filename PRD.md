# PRD — Overtake (Full Build)
**AI-Powered F1 Race Strategy & Performance Intelligence**

*This supersedes the scope defined for the original 15–20 day sprint plan. See Section 12 for exactly what changed and why — kept explicit rather than silently overwritten.*

---

## 1. Overview

Overtake is a decision-support system that replays historical Formula 1 races lap by lap and, at any point in the race, predicts tyre degradation and upcoming pace, simulates likely race outcomes under uncertainty, and recommends a pit-stop/tyre strategy — then lets you compare that recommendation against what actually happened.

It is not a live system. Historical replay is the core product, not a placeholder for one — see Section 4 for why live data specifically stays out of scope.

**One-line definition:** An AI race engineer that learns tyre and lap-pace behavior from real telemetry, simulates race futures under uncertainty, and recommends pit-stop strategy through multiple competing reasoning approaches — with every call checkable against what the real team actually did.

---

## 2. Problem / Motivation

Race strategy is a decision made under heavy uncertainty (tyre wear, safety cars, weather, rival strategy) with real consequences. Teams solve this with proprietary tooling and human judgment. This project builds the same kind of reasoning entirely from public data, makes every model's confidence and reasoning visible, and checks every recommendation against history rather than just asserting it works.

---

## 3. Goals

- Predict tyre degradation and short-horizon lap pace from real telemetry, trained on a genuinely broad multi-season dataset, not a handful of hand-picked races.
- Model lap pace as an *interaction-aware* problem — a car's pace depends on the cars around it, not just its own state in isolation.
- Simulate many possible race futures per candidate strategy (Monte Carlo), not just one deterministic guess.
- Build and honestly compare three different strategic reasoning approaches — exhaustive search, competitor-aware game theory, and reinforcement learning — rather than picking one and calling it done.
- Backtest every recommendation, from every engine, against real history and the best-possible hindsight strategy, across the full dataset.
- Analyze driver performance at the corner/mini-sector level from telemetry, not just lap aggregates.
- Present all of the above in a dashboard that reads as a purpose-built motorsport tool — see Design.md Section 10 for the specific design brief this is held to.

## 4. Non-Goals

Only one true non-goal remains, and it's a data-access constraint rather than a scope cut: **live/real-time F1 data ingestion**, which requires a paid data tier not available for this project. Everything else in the original plan — the GNN interaction model, the RL strategy engine, the competitor-aware game theory, full corner analysis, the complete dashboard, and deployment — is in scope. (An RL policy trained here for offline replay is fully in scope; only *live* re-planning against a live feed is excluded, since that's the piece that needs data this project can't access.)

---

## 5. Users & Use Cases

- **As a race analyst**, I want to see the predicted degradation curve for the current tyre stint, so I can judge how much longer it's safe to stay out.
- **As a race analyst**, I want a ranked pit/tyre recommendation with an expected time gain and a confidence figure, so I can decide whether to act now.
- **As a race analyst**, I want to see the probability spread of finishing outcomes for a given strategy, not just one number, so I understand the risk involved.
- **As a viewer**, I want to compare the exhaustive-search, game-theoretic, and RL strategy recommendations side by side at the same moment, so I can see how differently-reasoned approaches actually differ.
- **As a viewer/evaluator**, I want to see every recommendation compared against what actually happened, so I can judge whether the system is trustworthy — including when it's wrong.

---

## 6. Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | System ingests laps, sector times, telemetry, tyre compound/age, pit stops, race-control events, and weather across a broad dataset: target at least 3 full seasons (~60–70 races) spanning at least 15 distinct circuits, deliberately including wet races and safety-car-heavy races in enough number for those sub-models to actually generalize. |
| FR-2 | System predicts tyre degradation (XGBoost baseline, compared against a Bayesian state-space model for interpretability/uncertainty) with SHAP-based explainability surfaced in the UI. |
| FR-3 | System predicts next-lap pace via two compared model tracks: (a) an XGBoost/GRU baseline treating each driver independently, and (b) a graph neural network treating each race timestep as a graph of interacting cars (nodes = cars, edges = gap/DRS-range/wake relationships), producing interaction-aware pace predictions. |
| FR-4 | System replays a race lap by lap, exposing only data available at or before the current lap at every step (no-leakage is a hard constraint — see Design.md Section 5). |
| FR-5 | System models safety-car deployment probabilistically (empirical frequency by circuit/race-phase) and reflects its bunching effect during simulation. |
| FR-6 | System runs Monte Carlo simulation (≥1000 samples) per candidate strategy and reports a distribution over finishing outcomes. |
| FR-7 | System builds and exposes **three** strategy engines against the same Monte Carlo simulator: exhaustive search (baseline), a competitor-aware game-theoretic extension (Stackelberg-style, modeling rival response), and an RL-trained policy (DQN/PPO) — the dashboard lets a user toggle between them on the same race/lap. |
| FR-8 | System backtests all three engines across the full ingested dataset, reporting regret versus the best-possible hindsight strategy per engine, so the comparison in FR-7 is backed by evidence, not assertion. |
| FR-9 | System performs corner/mini-sector telemetry analysis, comparing a driver's speed/throttle/brake trace against a benchmark lap and quantifying time lost per corner. |
| FR-10 | Dashboard exposes all 6 views (Race, Strategy, Simulation, Tyre, Driver/Corner, Model) at production quality, with race selection covering the full ingested library, built to the visual identity specified in Design.md Section 10 — not a generic templated admin panel. |
| FR-11 | System is Dockerized for one-command local run, with cloud deployment as an actual delivered milestone rather than a stretch note. |

## 7. Non-Functional Requirements

- **No data leakage:** hard constraint, enforced everywhere through the shared guard in `Design.md` Section 5 — non-negotiable regardless of scope size.
- **Data breadth:** training data spans at least 3 full seasons and 15+ circuits, with enough wet and safety-car examples for those specific sub-models to be validated, not just demonstrated once.
- **Visual design quality:** the dashboard is evaluated against the design brief in `Design.md` Section 10, not against "does it work" — a functionally complete but generically-styled dashboard does not meet this requirement.
- **Explainability:** every prediction or recommendation shown in the UI carries a visible "why" (feature importance, confidence, or expected-value comparison) — never a bare unexplained number.
- **Reproducibility:** the full race list is versioned and cached locally; results don't change between runs due to an external API changing or being unavailable.

---

## 8. Full Scope & Build Order

Nothing below is deferred — this is a sequence, not a priority cut list, because later phases genuinely depend on earlier ones being solid.

1. Multi-season data pipeline (~60–70 races, 15+ circuits)
2. Tyre degradation model + SHAP explainability
3. Lap-time model — XGBoost/GRU baseline, then the GNN interaction-aware version
4. Replay engine + probabilistic safety-car ("ghost car") model
5. Monte Carlo simulation engine
6. Strategy engines, in order: exhaustive search → competitor-aware game theory → RL policy
7. Full backtesting across the entire dataset, all three engines compared
8. Corner/mini-sector driver-performance module
9. Dashboard — all 6 views, full design pass against Design.md Section 10
10. Dockerization and deployment
11. Documentation and writeup

## 9. Success Metrics / Acceptance Criteria

- Tyre and lap-time models (both tracks, including the GNN version) report MAE/RMSE on a held-out set spanning multiple circuits, not a single race.
- The backtest produces an honest three-way comparison of search vs. game-theory vs. RL across the full dataset — a result where the "advanced" engines don't win is a legitimate, reportable outcome.
- The dashboard passes a self-critique against the generic-design tells listed in `Design.md` Section 10 before being considered finished.
- The demo walkthrough runs across multiple races and multiple strategy-engine toggles without manual intervention.

---

## 10. Data Sources

- **FastF1** — primary source, telemetry/laps/timing from the 2018 season onward; the source for the ~60–70 race training set.
- **OpenF1** — optional supplement for dense in-session granularity on 2023+ races, if the replay feel benefits from it.
- **jolpica-f1** — season-level results/standings context (community successor to the retired Ergast API).

## 11. Risks & Assumptions

- Training and iteration time grows substantially with a 60–70 race dataset versus a handful — budget accordingly, especially for the GNN and RL models, which are the most compute- and iteration-hungry pieces.
- The three-engine comparison is a genuine open question, not a foregone conclusion — published research on RL strategy engines shows real gains over fixed strategies, but that doesn't guarantee it beats a well-built game-theoretic or search baseline on this specific dataset. Report whichever wins, honestly.
- This is now a multi-week undertaking, not a 15-day sprint — see Section 12.

## 12. What Changed From the Sprint Plan

The original 15–20 day plan explicitly deferred the GNN model, RL engine, game-theoretic extension, full corner analysis, and cloud deployment as P2 stretch items, in order to guarantee a working demo inside a hard timeline. This revision removes that constraint at the user's explicit direction: everything is now in scope, and the tradeoff is a longer build (weeks rather than days) and materially heavier data and compute requirements. That tradeoff is stated once, here, rather than repeated throughout the rest of this document.

## 13. Open Questions

- Final season/circuit list for the ~60–70 race dataset (aim for deliberate diversity, not just "most recent")
- Cloud hosting target for FR-11's deployment milestone
