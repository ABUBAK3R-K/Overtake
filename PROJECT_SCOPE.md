# Overtake: Project Scope

**Overview**
Overtake is an AI-powered Formula 1 race strategy and performance system. It acts as an "AI race engineer" by replaying historical races lap-by-lap, predicting tyre degradation, simulating race outcomes, and recommending pit-stop strategies based on public data.

## Core Objectives
- **Predictive Modeling:** Forecast tyre degradation and lap pace using real telemetry across a broad dataset (3+ seasons, 60-70 races, 15+ circuits).
- **Interaction-Aware Pace:** Model lap times based on the state of surrounding cars, not just the driver in isolation.
- **Simulation Under Uncertainty:** Run 1000+ Monte Carlo simulations per candidate strategy to report probabilistic finishing outcomes.
- **Strategy Engines:** Compare three distinct reasoning models against each other:
  1. Exhaustive Search (Baseline)
  2. Competitor-Aware Game Theory
  3. Reinforcement Learning (RL) Policy
- **Backtesting & Verification:** Compare every AI recommendation against what the real F1 team actually did.
- **Telemetry Analysis:** Provide driver performance analysis at the corner and mini-sector levels.

## Non-Goals (Out of Scope)
- **Live/Real-Time Data Ingestion:** The system does not connect to live F1 feeds. It strictly relies on historical data replays.

## Key Deliverables
1. **Data Pipeline:** Ingests laps, telemetry, weather, and race-control events (via FastF1, OpenF1, jolpica-f1).
2. **Predictive Models:** Tyre degradation (XGBoost/Bayesian) and Lap-time pace (XGBoost/GRU vs. Graph Neural Network).
3. **Simulation Engine:** Lap-by-lap race replay with probabilistic safety-car ("ghost car") modeling.
4. **Interactive Dashboard:** 6 specialized views (Race, Strategy, Simulation, Tyre, Driver/Corner, Model) displaying explainable predictions.
5. **Deployment:** Fully Dockerized for local execution and cloud deployment.

## Critical Constraints
- **Strict No Data Leakage:** The system must only access data available *at or before* the current simulated lap.
- **Explainability:** All predictions and recommendations must provide visible reasoning (e.g., feature importance, confidence scores).
