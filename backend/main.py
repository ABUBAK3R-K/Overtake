"""FastAPI Backend Server for Overtake (Design.md Section 6.9).

Exposes REST APIs for Race Replay, Tyre Degradation, Lap Prediction,
Monte Carlo Simulation, Strategy Optimization, and Backtesting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.evaluation.backtest import (
    ENGINES,
    HISTORICAL_BENCHMARKS,
    backtest_decision_point,
    run_full_backtest,
    run_multi_engine_backtest,
    summarize_backtest,
    summarize_multi_engine_backtest,
)
from src.ingestion.session import load_race_list
from src.ingestion.storage import connect
from src.models.lap_time import predict_lap_time
from src.models.lap_time_gnn import predict_lap_time_gnn
from src.models.tyre import predict_tyre_degradation, shap_values_for
from src.performance.corner import analyze_driver_lap_vs_benchmark
from src.preprocessing.graph import build_race_graph_from_state
from src.simulation.monte_carlo import run_monte_carlo
from src.simulation.replay import ReplaySession, build_state_at_lap, get_race_metadata, run_replay
from src.simulation.safety_car import get_circuit_safety_car_profile, safety_car_probability
from src.simulation.state import RaceState
from src.strategy.optimizer import get_strategy_recommendation
from src.strategy.gametheory import get_strategy_recommendation_gametheory
from src.strategy.rl_env import get_strategy_recommendation_rl

log = logging.getLogger("overtake.backend")

app = FastAPI(
    title="Overtake AI Race Strategy API",
    description="Formula 1 AI-Powered Race Strategy & Simulation Backend",
    version="1.0.0",
)

# Enable CORS for local dashboard development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "overtake-backend"}


@app.get("/api/races")
def get_races() -> list[dict[str, Any]]:
    """Return list of configured/ingested races with metadata."""
    try:
        races = load_race_list()
        con = connect()
        tables = {row[0] for row in con.sql("SHOW TABLES").fetchall()}
        if "races" in tables:
            df = con.sql("SELECT * FROM races").df()
            if not df.empty:
                return df.to_dict(orient="records")
        return races
    except Exception as e:
        log.exception("Error loading races: %s", e)
        return load_race_list()


_REPLAY_SESSIONS: dict[str, ReplaySession] = {}


def get_replay_session(race_id: str) -> ReplaySession:
    """Retrieve or initialize an in-memory ReplaySession for high-speed replay scrubbing."""
    if race_id not in _REPLAY_SESSIONS:
        _REPLAY_SESSIONS[race_id] = ReplaySession(race_id)
    return _REPLAY_SESSIONS[race_id]


@app.get("/api/replay/{race_id}/summary")
def get_replay_summary(race_id: str) -> dict[str, Any]:
    """Return race metadata, circuit, total laps, and available replay status."""
    try:
        session = get_replay_session(race_id)
        return {
            "race_id": race_id,
            "circuit": session.circuit,
            "total_laps": session.total_laps,
            "has_laps": not session.laps_df.empty,
            "total_drivers": int(session.laps_df["driver"].nunique()) if not session.laps_df.empty else 0,
        }
    except Exception as e:
        log.exception("Error loading replay summary for %s: %s", race_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/replay/{race_id}/events/{lap}")
def get_replay_events(race_id: str, lap: int) -> dict[str, Any]:
    """Return chronological race control and pit stop timeline strictly up to lap."""
    try:
        session = get_replay_session(race_id)
        state = session.seek(lap)
        return {
            "race_id": race_id,
            "lap": lap,
            "safety_car": state.safety_car,
            "status": state.status,
            "pit_stops": state.pit_stops_history,
            "race_control": state.race_control_events,
        }
    except Exception as e:
        log.exception("Error loading replay events for %s lap %d: %s", race_id, lap, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/replay/{race_id}/{lap}")
def get_replay_lap(race_id: str, lap: int) -> dict[str, Any]:
    """Return full RaceState at the specified lap."""
    try:
        session = get_replay_session(race_id)
        state = session.seek(lap)
        return state.to_dict()
    except Exception as e:
        log.exception("Error loading replay for %s lap %d: %s", race_id, lap, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tyre/{race_id}/{driver}/{lap}")
def get_tyre_prediction(
    race_id: str,
    driver: str,
    lap: int,
    horizon: int = Query(default=30, ge=1, le=60),
    include_shap: bool = Query(default=True),
) -> dict[str, Any]:
    """Return tyre degradation curves, wear prediction, and optional SHAP feature
    contributions for the driver's current stint. (Design.md §6.3 / FR-2)
    """
    try:
        state = build_state_at_lap(race_id, lap)
        current_comp, current_age = state.tyres.get(driver, ("MEDIUM", 1))
        circuit = state.circuit
        track_temp = state.weather.get("track_temp", 30.0)

        # Generate wear curve across ages 0 to 40 for Soft, Medium, Hard
        curves = {}
        for comp in ["SOFT", "MEDIUM", "HARD"]:
            curve = []
            for age in range(0, 41):
                try:
                    wear = predict_tyre_degradation(comp, age, circuit, track_temp)
                except Exception:
                    wear = age * (0.07 if comp == "SOFT" else (0.05 if comp == "MEDIUM" else 0.035))
                curve.append({"age": age, "pace_loss_seconds": round(wear, 3)})
            curves[comp] = curve

        current_loss = 0.0
        try:
            current_loss = predict_tyre_degradation(current_comp, current_age, circuit, track_temp)
        except Exception:
            current_loss = current_age * 0.05

        # SHAP contributions for current tyre state (FR-2 loose end)
        shap_contributions: dict[str, float] = {}
        if include_shap:
            try:
                shap_contributions = shap_values_for(
                    current_comp, current_age, circuit, track_temp
                )
            except Exception as shap_err:
                log.debug("SHAP unavailable: %s", shap_err)

        result: dict[str, Any] = {
            "race_id": race_id,
            "driver": driver,
            "current_lap": lap,
            "circuit": circuit,
            "compound": current_comp,
            "current_age": current_age,
            "track_temp": track_temp,
            "current_pace_loss_seconds": round(current_loss, 3),
            "degradation_curves": curves,
        }
        if shap_contributions:
            result["shap"] = shap_contributions
        return result
    except Exception as e:
        log.exception("Error generating tyre degradation for %s %s: %s", race_id, driver, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lap-prediction/{race_id}/{driver}/{lap}")
def get_lap_prediction(
    race_id: str,
    driver: str,
    lap: int,
    model: str = Query(default="baseline", pattern="^(baseline|gnn)$"),
) -> dict[str, Any]:
    """Return predicted next lap time for driver given current race state."""
    try:
        state = build_state_at_lap(race_id, lap)
        interaction_context = None

        if model == "gnn":
            graph = build_race_graph_from_state(state)
            gnn_preds = predict_lap_time_gnn(graph)
            pred_time = gnn_preds.get(driver, predict_lap_time(state, driver))

            # Extract driver-specific traffic & aerodynamic context
            d_edges = [e for e in graph.edges if e[1] == driver]
            wake_intensity = max([e[2].get("wake_effect", 0.0) for e in d_edges] + [0.0])
            in_drs = any(e[2].get("drs_range", False) and e[2].get("direction") == "ahead_to_follower" for e in d_edges)
            gap_ahead = min([e[2].get("gap", 999.0) for e in d_edges if e[2].get("direction") == "ahead_to_follower"] + [999.0])

            interaction_context = {
                "gap_ahead": round(gap_ahead, 3) if gap_ahead < 900.0 else None,
                "in_drs_range": in_drs,
                "wake_intensity": round(wake_intensity, 3),
                "in_pack": len(d_edges) >= 2,
            }
        else:
            pred_time = predict_lap_time(state, driver)

        base_time = state.last_lap_times.get(driver, pred_time)
        res = {
            "race_id": race_id,
            "driver": driver,
            "current_lap": lap,
            "model": model,
            "predicted_next_lap_time": round(pred_time, 3),
            "previous_lap_time": round(base_time, 3) if base_time else None,
            "predicted_delta": round(pred_time - base_time, 3) if base_time else 0.0,
            "safety_car": state.safety_car,
        }
        if interaction_context is not None:
            res["interaction_context"] = interaction_context
        return res
    except Exception as e:
        log.exception("Error predicting lap time for %s %s: %s", race_id, driver, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategy/{race_id}/{lap}")
def get_strategy(
    race_id: str,
    lap: int,
    driver: Optional[str] = Query(default=None),
    sims: int = Query(default=200, ge=20, le=1000),
    engine: str = Query(default="search", pattern="^(search|gametheory|rl)$"),
) -> dict[str, Any]:
    """Return strategy recommendation for a driver.

    ``?engine=search`` (default) — exhaustive candidate search via Monte Carlo.
    ``?engine=gametheory`` — Stackelberg competitor-aware game theory.
    ``?engine=rl`` — RL policy roll-out (greedy MC if no policy trained yet).
    All three return the same response shape so the dashboard toggle works
    without any frontend changes (Design.md §6.13).
    """
    try:
        state = build_state_at_lap(race_id, lap)
        target_driver = driver or (
            "VER" if "VER" in state.positions
            else next(iter(state.positions.keys()), "VER")
        )

        if engine == "gametheory":
            rec = get_strategy_recommendation_gametheory(
                state, rival_state=state,
                target_driver=target_driver,
                n_sims=sims,
            )
        elif engine == "rl":
            rec = get_strategy_recommendation_rl(
                state,
                policy=None,  # No trained policy at endpoint level; greedy MC
                target_driver=target_driver,
                n_sims=sims,
            )
        else:
            rec = get_strategy_recommendation(
                state, target_driver=target_driver, n_sims=sims
            )
            rec["engine"] = "search"

        return rec
    except Exception as e:
        log.exception("Error generating strategy for %s lap %d engine=%s: %s",
                      race_id, lap, engine, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulation/{race_id}/{lap}")
def get_simulation(
    race_id: str,
    lap: int,
    driver: Optional[str] = Query(default=None),
    pit_lap: Optional[int] = Query(default=None),
    compound: Optional[str] = Query(default=None),
    sims: int = Query(default=300, ge=50, le=1000),
) -> dict[str, Any]:
    """Return Monte Carlo finishing probability distribution for custom or baseline strategy."""
    try:
        state = build_state_at_lap(race_id, lap)
        target_driver = driver or ("VER" if "VER" in state.positions else next(iter(state.positions.keys()), "VER"))
        
        strategy = None
        if pit_lap and compound:
            strategy = {"pit_laps": [pit_lap], "compounds": [compound.upper()]}

        res = run_monte_carlo(state, strategy=strategy, target_driver=target_driver, n_sims=sims)
        return res
    except Exception as e:
        log.exception("Error running simulation for %s lap %d: %s", race_id, lap, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/{race_id}")
def get_race_backtest(
    race_id: str,
    engine: str = Query(default="search", pattern="^(search|gametheory|rl)$"),
) -> list[dict[str, Any]]:
    """Return backtest benchmarks for a specific race."""
    matching = [b for b in HISTORICAL_BENCHMARKS if b["race_id"] == race_id]
    if not matching:
        matching = [{"race_id": race_id, "driver": "VER", "decision_lap": 20, "actual_action": "BOX LAP 20", "actual_compound": "HARD", "actual_finish": 1, "circuit": "Circuit", "note": "Standard pit window"}]

    results = []
    for bm in matching:
        res = backtest_decision_point(race_id=race_id, driver=bm["driver"], decision_lap=bm["decision_lap"], n_sims=80, engine=engine)
        results.append(res)
    return results


@app.get("/api/backtest")
def get_full_backtest(
    engine: str = Query(default="search", pattern="^(search|gametheory|rl)$"),
) -> dict[str, Any]:
    """Return full historical calendar backtesting summary for one engine."""
    results = run_full_backtest(n_sims=60, engine=engine)
    return summarize_backtest(results)


@app.get("/api/backtest/compare/all")
def get_backtest_engine_comparison() -> dict[str, Any]:
    """Run the full backtest suite for all three FR-7 engines and return the
    three-way comparison (PRD FR-8 / Section 9's headline result). Slower
    than the single-engine endpoints above since it runs the whole benchmark
    set three times — intended for the dashboard's backtest summary view,
    not per-lap polling.
    """
    multi = run_multi_engine_backtest(n_sims=60, engines=ENGINES)
    return summarize_multi_engine_backtest(multi)


@app.get("/api/corner-analysis/{race_id}/{driver}/{lap}")
def get_corner_analysis(
    race_id: str,
    driver: str,
    lap: int,
    benchmark_driver: Optional[str] = Query(default=None),
    benchmark_lap: Optional[int] = Query(default=None),
) -> dict[str, Any]:
    """Per-corner time-loss breakdown for one driver's lap vs. a benchmark
    lap (PRD FR-9, Design.md Section 6.11). Defaults the benchmark to the
    race's overall fastest lap if not given.

    Loads a full FastF1 session on demand (cached in-process after the first
    call for a given race_id) rather than reading the ingested `telemetry`
    table, which only stores per-lap summaries — see src/performance/corner.py.
    The first request for a given race can take tens of seconds even from
    local cache; later requests for the same race are fast.
    """
    try:
        return analyze_driver_lap_vs_benchmark(
            race_id, driver, lap,
            benchmark_driver=benchmark_driver, benchmark_lap=benchmark_lap,
        )
    except Exception as e:
        log.exception("Error running corner analysis for %s %s lap %d", race_id, driver, lap)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/safety-car/{circuit}")
def get_safety_car_risk(
    circuit: str,
    lap_fraction: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    event_type: str = Query(default="ANY", pattern="^(ANY|SAFETY_CAR|VIRTUAL_SAFETY_CAR)$"),
) -> dict[str, Any]:
    """Return calibrated Safety Car risk profile and deployment probabilities for a circuit."""
    profile = get_circuit_safety_car_profile(circuit)
    if lap_fraction is not None:
        prob = safety_car_probability(circuit, lap_fraction, event_type=event_type)
        profile["queried_lap_fraction"] = lap_fraction
        profile["queried_event_type"] = event_type
        profile["current_lap_probability"] = prob
    return profile


# Serve static frontend if dist exists
frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
