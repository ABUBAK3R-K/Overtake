"""Backtesting and Historical Strategy Evaluation Engine (Design.md Section 6.8).

Evaluates Overtake's strategy recommendations against actual historical pit calls
and hindsight outcomes across the 2023 race calendar.
"""

from __future__ import annotations

import logging
from typing import Any
import pandas as pd

from src.ingestion.session import load_race_list
from src.ingestion.storage import connect
from src.preprocessing.leakage import as_of_lap
from src.simulation.replay import build_state_at_lap
from src.strategy.optimizer import get_strategy_recommendation
from src.strategy.gametheory import get_strategy_recommendation_gametheory
from src.strategy.rl_env import get_strategy_recommendation_rl

log = logging.getLogger("overtake.evaluation.backtest")

ENGINES = ("search", "gametheory", "rl")


def _run_engine(engine: str, state, driver: str, n_sims: int) -> dict[str, Any]:
    """Dispatch to one of the three FR-7 strategy engines with a uniform call
    shape, so backtesting doesn't special-case any of them (PRD FR-8)."""
    if engine == "gametheory":
        rec = get_strategy_recommendation_gametheory(
            state, rival_state=state, target_driver=driver, n_sims=n_sims
        )
    elif engine == "rl":
        rec = get_strategy_recommendation_rl(
            state, target_driver=driver, n_sims=n_sims
        )
    elif engine == "search":
        rec = get_strategy_recommendation(state, target_driver=driver, n_sims=n_sims)
        rec.setdefault("engine", "search")
    else:
        raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
    return rec


def _actual_outcome(race_id: str, driver: str, decision_lap: int, con) -> dict[str, Any]:
    """Derive what really happened from ingested data (pit_stops, laps), for
    any race/driver/decision point — not just the 6 curated in
    HISTORICAL_BENCHMARKS. Used so backtesting can scale to the full dataset
    (PRD FR-8) without a circular fallback (see the docstring note in
    ``backtest_decision_point`` about why the old fallback was unusable)."""
    pit = con.sql(
        "SELECT lap, compound_after FROM pit_stops "
        "WHERE race_id = ? AND driver = ? AND lap > ? ORDER BY lap LIMIT 1",
        params=[race_id, driver, decision_lap],
    ).fetchone()
    actual_action = f"BOX LAP {pit[0]}" if pit else "STAY OUT (no further stop)"
    actual_compound = pit[1] if pit else None

    finish = con.sql(
        "SELECT position FROM laps WHERE race_id = ? AND driver = ? AND position IS NOT NULL "
        "ORDER BY lap_number DESC LIMIT 1",
        params=[race_id, driver],
    ).fetchone()
    actual_finish = int(finish[0]) if finish and finish[0] is not None else None

    return {"actual_action": actual_action, "actual_compound": actual_compound, "actual_finish": actual_finish}


def decision_points_for_race(
    race_id: str, con=None, max_points: int = 2,
) -> list[tuple[str, int]]:
    """Pick (driver, decision_lap) pairs from a race's *real* pit stops — one
    lap before each of the ``max_points`` earliest real stops, by distinct
    driver. This is how full-dataset backtesting (PRD FR-8) chooses decision
    points without hand-curation: it evaluates the AI at the exact moment a
    real strategist actually had to decide.

    Races with no ingested pit stops (or none early enough to leave a
    meaningful lap window) return an empty list; the caller skips them.
    """
    if con is None:
        con = connect()
    # The EXISTS clause drops "stops" that were really retirements into the
    # pits (no laps after them) — the earliest "stop" in a race is often one,
    # and there's no strategy decision to score there.
    rows = con.sql(
        "SELECT p.driver, MIN(p.lap) AS lap FROM pit_stops p "
        "WHERE p.race_id = ? AND p.lap > 3 AND EXISTS ("
        "  SELECT 1 FROM laps l WHERE l.race_id = p.race_id AND l.driver = p.driver "
        "  AND l.lap_number > p.lap + 2) "
        "GROUP BY p.driver ORDER BY lap LIMIT ?",
        params=[race_id, max_points],
    ).fetchall()
    return [(driver, int(lap) - 1) for driver, lap in rows]


# Key historical decision points in 2023 races for benchmarking
HISTORICAL_BENCHMARKS = [
    {
        "race_id": "2023_bahrain",
        "circuit": "Sakhir",
        "driver": "ALO",
        "decision_lap": 14,
        "actual_action": "BOX LAP 15",
        "actual_compound": "HARD",
        "actual_finish": 3,
        "note": "Aston Martin undercut on Mercedes; secured Alonso podium",
    },
    {
        "race_id": "2023_monaco",
        "circuit": "Monaco",
        "driver": "ALO",
        "decision_lap": 54,
        "actual_action": "BOX LAP 54 (SLICKS IN RAIN)",
        "actual_compound": "MEDIUM",
        "actual_finish": 2,
        "note": "Infamous decision to pit for Medium slicks right as rain intensified",
    },
    {
        "race_id": "2023_britain",
        "circuit": "Silverstone",
        "driver": "HAM",
        "decision_lap": 32,
        "actual_action": "BOX LAP 33 (SAFETY CAR)",
        "actual_compound": "SOFT",
        "actual_finish": 3,
        "note": "Magnussen SC enabled Hamilton cheap pit stop to pass Piastri for P3",
    },
    {
        "race_id": "2023_singapore",
        "circuit": "Marina Bay",
        "driver": "RUS",
        "decision_lap": 43,
        "actual_action": "BOX LAP 44 (VSC 2-STOP)",
        "actual_compound": "MEDIUM",
        "actual_finish": 16,  # Crashed on last lap while hunting Sainz/Norris
        "note": "Aggressive Mercedes 2-stop under VSC gained massive pace advantage",
    },
    {
        "race_id": "2023_netherlands",
        "circuit": "Zandvoort",
        "driver": "PER",
        "decision_lap": 1,
        "actual_action": "BOX LAP 1 (INTERS)",
        "actual_compound": "INTERMEDIATE",
        "actual_finish": 4,
        "note": "Immediate Lap 1 pit stop for rain jumped Perez from P7 to P1",
    },
    {
        "race_id": "2023_spain",
        "circuit": "Barcelona",
        "driver": "VER",
        "decision_lap": 26,
        "actual_action": "BOX LAP 27",
        "actual_compound": "HARD",
        "actual_finish": 1,
        "note": "Dominant Red Bull 2-stop control",
    },
]


def backtest_decision_point(
    race_id: str,
    driver: str,
    decision_lap: int,
    con=None,
    n_sims: int = 150,
    engine: str = "search",
) -> dict[str, Any]:
    """Run one strategy engine at a specific historical decision point and compare
    against both the real outcome and the engine's own best-possible hindsight.

    ``engine``: "search" (Engine 1, default), "gametheory" (Engine 2), or "rl"
    (Engine 3) — see ``_run_engine``. All three PRD FR-7 engines route through
    this same evaluation and scoring path, which is what FR-8 requires.
    """
    if con is None:
        con = connect()

    # Look up benchmark note if available
    bm = next(
        (b for b in HISTORICAL_BENCHMARKS if b["race_id"] == race_id and b["driver"] == driver and abs(b["decision_lap"] - decision_lap) <= 2),
        None,
    )
    circuit = bm["circuit"] if bm else "Circuit"

    try:
        state = build_state_at_lap(race_id, decision_lap, con=con)
        rec = _run_engine(engine, state, driver, n_sims)
    except Exception as e:
        log.exception("Backtest failed for %s driver %s lap %d engine=%s", race_id, driver, decision_lap, engine)
        return {
            "race_id": race_id,
            "circuit": circuit,
            "driver": driver,
            "decision_lap": decision_lap,
            "engine": engine,
            "error": str(e),
            "verdict": "EVALUATION_FAILED",
        }

    ai_action = rec["action"]
    ai_tyre = rec["tyre"]
    expected_pos = rec["expected_position"]
    confidence = rec["confidence"]
    reasoning = rec["reasoning"]

    if bm:
        actual_action = bm["actual_action"]
        actual_compound = bm["actual_compound"]
        actual_finish = bm["actual_finish"]
    else:
        # No curated benchmark for this race/driver/lap — look up what
        # actually happened from ingested data rather than deriving "actual"
        # from the AI's own prediction (that was circular: pos_delta would
        # be ~0 by construction and every unbenchmarked point would silently
        # score as "MATCHED REAL STRATEGY").
        real = _actual_outcome(race_id, driver, decision_lap, con)
        actual_action = real["actual_action"]
        actual_compound = real["actual_compound"] or ai_tyre
        actual_finish = real["actual_finish"] if real["actual_finish"] is not None else int(round(expected_pos))

    # Outcome evaluation: positive pos_delta means the AI's simulated expected
    # finish beats what actually happened historically (lower position number
    # = better finish). No special-casing by circuit/driver — every point is
    # scored the same way, including cases where the real strategist won.
    pos_delta = actual_finish - expected_pos
    if abs(pos_delta) < 0.6:
        verdict = "MATCHED REAL STRATEGY"
    elif pos_delta >= 0.6:
        verdict = f"AI ADVANTAGE (+{pos_delta:.1f} POS)"
    else:
        verdict = f"REAL STRATEGY BETTER ({pos_delta:.1f} POS)"

    # Regret vs. best-possible hindsight (PRD FR-8): the lowest expected
    # position among every candidate this same engine actually evaluated at
    # this decision point. 0 means the engine picked its own best option;
    # >0 means a candidate it considered (or generated but discarded) would
    # have scored better — e.g. Engine 2/3's adversarial or exploratory
    # framing steering them away from the simulator's own top candidate.
    candidates = rec.get("candidates") or []
    hindsight_best_pos = (
        min(c["expected_position"] for c in candidates if "expected_position" in c)
        if candidates else expected_pos
    )
    regret_vs_hindsight = round(expected_pos - hindsight_best_pos, 2)

    return {
        "race_id": race_id,
        "circuit": circuit,
        "driver": driver,
        "decision_lap": decision_lap,
        "engine": engine,
        "actual_action": actual_action,
        "actual_compound": actual_compound,
        "actual_finish": actual_finish,
        "ai_action": ai_action,
        "ai_tyre": ai_tyre,
        "ai_expected_position": round(expected_pos, 1),
        "confidence": confidence,
        "verdict": verdict,
        "position_delta": round(pos_delta, 1),
        "regret_vs_hindsight": regret_vs_hindsight,
        "reasoning": reasoning,
    }


def run_full_backtest(con=None, n_sims: int = 100, engine: str = "search") -> list[dict[str, Any]]:
    """Run backtesting suite across all benchmark historical decision points."""
    if con is None:
        con = connect()

    results = []
    for bm in HISTORICAL_BENCHMARKS:
        res = backtest_decision_point(
            race_id=bm["race_id"],
            driver=bm["driver"],
            decision_lap=bm["decision_lap"],
            con=con,
            n_sims=n_sims,
            engine=engine,
        )
        results.append(res)
    return results


def run_multi_engine_backtest(
    con=None, n_sims: int = 100, engines: tuple[str, ...] = ENGINES,
) -> dict[str, list[dict[str, Any]]]:
    """Run the full backtest suite once per strategy engine (PRD FR-8: the
    three-way comparison is the central experiment of the whole project)."""
    if con is None:
        con = connect()
    return {engine: run_full_backtest(con=con, n_sims=n_sims, engine=engine) for engine in engines}


def run_dataset_backtest(
    race_ids: list[str] | None = None,
    con=None,
    n_sims: int = 60,
    engine: str = "search",
    decision_points_per_race: int = 2,
    on_result=None,
) -> list[dict[str, Any]]:
    """Run one engine across real decision points drawn from every given race
    (PRD FR-8: "across the full ingested dataset", not just the 6 curated
    HISTORICAL_BENCHMARKS points). Decision points come from
    ``decision_points_for_race`` — a real pit stop per driver — and the
    comparison outcome comes from ``_actual_outcome``, so this needs no
    hand-curation and scales to all 112 races.

    ``race_ids`` defaults to every race in configs/races.toml. Races with no
    ingested data, or no early pit stops to build decision points from, are
    skipped (not counted as failures — there was nothing to evaluate).

    ``on_result``, if given, is called with each result dict as soon as it's
    produced — this is how the CLI script below checkpoints progress for a
    112-race x 3-engine run that can take a long time and may be interrupted.
    """
    if con is None:
        con = connect()
    if race_ids is None:
        race_ids = [r["race_id"] for r in load_race_list()]

    results: list[dict[str, Any]] = []
    for race_id in race_ids:
        try:
            points = decision_points_for_race(race_id, con=con, max_points=decision_points_per_race)
        except Exception:
            log.warning("Skipping %s: could not build decision points (not ingested?)", race_id, exc_info=True)
            continue
        for driver, decision_lap in points:
            res = backtest_decision_point(
                race_id=race_id, driver=driver, decision_lap=decision_lap,
                con=con, n_sims=n_sims, engine=engine,
            )
            results.append(res)
            if on_result is not None:
                on_result(res)
    return results


def summarize_backtest(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate aggregate metrics across backtested races.

    Failed evaluations (verdict "EVALUATION_FAILED") are reported, not hidden
    or excluded from the count — PRD Section 9 requires an honest result, and
    a suppressed failure would misrepresent the success rate.
    """
    if not results:
        return {"total_evaluations": 0, "success_rate_pct": 0.0, "average_position_gain": 0.0}

    failed = sum(1 for r in results if r["verdict"] == "EVALUATION_FAILED")
    scored = [r for r in results if r["verdict"] != "EVALUATION_FAILED"]

    beats = sum(1 for r in scored if "ADVANTAGE" in r["verdict"])
    matches = sum(1 for r in scored if "MATCHED" in r["verdict"])
    worse = sum(1 for r in scored if "REAL STRATEGY BETTER" in r["verdict"])
    avg_gain = float(pd.Series([r["position_delta"] for r in scored]).mean()) if scored else None
    regrets = [r["regret_vs_hindsight"] for r in scored if "regret_vs_hindsight" in r]
    avg_regret = float(pd.Series(regrets).mean()) if regrets else None

    return {
        "total_evaluations": len(results),
        "failed_evaluations": failed,
        "strategies_improved": beats,
        "strategies_matched": matches,
        "strategies_worse": worse,
        "success_rate_pct": round(((beats + matches) / len(scored)) * 100, 1) if scored else 0.0,
        "average_position_gain": round(avg_gain, 2) if avg_gain is not None else None,
        "average_regret_vs_hindsight": round(avg_regret, 2) if avg_regret is not None else None,
        "details": results,
    }


def summarize_multi_engine_backtest(
    multi_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Three-way engine comparison (PRD FR-8 / Section 9): summarize each
    engine's backtest independently and report which one actually wins.

    An "advanced" engine (game theory / RL) losing to the exhaustive-search
    baseline is reported as-is — PRD Section 9 explicitly calls that a
    legitimate, reportable outcome, not something to massage.
    """
    per_engine = {engine: summarize_backtest(results) for engine, results in multi_results.items()}

    ranked = sorted(
        (e for e, s in per_engine.items() if s.get("success_rate_pct") is not None),
        key=lambda e: (
            -(per_engine[e]["success_rate_pct"] or 0.0),
            per_engine[e]["average_regret_vs_hindsight"] or 0.0,
        ),
    )
    best_engine = ranked[0] if ranked else None

    return {
        "engines": per_engine,
        "best_engine_by_success_rate": best_engine,
        "ranking": ranked,
    }


# ─── Fair three-engine scoring (PRD FR-8 / Section 9) ────────────────────────
#
# `regret_vs_hindsight` above compares each engine only against candidates it
# evaluated itself. Search and game theory both return the argmin of their
# own candidate list, so their regret is 0 by construction, and their
# self-reported expected positions carry a winner's-curse bias (the minimum of
# ~12 noisy estimates is optimistic). That can't support a three-way
# comparison. `score_decision_point_fair` instead re-simulates every engine's
# pick, the real team's strategy, and a shared reference set on one fresh seed
# that no engine selected with, and measures regret against the best of that
# shared set — the same yardstick for every engine.

FAIR_SEED_OFFSET = 7_919  # keep evaluation seeds away from run_monte_carlo's default 42
DRY_AND_WET = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}


def _strategy_key(strategy: dict[str, Any]) -> tuple:
    return (
        tuple(int(x) for x in strategy.get("pit_laps") or []),
        tuple(str(c).upper() for c in strategy.get("compounds") or []),
    )


def real_strategy(race_id: str, driver: str, decision_lap: int, con) -> dict[str, Any]:
    """What the team actually did after ``decision_lap``, as a strategy dict
    the simulator can replay (all remaining real stops, in order).

    ~4% of pit_stops rows have compound_after UNKNOWN/NULL (often a
    retirement into the pits). Those fall back to the tyres table's compound
    on the following laps; if that's missing too, ``unscorable`` is set and
    the real strategy is left out of the comparison rather than simulated on
    a made-up compound."""
    rows = con.sql(
        "SELECT lap, compound_after FROM pit_stops WHERE race_id = ? AND driver = ? AND lap > ? ORDER BY lap",
        params=[race_id, driver, decision_lap],
    ).fetchall()
    laps, compounds, unscorable = [], [], None
    for lap, comp in rows:
        comp = str(comp).upper() if comp is not None else "UNKNOWN"
        if comp not in DRY_AND_WET:
            nxt = con.sql(
                "SELECT compound FROM tyres WHERE race_id = ? AND driver = ? AND lap_number > ? "
                "AND compound IS NOT NULL ORDER BY lap_number LIMIT 1",
                params=[race_id, driver, int(lap)],
            ).fetchone()
            comp = str(nxt[0]).upper() if nxt else "UNKNOWN"
        if comp not in DRY_AND_WET:
            unscorable = f"unknown compound after lap {int(lap)} stop (likely a retirement)"
        laps.append(int(lap))
        compounds.append(comp)
    out = {"name": "REAL" if laps else "REAL_STAY_OUT", "pit_laps": laps, "compounds": compounds}
    if unscorable:
        out["unscorable"] = unscorable
    return out


def score_decision_point_fair(
    race_id: str,
    driver: str,
    decision_lap: int,
    con=None,
    engines: tuple[str, ...] = ENGINES,
    engine_sims: int = 60,
    eval_sims: int = 200,
) -> dict[str, Any]:
    """Score all engines at one real decision point on a common yardstick.

    Reference set = Engine 1's candidate generator + the RL action set + each
    engine's pick + the real strategy, deduplicated, each run for
    ``eval_sims`` on one shared seed. Regret for X = E[finish | X] minus the
    best in the reference set. The real strategy is scored the same way, so
    "engine vs. what the team did" is compared inside the same simulator.
    ``actual_finish`` is kept only to check how well the simulator itself
    matches reality.
    """
    import time
    import zlib

    from src.models.lap_time import make_lap_time_predictor
    from src.simulation.monte_carlo import run_monte_carlo
    from src.strategy.optimizer import generate_candidate_strategies
    from src.strategy.rl_env import _action_meta

    if con is None:
        con = connect()
    base = {"race_id": race_id, "driver": driver, "decision_lap": decision_lap}

    state = build_state_at_lap(race_id, decision_lap, con=con)
    if driver not in state.positions or state.total_laps - state.lap < 3:
        return {**base, "skipped": "driver not running or race nearly over"}
    base["circuit"] = state.circuit

    picks: dict[str, dict[str, Any]] = {}
    for engine in engines:
        t0 = time.time()
        try:
            rec = _run_engine(engine, state, driver, engine_sims)
            picks[engine] = {
                "strategy": rec["strategy"],
                "self_expected_position": float(rec["expected_position"]),
                "confidence": rec.get("confidence"),
                "seconds": round(time.time() - t0, 1),
            }
        except Exception as exc:  # noqa: BLE001 — one engine failing mustn't sink the point
            log.warning("engine %s failed at %s %s L%d", engine, race_id, driver, decision_lap, exc_info=True)
            picks[engine] = {"error": repr(exc), "seconds": round(time.time() - t0, 1)}

    real = real_strategy(race_id, driver, decision_lap, con)
    reference: dict[tuple, dict[str, Any]] = {}
    pool = (
        generate_candidate_strategies(state, driver)
        + [_action_meta(a, state.lap) for a in range(4)]
        + [p["strategy"] for p in picks.values() if "strategy" in p]
        + ([] if "unscorable" in real else [real])
    )
    for s in pool:
        reference.setdefault(_strategy_key(s), {k: s.get(k) for k in ("name", "pit_laps", "compounds")})

    try:
        predictor = make_lap_time_predictor(race_id, decision_lap)
    except Exception:
        predictor = None
    seed = FAIR_SEED_OFFSET + zlib.crc32(f"{race_id}|{driver}|{decision_lap}".encode()) % 1_000_000
    expected = {
        key: float(run_monte_carlo(state, s, target_driver=driver, n_sims=eval_sims,
                                   lap_time_predictor=predictor, seed=seed)["expected_position"])
        for key, s in reference.items()
    }
    best_key = min(expected, key=expected.get)
    best_pos = expected[best_key]

    def _score(strategy: dict[str, Any]) -> dict[str, Any]:
        pos = expected[_strategy_key(strategy)]
        return {"expected_position": round(pos, 3), "regret": round(pos - best_pos, 3),
                "picked_best": _strategy_key(strategy) == best_key}

    engines_out = {}
    for engine, p in picks.items():
        engines_out[engine] = {**p, **_score(p["strategy"])} if "strategy" in p else p

    return {
        **base,
        "eval_sims": eval_sims,
        "n_reference": len(reference),
        "hindsight_best": {**reference[best_key], "expected_position": round(best_pos, 3)},
        "engines": engines_out,
        "real": {"strategy": real, **({} if "unscorable" in real else _score(real))},
        "actual_finish": _actual_outcome(race_id, driver, decision_lap, con)["actual_finish"],
    }


def _bootstrap_ci(values, n_boot: int = 2000, seed: int = 0) -> list[float] | None:
    import numpy as np

    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def summarize_fair_backtest(
    rows: list[dict[str, Any]],
    held_out: set[str] | None = None,
    race_tags: dict[str, dict] | None = None,
    tie_margin: float = 0.25,
) -> dict[str, Any]:
    """Aggregate ``score_decision_point_fair`` rows. Regret numbers come with a
    95% bootstrap CI; "vs real" is a paired comparison inside the simulator,
    counting |difference| <= ``tie_margin`` positions as a tie."""
    import numpy as np

    scored = [r for r in rows if "engines" in r]
    contenders = sorted({e for r in scored for e in r["engines"]}) + ["real"]

    def _get(r, c):
        return r["real"] if c == "real" else r["engines"].get(c, {})

    def _block(subset: list[dict]) -> dict[str, Any]:
        out: dict[str, Any] = {"n_points": len(subset)}
        for c in contenders:
            ok = [r for r in subset if "regret" in _get(r, c)]
            reg = [_get(r, c)["regret"] for r in ok]
            entry: dict[str, Any] = {
                "n": len(ok),
                "failed": len(subset) - len(ok),
                "mean_regret": round(float(np.mean(reg)), 3) if reg else None,
                "mean_regret_ci95": _bootstrap_ci(reg),
                "median_regret": round(float(np.median(reg)), 3) if reg else None,
                "picked_best_pct": round(100 * float(np.mean([_get(r, c)["picked_best"] for r in ok])), 1) if ok else None,
                "mean_expected_position": round(float(np.mean([_get(r, c)["expected_position"] for r in ok])), 2) if ok else None,
            }
            paired = [r for r in ok if "expected_position" in r["real"]]
            if c != "real" and paired:
                diff = [_get(r, c)["expected_position"] - r["real"]["expected_position"] for r in paired]
                entry["vs_real"] = {
                    "n": len(paired),
                    "mean_diff": round(float(np.mean(diff)), 3),  # negative = engine finishes ahead of real strategy
                    "mean_diff_ci95": _bootstrap_ci(diff),
                    "better_pct": round(100 * float(np.mean([d < -tie_margin for d in diff])), 1),
                    "tie_pct": round(100 * float(np.mean([abs(d) <= tie_margin for d in diff])), 1),
                    "worse_pct": round(100 * float(np.mean([d > tie_margin for d in diff])), 1),
                }
                secs = [_get(r, c)["seconds"] for r in ok if "seconds" in _get(r, c)]
                entry["mean_seconds"] = round(float(np.mean(secs)), 1) if secs else None
                optimism = [_get(r, c)["self_expected_position"] - _get(r, c)["expected_position"] for r in ok]
                entry["self_report_optimism"] = round(float(np.mean(optimism)), 3)  # negative = engine over-promised
            out[c] = entry
        return out

    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "n_skipped": sum(1 for r in rows if "skipped" in r),
        "n_races": len({r["race_id"] for r in scored}),
        "all": _block(scored),
    }
    ranking = [c for c in contenders if summary["all"][c]["mean_regret"] is not None]
    summary["ranking_by_mean_regret"] = sorted(ranking, key=lambda c: summary["all"][c]["mean_regret"])

    if held_out:
        summary["held_out_races"] = _block([r for r in scored if r["race_id"] in held_out])
        summary["training_races"] = _block([r for r in scored if r["race_id"] not in held_out])
    if race_tags:
        for tag in ("wet", "sc_heavy"):
            summary[f"tag_{tag}"] = _block([r for r in scored if race_tags.get(r["race_id"], {}).get(tag)])

    calib = [(r["real"]["expected_position"], r["actual_finish"]) for r in scored
             if r.get("actual_finish") is not None and "expected_position" in r["real"]]
    if calib:
        sim, act = np.asarray(calib, dtype=float).T
        summary["simulator_calibration"] = {
            "n": len(calib),
            "mae_real_strategy_vs_actual_finish": round(float(np.mean(np.abs(sim - act))), 2),
            "correlation": round(float(np.corrcoef(sim, act)[0, 1]), 3) if len(calib) > 2 else None,
            "mean_bias": round(float(np.mean(sim - act)), 2),  # negative = simulator too optimistic
        }
    return summary
