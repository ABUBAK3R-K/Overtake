"""Run the FR-8 three-engine backtest across the full ingested race dataset
(Design.md Section 6.12: "the central experiment of the whole project").

At every real decision point (one lap before a driver's real pit stop, via
src.evaluation.backtest.decision_points_for_race), each engine picks a
strategy; then every pick, the real team's strategy, and a shared reference
set are re-simulated on one fresh common seed (score_decision_point_fair).
Regret = expected positions lost vs. the best strategy in that shared set —
the same yardstick for all three engines, unlike the per-engine
`regret_vs_hindsight` used by the fast /api/backtest* endpoints, which is 0
by construction for search and game theory.

Resumable and parallel (one race per worker): rows are checkpointed to
data/models/backtest/fair_checkpoint.jsonl, and races already finished are
skipped on restart. Keep --workers low: every worker loads its own models,
and 7 workers ran this machine out of memory.

Usage:
    python scripts/run_full_backtest.py                  # all races, all 3 engines, 3 workers
    python scripts/run_full_backtest.py --max-races 4    # a quick look
    python scripts/run_full_backtest.py --report-only    # rebuild the report from the checkpoint
    python scripts/run_full_backtest.py --fresh          # ignore any checkpoint
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.backtest import (  # noqa: E402
    ENGINES,
    decision_points_for_race,
    score_decision_point_fair,
    summarize_fair_backtest,
)
from src.ingestion.session import load_race_list  # noqa: E402
from src.ingestion.storage import connect  # noqa: E402
from src.strategy.rl_policy import held_out_races  # noqa: E402

OUT_DIR = ROOT / "data" / "models" / "backtest"
CHECKPOINT_PATH = OUT_DIR / "fair_checkpoint.jsonl"
RACE_TAGS_PATH = ROOT / "configs" / "race_tags.json"


def _load_checkpoint() -> list[dict]:
    if not CHECKPOINT_PATH.exists():
        return []
    return [json.loads(l) for l in CHECKPOINT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def _score_race(race_id: str, points_per_race: int, engines: tuple[str, ...],
                engine_sims: int, eval_sims: int) -> list[dict]:
    """Worker: every decision point in one race. Failures are recorded, not raised."""
    logging.disable(logging.WARNING)
    con = connect()
    rows = []
    try:
        points = decision_points_for_race(race_id, con=con, max_points=points_per_race)
    except Exception as exc:  # noqa: BLE001 — race not ingested
        points = []
        rows.append({"race_id": race_id, "skipped": f"no decision points: {exc!r}"})
    for driver, lap in points:
        try:
            rows.append(score_decision_point_fair(race_id, driver, lap, con=con, engines=engines,
                                                  engine_sims=engine_sims, eval_sims=eval_sims))
        except Exception as exc:  # noqa: BLE001
            rows.append({"race_id": race_id, "driver": driver, "decision_lap": lap, "error": repr(exc)})
    rows.append({"race_id": race_id, "race_done": True})
    return rows


def _fmt(x, suffix=""):
    return "–" if x is None else f"{x}{suffix}"


def _table(block: dict, contenders: list[str]) -> list[str]:
    lines = [
        "| Strategy | n | Mean regret (95% CI) | Median | Picked best | Mean E[finish] | vs real: better / tie / worse | Mean Δ vs real (95% CI) | Self-report optimism | s/decision |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in contenders:
        s = block.get(c)
        if not s or not s["n"]:
            continue
        ci = s["mean_regret_ci95"] or ["–", "–"]  # None when n < 2
        vr = s.get("vs_real")
        lines.append(
            f"| {c} | {s['n']} | {s['mean_regret']} ({ci[0]}–{ci[1]}) | {s['median_regret']} | "
            f"{s['picked_best_pct']}% | {s['mean_expected_position']} | "
            + (f"{vr['better_pct']}% / {vr['tie_pct']}% / {vr['worse_pct']}% | "
               f"{vr['mean_diff']} ({vr['mean_diff_ci95'][0]}–{vr['mean_diff_ci95'][1]}) | "
               f"{s['self_report_optimism']} | {_fmt(s.get('mean_seconds'))} |"
               if vr and vr["mean_diff_ci95"] else "– | – | – | – |")
        )
    return lines


def write_report(rows: list[dict], engines: tuple[str, ...], args) -> dict:
    tags = json.loads(RACE_TAGS_PATH.read_text(encoding="utf-8")) if RACE_TAGS_PATH.exists() else None
    summary = summarize_fair_backtest(rows, held_out=held_out_races(), race_tags=tags)
    errors = [r for r in rows if "error" in r]
    summary["n_errors"] = len(errors)
    (OUT_DIR / "dataset_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    contenders = list(engines) + ["real"]
    lines = [
        "# Overtake — Full-Dataset Backtest (FR-8)",
        "",
        f"{summary['all']['n_points']} decision points across {summary['n_races']} races "
        f"(one lap before a real pit stop; up to {args.decision_points_per_race} per race) · "
        f"engine sims {args.engine_sims} · common evaluation sims {args.eval_sims} · "
        f"{summary['n_skipped']} skipped · {len(errors)} errors",
        "",
        "Regret = expected finishing positions lost vs. the best strategy in a shared reference set, "
        "all re-simulated on one fresh seed no engine selected with. `real` = what the team actually did, "
        "scored inside the same simulator. Negative Δ vs real = engine finishes ahead. "
        "Self-report optimism = engine's own expected position minus the fresh re-score (negative = over-promised).",
        "",
        "## All races",
        "",
        *_table(summary["all"], contenders),
        "",
        f"**Ranking by mean regret:** {' < '.join(summary['ranking_by_mean_regret'])}",
    ]
    for key, title in (("held_out_races", "Held-out races (never seen by the RL policy or the frozen tyre split's training)"),
                       ("training_races", "Races in the RL policy's training set"),
                       ("tag_wet", "Wet races"), ("tag_sc_heavy", "Safety-car-heavy races")):
        if key in summary and summary[key]["n_points"]:
            lines += ["", f"## {title}", "", *_table(summary[key], contenders)]
    if "simulator_calibration" in summary:
        c = summary["simulator_calibration"]
        lines += ["", "## Simulator vs. reality",
                  "",
                  f"Real strategy re-simulated vs. the real classified finish ({c['n']} points): "
                  f"MAE {c['mae_real_strategy_vs_actual_finish']} positions, correlation {c['correlation']}, "
                  f"mean bias {c['mean_bias']} (negative = simulator too optimistic). "
                  "All regret numbers above are only as good as this."]
    (OUT_DIR / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-races", type=int, default=None)
    parser.add_argument("--decision-points-per-race", type=int, default=2)
    parser.add_argument("--engine-sims", type=int, default=60, help="sims each engine uses to choose")
    parser.add_argument("--eval-sims", type=int, default=200, help="sims per strategy in the common re-score")
    parser.add_argument("--engines", nargs="+", default=list(ENGINES), choices=list(ENGINES))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the report uses Δ; the Windows console default is cp1252
    engines = tuple(args.engines)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    rows = _load_checkpoint()

    if not args.report_only:
        done = {r["race_id"] for r in rows if r.get("race_done")}
        race_ids = [r["race_id"] for r in load_race_list()]
        if args.max_races:
            race_ids = race_ids[: args.max_races]
        todo = [rid for rid in race_ids if rid not in done]
        print(f"{len(done)} races checkpointed; scoring {len(todo)} on {args.workers} workers", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool, CHECKPOINT_PATH.open("a", encoding="utf-8") as f:
            futs = {pool.submit(_score_race, rid, args.decision_points_per_race, engines,
                                args.engine_sims, args.eval_sims): rid for rid in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                rid = futs[fut]
                try:
                    new = fut.result()
                except Exception as exc:  # noqa: BLE001 — worker died; leave race undone so a rerun retries it
                    print(f"[{n}/{len(todo)}] {rid} FAILED: {exc!r}", flush=True)
                    continue
                for r in new:
                    f.write(json.dumps(r, default=float) + "\n")
                f.flush()
                rows.extend(new)
                scored = sum(1 for r in new if "engines" in r)
                print(f"[{n}/{len(todo)}] {rid}: {scored} points ({time.time() - t0:.0f}s)", flush=True)

    write_report(rows, engines, args)
    print(f"\nWritten to {OUT_DIR / 'dataset_report.md'} and .json; checkpoint {CHECKPOINT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
