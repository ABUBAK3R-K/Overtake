"""Run the FR-8 three-engine backtest across the full ingested race dataset
(Design.md Section 6.12: "the central experiment of the whole project").

Unlike src/evaluation/backtest.py's HISTORICAL_BENCHMARKS (6 hand-curated 2023
decision points, used by the fast /api/backtest* endpoints), this scores every
engine at real pit-stop decision points drawn from every ingested race, via
src.evaluation.backtest.run_dataset_backtest / decision_points_for_race.

Resumable: progress is checkpointed to data/models/backtest/checkpoint.jsonl
after every decision point, so an interrupted run (Ctrl-C, crash) can be
restarted and will skip races that already have results for a given engine.

Usage:
    python scripts/run_full_backtest.py                  # all races, all 3 engines
    python scripts/run_full_backtest.py --max-races 20    # a subset, for a quick look
    python scripts/run_full_backtest.py --engines search  # one engine only
    python scripts/run_full_backtest.py --fresh            # ignore any checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.backtest import (  # noqa: E402
    ENGINES,
    run_dataset_backtest,
    summarize_multi_engine_backtest,
)
from src.ingestion.session import load_race_list  # noqa: E402
from src.ingestion.storage import connect  # noqa: E402

OUT_DIR = ROOT / "data" / "models" / "backtest"
CHECKPOINT_PATH = OUT_DIR / "checkpoint.jsonl"


def _load_checkpoint() -> list[dict]:
    if not CHECKPOINT_PATH.exists():
        return []
    lines = CHECKPOINT_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-races", type=int, default=None, help="only backtest the first N races (default: all)")
    parser.add_argument("--decision-points-per-race", type=int, default=2)
    parser.add_argument("--n-sims", type=int, default=60)
    parser.add_argument("--engines", nargs="+", default=list(ENGINES), choices=list(ENGINES))
    parser.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint and start over")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    race_ids = [r["race_id"] for r in load_race_list()]
    if args.max_races:
        race_ids = race_ids[: args.max_races]

    if args.fresh and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    prior_results = [] if args.fresh else _load_checkpoint()

    con = connect()
    checkpoint_file = CHECKPOINT_PATH.open("a", encoding="utf-8")
    multi: dict[str, list[dict]] = {e: [r for r in prior_results if r.get("engine") == e] for e in args.engines}

    def _record(res: dict) -> None:
        checkpoint_file.write(json.dumps(res, default=float) + "\n")
        checkpoint_file.flush()

    try:
        for engine in args.engines:
            done_races = {r["race_id"] for r in multi[engine]}
            remaining = [rid for rid in race_ids if rid not in done_races]
            print(f"[{engine}] {len(remaining)}/{len(race_ids)} races remaining "
                  f"({len(done_races)} already checkpointed)")
            new_results = run_dataset_backtest(
                race_ids=remaining, con=con, n_sims=args.n_sims, engine=engine,
                decision_points_per_race=args.decision_points_per_race,
                on_result=_record,
            )
            multi[engine].extend(new_results)
    finally:
        checkpoint_file.close()

    summary = summarize_multi_engine_backtest(multi)
    (OUT_DIR / "dataset_report.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    lines = [
        "# Overtake — Full-Dataset Backtest (FR-8)",
        "",
        f"Races: {len(race_ids)} · decision points/race: {args.decision_points_per_race} · n_sims: {args.n_sims}",
        "",
        "| Engine | Evaluations | Failed | Improved | Matched | Worse | Success % | Avg gain (pos) | Avg regret |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for engine, s in summary["engines"].items():
        lines.append(
            f"| {engine} | {s['total_evaluations']} | {s['failed_evaluations']} | "
            f"{s['strategies_improved']} | {s['strategies_matched']} | {s['strategies_worse']} | "
            f"{s['success_rate_pct']} | {s['average_position_gain']} | {s['average_regret_vs_hindsight']} |"
        )
    lines += [
        "",
        f"**Best engine by success rate:** {summary['best_engine_by_success_rate']}",
        f"**Ranking:** {' > '.join(summary['ranking'])}",
    ]
    (OUT_DIR / "dataset_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nWritten to {OUT_DIR / 'dataset_report.json'} and dataset_report.md")
    print(f"Checkpoint: {CHECKPOINT_PATH} ({sum(len(v) for v in multi.values())} total decision points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
