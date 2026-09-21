"""Freeze the Phase 2 tyre-model train/test split (PRD §8 phase 2).

Writes configs/tyre_split.toml. Run once, commit the result, and do not re-run
after seeing model results: the point of a frozen split is that the test set
was chosen without knowing how any model does on it.

Rule (uses only configs/race_tags.json and which races yield tyre training
rows; never any model output):
  1. Hold out ALL 2022-2024 races at HELD_OUT_CIRCUITS (unseen-circuit test).
  2. Hold out extra single races at circuits that stay in training
     (seen-circuit test): one wet race, plus four dry races at distinct
     circuits covering 2022, 2023 and 2024. Chosen with SEED.
  3. Only races that produce tyre training rows can be test races
     (fully-wet races such as 2022 Japan have no dry-compound running).
  4. 2018-2021 races are never in the test set.

Usage:  python scripts/make_tyre_split.py [--force]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.tyre import build_training_frame, load_lap_frame  # noqa: E402

OUT = ROOT / "configs" / "tyre_split.toml"
SEED = 2026
HELD_OUT_CIRCUITS = ["Monaco", "Monza", "Baku"]
N_WET_SEEN = 1
FIRST_SEASON = 2022


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    if OUT.exists() and not ap.parse_args().force:
        print(f"{OUT} exists; the split is frozen. Pass --force to overwrite.")
        return 1

    tags = json.loads((ROOT / "configs" / "race_tags.json").read_text(encoding="utf-8"))
    usable = set(build_training_frame(load_lap_frame())["race_id"])
    modern = {r: t for r, t in tags.items() if t["season"] >= FIRST_SEASON and r in usable}
    rng = np.random.default_rng(SEED)

    circuit_races = {c: sorted(r for r, t in tags.items()
                               if t["season"] >= FIRST_SEASON and t["circuit"] == c)
                     for c in {t["circuit"] for t in modern.values()}}
    unseen = sorted(r for c in HELD_OUT_CIRCUITS for r in circuit_races[c] if r in usable)

    # Seen-circuit candidates: circuit has all 3 seasons so >=2 stay in training.
    def eligible(r):
        t = tags[r]
        return (t["circuit"] not in HELD_OUT_CIRCUITS and len(circuit_races[t["circuit"]]) >= 3)

    wet_pool = sorted(r for r in modern if eligible(r) and modern[r]["wet"])
    dry_pool = sorted(r for r in modern if eligible(r) and not modern[r]["wet"] and not modern[r]["mixed"])

    seen, used_circuits = [], set()
    for r in rng.permutation(wet_pool)[:N_WET_SEEN]:
        seen.append(str(r)); used_circuits.add(tags[r]["circuit"])
    for season in (2022, 2023, 2024, None):     # one per season, then one free pick
        pool = [r for r in dry_pool if tags[r]["circuit"] not in used_circuits
                and (season is None or tags[r]["season"] == season)]
        r = str(rng.choice(pool))
        seen.append(r); used_circuits.add(tags[r]["circuit"])

    test = sorted(unseen + seen)
    n_modern = sum(t["season"] >= FIRST_SEASON for t in tags.values())
    wet_test = [r for r in test if tags[r]["wet"]]
    lines = [
        "# Frozen tyre-model evaluation split (scripts/make_tyre_split.py, seed "
        f"{SEED}).",
        "# Do NOT tune on these races or regenerate after seeing results.",
        "# Tuning uses leave-one-race/circuit-out inside the training set only.",
        "",
        f"seed = {SEED}",
        f"held_out_circuits = {json.dumps(HELD_OUT_CIRCUITS)}   # every 2022-2024 race here",
        f"seen_circuit_races = {json.dumps(sorted(seen))}   # single races; circuit stays in training",
        f"test_races = {json.dumps(test)}",
        f"wet_test_races = {json.dumps(wet_test)}",
        "",
        f"# {len(test)} test races = {len(test) / n_modern:.1%} of {n_modern} 2022-2024 races; "
        f"{len(wet_test)} wet.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for r in test:
        t = tags[r]
        print(f"  {r:28s} {t['circuit']:14s} wet={t['wet']!s:5} sc_heavy={t['sc_heavy']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
