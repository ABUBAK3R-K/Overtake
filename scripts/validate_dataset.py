"""Validate the ingested dataset and derive per-race condition tags (FR-1).

Writes:
  configs/race_tags.json        versioned tags (wet / safety-car heavy) used by later phases
  data/processed/dataset_report.md   human-readable summary + integrity warnings

Tags are DERIVED from ingested data, not hand-labelled. They describe whole
races, so like Race.weather_summary they are for stratifying evaluation and
picking races, never for model features (would leak future weather / SCs).
"""
import json
import re
import sys
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ingestion.storage import connect  # noqa: E402

WET_COMPOUND_SHARE = 0.10   # >=10% of laps on INTERMEDIATE/WET
WET_WEATHER_SHARE = 0.25    # >=25% of race laps flagged rainfall
MIXED_COMPOUND_SHARE = 0.02  # not wet, but some INTERMEDIATE/WET running...
MIXED_WEATHER_SHARE = 0.05   # ...or rainfall on >=5% of laps
SC_HEAVY_DEPLOYMENTS = 3    # SC + VSC deployments in a race
SC_HEAVY_LAP_SHARE = 0.15   # or >=15% of laps run under SC/VSC


def runs(flags: list[bool]) -> int:
    """Number of contiguous True runs."""
    return sum(1 for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1]))


def main() -> int:
    con = connect()
    cfg = {r["race_id"]: r for r in tomllib.load(open(ROOT / "configs" / "races.toml", "rb"))["races"]}
    races = con.sql("select * from races").df().set_index("race_id")
    laps = con.sql("select race_id, driver, lap_number, lap_time, position, track_status from laps").df()
    tyres = con.sql("select race_id, compound from tyres").df()
    weather = con.sql("select race_id, lap, is_wet from weather").df()
    pits = con.sql("select race_id, count(*) n from pit_stops group by 1").df().set_index("race_id")["n"]
    tele = con.sql("select race_id, count(*) n from telemetry group by 1").df().set_index("race_id")["n"]

    tags, warnings = {}, []
    for rid, r in races.iterrows():
        L = laps[laps.race_id == rid]
        T = tyres[tyres.race_id == rid]
        W = weather[weather.race_id == rid]
        max_lap = int(L.lap_number.max())
        ts = L.groupby("lap_number").track_status.apply(lambda s: "".join(s.dropna().astype(str)))
        by_lap = ts.reindex(range(1, max_lap + 1), fill_value="")
        sc = [("4" in x) for x in by_lap]
        vsc = [(("6" in x) or ("7" in x)) and "4" not in x for x in by_lap]
        red = [("5" in x) for x in by_lap]
        sc_dep, vsc_dep = runs(sc), runs(vsc)
        sc_lap_share = (sum(sc) + sum(vsc)) / max_lap
        wet_compound = T.compound.isin(["INTERMEDIATE", "WET"]).mean() if len(T) else 0.0
        wet_weather = W.is_wet.mean() if len(W) else 0.0
        wet = wet_compound >= WET_COMPOUND_SHARE or wet_weather >= WET_WEATHER_SHARE
        mixed = (not wet) and (wet_compound >= MIXED_COMPOUND_SHARE or wet_weather >= MIXED_WEATHER_SHARE)
        heavy = (sc_dep + vsc_dep) >= SC_HEAVY_DEPLOYMENTS or sc_lap_share >= SC_HEAVY_LAP_SHARE
        tags[rid] = {
            "season": int(r.season), "circuit": r.circuit, "total_laps": int(r.total_laps),
            "era": cfg[rid]["era"], "train_weight": cfg[rid]["train_weight"],
            "wet": bool(wet), "mixed": bool(mixed), "wet_compound_share": round(float(wet_compound), 3),
            "wet_weather_share": round(float(wet_weather), 3),
            "sc_deployments": sc_dep, "vsc_deployments": vsc_dep,
            "sc_lap_share": round(sc_lap_share, 3), "red_flag_laps": int(sum(red)),
            "sc_heavy": bool(heavy),
        }
        # --- integrity checks
        if L.duplicated(["driver", "lap_number"]).any():
            warnings.append(f"{rid}: duplicate (driver, lap) rows")
        if L.driver.nunique() < 18:
            warnings.append(f"{rid}: only {L.driver.nunique()} drivers")
        if max_lap < 0.9 * r.total_laps and not sum(red):
            warnings.append(f"{rid}: max lap {max_lap} << scheduled {r.total_laps}")
        null_lt = L.lap_time.isna().mean()
        if null_lt > 0.15:
            warnings.append(f"{rid}: {null_lt:.0%} of laps have no lap_time")
        if L.position.isna().mean() > 0.10:
            warnings.append(f"{rid}: {L.position.isna().mean():.0%} of laps have no position")
        if T.compound.isna().mean() > 0.05:
            warnings.append(f"{rid}: {T.compound.isna().mean():.0%} of laps have unknown compound")
        if len(W) < 0.8 * max_lap:
            warnings.append(f"{rid}: weather covers {len(W)}/{max_lap} laps")
        if pits.get(rid, 0) == 0:
            warnings.append(f"{rid}: no pit stops recorded")
        if tele.get(rid, 0) < 0.9 * len(L):
            warnings.append(f"{rid}: telemetry covers {tele.get(rid, 0)}/{len(L)} laps")

    (ROOT / "configs" / "race_tags.json").write_text(json.dumps(tags, indent=1, sort_keys=True))

    df = pd.DataFrame(tags).T
    out = [f"# Dataset report\n", f"- races ingested: **{len(df)}** of {len(cfg)} configured"]
    out.append(f"- distinct circuits: **{df.circuit.nunique()}**")
    out.append("- by season: " + ", ".join(f"{k}: {v}" for k, v in df.season.value_counts().sort_index().items()))
    for era in ("ground_effect", "pre_2022"):
        d = df[df.era == era]
        out.append(f"- {era}: {len(d)} races, wet {int(d.wet.sum())}, mixed {int(d.mixed.sum())}, SC-heavy {int(d.sc_heavy.sum())}")
    out.append(f"- wet races: **{int(df.wet.sum())}** -> {', '.join(df.index[df.wet.astype(bool)])}")
    out.append(f"- mixed/damp (brief rain, not wet): **{int(df.mixed.sum())}** -> {', '.join(df.index[df.mixed.astype(bool)])}")
    out.append(f"- SC-heavy races: **{int(df.sc_heavy.sum())}** -> {', '.join(df.index[df.sc_heavy.astype(bool)])}")
    out.append(f"\n## Warnings ({len(warnings)})\n" + "\n".join(f"- {w}" for w in warnings))
    (ROOT / "data" / "processed" / "dataset_report.md").write_text("\n".join(out))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
