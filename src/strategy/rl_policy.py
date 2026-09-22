"""Strategy Engine 3 — training and loading the RL policy (Design.md §6.10 / FR-7).

``RaceStrategyEnv`` in ``rl_env.py`` runs live Monte Carlo on every step
(~3.5 s for 100 sims), so training PPO against it directly would take days.
Instead, training is split in two:

1. ``build`` — sample decision states from real races (via replay, so
   ``as_of_lap()`` applies) and score all 4 actions on each with the same
   Monte Carlo simulator the other two engines use, using one seed per state
   so the 4 actions share their random draws. Resumable, parallel, written to
   ``data/models/rl/dataset.jsonl``.
2. ``train`` — ``CachedStrategyEnv`` replays those states; a step samples a
   finishing position from the chosen action's simulated distribution. PPO
   learns a policy obs → action that must generalise across states, unlike
   Engines 1–2 which search each state from scratch.

Races in the frozen split's ``test_races`` (configs/tyre_split.toml) are held
out of training; ``eval`` scores the policy on them against simple baselines.

    python -m src.strategy.rl_policy build            # ~30 min on 8 cores
    python -m src.strategy.rl_policy train
    python -m src.strategy.rl_policy eval
    python -m src.strategy.rl_policy all
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.strategy.rl_env import _COMPOUNDS, OBS_DIM, _make_observation

log = logging.getLogger("overtake.strategy.rl_policy")

ROOT = Path(__file__).resolve().parents[2]
RL_DIR = ROOT / "data" / "models" / "rl"
DATASET_PATH = RL_DIR / "dataset.jsonl"
POLICY_PATH = RL_DIR / "ppo_policy.zip"
META_PATH = RL_DIR / "policy_meta.json"
SPLIT_PATH = ROOT / "configs" / "tyre_split.toml"

N_ACTIONS = 4
ACTION_NAMES = ["STAY_OUT"] + [f"PIT_{c}" for c in _COMPOUNDS]
DRY = {"SOFT", "MEDIUM", "HARD"}


def held_out_races() -> set[str]:
    """The frozen tyre split's test races — never used to train the policy."""
    with open(SPLIT_PATH, "rb") as f:
        return set(tomllib.load(f)["test_races"])


def action_strategy(action: int, lap: int) -> dict[str, Any]:
    if action == 0:
        return {"name": "STAY_OUT", "pit_laps": [], "compounds": []}
    comp = _COMPOUNDS[action - 1]
    return {"name": f"PIT_{comp}", "pit_laps": [lap + 1], "compounds": [comp]}


# ─── 1. Dataset ──────────────────────────────────────────────────────────────


def sample_decision_points(
    race_id: str, con, rng: np.random.Generator, n_real: int = 3, n_random: int = 2,
) -> list[tuple[str, int]]:
    """Decision states for one race: ``n_real`` taken one lap before real pit
    stops (the states a strategist actually agonised over, and what the FR-8
    backtest scores), plus ``n_random`` random (driver, lap) pairs so the
    policy also sees states where staying out is clearly right."""
    stops = con.sql(
        "SELECT driver, lap FROM pit_stops WHERE race_id = ? AND lap > 4",
        params=[race_id],
    ).fetchall()
    laps = con.sql(
        "SELECT MAX(lap_number) FROM laps WHERE race_id = ?", params=[race_id],
    ).fetchone()
    total = int(laps[0]) if laps and laps[0] else 0
    if total < 15:
        return []

    points: list[tuple[str, int]] = []
    stops = [(d, int(l)) for d, l in stops if int(l) < total - 2]
    for i in rng.permutation(len(stops))[:n_real]:
        d, l = stops[i]
        points.append((d, l - 1))

    drivers = [r[0] for r in con.sql(
        "SELECT DISTINCT driver FROM laps WHERE race_id = ?", params=[race_id],
    ).fetchall()]
    for _ in range(n_random):
        if drivers:
            points.append((str(rng.choice(drivers)), int(rng.integers(5, total - 3))))
    return points


def score_state(race_id: str, driver: str, lap: int, n_sims: int, seed: int) -> dict[str, Any] | None:
    """Monte Carlo every action at one decision state. Returns None for states
    the policy can't act on (driver retired/absent, wet tyres, race over)."""
    from src.models.lap_time import make_lap_time_predictor
    from src.simulation.monte_carlo import run_monte_carlo
    from src.simulation.replay import build_state_at_lap

    state = build_state_at_lap(race_id, lap)
    if driver not in state.positions or driver not in state.tyres:
        return None
    if str(state.tyres[driver][0]).upper() not in DRY or state.total_laps - state.lap < 3:
        return None

    try:
        predictor = make_lap_time_predictor(race_id, lap)
    except Exception:
        predictor = None

    expected, dists = [], []
    for a in range(N_ACTIONS):
        res = run_monte_carlo(
            state, action_strategy(a, state.lap), target_driver=driver,
            n_sims=n_sims, lap_time_predictor=predictor, seed=seed,
        )
        expected.append(float(res["expected_position"]))
        dists.append([[int(p), float(q)] for p, q in res["finish_prob_by_position"].items()])

    return {
        "race_id": race_id, "driver": driver, "lap": int(state.lap),
        "total_laps": int(state.total_laps), "circuit": state.circuit,
        "obs": [round(float(x), 5) for x in _make_observation(state, driver)],
        "expected_position": expected, "finish_dist": dists,
    }


def _score_race(race_id: str, points: list[tuple[str, int]], n_sims: int, seed: int) -> list[dict]:
    """Worker: score every point in one race. One bad state never kills the race."""
    logging.disable(logging.WARNING)
    out = []
    for k, (driver, lap) in enumerate(points):
        try:
            rec = score_state(race_id, driver, lap, n_sims, seed + k)
        except Exception as exc:  # noqa: BLE001
            rec = {"race_id": race_id, "driver": driver, "lap": lap, "error": repr(exc)}
        if rec is not None:
            out.append(rec)
    out.append({"race_id": race_id, "race_done": True})
    return out


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if "obs" in r and len(r["obs"]) == OBS_DIM]


def build_dataset(n_sims: int = 150, workers: int | None = None, max_races: int | None = None,
                  seed: int = 2026) -> None:
    """Score decision states across every ingested race. Resumable: races
    already marked done in the dataset file are skipped."""
    from src.ingestion.session import load_race_list
    from src.ingestion.storage import connect

    RL_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if DATASET_PATH.exists():
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("race_done"):
                    done.add(r["race_id"])

    race_ids = [r["race_id"] for r in load_race_list()]
    if max_races:
        race_ids = race_ids[:max_races]
    con = connect()
    jobs = {}
    for i, race_id in enumerate(race_ids):
        if race_id in done:
            continue
        rng = np.random.default_rng([seed, i])
        try:
            points = sample_decision_points(race_id, con, rng)
        except Exception:
            log.warning("skipping %s: not ingested?", race_id)
            continue
        jobs[race_id] = points

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(done)} races already done; scoring {len(jobs)} races on {workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool, open(DATASET_PATH, "a", encoding="utf-8") as f:
        futs = {pool.submit(_score_race, rid, pts, n_sims, seed * 1000 + i): rid
                for i, (rid, pts) in enumerate(jobs.items())}
        for n, fut in enumerate(as_completed(futs), 1):
            rid = futs[fut]
            try:
                rows = fut.result()
            except Exception as exc:  # noqa: BLE001 — worker crash: leave race undone so a rerun retries it
                print(f"[{n}/{len(jobs)}] {rid} FAILED: {exc!r}", flush=True)
                continue
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            ok = sum(1 for r in rows if "obs" in r)
            print(f"[{n}/{len(jobs)}] {rid}: {ok} states ({time.time() - t0:.0f}s)", flush=True)


# ─── 2. Environment + training ───────────────────────────────────────────────

class CachedStrategyEnv(gym.Env):
    """Contextual-bandit env over pre-simulated decision states.

    reset() draws a state; step(a) samples a finishing position from action
    a's Monte Carlo distribution at that state. Reward = expected position
    if staying out − sampled position (positions gained vs. staying out),
    which removes the per-state baseline so PPO learns the *decision*, not
    how good the driver's grid slot is.
    """

    metadata = {"render_modes": []}

    def __init__(self, records: list[dict], seed: int | None = None) -> None:
        super().__init__()
        if not records:
            raise ValueError("CachedStrategyEnv needs at least one record")
        self._records = records
        self._obs = np.asarray([r["obs"] for r in records], dtype=np.float32)
        self._dists = []
        for r in records:
            per_action = []
            for d in r["finish_dist"]:
                pos = np.array([p for p, _ in d], dtype=np.float32)
                prob = np.array([q for _, q in d], dtype=np.float64)
                per_action.append((pos, prob / prob.sum()))
            self._dists.append(per_action)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self._i = 0
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._i = int(self._rng.integers(len(self._records)))
        return self._obs[self._i], {}

    def step(self, action: int):
        pos, prob = self._dists[self._i][int(action)]
        sampled = float(self._rng.choice(pos, p=prob))
        baseline = self._records[self._i]["expected_position"][0]
        return self._obs[self._i], baseline - sampled, True, False, {"finish_position": sampled}


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    held = held_out_races()
    return ([r for r in records if r["race_id"] not in held],
            [r for r in records if r["race_id"] in held])


def train_policy(records: list[dict], timesteps: int = 300_000, seed: int = 2026):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(lambda: CachedStrategyEnv(records), n_envs=8, seed=seed)
    # Single-step episodes: gamma is irrelevant, so keep it 0 (no bootstrapping).
    model = PPO("MlpPolicy", env, gamma=0.0, gae_lambda=1.0, n_steps=256, batch_size=256,
                ent_coef=0.01, learning_rate=3e-4, seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model


# ─── 3. Evaluation ───────────────────────────────────────────────────────────


def policy_actions(model, records: list[dict]) -> np.ndarray:
    obs = np.asarray([r["obs"] for r in records], dtype=np.float32)
    actions, _ = model.predict(obs, deterministic=True)
    return np.asarray(actions, dtype=int)


def evaluate(actions: np.ndarray, records: list[dict], train_records: list[dict]) -> dict[str, Any]:
    """Regret (expected positions lost vs. the best of the 4 actions at that
    state, per the same Monte Carlo) for the policy and simple baselines.
    The "oracle" is the per-state argmin, so it's biased slightly optimistic
    by Monte Carlo noise — every row shares that bias equally."""
    E = np.asarray([r["expected_position"] for r in records])
    best = E.min(axis=1)
    idx = np.arange(len(records))

    def row(a: np.ndarray) -> dict[str, float]:
        reg = E[idx, a] - best
        return {"mean_regret": round(float(reg.mean()), 3),
                "median_regret": round(float(np.median(reg)), 3),
                "picked_best_pct": round(float((reg < 1e-9).mean() * 100), 1)}

    train_E = np.asarray([r["expected_position"] for r in train_records])
    best_fixed = int(train_E.mean(axis=0).argmin())
    rng = np.random.default_rng(0)
    return {
        "n_states": len(records),
        "rl_policy": row(actions),
        "always_stay_out": row(np.zeros(len(records), dtype=int)),
        f"always_{ACTION_NAMES[best_fixed].lower()}": row(np.full(len(records), best_fixed)),
        "random": row(rng.integers(0, N_ACTIONS, len(records))),
        "rl_action_mix": {ACTION_NAMES[a]: int((actions == a).sum()) for a in range(N_ACTIONS)},
        "oracle_action_mix": {ACTION_NAMES[a]: int((E.argmin(axis=1) == a).sum()) for a in range(N_ACTIONS)},
    }


# ─── Loading (used by rl_env.get_strategy_recommendation_rl) ─────────────────


@functools.lru_cache(maxsize=1)
def load_policy():
    """The trained PPO policy, or None if it hasn't been trained yet."""
    if not POLICY_PATH.exists():
        return None
    try:
        from stable_baselines3 import PPO
        return PPO.load(POLICY_PATH, device="cpu")
    except Exception:
        log.warning("could not load RL policy from %s", POLICY_PATH, exc_info=True)
        return None


def action_probabilities(model, obs: np.ndarray) -> np.ndarray:
    """The policy's probability for each action at one observation."""
    import torch

    obs_t, _ = model.policy.obs_to_tensor(np.asarray(obs, dtype=np.float32))
    with torch.no_grad():
        probs = model.policy.get_distribution(obs_t).distribution.probs
    return probs.cpu().numpy().reshape(-1)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _train_and_eval(timesteps: int) -> None:
    records = load_dataset()
    train, test = split_records(records)
    print(f"dataset: {len(records)} states ({len(train)} train, {len(test)} held-out)")
    t0 = time.time()
    model = train_policy(train, timesteps=timesteps)
    RL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(POLICY_PATH)
    load_policy.cache_clear()
    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timesteps": timesteps, "train_seconds": round(time.time() - t0, 1),
        "n_train_states": len(train), "held_out_races": sorted(held_out_races()),
        "train": evaluate(policy_actions(model, train), train, train),
        "held_out": evaluate(policy_actions(model, test), test, train) if test else None,
    }
    META_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "train", "eval", "all"])
    ap.add_argument("--n-sims", type=int, default=150)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-races", type=int, default=None)
    ap.add_argument("--timesteps", type=int, default=300_000)
    args = ap.parse_args(argv)

    if args.cmd in ("build", "all"):
        build_dataset(n_sims=args.n_sims, workers=args.workers, max_races=args.max_races)
    if args.cmd in ("train", "all"):
        _train_and_eval(args.timesteps)
    if args.cmd == "eval":
        model = load_policy()
        if model is None:
            print("no trained policy; run `train` first")
            return 1
        train, test = split_records(load_dataset())
        print(json.dumps({"train": evaluate(policy_actions(model, train), train, train),
                          "held_out": evaluate(policy_actions(model, test), test, train)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
