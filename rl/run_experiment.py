"""
Single experiment run for the reward-sensitivity sweep and ablation study.

One process = one (trip, config, seed) training run. All reward-weight and
ablation switches are translated to environment variables and set BEFORE
rl.train_env is imported, because that module reads the weights at import time.
This is the only correct place to set them; the sweep launcher
(rl/run_campaign.py) just calls this script with different arguments.

Examples
--------
# w_P sensitivity point, forward trip, seed 0:
python -m rl.run_experiment --trip forward --seed 0 --wp 4 --epochs 100 \
       --outdir checkpoints/sweep/fwd_wp4_s0

# no-smoothness ablation, return trip, seed 3:
python -m rl.run_experiment --trip return --seed 3 --ws 0 --epochs 100 \
       --outdir checkpoints/abl/ret_nosmooth_s3
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trip", choices=["forward", "return"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--outdir", required=True, help="checkpoint dir for this run")
    # reward weights — omitted means use the train_env default
    p.add_argument("--wp", type=float, default=None, help="W_PACE")
    p.add_argument("--we", type=float, default=None, help="W_ENERGY")
    p.add_argument("--ws", type=float, default=None, help="W_SMOOTH")
    # ablation switches
    p.add_argument("--ablate-time", action="store_true",
                   help="hide schedule features (non-Markov ablation)")
    p.add_argument("--constant-entropy", action="store_true",
                   help="hold ent_coef fixed (no annealing) — entropy ablation")
    args = p.parse_args()

    # ── set env BEFORE importing rl.train_env (weights read at import) ──────
    if args.wp is not None:
        os.environ["RL_W_PACE"] = repr(args.wp)
    if args.we is not None:
        os.environ["RL_W_ENERGY"] = repr(args.we)
    if args.ws is not None:
        os.environ["RL_W_SMOOTH"] = repr(args.ws)
    if args.ablate_time:
        os.environ["RL_ABLATE_TIME"] = "1"

    import json
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rl.train_env import (NeTrainSimEnv, TRAINS_FILE, TRAINS_FILE_RETURN,
                              W_ENERGY, W_PACE, W_SMOOTH, ABLATE_TIME_FEATURES)
    from rl.train import run_training, build_policy
    from rl.evaluate import eval_policy
    import torch

    trains_file = TRAINS_FILE_RETURN if args.trip == "return" else TRAINS_FILE
    label = ("B→A return (Ho'jakent→Toshkent)" if args.trip == "return"
             else "A→B forward (Toshkent→Ho'jakent)")

    def make_env_fn():
        return NeTrainSimEnv(trains_file=trains_file)

    result = run_training(
        make_env_fn=make_env_fn,
        checkpoint_dir=args.outdir,
        trip_label=label,
        seed=args.seed,
        max_epoch=args.epochs,
        constant_entropy=args.constant_entropy,
        save_every=0,          # campaign keeps only best+final per run
    )

    # ── score the best checkpoint: deterministic + stochastic eval ─────────
    best_ckpt = os.path.join(args.outdir, "policy_best.pth")
    policy, *_ = build_policy()
    policy.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
    policy.eval()
    evals = {
        "deterministic": eval_policy(policy, trains_file, stochastic=False),
        "stochastic":    eval_policy(policy, trains_file, stochastic=True),
    }

    summary = {
        "trip": args.trip, "seed": args.seed, "epochs": args.epochs,
        "config": {
            "w_pace": W_PACE, "w_energy": W_ENERGY, "w_smooth": W_SMOOTH,
            "ablate_time": ABLATE_TIME_FEATURES,
            "constant_entropy": args.constant_entropy,
        },
        "best_reward": float(result["best_reward"]),
        "eval": evals,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[run_experiment] wrote {args.outdir}/results.json  "
          f"best_reward={summary['best_reward']:.1f}  "
          f"det={evals['deterministic']['energy_kwh']:.0f}kWh/"
          f"{evals['deterministic']['steps']}s  "
          f"stoch={evals['stochastic']['energy_kwh']:.0f}kWh/"
          f"{evals['stochastic']['steps']}s")


if __name__ == "__main__":
    main()
