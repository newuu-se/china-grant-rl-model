"""
Evaluate a trained RL policy on one full A→B trip and export the notch
profile + physics state to a CSV file.

The output CSV can be loaded into NeTrainSim (or plotted) to compare the
RL energy profile against a baseline (e.g. constant notch or human driver).

Usage:
    source venv/bin/activate
    python rl/evaluate.py                                      # uses latest checkpoint
    python rl/evaluate.py --checkpoint checkpoints/policy_epoch050.pth
    python rl/evaluate.py --checkpoint checkpoints/policy_final.pth
"""

import argparse
import csv
import os
import sys

import torch
import numpy as np
from tianshou.policy import PPOPolicy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import NeTrainSimEnv, NODES_FILE, LINKS_FILE
# Reuse training's policy factory + device so the eval policy can never drift
# from the trained one (same architecture and hyperparameters).
from rl.train import build_policy as build_ppo_policy, DEVICE

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")


def load_path_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build cumulative-distance → (x, y) lookup along the route.
    Returns three parallel arrays (cum_pos_m, x_m, y_m) so np.interp can
    map any position-along-path to (x_m, y_m) by linear interpolation.
    """
    nodes: dict[int, tuple[float, float]] = {}
    with open(NODES_FILE) as f:
        f.readline()                # header line
        f.readline()                # count / scales line
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            nid = int(parts[0])
            x = float(parts[1]); y = float(parts[2])
            nodes[nid] = (x, y)

    # Walk links in order; cumulative distance at each visited node.
    cum, xs, ys = [0.0], [], []
    first_id = None
    with open(LINKS_FILE) as f:
        f.readline()                # header
        f.readline()                # count / scales
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            from_id = int(parts[1]); to_id = int(parts[2])
            length  = float(parts[3])
            if first_id is None:
                first_id = from_id
                xs.append(nodes[from_id][0])
                ys.append(nodes[from_id][1])
            cum.append(cum[-1] + length)
            xs.append(nodes[to_id][0])
            ys.append(nodes[to_id][1])
    return np.array(cum), np.array(xs), np.array(ys)


def find_latest_checkpoint() -> str:
    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError(f"No checkpoints directory found at {CHECKPOINT_DIR}")
    files = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth"))
    if not files:
        raise FileNotFoundError("No .pth checkpoint files found in checkpoints/")
    if "policy_best.pth" in files:
        return os.path.join(CHECKPOINT_DIR, "policy_best.pth")
    if "policy_final.pth" in files:
        return os.path.join(CHECKPOINT_DIR, "policy_final.pth")
    return os.path.join(CHECKPOINT_DIR, files[-1])


def build_policy(checkpoint_path: str) -> PPOPolicy:
    policy, _actor, _critic, _optim = build_ppo_policy(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return policy


def run_episode(policy: PPOPolicy, output_path: str, stochastic: bool = False) -> None:
    env = NeTrainSimEnv()
    obs, _ = env.reset()

    cum_pos, xs, ys = load_path_coordinates()

    rows = []
    step = 0
    cum_energy = 0.0
    terminated = False
    truncated  = False

    print("Running evaluation episode...")
    print(f"{'Step':>6}  {'Pos(m)':>10}  {'X(m)':>12}  {'Y(m)':>12}  "
          f"{'Speed(m/s)':>10}  {'Notch':>5}  {'CumEnergy':>10}")
    print("-" * 80)

    while not (terminated or truncated):
        # Deterministic (argmax) by default; --stochastic samples from the policy.
        # NOTE: for a high-entropy policy the argmax can be unrepresentative (its mode
        # parks at the lowest-energy notch); --stochastic reflects the trained behavior.
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            probs, _ = policy.actor(obs_tensor)
            probs = probs.squeeze(0)
            notch = (int(torch.distributions.Categorical(probs).sample())
                     if stochastic else int(probs.argmax().item()))

        obs, reward, terminated, truncated, _ = env.step(notch)

        state = env._last_state
        position_m      = float(state["position_m"])
        speed_mps       = float(state["speed_mps"])
        step_energy_kwh = float(state["energy_kwh"])
        cum_energy     += step_energy_kwh

        # Interpolate (x, y) at the train's current position along the path
        x_m = float(np.interp(position_m, cum_pos, xs))
        y_m = float(np.interp(position_m, cum_pos, ys))

        rows.append({
            "position_m": round(position_m, 3),
            "x_m":       round(x_m, 3),
            "y_m":       round(y_m, 3),
            "speed_mps": round(speed_mps, 4),
            "notch":     notch,
        })

        step += 1
        if step % 100 == 0:
            print(f"{step:>6}  {position_m:>10.1f}  {x_m:>12.1f}  {y_m:>12.1f}  "
                  f"{speed_mps:>10.3f}  {notch:>5}  {env._cum_energy_kwh:>10.3f}")

    # Totals from the env (correct under action-repeat: sums every simulator second,
    # not just the last one of each control interval).
    total_energy = env._cum_energy_kwh
    sim_steps = env._step_count
    env.close()

    status = "ARRIVED" if terminated else "TIMEOUT"
    print(f"\n[{status}] decisions={step}  sim_steps={sim_steps}  total_energy={total_energy:.2f} kWh\n")

    # Write CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = ["position_m", "x_m", "y_m", "speed_mps", "notch"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Notch profile saved → {output_path}")
    print(f"Rows: {len(rows)} (control decisions)  |  Total energy: {total_energy:.2f} kWh  |  sim_steps: {sim_steps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint (default: latest in checkpoints/)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: results/notch_profile.csv)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions from the policy instead of argmax "
                             "(representative for a high-entropy policy)")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_latest_checkpoint()
    output_path = args.output or os.path.join(OUTPUT_DIR, "notch_profile.csv")

    policy = build_policy(checkpoint)
    run_episode(policy, output_path, stochastic=args.stochastic)


if __name__ == "__main__":
    main()
