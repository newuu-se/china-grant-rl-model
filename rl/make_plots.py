"""
Generate summary plots for the train-energy RL project (current physics:
GOST Davis + no regen). Outputs PNGs into results/plots/.

  1. energy_time_tradeoff.png  — constant-notch energy vs trip-time Pareto curve,
     deadline, and where the RL greedy policy landed.
  2. rl_policy_profile.png     — RL greedy policy: speed (vs speed limit), notch,
     and track grade along the route.
  3. training_curve.png        — test/best reward per epoch for the final run.

Usage:
    source venv/bin/activate
    python rl/make_plots.py
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINKS = os.path.join(_REPO, "data", "netrainsim_v2", "linksFile_v2_fixed_speed.dat")
PROFILE = os.path.join(_REPO, "results", "notch_profile.csv")
OUTDIR = os.path.join(_REPO, "results", "plots")
os.makedirs(OUTDIR, exist_ok=True)

# Measured constant-notch curve under the CURRENT physics (GOST Davis + no regen)
NOTCH = [8, 6, 5, 4, 3, 2]
STEPS = [5603, 5654, 5716, 5888, 6337, 7706]
ENERGY = [912.47, 909.28, 896.94, 881.53, 834.13, 783.52]
DEADLINE = 6500
RL_STEPS, RL_ENERGY = 7708, 784.16   # latest greedy eval (pace-penalty run)


def load_links():
    """cumulative distance -> (speed_limit, grade_pct) along the route."""
    cum, spd, grade = [0.0], [], []
    with open(LINKS) as f:
        f.readline(); f.readline()
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            length = float(p[3]); speed = float(p[4]); g = float(p[6])
            cum.append(cum[-1] + length); spd.append(speed); grade.append(g)
    return np.array(cum), np.array(spd), np.array(grade)


def load_profile():
    speed, notch = [], []
    with open(PROFILE) as f:
        for row in csv.DictReader(f):
            speed.append(float(row["speed_mps"])); notch.append(int(row["notch"]))
    speed = np.array(speed)
    dist = np.cumsum(speed)          # 1 s per row → travelled distance (m)
    return dist, speed, np.array(notch)


def load_training(log_path):
    epoch, test, best = [], [], []
    pat = re.compile(r"Epoch #(\d+): test_reward: (-?[0-9.]+).*best_reward: (-?[0-9.]+)")
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                epoch.append(int(m.group(1))); test.append(float(m.group(2))); best.append(float(m.group(3)))
    return np.array(epoch), np.array(test), np.array(best)


def plot_tradeoff():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(STEPS, ENERGY, "o-", color="#1f77b4", lw=2, ms=8, label="constant notch")
    for n, s, e in zip(NOTCH, STEPS, ENERGY):
        ax.annotate(f"N{n}", (s, e), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axvline(DEADLINE, color="crimson", ls="--", lw=1.5, label=f"schedule deadline ({DEADLINE} s)")
    ax.scatter([RL_STEPS], [RL_ENERGY], color="darkorange", s=140, marker="*",
               zorder=5, label=f"RL greedy ({RL_ENERGY:.0f} kWh, {RL_STEPS} s — late)")
    # highlight eco target (notch 3, the slowest still on-time)
    ax.scatter([6337], [834.13], facecolors="none", edgecolors="green", s=200, lw=2,
               zorder=5, label="eco target N3 (834 kWh, on-time)")
    ax.set_xlabel("trip time (s)"); ax.set_ylabel("total energy (kWh)")
    ax.set_title("Energy–time tradeoff (GOST Davis, no regen)\nlower notch = less energy but slower")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); p = os.path.join(OUTDIR, "energy_time_tradeoff.png")
    fig.savefig(p, dpi=130); plt.close(fig); return p


def plot_profile():
    cum, spd_lim, grade = load_links()
    dist, speed, notch = load_profile()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(dist / 1000, speed, color="#1f77b4", lw=1, label="train speed")
    axes[0].step(cum[1:] / 1000, spd_lim, where="post", color="crimson", lw=1, alpha=.7, label="speed limit")
    axes[0].set_ylabel("speed (m/s)"); axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"RL greedy policy along the route  ({RL_ENERGY:.0f} kWh, {RL_STEPS} s)")
    axes[0].grid(alpha=0.3)

    axes[1].step(dist / 1000, notch, where="post", color="darkorange", lw=1)
    axes[1].set_ylabel("notch (0–8)"); axes[1].set_ylim(-0.5, 8.5); axes[1].grid(alpha=0.3)

    axes[2].fill_between(cum[1:] / 1000, grade, color="gray", alpha=0.5, step="post")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_ylabel("grade (%)"); axes[2].set_xlabel("distance along route (km)"); axes[2].grid(alpha=0.3)

    fig.tight_layout(); p = os.path.join(OUTDIR, "rl_policy_profile.png")
    fig.savefig(p, dpi=130); plt.close(fig); return p


def plot_training():
    logs = sorted(f for f in os.listdir(os.path.join(_REPO, "logs")) if f.startswith("train_run_") and f.endswith(".log"))
    if not logs:
        return None
    log_path = os.path.join(_REPO, "logs", logs[-1])
    epoch, test, best = load_training(log_path)
    if len(epoch) == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch, test, color="#1f77b4", lw=1, alpha=0.6, label="test reward (per epoch)")
    ax.plot(epoch, best, color="green", lw=2, label="best reward (running)")
    ax.set_xlabel("epoch"); ax.set_ylabel("episode reward")
    ax.set_title(f"Training progress — {logs[-1]}\n(pace-penalty reward; stable, plateaus early)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); p = os.path.join(OUTDIR, "training_curve.png")
    fig.savefig(p, dpi=130); plt.close(fig); return p


if __name__ == "__main__":
    for fn in (plot_tradeoff, plot_profile, plot_training):
        out = fn()
        print(f"wrote {out}" if out else f"skipped {fn.__name__} (missing data)")
