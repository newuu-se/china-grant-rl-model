"""
Summary plots for the train-energy project (current physics: GOST Davis + no regen).
Outputs PNGs into results/plots/.

  1. energy_time_tradeoff.png  — polished energy-vs-trip-time curve over constant
     notches, with the RL greedy policy and NeTrainSim's native driver overlaid.
  2. netrainsim_trajectory.png — the trajectory NeTrainSim itself produces (the data
     its GUI plots): speed vs limit, notch, tractive/resistance forces, power,
     cumulative energy and grade along the route.
  3. rl_policy_profile.png     — RL greedy policy speed/notch/grade along the route.
  4. training_curve.png        — test/best reward per epoch for the final run.

Usage:
    source venv/bin/activate
    python rl/make_plots.py
"""
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINKS = os.path.join(_REPO, "data", "netrainsim_v2", "linksFile_v2_fixed_speed.dat")
PROFILE = os.path.join(_REPO, "results", "notch_profile.csv")
NTS_GLOB = os.path.join(_REPO, "NeTrainSim-adjusted", "res", "trainTrajectory_*.csv")
OUTDIR = os.path.join(_REPO, "results", "plots")
os.makedirs(OUTDIR, exist_ok=True)

# Measured constant-notch curve under the CURRENT physics (GOST Davis + no regen)
NOTCH = [8, 6, 5, 4, 3, 2]
STEPS = [5603, 5654, 5716, 5888, 6337, 7706]
ENERGY = [912.47, 909.28, 896.94, 881.53, 834.13, 783.52]
RL_STEPS, RL_ENERGY = 7708, 784.16       # RL greedy eval (pace-penalty run)

# ── shared "advanced" styling ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfbfd",
    "axes.edgecolor": "#b8b8c0",
    "axes.linewidth": 1.1,
    "axes.grid": True,
    "grid.color": "#e3e3ea",
    "grid.linewidth": 0.9,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "medium",
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})
ACCENT, RLC, NTSC = "#2563eb", "#f59e0b", "#dc2626"


def load_links():
    cum, spd, grade = [0.0], [], []
    with open(LINKS) as f:
        f.readline(); f.readline()
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            cum.append(cum[-1] + float(p[3])); spd.append(float(p[4])); grade.append(float(p[6]))
    return np.array(cum), np.array(spd), np.array(grade)


def load_profile():
    speed, notch = [], []
    with open(PROFILE) as f:
        for row in csv.DictReader(f):
            speed.append(float(row["speed_mps"])); notch.append(int(row["notch"]))
    speed = np.array(speed)
    return np.cumsum(speed), speed, np.array(notch)


def load_nts():
    files = sorted(glob.glob(NTS_GLOB), key=os.path.getmtime)
    if not files:
        return None
    rows = list(csv.DictReader(open(files[-1])))
    g = lambda k: np.array([float(r[k]) for r in rows])
    return {
        "dist": g("TravelledDistance_m") / 1000.0,
        "speed": g("Speed_mps"),
        "limit": g("LinkMaxSpeed_mps"),
        "notch": g("FirstLocoNotchPosition"),
        "tractive": g("tractiveForce_N") / 1000.0,
        "resist": g("ResistanceForces_N") / 1000.0,
        "power": g("CurrentUsedTractivePower_kw"),
        "energy_cum": np.cumsum(g("EnergyConsumption_KWH")),
        "grade": g("GradeAtTip_Perc"),
        "steps": len(rows),
    }


def load_training(log_path):
    e, t, b = [], [], []
    pat = re.compile(r"Epoch #(\d+): test_reward: (-?[0-9.]+).*best_reward: (-?[0-9.]+)")
    for line in open(log_path):
        m = pat.search(line)
        if m:
            e.append(int(m.group(1))); t.append(float(m.group(2))); b.append(float(m.group(3)))
    return np.array(e), np.array(t), np.array(b)


def plot_tradeoff():
    nts = load_nts()
    fig, ax = plt.subplots(figsize=(9, 6))

    # shaded region under the constant-notch frontier
    ax.fill_between(STEPS, ENERGY, min(ENERGY) - 15, color=ACCENT, alpha=0.06, zorder=0)
    ax.plot(STEPS, ENERGY, "-", color=ACCENT, lw=2.2, alpha=0.55, zorder=2)
    sc = ax.scatter(STEPS, ENERGY, c=NOTCH, cmap="viridis", s=170, zorder=3,
                    edgecolors="white", linewidths=1.6)
    for n, s, en in zip(NOTCH, STEPS, ENERGY):
        ax.annotate(f"N{n}", (s, en), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=10, fontweight="bold", color="#333")
    cbar = fig.colorbar(sc, ax=ax, pad=0.015); cbar.set_label("throttle notch", fontsize=11)

    # RL greedy policy
    ax.scatter([RL_STEPS], [RL_ENERGY], color=RLC, s=320, marker="*", zorder=5,
               edgecolors="white", linewidths=1.5, label=f"RL policy  ({RL_ENERGY:.0f} kWh, {RL_STEPS:,} s)")
    # NeTrainSim native driver
    if nts:
        nts_e = float(nts["energy_cum"][-1]); nts_s = nts["steps"]
        ax.scatter([nts_s], [nts_e], color=NTSC, s=180, marker="D", zorder=5,
                   edgecolors="white", linewidths=1.5,
                   label=f"NeTrainSim driver  ({nts_e:.0f} kWh, {nts_s:,} s)")

    ax.set_xlabel("trip time  (s)"); ax.set_ylabel("total trip energy  (kWh)")
    ax.set_title("Energy vs Trip Time — ER9E on Toshkent→Ho'jakent\nGOST resistance, regenerative braking removed")
    ax.legend(loc="upper right", fontsize=10.5)
    ax.set_xlim(5350, 8050); ax.set_ylim(752, 930)
    p = os.path.join(OUTDIR, "energy_time_tradeoff.png"); fig.savefig(p); plt.close(fig); return p


def plot_netrainsim():
    d = load_nts()
    if d is None:
        return None
    x = d["dist"]
    fig, ax = plt.subplots(5, 1, figsize=(12, 11), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1.2, 1.6, 1.4, 1.2]})
    fig.suptitle(f"NeTrainSim trajectory — native driver  ({d['energy_cum'][-1]:.0f} kWh, {d['steps']:,} s, {x[-1]:.1f} km)",
                 fontsize=15, fontweight="bold", y=0.995)

    ax[0].plot(x, d["speed"], color=ACCENT, lw=1.1, label="train speed")
    ax[0].plot(x, d["limit"], color=NTSC, lw=1.0, alpha=0.7, ls="--", label="speed limit")
    ax[0].set_ylabel("speed (m/s)"); ax[0].legend(loc="upper right", fontsize=9.5)

    ax[1].step(x, d["notch"], where="post", color="#7c3aed", lw=1.1)
    ax[1].set_ylabel("notch"); ax[1].set_ylim(-0.5, 8.5); ax[1].yaxis.set_major_locator(MultipleLocator(2))

    ax[2].plot(x, d["tractive"], color="#059669", lw=1.0, label="tractive force")
    ax[2].plot(x, d["resist"], color="#b45309", lw=1.0, alpha=0.8, label="resistance")
    ax[2].axhline(0, color="#999", lw=0.6); ax[2].set_ylabel("force (kN)"); ax[2].legend(loc="upper right", fontsize=9.5)

    ax[3].plot(x, d["energy_cum"], color="#be123c", lw=1.6)
    ax[3].fill_between(x, d["energy_cum"], color="#be123c", alpha=0.10)
    ax[3].set_ylabel("cum. energy (kWh)")

    ax[4].fill_between(x, d["grade"], color="#6b7280", alpha=0.55, step="mid")
    ax[4].axhline(0, color="k", lw=0.6); ax[4].set_ylabel("grade (%)"); ax[4].set_xlabel("distance along route (km)")

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = os.path.join(OUTDIR, "netrainsim_trajectory.png"); fig.savefig(p); plt.close(fig); return p


def plot_profile():
    cum, spd_lim, grade = load_links()
    dist, speed, notch = load_profile()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
    axes[0].plot(dist / 1000, speed, color=ACCENT, lw=1, label="train speed")
    axes[0].plot(cum[1:] / 1000, spd_lim, color=NTSC, lw=1, alpha=.7, ls="--", label="speed limit")
    axes[0].set_ylabel("speed (m/s)"); axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"RL greedy policy along the route  ({RL_ENERGY:.0f} kWh, {RL_STEPS:,} s)")
    axes[1].step(dist / 1000, notch, where="post", color=RLC, lw=1)
    axes[1].set_ylabel("notch"); axes[1].set_ylim(-0.5, 8.5)
    axes[2].fill_between(cum[1:] / 1000, grade, color="#6b7280", alpha=.5, step="mid")
    axes[2].axhline(0, color="k", lw=0.6); axes[2].set_ylabel("grade (%)"); axes[2].set_xlabel("distance (km)")
    fig.tight_layout(); p = os.path.join(OUTDIR, "rl_policy_profile.png"); fig.savefig(p); plt.close(fig); return p


def plot_training():
    logs = sorted(glob.glob(os.path.join(_REPO, "logs", "train_run_*.log")), key=os.path.getmtime)
    if not logs:
        return None
    epoch, test, best = load_training(logs[-1])
    if len(epoch) == 0:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(epoch, test, color=ACCENT, lw=1, alpha=0.55, label="test reward")
    ax.plot(epoch, best, color="#059669", lw=2.2, label="best reward (running)")
    ax.set_xlabel("epoch"); ax.set_ylabel("episode reward")
    ax.set_title(f"Training progress — {os.path.basename(logs[-1])}")
    ax.legend(loc="lower right", fontsize=10)
    p = os.path.join(OUTDIR, "training_curve.png"); fig.savefig(p); plt.close(fig); return p


if __name__ == "__main__":
    for fn in (plot_tradeoff, plot_netrainsim, plot_profile, plot_training):
        out = fn()
        print(f"wrote {out}" if out else f"skipped {fn.__name__} (missing data)")
