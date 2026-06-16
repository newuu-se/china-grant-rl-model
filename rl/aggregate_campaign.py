"""
Aggregate the campaign into the statistics the paper reports (reviewer #1, #2):
per-configuration mean +/- std and 95% CI across seeds, a reward-sensitivity
curve vs w_P, an ablation comparison, and permutation-test p-values. Reads every
checkpoints/campaign/<name>/results.json that exists (works on a partial run).

No scipy dependency: 95% CIs use a small hardcoded Student-t table; significance
uses a two-sided permutation test (assumption-light, appropriate for n=5).

Outputs to results/campaign/: sweep_summary.csv, ablation_summary.csv,
significance.csv; figures fig_sensitivity.png and fig_ablation.png into
results/plots/ and latex/figures/.

Usage:  python rl/aggregate_campaign.py [--metric stochastic|deterministic]
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_ROOT = os.path.join(_REPO, "checkpoints", "campaign")
OUTDIR = os.path.join(_REPO, "results", "campaign")
DEADLINE_S = 6500

# two-sided 95% t-multipliers by dof (n-1); fall back to 1.96 for large n
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def ci95(vals):
    a = np.asarray(vals, float)
    n = len(a)
    if n <= 1:
        return (float(a.mean()) if n else float("nan"), 0.0, 0.0, float("nan"))
    mean, sd = a.mean(), a.std(ddof=1)
    half = _T95.get(n - 1, 1.96) * sd / np.sqrt(n)
    return float(mean), float(sd), float(half), n


def perm_test(a, b, iters=20000, seed=0):
    """Two-sided permutation test on the difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(iters):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs - 1e-12:
            count += 1
    return count / iters


def load_runs(metric):
    runs = []
    for rj in glob.glob(os.path.join(CKPT_ROOT, "*", "results.json")):
        with open(rj) as f:
            d = json.load(f)
        ev = d["eval"][metric]
        cfg = d["config"]
        runs.append({
            "trip": d["trip"], "seed": d["seed"],
            "w_pace": cfg["w_pace"], "w_smooth": cfg["w_smooth"],
            "ablate_time": cfg["ablate_time"], "constant_entropy": cfg["constant_entropy"],
            "best_reward": d["best_reward"],
            "energy": ev["energy_kwh"], "steps": ev["steps"],
            "lateness": ev["lateness_frac"], "on_schedule": int(ev["on_schedule"]),
        })
    return runs


def is_baseline(r):
    return (r["w_pace"] == 2.0 and r["w_smooth"] == 0.15
            and not r["ablate_time"] and not r["constant_entropy"])


def summarize(runs, keyfn, fields):
    groups = {}
    for r in runs:
        groups.setdefault(keyfn(r), []).append(r)
    rows = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = {"group": key, "n": len(rs)}
        for fld in fields:
            m, sd, half, _ = ci95([r[fld] for r in rs])
            row[f"{fld}_mean"] = round(m, 3)
            row[f"{fld}_std"] = round(sd, 3)
            row[f"{fld}_ci95"] = round(half, 3)
        row["on_sched_frac"] = round(np.mean([r["on_schedule"] for r in rs]), 2)
        rows.append((key, rs, row))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []                      # union of keys, preserving first-seen order
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["stochastic", "deterministic"], default="stochastic")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    runs = load_runs(args.metric)
    if not runs:
        print(f"No results.json found under {CKPT_ROOT} yet.")
        return
    print(f"Loaded {len(runs)} runs ({args.metric} eval).")

    FIELDS = ["energy", "steps", "lateness", "best_reward"]

    # ── sweep: group by (trip, w_pace), only the non-ablated sweep runs ────
    sweep = [r for r in runs if r["w_smooth"] == 0.15 and not r["ablate_time"]
             and not r["constant_entropy"]]
    sweep_rows = summarize(sweep, lambda r: (r["trip"], r["w_pace"]), FIELDS)
    write_csv(os.path.join(OUTDIR, "sweep_summary.csv"), [row for _, _, row in sweep_rows])

    # ── ablations (forward): baseline vs each ablation, with p-values ──────
    fwd = [r for r in runs if r["trip"] == "forward"]
    base = [r for r in fwd if is_baseline(r)]
    abl_groups = {
        "baseline": base,
        "no_smoothness": [r for r in fwd if r["w_smooth"] == 0.0 and not r["ablate_time"] and not r["constant_entropy"]],
        "constant_entropy": [r for r in fwd if r["constant_entropy"]],
        "no_time_features": [r for r in fwd if r["ablate_time"]],
        "no_pace (w_P=0)": [r for r in fwd if r["w_pace"] == 0.0 and r["w_smooth"] == 0.15 and not r["ablate_time"] and not r["constant_entropy"]],
    }
    abl_rows = []
    for name, rs in abl_groups.items():
        if not rs:
            continue
        row = {"ablation": name, "n": len(rs)}
        for fld in ("energy", "steps", "lateness", "best_reward"):
            m, sd, half, _ = ci95([r[fld] for r in rs])
            row[f"{fld}_mean"] = round(m, 2); row[f"{fld}_ci95"] = round(half, 2)
        row["on_sched_frac"] = round(np.mean([r["on_schedule"] for r in rs]), 2)
        if base and name != "baseline":
            row["p_energy_vs_base"] = round(perm_test([r["energy"] for r in base],
                                                      [r["energy"] for r in rs]), 4)
            row["p_lateness_vs_base"] = round(perm_test([r["lateness"] for r in base],
                                                        [r["lateness"] for r in rs]), 4)
        abl_rows.append(row)
    write_csv(os.path.join(OUTDIR, "ablation_summary.csv"), abl_rows)

    # ── significance across the sweep: each w_P vs chosen w_P=2 ────────────
    sig_rows = []
    for trip in ("forward", "return"):
        base2 = [r for r in sweep if r["trip"] == trip and r["w_pace"] == 2.0]
        for wp in sorted({r["w_pace"] for r in sweep if r["trip"] == trip}):
            if wp == 2.0:
                continue
            grp = [r for r in sweep if r["trip"] == trip and r["w_pace"] == wp]
            sig_rows.append({
                "trip": trip, "w_pace": wp, "n": len(grp),
                "p_energy_vs_wp2": round(perm_test([r["energy"] for r in base2],
                                                   [r["energy"] for r in grp]), 4),
                "p_lateness_vs_wp2": round(perm_test([r["lateness"] for r in base2],
                                                     [r["lateness"] for r in grp]), 4),
            })
    write_csv(os.path.join(OUTDIR, "significance.csv"), sig_rows)

    _plots(sweep_rows, abl_groups, base)
    print(f"Wrote {OUTDIR}/ (sweep_summary, ablation_summary, significance) + figures.")
    # console preview
    print("\n— sweep (energy mean±CI95 / on-schedule frac) —")
    for key, _, row in sweep_rows:
        print(f"  {key}: {row['energy_mean']:.1f}±{row['energy_ci95']:.1f} kWh, "
              f"{row['steps_mean']:.0f}s, on-sched {row['on_sched_frac']}")


def _plots(sweep_rows, abl_groups, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # sensitivity: energy & lateness vs w_P, per trip, with 95% CI bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {"forward": "#2563eb", "return": "#059669"}
    for trip in ("forward", "return"):
        pts = sorted([(key[1], row) for key, _, row in sweep_rows if key[0] == trip])
        if not pts:
            continue
        wp = [p[0] for p in pts]
        en = [p[1]["energy_mean"] for p in pts]; enc = [p[1]["energy_ci95"] for p in pts]
        lt = [p[1]["lateness_mean"] * 100 for p in pts]; ltc = [p[1]["lateness_ci95"] * 100 for p in pts]
        axes[0].errorbar(wp, en, yerr=enc, marker="o", lw=1.6, capsize=3,
                         color=colors[trip], label=trip)
        axes[1].errorbar(wp, lt, yerr=ltc, marker="o", lw=1.6, capsize=3,
                         color=colors[trip], label=trip)
    axes[0].set_xlabel("pace weight $w_P$"); axes[0].set_ylabel("trip energy (kWh)")
    axes[0].set_title("Energy vs $w_P$ (mean $\\pm$ 95% CI, 5 seeds)", fontsize=10.5)
    axes[1].set_xlabel("pace weight $w_P$"); axes[1].set_ylabel("lateness (% past deadline)")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("Schedule violation vs $w_P$", fontsize=10.5)
    for ax in axes:
        ax.axvline(2.0, color="red", ls="--", lw=1.1, alpha=0.7)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    for d in (os.path.join(_REPO, "results", "plots"), os.path.join(_REPO, "latex", "figures")):
        if os.path.isdir(d):
            fig.savefig(os.path.join(d, "fig_sensitivity.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ablation bars (forward): energy and lateness with CI
    order = ["baseline", "no_smoothness", "constant_entropy", "no_time_features", "no_pace (w_P=0)"]
    present = [(k, abl_groups[k]) for k in order if abl_groups.get(k)]
    if present:
        fig, ax2 = plt.subplots(1, 2, figsize=(11, 4.2))
        labels = [k.replace("_", "\n") for k, _ in present]
        for ax, fld, title, scale in ((ax2[0], "energy", "Trip energy (kWh)", 1.0),
                                      (ax2[1], "lateness", "Lateness (% past deadline)", 100.0)):
            means, errs = [], []
            for _, rs in present:
                m, sd, half, _ = ci95([r[fld] for r in rs])
                means.append(m * scale); errs.append(half * scale)
            bars = ax.bar(range(len(present)), means, yerr=errs, capsize=4,
                          color=["#2563eb"] + ["#9ca3af"] * (len(present) - 1))
            ax.set_xticks(range(len(present))); ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(title, fontsize=10.5); ax.grid(alpha=0.3, axis="y")
        fig.suptitle("Forward-trip ablations (mean $\\pm$ 95% CI, 5 seeds)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        for d in (os.path.join(_REPO, "results", "plots"), os.path.join(_REPO, "latex", "figures")):
            if os.path.isdir(d):
                fig.savefig(os.path.join(d, "fig_ablation.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
