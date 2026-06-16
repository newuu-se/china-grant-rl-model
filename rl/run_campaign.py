"""
Reward-sensitivity sweep + ablation campaign (addresses reviewer comments #1, #2).

Launches many independent `rl.run_experiment` runs — each a (trip, config, seed)
training run that writes its own results.json — at a bounded parallelism, and is
fully resumable (a run whose results.json already exists is skipped). Aggregate
afterwards with rl/aggregate_campaign.py.

Matrix (ordered so the headline + variance results land first):
  1. w_P=2  (chosen) x 5 seeds, both trips        -> the main result + its CI
  2. w_P in {0,1,4,8} x 5 seeds, both trips        -> sensitivity curve
  3. ablations {no-smoothness, constant-entropy,
     no-time-features} x 5 seeds, forward          -> diagnostic chain
(no-pace is the w_P=0 sweep point, so it is not repeated.)

Usage:
    python rl/run_campaign.py                 # run/resume the full matrix
    python rl/run_campaign.py --parallel 2 --epochs 100 --seeds 5
    python rl/run_campaign.py --dry-run       # print the matrix and exit
"""
import argparse
import csv
import os
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_ROOT = os.path.join(_REPO, "checkpoints", "campaign")
LOG_ROOT = os.path.join(_REPO, "logs", "campaign")
MANIFEST = os.path.join(_REPO, "results", "campaign", "manifest.csv")


def build_matrix(seeds, epochs):
    """Return an ordered list of run specs: (name, extra_cli_args)."""
    runs = []

    def add(name, trip, seed, extra):
        runs.append((name, ["--trip", trip, "--seed", str(seed),
                            "--epochs", str(epochs), *extra]))

    # 1. chosen w_P=2 x seeds, both trips (headline + variance)
    for trip in ("forward", "return"):
        for s in seeds:
            add(f"sweep_{trip}_wp2_s{s}", trip, s, ["--wp", "2"])
    # 2. sweep other w_P, both trips
    for trip in ("forward", "return"):
        for wp in (0, 1, 4, 8):
            for s in seeds:
                add(f"sweep_{trip}_wp{wp}_s{s}", trip, s, ["--wp", str(wp)])
    # 3. ablations (forward), default w_P=2
    for s in seeds:
        add(f"abl_forward_nosmooth_s{s}",  "forward", s, ["--wp", "2", "--ws", "0"])
        add(f"abl_forward_constent_s{s}",  "forward", s, ["--wp", "2", "--constant-entropy"])
        add(f"abl_forward_notime_s{s}",    "forward", s, ["--wp", "2", "--ablate-time"])
    return runs


def is_done(name):
    return os.path.isfile(os.path.join(CKPT_ROOT, name, "results.json"))


def launch(name, extra):
    outdir = os.path.join(CKPT_ROOT, name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    logf = open(os.path.join(LOG_ROOT, f"{name}.log"), "w")
    cmd = [sys.executable, "-m", "rl.run_experiment", "--outdir", outdir, *extra]
    proc = subprocess.Popen(cmd, cwd=_REPO, stdout=logf, stderr=subprocess.STDOUT)
    return proc, logf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=2, help="max concurrent runs (8 envs each)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5, help="seeds 0..N-1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    matrix = build_matrix(seeds, args.epochs)
    todo = [(n, e) for (n, e) in matrix if not is_done(n)]
    done = [n for (n, _) in matrix if is_done(n)]

    print(f"Campaign: {len(matrix)} runs total | {len(done)} already done | "
          f"{len(todo)} to run | parallel={args.parallel} | epochs={args.epochs}")
    if args.dry_run:
        for n, e in matrix:
            print(f"  [{'done' if is_done(n) else 'todo'}] {n}: {' '.join(e)}")
        return
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)

    running = []   # (name, proc, logf, start)
    t0 = time.time()
    completed = len(done)
    i = 0
    while i < len(todo) or running:
        # fill the pool
        while len(running) < args.parallel and i < len(todo):
            name, extra = todo[i]; i += 1
            proc, logf = launch(name, extra)
            running.append((name, proc, logf, time.time()))
            print(f"[{time.strftime('%H:%M:%S')}] launched {name} "
                  f"({len(running)} running, {i}/{len(todo)} dispatched)", flush=True)
        # reap finished
        still = []
        for name, proc, logf, st in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, proc, logf, st)); continue
            logf.close()
            completed += 1
            mins = (time.time() - st) / 60
            ok = is_done(name) and rc == 0
            print(f"[{time.strftime('%H:%M:%S')}] {'✓' if ok else '✗ FAILED'} {name} "
                  f"({mins:.1f} min, rc={rc}) — {completed}/{len(matrix)} done", flush=True)
            with open(MANIFEST, "a", newline="") as f:
                csv.writer(f).writerow([name, "ok" if ok else "fail", rc, f"{mins:.1f}",
                                        time.strftime("%Y-%m-%d %H:%M:%S")])
        running = still
        if running:
            time.sleep(5)

    print(f"\nCampaign finished in {(time.time()-t0)/60:.1f} min. "
          f"Aggregate with: python rl/aggregate_campaign.py")


if __name__ == "__main__":
    main()
