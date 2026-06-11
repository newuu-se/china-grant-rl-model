"""
Measure the constant-notch energy/time frontier on the CURRENT physics + data.

Run after any physics or input-data change; these numbers calibrate the reward
comments in train_env.py, DEADLINE_STEPS, and the plot constants in
make_plots.py.

Usage:
    source venv/bin/activate
    python rl/run_baselines.py [notches...]   # default: 2 3 4 5 6 8
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import NeTrainSimEnv


def run_constant_notch(notch: int):
    env = NeTrainSimEnv()
    env.reset()
    terminated = truncated = False
    max_step_energy = 0.0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(notch)
        max_step_energy = max(max_step_energy,
                              float(env._last_state["energy_kwh"]))
    steps, energy = env._step_count, env._cum_energy_kwh
    env.close()
    return steps, energy, max_step_energy, terminated


def main():
    notches = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5, 6, 8]
    print(f"{'notch':>5}  {'steps(s)':>9}  {'energy(kWh)':>12}  "
          f"{'maxE/step':>9}  {'arrived':>7}")
    results = []
    for n in notches:
        steps, energy, max_e, arrived = run_constant_notch(n)
        results.append((n, steps, energy))
        print(f"{n:>5}  {steps:>9,}  {energy:>12.2f}  {max_e:>9.3f}  "
              f"{str(arrived):>7}")
    print("\nmake_plots.py constants:")
    print(f"NOTCH  = {[r[0] for r in results]}")
    print(f"STEPS  = {[r[1] for r in results]}")
    print(f"ENERGY = {[round(r[2], 2) for r in results]}")


if __name__ == "__main__":
    main()
