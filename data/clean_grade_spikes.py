#!/usr/bin/env python3
"""
Remove DEM elevation-noise spikes from the v2 links file grades.

Problem: linksFile_v2_fixed_speed.dat contains physically impossible grades
(up to ±17% on 50 m segments, in adjacent opposite-sign pairs, e.g. links
887/888 = -12.4%/+17.0%). Real rail grades top out around 3-4%. These are
single-node elevation errors from the DEM source. They poison the physics:
on a +17% link the grade resistance (~620 kN) exceeds the train's total
adhesion (~440 kN), force balance breaks, and NeTrainSim's virtual-power
energy accounting charges up to 3.76 kWh in one second (11.8 MW from a
3.6 MW consist).

Method (elevation-preserving):
  1. Integrate grades into an elevation profile: elev[i+1] = elev[i] + g%/100 * len.
  2. Median-filter (window 3) the interior elevations N passes — removes
     isolated single-node spikes exactly, leaves monotone slopes untouched,
     endpoints pinned so the route's net climb is preserved bit-for-bit.
  3. One light moving-average pass (window 3, interior only) to soften the
     residual staircase.
  4. Re-derive grades; write linksFile_v2_clean.dat (only column 7 changes).

Usage: python data/clean_grade_spikes.py
"""

import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(DATA_DIR, "netrainsim_v2", "linksFile_v2_fixed_speed.dat")
DST = os.path.join(DATA_DIR, "netrainsim_v2", "linksFile_v2_clean.dat")

MEDIAN_PASSES = 3   # enough to collapse 1-2-node-wide spikes
SMOOTH_PASSES = 1


def median3(a, b, c):
    return sorted((a, b, c))[1]


def main():
    with open(SRC) as f:
        lines = f.read().splitlines()

    header, count_line, rows = lines[0], lines[1], lines[2:]
    parsed = []  # (fields list, length, grade)
    for line in rows:
        p = line.split("\t")
        if len(p) < 8:
            continue
        parsed.append(p)

    lengths = [float(p[3]) for p in parsed]
    grades = [float(p[6]) for p in parsed]
    n = len(parsed)

    # 1. integrate to elevation (meters); grade column is percent
    elev = [0.0]
    for g, L in zip(grades, lengths):
        elev.append(elev[-1] + g / 100.0 * L)
    net_before = elev[-1]

    # 2. median filter interior nodes (endpoints pinned)
    for _ in range(MEDIAN_PASSES):
        e = elev[:]
        for i in range(1, len(elev) - 1):
            e[i] = median3(elev[i - 1], elev[i], elev[i + 1])
        elev = e

    # 3. light moving average, interior only
    for _ in range(SMOOTH_PASSES):
        e = elev[:]
        for i in range(1, len(elev) - 1):
            e[i] = (elev[i - 1] + elev[i] + elev[i + 1]) / 3.0
        elev = e

    # 4. re-derive grades
    new_grades = [
        (elev[i + 1] - elev[i]) / lengths[i] * 100.0 for i in range(n)
    ]

    for p, g in zip(parsed, new_grades):
        p[6] = f"{g:.9f}"

    with open(DST, "w") as f:
        f.write(header + "\n")
        f.write(count_line + "\n")
        for p in parsed:
            f.write("\t".join(p) + "\n")

    gmax_b, gmin_b = max(grades), min(grades)
    gmax_a, gmin_a = max(new_grades), min(new_grades)
    big_b = sum(1 for g in grades if abs(g) > 3)
    big_a = sum(1 for g in new_grades if abs(g) > 3)
    print(f"links: {n}")
    print(f"grade range before: [{gmin_b:.2f}, {gmax_b:.2f}] %   |g|>3%: {big_b}")
    print(f"grade range after : [{gmin_a:.2f}, {gmax_a:.2f}] %   |g|>3%: {big_a}")
    print(f"net elevation: before {net_before:.3f} m, after {elev[-1] - elev[0]:.3f} m (preserved)")
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
