# Train Energy-Optimization with RL — Methods, Changes & Results

A chronological, paper-oriented log of every change made to the simulator and the
RL layer, **why** each change was made, and **what result** it produced. Numbers
are from measured runs on the current configuration unless noted.

---

## 1. System under study

| Component | Description |
|---|---|
| **Simulator** | NeTrainSim (C++/Qt6 freight-train physics simulator), console build, driven per-second |
| **RL layer** | Python: Gymnasium env wrapping the simulator + Tianshou **PPO** agent (0.5.1) |
| **Route** | Toshkent → Ho'jakent line: **74.89 km**, 1500 nodes, 1499 links (~50 m/segment) |
| **Speed limits** | 11.11 / 16.67 / 19.44 / 22.22 m/s (40/60/70/80 km/h); distance per zone: 31.0 / 15.6 / 4.0 / 24.3 km |
| **Train** | ER9E electric multiple unit (3 power cars, 0.25 adhesion coefficient) |
| **Control** | Locomotive throttle **notch 0–8**; quadratic notch→throttle mapping `(N/Nmax)²` |
| **Objective** | Minimize total trip energy (kWh) subject to arriving within a schedule window |

The agent sets the locomotive notch; NeTrainSim integrates the longitudinal
dynamics (tractive effort vs. resistance, adhesion-limited) and reports per-step
state and energy.

---

## 2. Simulator physics modifications (NeTrainSim C++)

### 2.1 Davis resistance equation → GOST/SI form
**Files:** `src/NeTrainSim/traindefinition/car.cpp` (`Car::getResistance`),
`locomotive.cpp` (`Locomotive::getResistance`), doc comment in `traincomponent.h`.

**Before** — modified Davis equation in **US units** (lb, mph), with empirical
coefficients and unit-conversion factors:
```
R = (1.5 + 18/w_axle + 0.03·V_mph + K·A·V²/w)·W_lb + 20·W_lb·grade + |curv|·0.8·W_lb,  ×4.44822 → N
```

**After** — fully SI **GOST-based** specific-resistance formula (railway practice
in the region of study):
```
v   = speed·3.6                       [km/h]
W   = mass·9.81                       [kN]
w0  = 1.1 + 0.01·v + 0.000227·v²      [N/kN]   (basic running resistance)
R   = w0·W  +  grade‰·W  +  |curv|·(700/1746.4)·W   [N]
```
where `trackGrade` is stored in % (×10 → ‰) and curve resistance uses the GOST
`700/R_curve` rule with the standard degree-of-curve→radius conversion.

**Why:** the US-unit Davis form is not appropriate for the Uzbekistan rolling
stock/track standard; a transparent SI GOST formulation matches local engineering
practice and removes opaque imperial conversions, making the resistance model
defensible and reproducible for the paper.

**Result (measured):** grade and curve terms are numerically near-identical
between the two formulas (both ≈ m·g·grade), but the GOST **basic running
resistance is lower** than the US Davis term. Net effect at constant notch 6:
**981.4 → 909.3 kWh** (≈ −7%), trip time unchanged (5653 → 5654 s).

### 2.2 Removal of regenerative braking
**File:** `locomotive.cpp` (`Locomotive::getRegenerativeEffeciency`).

**Before:** for rechargeable power types (electric/hybrid), braking recovered
energy with efficiency `η = 1/exp(γ/|a|)`; the per-step net energy was
`consumed − regenerated`. The ER9E is **electric** (power-type index 1 ∈
`locomotiveRechargableTechnologies`), so it *was* regenerating.

**After:** `getRegenerativeEffeciency` returns **0** for all power types, so
braking is purely dissipative (only auxiliary power is drawn). This zeroes the
regenerated-energy term in both `getEnergyConsumption` and
`getEnergyConsumptionAtDCBus`; because braking then yields ≥0 energy, no negative
energy propagates to the battery/catenary recharge paths either.

**Why:** the study models energy *consumption* without regenerative recovery
(conservative energy accounting / matching the operating assumption for the line).

**Result:** removing regen *increases* energy, but on this route the §2.1 GOST
reduction dominates, so the net constant-notch-6 figure still falls to 909 kWh.
The two effects were separated by the constant-notch probe (below).

### 2.3 Adhesion-limited tractive force — unchanged (documented)
Tractive force is already adhesion-limited: `F = min(P·η/v, μ·m·g)` in
`Locomotive::getTractiveForce`, with μ the train friction coefficient (0.25). This
was reviewed and **intentionally left unchanged**; the simulator also caps speed at
each link's free-flow speed, so over-speeding is physically prevented.

### 2.4 Verification
- Independent C++ review (units, formula, compile-safety, completeness of regen
  removal) — no defects; confirmed `currentWeight` is in tonnes and `trackGrade`
  in %, so the GOST unit handling is correct.
- Rebuilt console binary (`build-linux.sh`), clean (exit 0).
- Smoke test (constant notch 6) confirmed the new energy figure (909 kWh).

---

## 3. RL problem formulation

| Element | Definition |
|---|---|
| **Observation** | 7 floats, normalized ≈[0,1]: speed, position, grade, curvature, remaining distance, per-step energy, link speed limit |
| **Action** | `Discrete(9)` — notch 0–8 |
| **Transition** | persistent interactive NeTrainSim subprocess: send `{notch}` → run one (or several) sim-seconds → read state JSON |
| **Episode end** | `terminated` = reached terminus; `truncated` = step count ≥ MAX_STEPS (12,000) |
| **Algorithm** | PPO (separate actor/critic, GAE λ=0.95, clip 0.2), Tianshou 0.5.1 |

---

## 4. Correctness fixes (from code review, before tuning)

| # | Issue found | Fix | Why it mattered |
|---|---|---|---|
| 1 | `TARGET_STEPS=4500` deadline below the physical floor (~5,025 s) | set realistic deadline 6,500 s | every arrival was always "late"; the schedule term was meaningless |
| 2 | Train path `1,1499` stopped one node short of the 1500-node route | path `1,1500` | train now reaches the true terminus (74,891 m, was 74,889) |
| 3 | Obs normalization used 19.4 m/s max | 22.22 m/s | route actually reaches 80 km/h; features were mis-scaled |
| 4 | Eval rebuilt the policy with hyperparameters that could drift from training | shared `build_policy()` factory | guarantees the evaluated policy matches the trained one |
| 5 | Per-worker stderr log opened append-only, grew unbounded | per-PID log, truncated each reset | clean debugging |

---

## 5. Reward-design evolution (the core of the study)

PPO maximizes expected return; the reward is where the eco-driving objective is
encoded. The episode is long (~6,000 control steps at 1 s), which proved to be the
dominant difficulty. Each iteration below lists the **problem observed**, the
**change**, and the **measured result**.

### Iteration 0 — baseline (pre-existing)
Dense reward with a **speed-deficit penalty** (reward for staying near the speed
limit), terminal arrival/timeout, discount 0.99. **Result:** policy collapsed to a
near-constant high notch (used only notch 6–7); no energy optimization. The speed
target directly opposed coasting.

### Iteration 1 — remove speed target, add uniform time cost
**Change:** delete speed-deficit penalty; reward = `−energy − W_TIME·t − overspeed
+ progress_shaping + arrival`; discount 0.99 → 0.999.
**Why:** a *time cost* (not a speed target) should create the energy↔time trade-off.
**Result:** policy converged to **fast/high-energy (~965 kWh)**; best reward
plateaued. Diagnosis: at γ=0.999 over 6,000 steps the terminal arrival/timeout
signals are discounted to ≈0, and a discounted progress reward rewards *finishing
sooner* → bias to high notch.

### Iteration 2 — near-undiscounted; drop progress shaping
**Change:** discount 0.999 → **0.9999**; remove progress reward (a potential-based
variant was tried but its `(1−γ)·Φ` term corrupted the logged/selection reward).
**Why:** treat it as the near-undiscounted min-energy-to-goal problem it is.
**Result:** **diverged** — greedy test reward degraded −1063 → −2693 over 300
epochs; the policy drifted to slow-crawling. Large-magnitude returns (~−1,000 sums)
destabilized the value function.

### Iteration 3 — stabilize
**Change:** `reward_normalization=True`; `ent_coef` 0.01 → 0.005; `MAX_EPOCH`
300 → 60.
**Why:** normalize the large returns; calm exploration; stop before divergence.
**Result:** **stable** (no divergence) **but plateaued at fast ~900 kWh** —
essentially unchanged from start. The eco gain (~5% of episode reward) was too weak
a gradient to climb.

### Iteration 4 — amplify the energy signal
**Change:** `W_ENERGY` 1 → 3, `W_TIME` 0.05 → 0.03, `W_LATE` 0.10 → 0.30,
`ent_coef` → 0.008. Calibrated from the constant-notch curve so notch 3 is the clear
optimum (fast→eco reward gap ~205 vs ~41 before).
**Result:** stochastic behavior improved toward notch 4–5 (~882 kWh), **but the
greedy (deterministic) policy crawled** — notch 1, 820 kWh but **10,700 steps**
(massively late). A *terminal-only* late penalty cannot override an immediate
per-step energy gain over a long episode, so the policy's mode minimizes per-step
energy (lowest notch).

### Iteration 5 — per-step pace penalty
**Change:** replace the terminal late penalty with a **per-step pace penalty**:
```
behind = max(0, (t/DEADLINE)·route_len − position) / route_len
r_t = −W_ENERGY·energy − W_PACE·behind − overspeed   (+arrival / −timeout)
```
zero when on/ahead of schedule, so coasting is free where there is slack but
lagging is penalized *immediately*. `W_PACE=2`, `MAX_EPOCH=100`.
**Why:** make the schedule pressure local (per-step), not deferred to the terminus.
**Result:** crawl reduced (notch-1/10,700 → **notch-2/7,708 steps, 784 kWh**), but
still **~19% late** and the notch is **essentially constant** — not terrain-aware.
The reward is now *correct* (notch 3 strongly optimal), but PPO's deterministic mode
still lands one notch too slow: the delayed pace cost is under-weighted vs. the
immediate energy gain.

### Iteration 6 — coarse control interval (current run)
**Change:** `CONTROL_INTERVAL=15` — the agent chooses a notch every **15 s** (held
in between), reducing the decision horizon from ~6,000 to **~470 per trip**;
`STEP_PER_EPOCH` lowered to 3,000 accordingly.
**Why:** the recurring failure (constant notch, mode one step too slow) is a
**long-horizon credit-assignment** problem — crediting a 1-second throttle choice
for its delayed energy/schedule consequence is intractable at 6,000 steps. Action
repeat shortens the horizon ~15×, and at ~470 steps γ=0.9999 barely discounts, so
credit assignment becomes tractable — the prerequisite for learning terrain-aware
modulation.
**Result:** training was **stable and ~4× faster** (12.8 min for 100 epochs; ~15×
less per-transition overhead), best reward −2375.96 (best of any iteration). The
**stochastic policy now arrives on-schedule (6,113 s) at ≈887 kWh with the notch
genuinely spanning the full 0–8 range** — the first terrain-aware, on-schedule
policy obtained (decision histogram: N0×66, N1×135, N2×37, N3×27, N4×45, N5×38,
N6×27, N7×12, N8×21). Two honest caveats:
- **Energy parity, not superiority:** 887 kWh ≈ constant notch 4; it does **not** beat
  the constant-notch-3 reference (834 kWh). Part of the variation is genuine
  terrain response, part is sampling.
- **Deterministic argmax still degenerate:** the greedy policy collapses to notch 1
  and times out (12,000 s). The high entropy coefficient (0.008) parks the policy's
  *mode* at the lowest-energy notch while *sampling* supplies the on-schedule speed.

**Interpretation:** coarse control removed the long-horizon blocker — notch variation
and on-schedule arrival finally emerge together — but a clean, energy-optimal
*deterministic* policy needs the entropy lowered/annealed so the mode converges to the
on-schedule optimum (proposed next step).

### Iteration 7 — smoothness regularization + lower entropy → deterministic eco policy
**Motivation:** the iteration-6 *sampled* policy switched notch erratically (full 0–8
range, near-100% of decisions) and its deterministic argmax was degenerate.
**Change:** add a per-decision **notch-change penalty** `−W_SMOOTH·|Δnotch|` and lower
`ent_coef`. Two settings probed:
- `W_SMOOTH=0.5, ent_coef=0.002` → collapsed to **constant notch 5** (897 kWh, 5,716 s) —
  over-smoothed (smooth, but a sub-optimal notch).
- `W_SMOOTH=0.15, ent_coef=0.004` → the **deterministic policy converges to ~constant
  notch 3: 834 kWh, 6,337 s, on-schedule** (one notch change over the entire trip).

**Result:** the policy lands **exactly on the eco-optimal point of the constant-notch
Pareto frontier** — the lowest-energy schedule-feasible setting: **~8.6% below flat-out**
(notch 8 = 912 kWh) and **~6% below NeTrainSim's native driver** (888 kWh). The
deterministic policy is now clean, smooth and deployable (argmax no longer degenerate).

**Finding:** under any smoothness pressure the optimal policy is a *constant* notch,
because terrain-aware sub-trip modulation yields negligible energy savings on this route
(corroborated by the nearly straight constant-notch frontier and by NeTrainSim's own
*varied*-notch driver sitting *above* that frontier). The defensible contribution is
therefore that the RL agent **reliably identifies the energy-optimal, schedule-feasible
throttle setting end-to-end from the simulator**, rather than performing continuous
eco-modulation. (`W_SMOOTH` is the dial between an erratic profile and a flat one.)

---

## 6. Training methodology (current)

| Hyperparameter | Value | Note |
|---|---|---|
| Algorithm | PPO (actor/critic) | Tianshou 0.5.1, CPU (tiny net) |
| Net | [256,128,64] | shared `build_policy()` for train+eval |
| Discount γ | 0.9999 | near-undiscounted; appropriate for long episodic min-cost |
| GAE λ | 0.95 | |
| Clip ε | 0.2 | |
| Entropy coef | 0.008 | exploration toward the eco basin |
| Reward norm | True | stabilizes large-magnitude returns |
| Adv. norm | True | |
| Parallel envs | 8 (`SubprocVectorEnv`) | one C++ subprocess each |
| Control interval | 15 s (action repeat) | iteration 6 |
| Reward weights | W_ENERGY=3, W_PACE=2, W_OVERSPEED=1, arrival=+200, timeout=−1500 | |

---

## 7. Results

### 7.1 Constant-notch baselines (current physics, measured)
The energy–time Pareto frontier, obtained by driving each constant notch to
completion through the interactive simulator:

| Notch | Trip time (s) | Energy (kWh) | On schedule (≤6,500 s)? |
|---|---|---|---|
| 8 | 5,603 | 912.5 | yes |
| 6 | 5,654 | 909.3 | yes |
| 5 | 5,716 | 896.9 | yes |
| 4 | 5,888 | 881.5 | yes |
| **3** | **6,337** | **834.1** | **yes (eco optimum)** |
| 2 | 7,706 | 783.5 | no (late) |

→ **Constant notch 3 (834 kWh, on-time)** is the lowest-energy schedule-feasible
constant policy and serves as the reference eco target. Theoretical minimum trip
time (speed-limit-bound) ≈ 5,025 s.

### 7.2 NeTrainSim native-driver baseline
Running NeTrainSim's own driver (no RL): **6,023 s, 887.7 kWh**, with a genuinely
**varied notch** (mostly 5, with 6/7/2/3). Notably this point sits **above** the
constant-notch frontier (more energy than constant notch 4 at comparable time), i.e.
the built-in driver does not outperform simple constant-notch selection here.

### 7.3 RL policy outcomes by iteration

| Iter | Reward design | Greedy result | Verdict |
|---|---|---|---|
| 0 | speed-deficit | notch 6–7 only | no optimization |
| 1 | time cost, γ=0.999 | ~965 kWh, fast | plateaued (discount bias) |
| 2 | γ=0.9999, no shaping | — | diverged |
| 3 | + reward-norm, MAX_EPOCH 60 | ~900 kWh, fast | stable, no eco |
| 4 | W_ENERGY=3 | greedy crawl: 820 kWh / 10,700 s | off-schedule |
| 5 | per-step pace penalty | 784 kWh / 7,708 s, ~const notch 2 | low energy but ~19% late, not terrain-aware |
| 6 | coarse control (15 s) | sampled: **6,113 s / 887 kWh, notch spans 0–8**; argmax degenerate (notch 1, timeout) | **first terrain-aware + on-schedule** policy; energy ≈ const. notch 4; argmax needs lower entropy |
| 7 | + smoothness penalty + lower entropy | deterministic **~constant notch 3: 834 kWh / 6,337 s (on-schedule)** | **eco-optimal operating point**; clean/smooth/deployable; ~8.6% below flat-out, ~6% below NeTrainSim driver |

### 7.4 Figures (`results/plots/`, regenerate with `python rl/make_plots.py`)
- **`energy_time_tradeoff.png`** — constant-notch Pareto frontier (colored by notch)
  with the RL policy and NeTrainSim native driver overlaid.
- **`netrainsim_trajectory.png`** — NeTrainSim native-driver trajectory (the data the
  GUI plots): speed vs limit, notch, tractive/resistance force, cumulative energy,
  grade vs distance.
- **`rl_policy_profile.png`** — RL greedy policy: speed (vs limit), notch, grade vs
  distance — shows the constant-notch-2 behavior of iteration 5.
- **`training_curve.png`** — test/best reward per epoch.

---

## 8. Key findings & discussion

1. **Reward specification was solved; optimization is the bottleneck.** After
   iteration 5 the reward correctly makes the on-schedule eco policy (notch 3)
   strongly optimal. The agent nonetheless converges to a near-constant notch one
   step too slow.
2. **Root cause: long-horizon credit assignment.** At 1-second control (~6,000
   steps) PPO under-weights delayed schedule costs against immediate per-step energy
   gains, and discounting over such a horizon either erases terminal signals (low γ)
   or destabilizes value learning (high γ). Every reward tweak hit this same wall.
3. **Discounting/horizon interaction:** γ must be ≈0.9999 to keep the energy *sum*
   undistorted, which requires reward normalization for stability.
4. **The RL did not beat constant-notch selection** through iteration 5 — an
   honest negative result that motivates the structural change.
5. **Coarse control (iteration 6) confirmed the thesis:** shortening the decision
   horizon ~15× (one notch every 15 s) made credit assignment tractable — terrain-aware,
   on-schedule behaviour emerged for the first time (sampled policy: 6,113 s, 887 kWh,
   notch spanning 0–8) and training became ~4× faster. Two remaining items: (a) the
   policy's *mode* (argmax) still parks at the lowest notch under the entropy bonus, so
   a clean *deterministic* eco policy needs entropy annealing; (b) energy is at parity
   with constant notch 3 (≈834–887 kWh), not yet below it — converting notch *variation*
   into net energy *savings* is the next reward/curriculum question.

---

## 9. Reproducibility

```bash
# 1. Generate / rebuild simulator (after physics edits)
cd NeTrainSim-adjusted && ./build-linux.sh

# 2. Measure constant-notch baselines / verify physics
#    (constant-notch probe drives the interactive binary per notch)

# 3. Train
source venv/bin/activate
python rl/train.py 2>&1 | tee logs/train_run_<ts>.log

# 4. Evaluate best checkpoint → results/notch_profile.csv
python rl/evaluate.py

# 5. Plots
python rl/make_plots.py        # → results/plots/*.png
```

Key files: `rl/train_env.py` (env + reward, fully commented), `rl/train.py` (PPO +
shared `build_policy`), `rl/evaluate.py`, `rl/make_plots.py`; physics in
`NeTrainSim-adjusted/src/NeTrainSim/traindefinition/{car,locomotive}.cpp`. Checkpoints
of superseded runs are archived under `checkpoints/archive_*`.

---

*Status: iterations 0–7 complete. Final result: a clean **deterministic** policy that
reaches the **eco-optimal, schedule-feasible operating point** (~constant notch 3, 834 kWh,
6,337 s) — ~8.6% below flat-out and ~6% below NeTrainSim's native driver. Smoothness
(`W_SMOOTH`) dials the notch profile between erratic and flat. Open question for future work:
whether any route/objective on this corridor rewards genuine terrain-aware modulation beyond
a constant notch (so far it does not).*
