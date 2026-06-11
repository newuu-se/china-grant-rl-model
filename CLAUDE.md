# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-based train energy optimization: a Tianshou PPO agent controls locomotive throttle
(notch 0–8) each second to minimize total energy consumption over a full A→B trip while
respecting speed limits and arriving within the scheduled time window.

Three components:
1. **NeTrainSim** (`NeTrainSim-adjusted/`) — C++/Qt6 train simulator (GOST resistance,
   regenerative braking removed, interactive RL mode via `-I`)
2. **RL layer** (`rl/`) — Python: Gymnasium env + Tianshou PPO training/eval scripts
3. **Docs** (`gymnasium-docs/`, `tianshou-docs/`) — API reference (do not modify)

## Repository Layout

```
data/
  netrainsim_v2/                  ACTIVE dataset — Toshkent→Ho'jakent, 1500 nodes/1499 links
    nodesFile_v2_fixed.dat        nodes
    linksFile_v2_fixed_speed.dat  links with real speed zones (RAW grades — has DEM spikes)
    linksFile_v2_clean.dat        ← what training uses: spike-cleaned grades
    trainsFile_rl.dat             ER9E EMU: 3 locos (1213 kW, 60 t, electric) + 3 cars = 373 t
  clean_grade_spikes.py           regenerates linksFile_v2_clean.dat from _fixed_speed
  coordinates.csv / data.csv      LEGACY v1 source data (750 nodes, diesel demo)
  generate_netrainsim_input.py    LEGACY — generates data/netrainsim/ (v1) only
  netrainsim/                     LEGACY v1 .dat files (not used by training)

NeTrainSim-adjusted/
  src/
    NeTrainSimConsole/main.cpp    CLI entry; `-I/--interactive` RL JSON loop lives here
    NeTrainSim/
      simulator.h / simulator.cpp Time-step loop, CSV output, runOneTimeStep API
      simulatorapi.h              C++ programmatic API (singleton)
      traindefinition/
        train.h / train.cpp       Physics; per-step energyStat = NEC − NER
        locomotive.h / locomotive.cpp  rlOverrideEnabled/rlOverrideThrottle injection,
                                       GOST resistance, regen removed (returns 0)
        car.cpp                   GOST resistance for cars
      network/
        readwritenetwork.cpp      Parses .dat files; speed in m/s, length in meters
        netlink.cpp               freeFlowSpeed stored and used as m/s
  build-mac.sh / build-linux.sh  Build scripts (macOS / Linux)

rl/
  train_env.py                   NeTrainSimEnv (gymnasium.Env, interactive subprocess)
  train.py                       Tianshou PPO training script (build_policy = source of truth)
  evaluate.py                    Loads checkpoint, runs greedy/stochastic episode → CSV
  run_baselines.py               Constant-notch energy/time frontier measurement
  make_plots.py                  Paper plots into results/plots/
  requirements.txt               Pinned deps matching installed venv

venv/                            Python virtualenv (gymnasium 1.3.0, tianshou 0.5.1, torch 2.x)
```

## Build & Run

### NeTrainSim (C++)

Linux (this machine): `./build-linux.sh` from `NeTrainSim-adjusted/` → binary at
`build-linux/src/NeTrainSimConsole/NeTrainSim`.
macOS: `brew install qt cmake`, then `./build-mac.sh`.

Run with the active data + trajectory export (run from inside `NeTrainSim-adjusted/`):
```bash
./build-linux/src/NeTrainSimConsole/NeTrainSim \
  -n ../data/netrainsim_v2/nodesFile_v2_fixed.dat \
  -l ../data/netrainsim_v2/linksFile_v2_clean.dat \
  -t ../data/netrainsim_v2/trainsFile_rl.dat \
  -o res -e true -p 1.0
```

CLI flags: `-n` nodes, `-l` links, `-t` trains, `-o` output dir, `-p` timestep in seconds
(default 1.0), `-z` enable optimizer, **`-I` interactive RL mode** (JSON over stdin/stdout).
**`-e true`** exports the trajectory CSV — `-e` takes a value, NOT a bare flag. `-e` alone uses
default `false` and produces no CSV.

### Data preparation

```bash
python data/clean_grade_spikes.py          # regenerates linksFile_v2_clean.dat (ACTIVE data)
python data/generate_netrainsim_input.py   # LEGACY: regenerates v1 data/netrainsim/*.dat
```

The v2 nodes/links/trains files are hand-derived (no generator); only the grade-cleaning
step is scripted. Re-run `clean_grade_spikes.py` if `linksFile_v2_fixed_speed.dat` changes,
then re-measure baselines: `python rl/run_baselines.py`.

### Python RL layer

```bash
source venv/bin/activate
pip install -r rl/requirements.txt   # if venv is fresh
python rl/train.py
```

Validate Gymnasium env compliance before training:
```bash
python -c "
import sys; sys.path.insert(0,'.')
from rl.train_env import NeTrainSimEnv
from gymnasium.utils.env_checker import check_env
check_env(NeTrainSimEnv())
print('OK')
"
```

## Data Files

### Source data (in `data/`) — LEGACY v1 dataset (active training data is `data/netrainsim_v2/`)

**`coordinates.csv`** — tab-separated, no header, 750 rows:
```
node_id   x_meters    y_meters
1         -27375.11   -21357.03
...
750        29742.51    18689.05
```
Total route: 74.87 km straight-line path (A=node 1, B=node 750).
Consecutive nodes are ~100 m apart.

**`data.csv`** — comma-separated, has header, 749 rows (one per link segment):
```
,Grade,Curvature,Speed limit
1,-0.145,0.042,1.0
...
```
- **Grade**: **per mille (‰)** — divided by 10 when writing to linksFile.dat so NeTrainSim sees %
  (NeTrainSim's Davis formula `20 × weight_tons × grade` expects grade in %). Max: ±6.28‰ = ±0.628%
- **Curvature**: unit passed through to NeTrainSim as-is
- **Speed limit**: **m/s** — values are km/h ÷ 3.6: 1.0, 3.0, 11.1(40), 16.6(60), 19.4(70), 22.2(80)

### Generated NeTrainSim input (in `data/netrainsim/`) — LEGACY v1 (format reference still valid for v2 files)

**`nodesFile.dat`** format (ASCII, tab-separated):
```
This is the node file of route1		
<count>  <xScale>  <yScale>           ← scales=1 (coords already in meters)
<id>  <x>  <y>  <isTerminal>  <dwellTime>  <desc>
```
Nodes 1 and 750 are marked `isTerminal=1`.

**`linksFile.dat`** format (ASCII, tab-separated):
```
This is the link file of route1     (many tabs)
<count>  <lengthScale>  <speedScale>   ← both=1
<id>  <from>  <to>  <length_m>  <speed_mps>  <signalNo>  <grade_pct>  <curvature>
       <directions>  <speedVariation>  <hasCatenary>
```
Link lengths are Euclidean distances between consecutive coordinate pairs (meters).
Speed is in m/s (confirmed in netlink.cpp: `length/freeFlowSpeed` gives seconds).
Grade is stored as percent (%) — generator divides data.csv ‰ values by 10.
`directions=1` (unidirectional A→B), `hasCatenary=0` (diesel).

**`trainsFile.dat`** format (ASCII):
```
Automatic Trains Definition
1                                           ← number of trains
1  <path>  <startTime>  <frictionCoef>  <loco_defs>  <car_defs>
```
Loco field order: `Count, Power(kW), TransmissionEff, NoOfAxles, AirDragCoeff, FrontalArea(m²), Length(m), GrossWeight(t), Type`
Car field order:  `Count, NoOfAxles, AirDragCoeff, FrontalArea(m²), Length(m), GrossWeight(t), TareWeight(t), Type`
Current config: 1 loco (5000 kW, 90 t, 4 axles) + 4 cars (50 t each) = ~290 t total.
Locomotive default max speed: 33.33 m/s (100/3); route speed limits are the binding constraint.

## Architecture

### C++ Simulation Engine

**Entry point:** `src/NeTrainSimConsole/main.cpp` — QCommandLineParser, calls
`SimulatorAPI::ContinuousMode::createNewSimulationEnvironmentFromFiles()`, then
`sim->runSimulation()` (blocking).

**Core classes:**
- **`Simulator`** (`simulator.h/cpp`) — time-step loop: `runSimulation()` calls
  `runOneTimeStep()` → `playTrainOneTimeStep()` → writes one CSV row per train per step.
  Exposes `pauseSimulation()` / `resumeSimulation()` / `runOneTimeStep()` (Q_INVOKABLE).
- **`Network`** (`network/network.h`) — nodes + links + signals; train route is fixed.
- **`Train`** (`traindefinition/train.h`) — `getStepAcceleration()` → `moveTrain()`;
  holds `optimumThrottleLevels` queue (built-in optimizer injection point).
- **`Locomotive`** (`locomotive.h`) — `currentLocNotch` (int 0–8), `throttleLevel`
  (double 0–1). Notch-throttle mapping: quadratic `(N/8)^2`.

**Trajectory CSV columns** (written per timestep when `-e` flag used):
```
TrainNo, TStep_s, TravelledDistance_m, Acceleration_mps2, Speed_mps,
LinkMaxSpeed_mps, EnergyConsumption_KWH, DelayTimeToEach_s, DelayTime_s,
Stoppings, tractiveForce_N, ResistanceForces_N, CurrentUsedTractivePower_kw,
GradeAtTip_Perc, CurvatureAtTip_Perc, FirstLocoNotchPosition, optimizationEnabled
```
`EnergyConsumption_KWH` is the **per-step** net energy consumed this timestep (kWh), not cumulative.
It equals `train->energyStat = NEC - NER` where NEC/NER are reset at the start of each step
(`resetTrainEnergyConsumption`, train.cpp). NER ≡ 0 since regen removal.
Values range ~0–1.4 kWh depending on throttle; a full trip totals ~760–860 kWh (clean data).
The RL env tracks `_cum_energy_kwh` in Python by summing per-step values.
NOTE: energy is charged from *virtual* tractive power `(m·a + R)·v` — with impossible grades
(the raw v2 file's ±17% DEM spikes) this exceeded installed power 3×; clean data keeps it sane.

### Integration Strategy

**Phase 2 — interactive per-step control (IMPLEMENTED, the only mode in use):**
- `main.cpp` `-I/--interactive`: loops `read action JSON from stdin → set
  rlOverrideThrottle on every locomotive → runOneTimeStep() → write state JSON
  to stdout` (line-prefixed `NTS_JSON `).
- The RL override (`locomotive.cpp`) bypasses both the built-in A* optimizer and
  the discretized throttle governor: `getThrottleLevel()` returns
  `rlOverrideThrottle`, `updateLocNotch()` is a no-op, `train->optimize=false`.
  Notch→throttle mapping applied in main.cpp: `(notch/Nmax)²` — identical to
  `defineThrottleLevels()`.
- `NeTrainSimEnv.reset()` spawns the binary; a bootstrap `{"notch": 0}` fetches
  the initial state. A reader thread + queue (not select+readline) consumes
  stdout — immune to buffered-line races.
- Env-level action repeat: each `step()` holds the notch for CONTROL_INTERVAL=15
  simulator seconds (~430 decisions per on-schedule trip).

**stdout state JSON (simulator → Python):**
```json
{
  "timestep": 42, "speed_mps": 15.3, "position_m": 1230.0,
  "grade_perc": 1.2, "curvature_perc": 0.0, "remaining_dist_m": 73639.6,
  "energy_kwh": 0.4, "link_max_speed_mps": 11.1, "terminated": false, "notch": 3
}
```
`energy_kwh` is the energy of THIS step only (see trajectory CSV note above).
**stdin action JSON (Python → simulator):** `{"notch": 6}`

(Phase 1 — replaying a pre-computed trajectory CSV — is obsolete and removed.)

### Gymnasium Environment (`rl/train_env.py`)

**Observation space** (9 floats, `Box`, all normalized ~[-1,1]/[0,1]):
```
[speed_mps, position_m, grade_perc, curvature_perc,
 remaining_dist_m, energy_kwh, link_max_speed_mps,
 time_frac, behind_frac]
```
`time_frac` = step/DEADLINE_STEPS, `behind_frac` = fraction of route behind the
deadline pace. These make the clock visible — the pace penalty depends on the
step counter, so without them the reward would be non-Markov in the observation.
Normalization denominators (`_GRADE_MAX=4.5`, `_ENERGY_MAX=1.5`, …) are
calibrated to the CLEAN v2 data; recheck them whenever physics/data change.

**Action space:** `Discrete(9)` — notch 0–8, held for CONTROL_INTERVAL=15 sim-seconds
per decision (action repeat; credit assignment was intractable at 1 s decisions).

**Reward (per sim-second, summed over the 15-s decision)** — objective is
*minimize trip energy subject to the schedule deadline*:
```
r = -W_ENERGY * energy_kwh                          # energy this step       (W_ENERGY=3.0)
  - W_PACE * behind_frac                            # lagging deadline pace  (W_PACE=2.0; 0 when on pace)
  - W_OVERSPEED * max(0, speed - link_limit)        # safety net; sim caps speed  (1.0)
  + ARRIVAL_BONUS   on terminated                   # +200
  - TIMEOUT_PENALTY on truncated                    # -1500
plus, once per decision: -W_SMOOTH * |Δnotch|       # smoothness         (0.15)
```
The per-step pace penalty (not a terminal late fee) is what stops the
crawl-at-notch-1 failure mode; zero penalty when on/ahead of schedule preserves
the freedom to coast. Eco target = slowest on-schedule constant notch ≈ n3
(789 kWh / 6,442 s on clean data); the agent should beat it by modulating with
grade/speed zone.

**Episode boundaries:**
- `terminated=True`: simulator reports the train reached its destination, or
  `position_m >= TOTAL_ROUTE_LENGTH_M` (74,891.29 m)
- `truncated=True`: `_step_count >= MAX_STEPS` (12,000) without arriving

### Tianshou PPO (`rl/train.py`) — tianshou 0.5.1 API

`build_policy()` in `rl/train.py` is the single source of truth (imported by
`evaluate.py` so train/eval can never drift):

```python
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic  # use Actor, not DiscreteActor

# separate actor/critic nets, HIDDEN_SIZES=[256,128,64], OBS_SHAPE=(9,), 9 actions
policy = PPOPolicy(actor, critic, optim, dist_fn=torch.distributions.Categorical,
                   discount_factor=0.9999, eps_clip=0.2, ent_coef=0.004,
                   gae_lambda=0.95, vf_coef=0.5, max_grad_norm=0.5,
                   advantage_normalization=True, reward_normalization=True)
```
**Hard-won hyperparameter rationale (don't regress these):**
- `discount=0.9999`: episodes are ~6k sim-steps of energy SUM; at γ≤0.999 the
  discounting biases toward finishing fast (high notch). Near-undiscounted keeps
  the energy objective honest.
- `reward_normalization=True` required for stability at that γ (returns ~±1000).
- `ent_coef=0.004`: 0.008 leaves the argmax parked at the energy floor; 0.002
  collapses to a constant notch.
- `episode_per_collect` (not `step_per_collect`): full episodes per update.
- `train_fn` fires at epoch START, `test_fn` after the epoch's updates —
  checkpoints are saved in `test_fn` so labels match content.

## RL Design Decisions

| Dimension | Choice | Rationale |
|-----------|--------|-----------|
| Action space | `Discrete(9)` notch 0–8, 15 s action repeat | Matches simulator internals; coarse control makes long-horizon credit assignment tractable |
| State space | 9 normalized features (7 physics + 2 schedule) | Pace-penalty reward needs the clock in the obs (Markov) |
| Reward | Dense per-step energy + pace penalty + terminal | Terminal-only deadline was ignored; raw progress reward biased toward speed |
| Algorithm | PPO (`PPOPolicy`) | REINFORCE diverged; PPO + reward-norm stable |
| Parallelism | `SubprocVectorEnv(8)` | Each env = one C++ subprocess |
| Control mode | Phase 2 interactive (stdin/stdout JSON) | Real per-step physics control |

## C++ Modifications (Phase 2 — DONE)

Implemented in:
- `src/NeTrainSimConsole/main.cpp` — `-I` flag, JSON loop, notch→`(N/Nmax)²` throttle,
  sets `rlOverrideEnabled/rlOverrideThrottle` + `currentLocNotch` on all locomotives,
  `train->optimize=false`, drains Qt events each step.
- `src/NeTrainSim/traindefinition/locomotive.{h,cpp}` — override fields;
  `getThrottleLevel()` returns the override; `updateLocNotch()` no-op under override.
- Physics edits (2026-06-10): GOST/SI resistance `w0 = 1.1 + 0.01v + 0.000227v²`
  (v in km/h) in `Car::getResistance` and `Locomotive::getResistance`; grade term
  `trackGrade(%)×10×W_kN`; curve term `|curv°|×(700/1746.4)×W_kN`; regenerative
  braking removed (`getRegenerativeEffeciency` returns 0).

Rebuild after C++ changes: `cd NeTrainSim-adjusted && ./build-linux.sh`

## Key Data Facts

- Route: 74.89 km (Toshkent → Ho'jakent line), 1500 nodes, 1499 links, ~50 m per segment.
  Active input files: `data/netrainsim_v2/{nodesFile_v2_fixed, linksFile_v2_clean, trainsFile_rl}.dat`
- Train: ER9E electric multiple unit — 3 locos (1213 kW, 60 t, type 1 = electric) +
  3 cars (2×78 t + 37 t) = 373 t; links have `hasCatenary=1`
- Speed limits: 11.11, 16.67, 19.44, 22.22 m/s (40, 60, 70, 80 km/h).
  Distance per zone: 31.0 km @ 40, 15.6 km @ 60, 4.0 km @ 70, 24.3 km @ 80.
- Grades: RAW v2 file had DEM noise spikes to ±17% (links 887/888, 1204/1205…);
  `linksFile_v2_clean.dat` (median-filtered elevation, net climb 346.6 m preserved)
  spans −2.7…+4.5%. ALWAYS use the clean file — the spikes break force balance
  and triple apparent power draw.
- Trip-time bounds (clean data): theoretical floor ≈ 5,025 s (speed-limit-bound); fastest
  feasible ≈ 5,602 s (constant notch 8); RL schedule deadline = 6,500 s (1 step = 1 second)
- Constant-notch energy/time (clean data, `rl/run_baselines.py`, 2026-06-11):
  n8=5,602 s/862 kWh, n6=5,652/853, n5=5,721/849, n4=5,910/834, n3=6,442/789,
  n2=8,492/758 — lower notch = less energy, more time. Eco target ≈ n3.
- The simulator hard-caps speed at each link's freeFlowSpeed, so over-speeding is rare
- At standstill (v=0) the loco applies full adhesion force regardless of notch
  (original NeTrainSim behavior) — the train cannot be parked mid-route
- NeTrainSim speed unit in .dat files: **m/s** (confirmed in netlink.cpp)
- NeTrainSim .dat files: ASCII text, tab-separated, NOT binary

## Testing & Validation

```bash
# 1. Regenerate clean grades (only needed if the raw links file changed)
python data/clean_grade_spikes.py

# 2. Verify simulator runs with the active data (run from inside NeTrainSim-adjusted/)
cd NeTrainSim-adjusted && mkdir -p res
./build-linux/src/NeTrainSimConsole/NeTrainSim \
  -n ../data/netrainsim_v2/nodesFile_v2_fixed.dat \
  -l ../data/netrainsim_v2/linksFile_v2_clean.dat \
  -t ../data/netrainsim_v2/trainsFile_rl.dat \
  -o res -e true -p 1.0        # note: -e takes 'true', not a bare flag
ls res/trainTrajectory_*.csv

# 3. Validate Gymnasium env
python -c "
import sys; sys.path.insert(0,'.')
from rl.train_env import NeTrainSimEnv
from gymnasium.utils.env_checker import check_env
check_env(NeTrainSimEnv())
print('OK')
"

# 4. Measure constant-notch baselines (recalibrates reward comments/plots)
python rl/run_baselines.py

# 5. Run training (use tmux for long runs), then evaluate
python rl/train.py
python rl/evaluate.py          # → results/notch_profile.csv (prefers policy_best.pth)
```
