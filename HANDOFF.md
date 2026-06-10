# Session Handoff — 2026-06-09

Written so you can resume from home. (Updated 2026-06-10 — **training has finished**.)

## ⚠️ RESULT — training finished (287 min, 300 epochs) and it DIVERGED
- **Best checkpoint = epoch 30** (`policy_best.pth`, reward −1009.5 ≈ **~900 kWh, near-schedule**,
  ~8% better than the fast baseline of 981 kWh). best_reward never improved after epoch 30.
- **Training was unstable**: greedy test reward got steadily *worse* after epoch 30
  (−1394 → −2973 → −2693 by epoch 300). The **final** policy now **crawls** — ~11,300 steps to
  arrive vs the 6,500 deadline — using *more* energy AND far more time than constant notch 3.
  So it over-corrected from the previous "too fast" run into "too slow / diverged."
- **Use `policy_best.pth`, NOT `policy_final.pth`.** (`evaluate.py` already prefers best.)
- The mobile push did not reach you — **Remote Control was inactive** — so only a desktop
  notification fired. This file + `results/training_summary_20260609_224708.txt` are the record.

### Do this first (from home)
```bash
python rl/evaluate.py        # uses policy_best.pth → results/notch_profile.csv
cut -d, -f4 results/notch_profile.csv | tail -n +2 | sort | uniq -c   # notch histogram
```
If the notch histogram shows real spread (not just 2 values) and total energy is well under
965 kWh, the best checkpoint is a genuine eco policy worth keeping.

### To fix the instability on the next run (my recommendation)
The policy degraded after epoch 30 → instability, plus weak schedule enforcement let it drift slow:
1. `rl/train.py`: turn **`REWARD_NORM = True`** (returns are large ~−1000 sums at γ=0.9999; normalizing stabilizes the value fn).
2. `rl/train.py`: lower **`ENT_COEF` 0.01 → 0.005** (0.01 let it wander).
3. `rl/train_env.py`: strengthen the schedule so crawling isn't cheap — **`W_TIME` 0.05 → 0.10**, **`W_LATE` 0.10 → 0.5**.
4. Consider **`MAX_EPOCH` 300 → ~60** (best was epoch 30; the extra 270 epochs only made it worse), or `DISCOUNT` 0.9999 → 0.9995 as a stability/accuracy middle ground.
Re-run, then re-check the notch histogram. I can apply all four and relaunch when you're back.

## TL;DR
- A PPO training run is live in tmux session **`rltrain`** (started 17:59, ~2.5–3 h for 300 epochs).
- A detached watcher in tmux **`rltrain-save`** will write `results/training_summary_*.txt` + a log copy the moment training ends — this works even if the Claude session is closed.
- This session reviewed the project, fixed a list of bugs, and **redesigned the RL reward** after two earlier runs failed to learn energy-saving.

## What's running right now
| tmux session | what | notes |
|---|---|---|
| `rltrain` | `python rl/train.py` (PPO, 300 epochs) | log: `logs/train_run_20260609_175937.log` |
| `rltrain-save` | `rl/_train_watcher.sh` | writes summary when training reaches a terminal state |
| `1` | pre-existing unrelated session | leave alone |

### Check progress from home
```bash
# SSH into this same machine, then:
tmux attach -t rltrain                 # live view (detach: Ctrl-b then d)
tail -f logs/train_run_20260609_175937.log
tmux ls                                # confirm rltrain + rltrain-save still up
```
As of handoff: **epoch 7/300**, energy easing 981→957 kWh, best_reward ≈ −1054.

## What success looks like (how to judge when done)
The goal is **eco-driving**: minimize trip energy while arriving on schedule.
- **reward climbs** from ~−1054 toward −990 and beyond (reward = −energy − 0.05·steps + 200 arrival; higher = less energy).
- **energy drops** below ~965 toward the constant-notch-3 baseline (~868 kWh) **or lower**.
- After training: `python rl/evaluate.py` writes `results/notch_profile.csv`. Success = **notch varies with grade/speed-zone** (the previous broken run used only notch 6 & 7).

### Constant-notch baselines (measured this session)
| notch | steps (s) | energy (kWh) |
|---|---|---|
| 8 | 5,600 | 986.9 |
| 6 | 5,653 | 981.4 |
| 4 | 5,936 | 935.8 |
| 3 | 6,446 | 868.0 |
| 2 | 7,897 | 820.1 |
Theoretical min trip time ≈ 5,025 s; schedule deadline set to 6,500 s.

## Changes made this session (all in working tree, NOT committed)
**Bug fixes from the review:**
- `rl/train_env.py`: removed speed-deficit penalty; realistic deadline 6,500 steps (was an impossible 4,500); normalization fixed to 22.22 m/s; per-process stderr log.
- `rl/train.py` + `rl/evaluate.py`: shared `build_policy()` factory so train/eval hyperparameters can't drift; fixed stale prints.
- `data/netrainsim_v2/trainsFile_rl.dat`: path `1,1499` → `1,1500` (full route; train now reaches 74,891 m).
- `CLAUDE.md`: refreshed stale facts (1499 links, ER9E, 22.22 m/s, reward block).

**Reward redesign (the main story):**
- Two earlier runs converged to "drive fast, high energy" (~965 kWh) and never coasted. Root cause: **discounting distorts the ~6,000-step energy sum** — at γ≤0.999, finishing sooner outweighed the energy saved, so high notch won. A progress-reward made it worse; a potential-based variant fixed the bias but corrupted the logged/selection reward.
- **Final design (now running):** γ = **0.9999** (near-undiscounted min-energy-to-goal), **no progress shaping**, ent_coef = **0.01**. Reward = `−energy − 0.05·time − overspeed (+200 arrival / −1500 timeout)`. This is a clean, interpretable proxy for the objective, and `best_reward` now correctly prefers low-energy trips.
- Rationale is documented in code comments in `rl/train.py` (DISCOUNT) and `rl/train_env.py` (reward block).

## Notifications
- **Phone push on finish/crash** requires this Claude session to be **alive** (it's driven by a persistent monitor). If you close the work session, the push won't fire — but the **watcher still writes `results/training_summary_*.txt`**, so that file is the reliable record.
- When you resume from home (`claude --resume` in this repo), if training is still going, ask me to **re-arm the push**; if it already finished, read the summary file.

## Next steps (from home)
1. Confirm `rltrain` still running; tail the log.
2. When done: `python rl/evaluate.py` → inspect `results/notch_profile.csv` for notch variety + total energy vs the baselines above.
3. If eco-driving emerged: decide whether to **commit** the run (code + checkpoints + results). If it plateaued again, the next lever is more entropy or revisiting the time-cost weight.

## Housekeeping notes
- `git status` shows ~31 checkpoint files as **deleted** — they were *moved*, not lost, into `checkpoints/archive_pre_redesign_*/` (old reward) and `checkpoints/archive_plateau_*/` (the failed run). Fresh checkpoints land in `checkpoints/`.
- `rl/_train_watcher.sh` is a temporary helper (untracked).
- Nothing has been committed; the working tree holds all changes for your review.

## Resume the chat
```bash
cd /home/asilbek/china-grant-rl-model
claude --resume          # pick this session from the list
```
