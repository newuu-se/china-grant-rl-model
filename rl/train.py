"""
PPO training on NeTrainSimEnv using Tianshou 0.5.1.

Improvements over REINFORCE:
  - PPO clips policy updates (stable learning, no entropy collapse)
  - Critic (value network) reduces gradient variance vs pure REINFORCE
  - ent_coef keeps exploration alive throughout training (see ENT_COEF)
  - repeat_per_collect reuses each batch of episodes (more efficient)

Reward (defined in train_env.py): minimize trip energy subject to a schedule
deadline — uniform per-step time cost trades off against energy, no speed-limit
target. discount=0.999 so the deadline/arrival signals reach the early steps.

Run:
    source venv/bin/activate
    python rl/train.py 2>&1 | tee train_log.txt
"""

import os
import sys
import time

import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import NeTrainSimEnv, ARRIVAL_BONUS, DEADLINE_STEPS

# CPU is faster: network is tiny and inference runs one step at a time,
# so GPU kernel-launch overhead exceeds any compute gain.
DEVICE = "cpu"

NUM_TRAIN_ENVS = 8
NUM_TEST_ENVS  = 1

HIDDEN_SIZES = [256, 128, 64]
LR           = 3e-4
DISCOUNT     = 0.9999  # episodes are ~6k steps (1 step = 1 s); this is a near-undiscounted
                       # min-energy-to-goal problem. At 0.99 terminal signals vanished; at 0.999
                       # the discounting still distorted the long energy SUM and biased toward
                       # finishing sooner (= high notch, high energy — a run plateaued at ~965 kWh).
                       # 0.9999 (effective horizon ~10k > episode) keeps the energy objective
                       # ~undistorted and the arrival/timeout terminal signals visible.
ENT_COEF     = 0.004   # middle ground: 0.008 was too jumpy, 0.002 (+smoothness) collapsed to a
                       # constant notch. 0.004 keeps enough exploration for terrain-driven variation
                       # while the deterministic policy stays usable
MAX_EPOCH    = 100     # stable now (reward-norm); 100 epochs to let the pace-penalty policy
                       # converge to a clean on-schedule eco operating point

EPISODES_PER_COLLECT = NUM_TRAIN_ENVS   # 8 parallel episodes before each update
EPISODES_PER_TEST    = 1
REPEAT_PER_COLLECT   = 4                # PPO reuses each collected batch 4×
BATCH_SIZE           = 2048
STEP_PER_EPOCH       = 3_000   # env-steps/epoch. With CONTROL_INTERVAL=15 an episode is ~470
                               # env-steps (not ~7k), so 3k ≈ one 8-episode collect per epoch

# PPO / network architecture — single source of truth, also imported by evaluate.py
# so the eval-time policy can never drift from the trained one.
OBS_SHAPE     = (7,)
N_ACTIONS     = 9
EPS_CLIP      = 0.2     # clip ratio — prevents large destructive updates
VF_COEF       = 0.5     # value-loss weight
GAE_LAMBDA    = 0.95    # GAE smoothing for advantage estimates
MAX_GRAD_NORM = 0.5
ADV_NORM      = True    # advantage normalization
REWARD_NORM   = True    # ON — at γ=0.9999 returns are large (~-1000) sums; normalizing the
                        # returns stabilizes the value fn (the prior run diverged with it off)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")

_train_start = time.time()


def make_env():
    return NeTrainSimEnv()


def train_fn(epoch: int, env_step: int) -> None:
    elapsed = (time.time() - _train_start) / 60
    print(
        f"\n{'━'*65}\n"
        f"  Epoch {epoch:3d}/{MAX_EPOCH}  │  {env_step:>10,} steps  │  {elapsed:.1f} min elapsed\n"
        f"{'━'*65}",
        flush=True,
    )
    if epoch % 10 == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, f"policy_epoch{epoch:03d}.pth")
        torch.save(policy.state_dict(), path)
        print(f"  → checkpoint: {path}", flush=True)


def test_fn(epoch: int, env_step: int) -> None:
    print(f"  [test]", flush=True)


def save_best_fn(pol) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_path = os.path.join(CHECKPOINT_DIR, "policy_best.pth")
    torch.save(pol.state_dict(), best_path)
    print(f"  → new best saved: {best_path}", flush=True)


def build_policy(device: str = DEVICE):
    """Construct the PPO policy (and its actor/critic/optim) from the shared
    hyperparameters above. Used by both training and evaluation so the two can
    never diverge. Returns (policy, actor, critic, optim)."""
    net_actor  = Net(state_shape=OBS_SHAPE, hidden_sizes=HIDDEN_SIZES, device=device)
    net_critic = Net(state_shape=OBS_SHAPE, hidden_sizes=HIDDEN_SIZES, device=device)
    actor  = Actor(net_actor,  action_shape=N_ACTIONS, softmax_output=True, device=device).to(device)
    critic = Critic(net_critic, device=device).to(device)
    optim  = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        discount_factor=DISCOUNT,
        eps_clip=EPS_CLIP,
        advantage_normalization=ADV_NORM,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        gae_lambda=GAE_LAMBDA,
        reward_normalization=REWARD_NORM,
        max_grad_norm=MAX_GRAD_NORM,
    )
    return policy, actor, critic, optim


def main():
    global policy

    print("━" * 65)
    print("  PPO Training — NeTrainSim Energy Optimization")
    print("━" * 65)
    print(f"  Device       : {DEVICE}")
    print(f"  Train envs   : {NUM_TRAIN_ENVS} parallel C++ simulators")
    print(f"  Epochs       : {MAX_EPOCH}  (checkpoint every 10)")
    print(f"  Algorithm    : PPO  (eps_clip=0.2, ent_coef={ENT_COEF}, discount={DISCOUNT}, repeat={REPEAT_PER_COLLECT}×)")
    print(f"  Reward       : -energy/step  -time/step  -overspeed  +progress  +{ARRIVAL_BONUS:.0f} arrival (deadline {DEADLINE_STEPS} steps)")
    print("━" * 65 + "\n")

    train_envs = SubprocVectorEnv([make_env] * NUM_TRAIN_ENVS)
    test_envs  = SubprocVectorEnv([make_env] * NUM_TEST_ENVS)

    # Actor/critic/optim/policy from the shared factory (see build_policy).
    policy, actor, critic, optim = build_policy(DEVICE)

    buffer = VectorReplayBuffer(
        total_size=8_000 * NUM_TRAIN_ENVS,
        buffer_num=NUM_TRAIN_ENVS,
    )
    train_collector = Collector(policy, train_envs, buffer)
    test_collector  = Collector(policy, test_envs)

    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=MAX_EPOCH,
        step_per_epoch=STEP_PER_EPOCH,
        repeat_per_collect=REPEAT_PER_COLLECT,
        episode_per_test=EPISODES_PER_TEST,
        batch_size=BATCH_SIZE,
        episode_per_collect=EPISODES_PER_COLLECT,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        verbose=True,
        show_progress=True,
    )

    result = trainer.run()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    final_path = os.path.join(CHECKPOINT_DIR, "policy_final.pth")
    torch.save(policy.state_dict(), final_path)

    total_min = (time.time() - _train_start) / 60
    print(f"\n{'━'*65}")
    print(f"  Training complete in {total_min:.1f} min")
    print(f"  Best reward : {result['best_reward']:.3f}")
    print(f"  Final model : {final_path}")
    print(f"{'━'*65}")


if __name__ == "__main__":
    main()
