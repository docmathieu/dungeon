"""train_ppo.py — entraînement PPO (Stable-Baselines3) pour DungeonEnv.

Avantage vs DQN : algorithme on-policy avec mise à jour bornée (clip_range),
moins susceptible d'écraser les politiques précédentes → meilleure résistance
au catastrophic forgetting multi-seeds.

Utilisation :
    python src/train_ppo.py --timesteps 500000 --seed-pool 0,1,2,3,4,5,6,7,8,9
    python src/train_ppo.py --timesteps 200000 --seed 42
    python src/train_ppo.py --timesteps 500000 --seed-pool 0,1,...,99 --n-envs 4
    python src/train_ppo.py --timesteps 500000 --seed-pool 0,...,9 \\
                            --pretrained models/.../final.zip

Produits (regroupés dans un dossier {ts}_run/) :
    logs/{ts}_run/{ts}_{label}_ts{N}.jsonl
    models/{ts}_run/{ts}_{label}_ts{N}/ppo_{k}_steps.zip + final.zip
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(__file__))

from dungeon_env import DungeonEnv, ACTIONS, MAX_STEPS


# ---------------------------------------------------------------------------
# Hyperparamètres PPO
# ---------------------------------------------------------------------------

TIMESTEPS       = 500_000
N_ENVS          = 1
LEARNING_RATE   = 3e-4
N_STEPS         = 2048    # steps par env entre deux updates
BATCH_SIZE      = 64
N_EPOCHS        = 10
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
NET_ARCH        = [256, 128, 64]
CHECKPOINT_FREQ = 50_000  # timesteps entre checkpoints


# ---------------------------------------------------------------------------
# Environnement gymnasium
# ---------------------------------------------------------------------------

class DungeonGymEnv(gym.Env):
    """Wrapper gymnasium/SB3 autour de DungeonEnv.

    Observation : vecteur 304 floats (encode_obs_pure — sans seed one-hot).
    Action      : entier 0–3 → LEFT / RIGHT / UP / DOWN.
    """

    OBS_DIM = 304

    def __init__(self, seed: int | None = None, seed_pool: list[int] | None = None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        self._env = DungeonEnv(seed=seed, seed_pool=seed_pool, max_steps=MAX_STEPS)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs_dict = self._env.reset()
        return self._encode(obs_dict), {}

    def step(self, action: int):
        obs_dict, reward, done, info = self._env.step(ACTIONS[int(action)])
        obs            = self._encode(obs_dict)
        terminated     = bool(info["won"])
        truncated      = done and not terminated
        info["seed"]   = self._env.current_seed
        return obs, float(reward), terminated, truncated, info

    @staticmethod
    def _encode(obs_dict: dict) -> np.ndarray:
        """Encode l'observation en 304 floats (identique à encode_obs_pure)."""
        one_hot = np.zeros(300, dtype=np.float32)
        for i, tile in enumerate(obs_dict["grid"]):
            one_hot[i * 3 + tile] = 1.0
        cx, cy = obs_dict["char_pos"]
        ex, ey = obs_dict["exit_pos"]
        pos = np.array([cx / 9.0, cy / 9.0, ex / 9.0, ey / 9.0], dtype=np.float32)
        return np.concatenate([one_hot, pos])


# ---------------------------------------------------------------------------
# Callback : log JSON + affichage périodique
# ---------------------------------------------------------------------------

class LogCallback(BaseCallback):
    """Log une ligne JSON par épisode terminé + affichage toutes les 10k steps."""

    PRINT_FREQ   = 10_000
    WIN_RATE_WIN = 100   # fenêtre glissante pour le win rate affiché

    def __init__(self, log_path: Path, total_timesteps: int, t0: float):
        super().__init__()
        self._log_path        = log_path
        self._total           = total_timesteps
        self._t0              = t0
        self._ep              = 0
        self._last_print_ts   = 0
        self._recent_scores: deque[int] = deque(maxlen=self.WIN_RATE_WIN)
        self._ep_rewards: np.ndarray | None = None   # accumulateur reward par env
        self._log_file        = open(log_path, "w")

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_rewards = np.zeros(n, dtype=np.float32)
        meta = {
            "type": "meta",
            "hyperparams": {
                "timesteps":  self._total,
                "lr":         LEARNING_RATE,
                "n_steps":    N_STEPS,
                "batch_size": BATCH_SIZE,
                "n_epochs":   N_EPOCHS,
                "gamma":      GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "clip_range": CLIP_RANGE,
                "net_arch":   NET_ARCH,
            },
        }
        self._log_file.write(json.dumps(meta) + "\n")
        self._log_file.flush()

    def _on_step(self) -> bool:
        # Accumule les rewards de l'étape courante pour chaque env
        self._ep_rewards += self.locals["rewards"]

        for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            if done:
                won       = bool(info.get("won", False))
                score     = int(info.get("score", 0))
                ep_reward = round(float(self._ep_rewards[i]), 4)
                self._ep_rewards[i] = 0.0   # reset pour le prochain épisode
                self._ep += 1
                self._recent_scores.append(score)
                entry = {
                    "episode":  self._ep,
                    "timestep": self.num_timesteps,
                    "seed":     info.get("seed"),
                    "won":      won,
                    "score":    score,
                    "moves":    int(info.get("moves", 0)),
                    "reward":   ep_reward,
                }
                self._log_file.write(json.dumps(entry) + "\n")

        if self.num_timesteps - self._last_print_ts >= self.PRINT_FREQ:
            self._last_print_ts = self.num_timesteps
            win_rate = (
                sum(1 for s in self._recent_scores if s > 0) / len(self._recent_scores) * 100
                if self._recent_scores else 0.0
            )
            elapsed = time.time() - self._t0
            print(
                f"  ts {self.num_timesteps:>8}/{self._total}"
                f"  ep={self._ep}"
                f"  wr={win_rate:.1f}%"
                f"  t={elapsed:.0f}s"
            )
            self._log_file.flush()

        return True

    def _on_training_end(self) -> None:
        self._log_file.close()


# ---------------------------------------------------------------------------
# Nommage des runs (cohérence avec train.py)
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y%m%d_%H%M")


def _run_label(seed: int | None, seed_pool: list[int] | None) -> str:
    if seed_pool is not None:
        return f"ppo_pool{len(seed_pool)}"
    if seed is not None:
        return f"ppo_seed{seed}"
    return "ppo_random"


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def train(
    timesteps:  int                  = TIMESTEPS,
    seed:       int | None           = None,
    seed_pool:  list[int] | None     = None,
    lr:         float                = LEARNING_RATE,
    n_envs:     int                  = N_ENVS,
    pretrained: Path | None          = None,
    log_path:   Path                 = Path("logs/train_ppo.jsonl"),
    model_dir:  Path                 = Path("models/ppo"),
    verbose:    bool                 = True,
) -> PPO:
    """Entraîne un agent PPO. Retourne le modèle SB3 entraîné."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        (lambda s=seed, sp=seed_pool: DungeonGymEnv(seed=s, seed_pool=sp))
        for _ in range(n_envs)
    ]
    env = DummyVecEnv(env_fns)

    policy_kwargs = dict(net_arch=NET_ARCH)

    if pretrained is not None:
        model = PPO.load(str(pretrained), env=env, learning_rate=lr)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )

    t0 = time.time()
    callbacks = [
        LogCallback(log_path, timesteps, t0),
        CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // n_envs, 1),
            save_path=str(model_dir),
            name_prefix="ppo",
            verbose=0,
        ),
    ]

    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(str(model_dir / "final"))

    if verbose:
        elapsed = time.time() - t0
        print(f"Terminé — {timesteps} timesteps en {elapsed:.1f}s")

    env.close()
    return model


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraînement PPO — Dungeon POC")
    p.add_argument("--timesteps",  type=int,   default=TIMESTEPS,
                   help=f"Nombre de timesteps (défaut : {TIMESTEPS})")
    p.add_argument("--seed",       type=int,   default=None,
                   help="Seed fixe — même terrain à chaque épisode")
    p.add_argument("--seed-pool",  type=str,   default=None,
                   help="Pool de seeds, ex : 0,1,2,3 ou plage 0-99")
    p.add_argument("--lr",         type=float, default=LEARNING_RATE,
                   help=f"Learning rate (défaut : {LEARNING_RATE})")
    p.add_argument("--n-envs",     type=int,   default=N_ENVS,
                   help=f"Nombre d'environnements parallèles (défaut : {N_ENVS})")
    p.add_argument("--pretrained", type=Path,  default=None,
                   help="Checkpoint .zip SB3 à utiliser comme point de départ")
    return p.parse_args()


def _parse_pool(pool_str: str) -> list[int]:
    """Accepte '0,1,2,3' ou une plage '0-99'."""
    if "-" in pool_str and "," not in pool_str:
        start, end = pool_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in pool_str.split(",")]


if __name__ == "__main__":
    args  = _parse_args()
    pool  = _parse_pool(args.seed_pool) if args.seed_pool else None
    ts    = _now()
    label = _run_label(args.seed, pool)
    run   = f"{ts}_{label}_ts{args.timesteps}"
    run_dir = f"{ts}_run"

    train(
        timesteps  = args.timesteps,
        seed       = args.seed,
        seed_pool  = pool,
        lr         = args.lr,
        n_envs     = args.n_envs,
        pretrained = args.pretrained,
        log_path   = Path("logs")   / run_dir / f"{run}.jsonl",
        model_dir  = Path("models") / run_dir / run,
    )
