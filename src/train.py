"""train.py — boucle d'entraînement DQN pour DungeonEnv.

Utilisation :
    python src/train.py                                        # 5000 épisodes, terrain aléatoire
    python src/train.py --episodes 2000 --seed 42              # seed fixe, 2000 épisodes
    python src/train.py --episodes 2000 --seed-pool 0,1,2,...  # pool de seeds fixes
    python src/train.py --episodes 2000 --lr 3e-4              # learning rate réduit
    python src/train.py --episodes 2000 --seed-pool 0,1,2,... \
                        --lr 3e-4 \
                        --pretrained models/20260527_1222_seed42_ep2000/final.pt

Produits :
    logs/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}].jsonl   une ligne JSON par épisode
    models/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}]/      dossier du run
        ep500.pt … ep<N>.pt                                      checkpoints périodiques
        final.pt                                                 checkpoint final
"""

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Permet l'import depuis src/ quand lancé comme script
sys.path.insert(0, os.path.dirname(__file__))

from dungeon_env import DungeonEnv, ACTIONS
from model import DQNetwork, INPUT_DIM


# ---------------------------------------------------------------------------
# Hyperparamètres
# ---------------------------------------------------------------------------

EPISODES           = 5_000
MAX_STEPS          = 100       # identique à DungeonEnv.MAX_STEPS
BATCH_SIZE         = 64
LEARNING_RATE      = 1e-3
GAMMA              = 0.99      # facteur d'actualisation des récompenses futures
EPSILON_START      = 1.0       # exploration initiale (100 % aléatoire)
EPSILON_END        = 0.05      # exploration minimale (5 % aléatoire)
EPSILON_DECAY      = 0.995     # valeur de référence (remplacée par decay adaptatif dans train())
BUFFER_SIZE        = 10_000    # capacité du replay buffer
TARGET_UPDATE_FREQ = 100       # synchronisation réseau cible (épisodes)
CHECKPOINT_FREQ    = 500       # sauvegarde checkpoint (épisodes)


# ---------------------------------------------------------------------------
# Encodage de l'observation
# ---------------------------------------------------------------------------

def encode_obs(obs: dict) -> torch.Tensor:
    """Convertit une observation DungeonEnv en tenseur float32 de taille 304.

    Encodage :
        grille    : one-hot 100×3 = 300 floats  (index = tile_idx*3 + type)
        char_pos  : (x/9, y/9)   =   2 floats  normalisés dans [0, 1]
        exit_pos  : (x/9, y/9)   =   2 floats  normalisés dans [0, 1]
    """
    one_hot = [0.0] * 300
    for i, tile in enumerate(obs["grid"]):   # tile ∈ {0=herbe, 1=roche, 2=eau}
        one_hot[i * 3 + tile] = 1.0

    cx, cy = obs["char_pos"]
    ex, ey = obs["exit_pos"]
    features = one_hot + [cx / 9.0, cy / 9.0, ex / 9.0, ey / 9.0]
    return torch.tensor(features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """Buffer circulaire FIFO stockant les transitions (s, a, r, s', done)."""

    def __init__(self, capacity: int = BUFFER_SIZE):
        self._buf = collections.deque(maxlen=capacity)

    def push(
        self,
        state:      torch.Tensor,
        action:     int,
        reward:     float,
        next_state: torch.Tensor,
        done:       bool,
    ) -> None:
        self._buf.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        """Tire batch_size transitions au hasard (sans remise).
        Lève ValueError si le buffer contient moins de batch_size éléments.
        """
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def capacity(self) -> int:
        return self._buf.maxlen


# ---------------------------------------------------------------------------
# Agent DQN
# ---------------------------------------------------------------------------

class DQNAgent:
    """Agent DQN : epsilon-greedy + réseau cible + apprentissage par batch."""

    def __init__(
        self,
        lr:         float = LEARNING_RATE,
        gamma:      float = GAMMA,
        epsilon:    float = EPSILON_START,
        eps_end:    float = EPSILON_END,
        eps_decay:  float = EPSILON_DECAY,
    ):
        self.gamma      = gamma
        self.epsilon    = epsilon
        self._eps_end   = eps_end
        self._eps_decay = eps_decay

        self.q_net      = DQNetwork()
        self.target_net = DQNetwork()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn    = nn.MSELoss()

    def select_action(self, state: torch.Tensor) -> int:
        """Epsilon-greedy : explore aléatoirement ou exploite le réseau."""
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))
        with torch.no_grad():
            q_values = self.q_net(state.unsqueeze(0))
            return int(q_values.argmax(dim=1).item())

    def learn(
        self, buffer: ReplayBuffer, batch_size: int = BATCH_SIZE
    ) -> float | None:
        """Un pas de gradient depuis un batch du replay buffer.

        Retourne la loss (float) ou None si le buffer est trop petit.
        """
        if len(buffer) < batch_size:
            return None

        batch      = buffer.sample(batch_size)
        states     = torch.stack([t.state      for t in batch])
        actions    = torch.tensor([t.action    for t in batch], dtype=torch.long)
        rewards    = torch.tensor([t.reward    for t in batch], dtype=torch.float32)
        next_states = torch.stack([t.next_state for t in batch])
        dones      = torch.tensor([t.done      for t in batch], dtype=torch.float32)

        # Q(s, a) estimé par le réseau principal
        q_current = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Cible de Bellman : r + γ × max_a' Q_target(s', a')  (annulé si terminal)
        with torch.no_grad():
            q_next   = self.target_net(next_states).max(dim=1).values
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self) -> None:
        """Réduit epsilon après chaque épisode (jamais en dessous de eps_end)."""
        self.epsilon = max(self._eps_end, self.epsilon * self._eps_decay)

    def sync_target(self) -> None:
        """Copie les poids du réseau principal vers le réseau cible."""
        self.target_net.load_state_dict(self.q_net.state_dict())


# ---------------------------------------------------------------------------
# Nommage des runs
# ---------------------------------------------------------------------------

def _now() -> str:
    """Retourne le timestamp courant au format yyyymmdd_hhmm."""
    return time.strftime("%Y%m%d_%H%M")


def _run_label(seed: int | None, seed_pool: list[int] | None) -> str:
    """Retourne le label de seed : 'seed42', 'pool10', ou 'random'."""
    if seed_pool is not None:
        return f"pool{len(seed_pool)}"
    if seed is not None:
        return f"seed{seed}"
    return "random"


def _pretrained_label(pretrained: "Path | None") -> str:
    """Extrait le timestamp du dossier pretrained pour le suffixe _from_.

    Path('models/20260527_1222_seed42_ep2000/final.pt') → '20260527_1222'
    None → ''
    """
    if pretrained is None:
        return ""
    return pretrained.parent.name[:13]


def _run_name(
    timestamp:        str,
    episodes:         int,
    seed:             int | None,
    seed_pool:        list[int] | None,
    pretrained_label: str = "",
) -> str:
    """Construit l'identifiant unique d'un run.

    Sans pretrained : '20260527_1430_seed42_ep3000'
    Avec pretrained : '20260527_1300_pool10_ep2000_from_20260527_1222'
    """
    name = f"{timestamp}_{_run_label(seed, seed_pool)}_ep{episodes}"
    if pretrained_label:
        name += f"_from_{pretrained_label}"
    return name


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def _run_episode(
    env:   "DungeonEnv",
    agent: DQNAgent,
    buf:   ReplayBuffer,
) -> tuple[list[str], float, dict]:
    """Joue un épisode complet. Retourne (moves, ep_reward, info)."""
    obs       = env.reset()
    state     = encode_obs(obs)
    done      = False
    moves:    list[str] = []
    ep_reward = 0.0

    while not done:
        action_idx                   = agent.select_action(state)
        action                       = ACTIONS[action_idx]
        next_obs, reward, done, info = env.step(action)
        next_state                   = encode_obs(next_obs)

        buf.push(state, action_idx, reward, next_state, done)
        agent.learn(buf)

        state      = next_state
        ep_reward += reward
        moves.append(action)

    return moves, ep_reward, info


def _log_episode(
    log_file,
    ep:       int,
    info:     dict,
    moves:    list[str],
    agent:    DQNAgent,
    ep_reward: float,
) -> None:
    """Écrit une ligne JSON dans le fichier de log."""
    entry = {
        "episode": ep,
        "score":   info["score"],
        "moves":   moves,
        "epsilon": round(agent.epsilon, 4),
        "reward":  round(ep_reward, 4),
    }
    log_file.write(json.dumps(entry) + "\n")


def _save_checkpoint(
    agent:     DQNAgent,
    model_dir: Path,
    ep:        int,
    episodes:  int,
    score:     int,
    t0:        float,
    verbose:   bool,
    final:     bool = False,
) -> None:
    """Sauvegarde les poids du réseau. final=True pour le checkpoint final."""
    path = model_dir / ("final.pt" if final else f"ep{ep}.pt")
    torch.save(agent.q_net.state_dict(), path)
    if verbose:
        elapsed = time.time() - t0
        if final:
            print(f"Terminé — {episodes} épisodes en {elapsed:.1f}s")
        else:
            print(
                f"Ep {ep:>6}/{episodes}  "
                f"score={score:>3}  "
                f"eps={agent.epsilon:.3f}  "
                f"t={elapsed:.0f}s"
            )


def train(
    episodes:   int              = EPISODES,
    seed:       int | None       = None,
    seed_pool:  list[int] | None = None,
    lr:         float            = LEARNING_RATE,
    pretrained: Path | None      = None,
    log_path:   Path             = Path("logs/train.jsonl"),
    model_dir:  Path             = Path("models"),
    verbose:    bool             = True,
) -> DQNAgent:
    """Entraîne un agent DQN sur DungeonEnv. Retourne l'agent entraîné."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Decay adaptatif : epsilon atteint EPSILON_END à la moitié des épisodes,
    # quelle que soit la taille du pool.
    eps_decay = (EPSILON_END / EPSILON_START) ** (2.0 / episodes)

    env   = DungeonEnv(seed=seed, seed_pool=seed_pool, max_steps=MAX_STEPS)
    agent = DQNAgent(lr=lr, eps_decay=eps_decay)

    if pretrained is not None:
        agent.q_net.load_state_dict(torch.load(pretrained, weights_only=True))
        agent.sync_target()
    buf   = ReplayBuffer(BUFFER_SIZE)
    t0    = time.time()

    with open(log_path, "w") as log_file:
        for ep in range(1, episodes + 1):
            moves, ep_reward, info = _run_episode(env, agent, buf)
            agent.decay_epsilon()

            if ep % TARGET_UPDATE_FREQ == 0:
                agent.sync_target()

            _log_episode(log_file, ep, info, moves, agent, ep_reward)

            if ep % CHECKPOINT_FREQ == 0:
                _save_checkpoint(agent, model_dir, ep, episodes, info["score"], t0, verbose)

    _save_checkpoint(agent, model_dir, episodes, episodes, info["score"], t0, verbose, final=True)
    return agent


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraînement DQN — Dungeon POC")
    p.add_argument("--episodes",   type=int,   default=EPISODES,
                   help=f"Nombre d'épisodes (défaut : {EPISODES})")
    p.add_argument("--seed",       type=int,   default=None,
                   help="Seed fixe — même terrain à chaque épisode")
    p.add_argument("--seed-pool",  type=str,   default=None,
                   help="Pool de seeds, ex : 0,1,2,3")
    p.add_argument("--lr",         type=float, default=LEARNING_RATE,
                   help=f"Learning rate (défaut : {LEARNING_RATE})")
    p.add_argument("--pretrained", type=Path,  default=None,
                   help="Checkpoint .pt à utiliser comme point de départ")
    return p.parse_args()


if __name__ == "__main__":
    args      = _parse_args()
    pool      = [int(s) for s in args.seed_pool.split(",")] if args.seed_pool else None
    pre_label = _pretrained_label(args.pretrained)
    run       = _run_name(_now(), args.episodes, args.seed, pool, pre_label)
    train(
        episodes   = args.episodes,
        seed       = args.seed,
        seed_pool  = pool,
        lr         = args.lr,
        pretrained = args.pretrained,
        log_path   = Path("logs") / f"{run}.jsonl",
        model_dir  = Path("models") / run,
    )
