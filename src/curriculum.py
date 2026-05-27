"""curriculum.py — entraînement DQN par curriculum progressif.

Principe : élargir le pool de seeds par étapes, en passant à l'étape suivante
quand l'agent atteint le win rate cible ou quand max_episodes_per_stage est épuisé.

Utilisation :
    python src/curriculum.py --pool 0,1,2,3,4,5,6,7,8,9 \\
                             --stages 1,3,6,10 \\
                             --max-episodes-per-stage 2000 \\
                             --win-rate-threshold 0.8 \\
                             --lr 3e-4

Produits (un par étape) :
    logs/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}].jsonl
    models/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}]/ep<N>.pt + final.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))

from train import (
    DQNAgent,
    ReplayBuffer,
    DungeonEnv,
    _log_episode,
    _now,
    _pretrained_label,
    _run_episode,
    _run_name,
    _save_checkpoint,
    BUFFER_SIZE,
    CHECKPOINT_FREQ,
    EPSILON_END,
    EPSILON_START,
    LEARNING_RATE,
    MAX_STEPS,
    TARGET_UPDATE_FREQ,
)


WIN_RATE_WINDOW = 100   # nombre d'épisodes pour calculer le win rate


def _win_rate(scores: list[int], window: int = WIN_RATE_WINDOW) -> float:
    """Calcule le taux de victoire sur les `window` derniers scores.

    Un score > 0 indique une victoire. Retourne 0.0 si la liste est vide.
    """
    recent = scores[-window:]
    if not recent:
        return 0.0
    return sum(1 for s in recent if s > 0) / len(recent)


def _train_stage(
    stage_pool:          list[int],
    max_episodes:        int,
    win_rate_threshold:  float,
    lr:                  float,
    pretrained:          Path | None,
    log_path:            Path,
    model_dir:           Path,
    verbose:             bool,
) -> Path:
    """Entraîne un agent sur un pool donné jusqu'à maîtrise ou max_episodes.

    Retourne le chemin vers final.pt.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    eps_decay = (EPSILON_END / EPSILON_START) ** (2.0 / max_episodes)
    env       = DungeonEnv(seed_pool=stage_pool, max_steps=MAX_STEPS)
    agent     = DQNAgent(lr=lr, eps_decay=eps_decay)
    buf       = ReplayBuffer(BUFFER_SIZE)
    t0        = time.time()

    if pretrained is not None:
        agent.q_net.load_state_dict(torch.load(pretrained, weights_only=True))
        agent.sync_target()

    scores:   list[int] = []
    mastered: bool      = False

    with open(log_path, "w") as log_file:
        for ep in range(1, max_episodes + 1):
            moves, ep_reward, info = _run_episode(env, agent, buf)
            agent.decay_epsilon()

            if ep % TARGET_UPDATE_FREQ == 0:
                agent.sync_target()

            _log_episode(log_file, ep, info, moves, agent, ep_reward)
            scores.append(info["score"])

            if ep % CHECKPOINT_FREQ == 0:
                _save_checkpoint(agent, model_dir, ep, max_episodes,
                                 info["score"], t0, verbose)

            if ep >= WIN_RATE_WINDOW and _win_rate(scores) >= win_rate_threshold:
                mastered = True
                if verbose:
                    print(f"  -> Maitrise atteinte a ep {ep} "
                          f"(win rate={_win_rate(scores):.0%})")
                break

    if not mastered and verbose:
        print(f"  -> Seuil non atteint apres {len(scores)} episodes "
              f"(win rate={_win_rate(scores):.0%})")

    _save_checkpoint(agent, model_dir, len(scores), max_episodes,
                     info["score"], t0, verbose, final=True)
    return model_dir / "final.pt"


def run_curriculum(
    pool:                   list[int],
    stages:                 list[int],
    max_episodes_per_stage: int   = 2000,
    win_rate_threshold:     float = 0.8,
    lr:                     float = LEARNING_RATE,
    log_dir:                Path  = Path("logs"),
    model_dir:              Path  = Path("models"),
    verbose:                bool  = True,
) -> Path:
    """Lance le curriculum complet. Retourne le chemin vers le dernier final.pt."""
    pretrained: Path | None = None

    for stage_idx, n_seeds in enumerate(stages):
        stage_pool = pool[:n_seeds]
        timestamp  = _now()
        pre_label  = _pretrained_label(pretrained)

        seed      = stage_pool[0] if n_seeds == 1 else None
        seed_pool = None           if n_seeds == 1 else stage_pool
        run       = _run_name(timestamp, max_episodes_per_stage,
                              seed, seed_pool, pre_label)

        if verbose:
            label  = f"seed{stage_pool[0]}" if n_seeds == 1 else f"pool{n_seeds}"
            suffix = f" <- {pre_label}" if pre_label else ""
            print(f"\n=== Etape {stage_idx + 1}/{len(stages)} -- {label}{suffix} ===")

        pretrained = _train_stage(
            stage_pool         = stage_pool,
            max_episodes       = max_episodes_per_stage,
            win_rate_threshold = win_rate_threshold,
            lr                 = lr,
            pretrained         = pretrained,
            log_path           = log_dir  / f"{run}.jsonl",
            model_dir          = model_dir / run,
            verbose            = verbose,
        )

    return pretrained


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curriculum DQN — Dungeon POC")
    p.add_argument("--pool",                   type=str,   required=True,
                   help="Seeds disponibles, ex : 0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--stages",                 type=str,   default="1,3,6,10",
                   help="Nombre de seeds par étape (défaut : 1,3,6,10)")
    p.add_argument("--max-episodes-per-stage", type=int,   default=2000,
                   help="Episodes maximum par étape (défaut : 2000)")
    p.add_argument("--win-rate-threshold",     type=float, default=0.8,
                   help="Taux de victoire cible pour progresser (défaut : 0.8)")
    p.add_argument("--lr",                     type=float, default=LEARNING_RATE,
                   help=f"Learning rate (défaut : {LEARNING_RATE})")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    pool   = [int(s) for s in args.pool.split(",")]
    stages = [int(s) for s in args.stages.split(",")]
    final  = run_curriculum(
        pool                   = pool,
        stages                 = stages,
        max_episodes_per_stage = args.max_episodes_per_stage,
        win_rate_threshold     = args.win_rate_threshold,
        lr                     = args.lr,
    )
    print(f"\nCurriculum termine. Modele final : {final}")
