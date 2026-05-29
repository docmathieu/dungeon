"""curriculum.py — entraînement DQN par curriculum progressif.

Principe : élargir le pool de seeds par étapes, en passant à l'étape suivante
quand l'agent atteint le win rate cible ou quand max_episodes_per_stage est épuisé.

Utilisation :
    python src/curriculum.py --pool 0,1,2,3,4,5,6,7,8,9 \\
                             --stages 1,3,6,10 \\
                             --max-episodes-per-stage 2000 \\
                             --win-rate-threshold 0.8 \\
                             --lr 3e-4,1e-4

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
    ARCHITECTURES,
    DQNAgent,
    ReplayBuffer,
    StratifiedReplayBuffer,
    DungeonEnv,
    _log_episode,
    _log_meta,
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


def _pad_lr(lrs: list[float], n: int) -> list[float]:
    """Normalise lrs à exactement n éléments.

    Si lrs est plus court que n, répète la dernière valeur.
    Si lrs est plus long que n, tronque.

    Exemples :
        _pad_lr([3e-4], 4)           -> [3e-4, 3e-4, 3e-4, 3e-4]
        _pad_lr([3e-4, 1e-4], 4)     -> [3e-4, 1e-4, 1e-4, 1e-4]
        _pad_lr([3e-4, 1e-4, 5e-5], 2) -> [3e-4, 1e-4]
    """
    return (lrs + [lrs[-1]] * n)[:n]


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
    arch:                str         = "film",
    stage_meta:          dict | None = None,
) -> Path:
    """Entraîne un agent sur un pool donné jusqu'à maîtrise ou max_episodes.

    Retourne le chemin vers final.pt.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    eps_decay = (EPSILON_END / EPSILON_START) ** (2.0 / max_episodes)
    env       = DungeonEnv(seed_pool=stage_pool, max_steps=MAX_STEPS)
    agent     = DQNAgent(lr=lr, eps_decay=eps_decay, arch=arch)
    buf       = (StratifiedReplayBuffer(BUFFER_SIZE, n_seeds=len(stage_pool))
                 if len(stage_pool) > 1 else ReplayBuffer(BUFFER_SIZE))
    t0        = time.time()

    if pretrained is not None:
        agent.q_net.load_state_dict(torch.load(pretrained, weights_only=True))
        agent.sync_target()

    scores:   list[int] = []
    mastered: bool      = False

    seed      = stage_pool[0] if len(stage_pool) == 1 else None
    seed_pool = None if len(stage_pool) == 1 else stage_pool

    with open(log_path, "w") as log_file:
        _log_meta(log_file, agent, max_episodes, lr, seed, seed_pool,
                  pretrained, extra=stage_meta)
        for ep in range(1, max_episodes + 1):
            moves, ep_reward, info, ep_seed = _run_episode(env, agent, buf)
            agent.decay_epsilon()

            if ep % TARGET_UPDATE_FREQ == 0:
                agent.sync_target()

            _log_episode(log_file, ep, info, moves, agent, ep_reward, seed=ep_seed)
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
    max_episodes_per_stage: int              = 2000,
    win_rate_threshold:     float            = 0.8,
    lr:                     list[float] | None = None,
    arch:                   str              = "film",
    pretrained:             Path | None      = None,
    log_dir:                Path             = Path("logs"),
    model_dir:              Path             = Path("models"),
    verbose:                bool             = True,
) -> Path:
    """Lance le curriculum complet. Retourne le chemin vers le dernier final.pt.

    lr : learning rate(s) par étape. Si la liste est plus courte que stages,
         la dernière valeur est répétée. Défaut : [LEARNING_RATE] pour toutes.
    pretrained : checkpoint .pt de départ pour la première étape (optionnel).
    """
    import sys
    if lr is None:
        lr = [LEARNING_RATE]
    lrs = _pad_lr(lr, len(stages))

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
            print(f"\n=== Etape {stage_idx + 1}/{len(stages)} -- {label}{suffix}"
                  f"  lr={lrs[stage_idx]:.0e} ===")

        stage_meta = {
            "curriculum_command":    " ".join(sys.argv),
            "stage":                 stage_idx + 1,
            "stage_total":           len(stages),
            "full_pool":             pool,
            "win_rate_threshold":    win_rate_threshold,
            "max_episodes_per_stage": max_episodes_per_stage,
            "all_lrs":               lrs,
        }

        pretrained = _train_stage(
            stage_pool         = stage_pool,
            max_episodes       = max_episodes_per_stage,
            win_rate_threshold = win_rate_threshold,
            lr                 = lrs[stage_idx],
            pretrained         = pretrained,
            log_path           = log_dir  / f"{run}.jsonl",
            model_dir          = model_dir / run,
            verbose            = verbose,
            arch               = arch,
            stage_meta         = stage_meta,
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
    p.add_argument("--lr",                     type=str,   default=str(LEARNING_RATE),
                   help=f"Learning rate(s) par etape, ex : 3e-4,1e-4 (defaut : {LEARNING_RATE})")
    p.add_argument("--pretrained",             type=str,   default=None,
                   help="Checkpoint .pt de depart (optionnel)")
    p.add_argument("--architecture",           type=str,   default="film",
                   choices=list(ARCHITECTURES),
                   help="Architecture du réseau (défaut : film)")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    pool   = [int(s)   for s in args.pool.split(",")]
    stages = [int(s)   for s in args.stages.split(",")]
    lrs    = [float(s) for s in args.lr.split(",")]
    final  = run_curriculum(
        pool                   = pool,
        stages                 = stages,
        max_episodes_per_stage = args.max_episodes_per_stage,
        win_rate_threshold     = args.win_rate_threshold,
        lr                     = lrs,
        arch                   = args.architecture,
        pretrained             = Path(args.pretrained) if args.pretrained else None,
    )
    print(f"\nCurriculum termine. Modele final : {final}")
