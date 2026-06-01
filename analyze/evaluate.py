"""evaluate.py — Évalue un checkpoint sur un ensemble de seeds.

Supporte les checkpoints DQN (.pt) et PPO SB3 (.zip).

Utilisation :
    cd dungeon/claude
    .venv\\Scripts\\python.exe analyze/evaluate.py --checkpoint models/.../final.pt  --seeds 100-299
    .venv\\Scripts\\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 0-9
    .venv\\Scripts\\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 0-9 --verbose

Options :
    --checkpoint  Chemin vers un fichier .pt (DQN) ou .zip (PPO SB3)
    --seeds       Plage (ex: 100-299) ou liste (ex: 0,1,5,10)
    --verbose     Affiche le résultat de chaque épisode (won/score)
"""

import argparse
import sys
from pathlib import Path

# Accès aux modules src/ depuis le dossier analyze/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exploit import load_net, run_one_episode_info  # noqa: E402


def _parse_seeds(seeds_arg: str) -> list[int]:
    """Accepte '100-299' ou '0,1,5,10'."""
    if "-" in seeds_arg and "," not in seeds_arg:
        start, end = seeds_arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in seeds_arg.split(",")]


def _run_episode_ppo(model, seed: int) -> tuple[bool, int]:
    """Joue un épisode complet avec un modèle PPO SB3.

    Retourne (won, score).
    """
    from train_ppo import DungeonGymEnv
    import numpy as np

    env = DungeonGymEnv(seed=seed)
    obs, _ = env.reset()
    terminated, truncated = False, False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(int(action))

    return bool(info["won"]), int(info["score"])


def evaluate(checkpoint: Path, seeds: list[int], verbose: bool = False) -> dict:
    """Évalue le checkpoint sur la liste de seeds. Retourne un dict de métriques.

    Détecte automatiquement le format : .pt → DQN, .zip → PPO SB3.
    """
    # Chargement selon le format
    if checkpoint.suffix == ".zip":
        from stable_baselines3 import PPO
        model = PPO.load(str(checkpoint))
        is_ppo = True
    else:
        model = load_net(checkpoint)
        is_ppo = False

    wins = 0
    scores: list[int] = []
    score_wins: list[int] = []

    for seed in seeds:
        if is_ppo:
            won, score = _run_episode_ppo(model, seed)
        else:
            _trail, won, score = run_one_episode_info(model, seed=seed, seed_idx=0)

        wins += int(won)
        scores.append(score)
        if won:
            score_wins.append(score)

        if verbose:
            status = f"WIN  score={score:3d}" if won else "LOSS score=  0"
            print(f"  seed {seed:5d} : {status}")

    n = len(seeds)
    win_rate = wins / n * 100
    score_mean_all = sum(scores) / n
    score_mean_wins = sum(score_wins) / len(score_wins) if score_wins else 0.0

    return {
        "n": n,
        "wins": wins,
        "win_rate": win_rate,
        "score_mean_all": score_mean_all,
        "score_mean_wins": score_mean_wins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue un checkpoint sur un ensemble de seeds")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Chemin vers un fichier .pt (DQN) ou .zip (PPO)")
    parser.add_argument("--seeds", required=True, help="Plage (ex: 100-299) ou liste (ex: 0,1,5)")
    parser.add_argument("--verbose", action="store_true", help="Détail par seed")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Erreur : checkpoint introuvable : {args.checkpoint}")
        sys.exit(1)

    seeds = _parse_seeds(args.seeds)
    fmt = "PPO (.zip)" if args.checkpoint.suffix == ".zip" else "DQN (.pt)"
    print(f"Checkpoint : {args.checkpoint}  [{fmt}]")
    print(f"Seeds      : {len(seeds)} seeds ({args.seeds})")
    print()

    if args.verbose:
        print("Résultats par seed :")
    metrics = evaluate(args.checkpoint, seeds, verbose=args.verbose)
    if args.verbose:
        print()

    print(f"Victoires        : {metrics['wins']}/{metrics['n']}  ({metrics['win_rate']:.1f}%)")
    print(f"Score moyen      : {metrics['score_mean_all']:.1f}  (tous épisodes, échecs = 0)")
    print(f"Score moyen wins : {metrics['score_mean_wins']:.1f}  (victoires uniquement)")


if __name__ == "__main__":
    main()
