"""evaluate.py — Évalue un checkpoint sur un ensemble de seeds.

Supporte les checkpoints DQN (.pt) et PPO SB3 (.zip).

Utilisation :
    cd dungeon/claude
    .venv\\Scripts\\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 0-9
    .venv\\Scripts\\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 100-299
    .venv\\Scripts\\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 0-9 --n-episodes 5
    .venv\\Scripts\\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 0-9 --stochastic --n-episodes 10
    .venv\\Scripts\\python.exe analyze/evaluate.py --checkpoint models/.../final.pt  --seeds 100-299 --verbose

Options :
    --checkpoint  Chemin vers un fichier .pt (DQN) ou .zip (PPO SB3)
    --seeds       Plage (ex: 100-299) ou liste (ex: 0,1,5,10)
    --n-episodes  Épisodes par seed (défaut : 1 ; utile en mode stochastique)
    --stochastic  Politique stochastique PPO (par défaut : déterministe, ignoré pour DQN)
    --verbose     Affiche le résultat par seed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exploit import load_model, run_one_episode_info  # noqa: E402


def _parse_seeds(seeds_arg: str) -> list[int]:
    """Accepte '100-299' ou '0,1,5,10'."""
    if "-" in seeds_arg and "," not in seeds_arg:
        start, end = seeds_arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in seeds_arg.split(",")]


def _run_ppo_episode(model, seed: int, deterministic: bool = True) -> tuple[bool, int]:
    """Joue un épisode avec un modèle PPO SB3. Retourne (won, score).

    deterministic=False → politique stochastique (résultats différents à chaque appel).
    """
    from train_ppo import DungeonGymEnv
    env = DungeonGymEnv(seed=seed)
    obs, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, info = env.step(int(action))
    return bool(info["won"]), int(info["score"])


def evaluate(
    checkpoint: Path,
    seeds: list[int],
    n_episodes: int = 1,
    stochastic: bool = False,
    verbose: bool = False,
) -> dict:
    """Évalue le checkpoint sur la liste de seeds. Retourne un dict de métriques.

    Détecte automatiquement le format : .pt → DQN, .zip → PPO SB3.

    n_episodes : nombre d'épisodes par seed (>1 utile en mode stochastique pour PPO).
    stochastic : PPO uniquement — politique stochastique au lieu de déterministe.
    """
    model = load_model(checkpoint)
    is_ppo = hasattr(model, "predict")
    deterministic = not stochastic

    total_wins = 0
    all_scores: list[int] = []
    win_scores: list[int] = []

    for seed in seeds:
        seed_wins = 0
        seed_scores: list[int] = []

        for _ in range(n_episodes):
            if is_ppo:
                won, score = _run_ppo_episode(model, seed, deterministic=deterministic)
            else:
                _trail, won, score = run_one_episode_info(model, seed=seed, seed_idx=0)

            seed_wins += int(won)
            seed_scores.append(score)
            all_scores.append(score)
            if won:
                win_scores.append(score)

        total_wins += seed_wins

        if verbose:
            wr = seed_wins / n_episodes * 100
            sm = sum(seed_scores) / n_episodes
            if n_episodes == 1:
                line = f"WIN  score={seed_scores[0]:3d}" if seed_wins else "LOSS score=  0"
            else:
                line = f"wr={wr:5.1f}%  score_moy={sm:5.1f}"
            print(f"  seed {seed:5d} : {line}")

    total_episodes = len(seeds) * n_episodes
    win_rate       = total_wins / total_episodes * 100
    score_mean_all  = sum(all_scores) / total_episodes
    score_mean_wins = sum(win_scores) / len(win_scores) if win_scores else 0.0

    return {
        "n_seeds":        len(seeds),
        "n_episodes":     n_episodes,
        "total_episodes": total_episodes,
        "wins":           total_wins,
        "win_rate":       win_rate,
        "score_mean_all":  score_mean_all,
        "score_mean_wins": score_mean_wins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue un checkpoint sur un ensemble de seeds")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Chemin vers un fichier .pt (DQN) ou .zip (PPO)")
    parser.add_argument("--seeds", required=True,
                        help="Plage (ex: 100-299) ou liste (ex: 0,1,5)")
    parser.add_argument("--n-episodes", type=int, default=1,
                        help="Épisodes par seed (défaut : 1 ; utile en mode stochastique)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Politique stochastique PPO (par défaut : déterministe)")
    parser.add_argument("--verbose", action="store_true",
                        help="Détail par seed")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Erreur : checkpoint introuvable : {args.checkpoint}")
        sys.exit(1)

    seeds = _parse_seeds(args.seeds)
    fmt   = "PPO (.zip)" if args.checkpoint.suffix == ".zip" else "DQN (.pt)"
    mode  = "stochastique" if args.stochastic else "déterministe"

    print(f"Checkpoint : {args.checkpoint}  [{fmt}]")
    print(f"Seeds      : {len(seeds)} seeds ({args.seeds})")
    if args.n_episodes > 1 or args.stochastic:
        print(f"Mode       : {mode}  ×{args.n_episodes} épisodes/seed  "
              f"({len(seeds) * args.n_episodes} épisodes total)")
    print()

    if args.verbose:
        print("Résultats par seed :")

    metrics = evaluate(
        args.checkpoint,
        seeds,
        n_episodes=args.n_episodes,
        stochastic=args.stochastic,
        verbose=args.verbose,
    )

    if args.verbose:
        print()

    total_ep  = metrics["total_episodes"]
    label_ep  = f"  ({total_ep} épisodes)" if args.n_episodes > 1 else ""

    print(f"Victoires        : {metrics['wins']}/{total_ep}  "
          f"({metrics['win_rate']:.1f}%){label_ep}")
    print(f"Score moyen      : {metrics['score_mean_all']:.1f}  "
          f"(tous épisodes, échecs = 0)")
    print(f"Score moyen wins : {metrics['score_mean_wins']:.1f}  "
          f"(victoires uniquement)")


if __name__ == "__main__":
    main()
