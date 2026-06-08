"""analyze_failures.py — Compare les profils terrain des victoires vs échecs du modèle.

Pour chaque seed évalué, calcule :
  - résultat du modèle (gagné/perdu, score)
  - métriques terrain : coût optimal, détour rochers, cases eau, distance Manhattan

Affiche les distributions comparées victoires vs échecs pour diagnostiquer
ce qui rend un terrain difficile pour le modèle.

Utilisation :
    cd dungeon/claude
    .venv\\Scripts\\python.exe analyze/analyze_failures.py \\
        --checkpoint models/.../final.zip --seeds 100-499

Options :
    --checkpoint  Chemin vers un fichier .zip (PPO SB3)
    --seeds       Plage (ex: 100-499) ou liste (ex: 0,1,5,10)
    --n-bins      Nombre de tranches pour les histogrammes (défaut : 5)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dungeon_env import DungeonEnv          # noqa: E402
from pathfinder  import PathFinder          # noqa: E402
from grid        import TileType            # noqa: E402
from directions  import DIRECTIONS          # noqa: E402
from exploit     import load_model          # noqa: E402


# ---------------------------------------------------------------------------
# Métriques terrain (repris de search_seeds.py)
# ---------------------------------------------------------------------------

def terrain_profile(seed: int) -> dict:
    """Calcule les métriques structurelles d'un terrain pour un seed donné."""
    env = DungeonEnv(seed=seed)
    env.reset()
    gs  = env._state
    pf  = PathFinder()

    optimal_path = pf.find_shortest_path(gs.grid, gs.char_pos, gs.exit_pos)
    optimal_cost = pf.shortest_cost(gs.grid, gs.char_pos, gs.exit_pos)

    cx, cy = gs.char_pos
    water_on_path = 0
    for direction in (optimal_path or []):
        dx, dy = DIRECTIONS[direction]
        cx += dx
        cy += dy
        if gs.grid.get_tile(cx, cy) == TileType.WATER:
            water_on_path += 1

    ex, ey = gs.exit_pos
    manhattan   = abs(gs.char_pos[0] - ex) + abs(gs.char_pos[1] - ey)
    rock_detour = len(optimal_path) - manhattan if optimal_path else 0

    # Compte total des cases eau et rochers sur la grille
    total_rocks = sum(
        1 for x in range(10) for y in range(10)
        if gs.grid.get_tile(x, y) == TileType.ROCK
    )
    total_water = sum(
        1 for x in range(10) for y in range(10)
        if gs.grid.get_tile(x, y) == TileType.WATER
    )

    return {
        "seed":          seed,
        "optimal_cost":  optimal_cost or 0,
        "optimal_moves": len(optimal_path) if optimal_path else 0,
        "manhattan":     manhattan,
        "rock_detour":   rock_detour,
        "water_on_path": water_on_path,
        "total_rocks":   total_rocks,
        "total_water":   total_water,
    }


# ---------------------------------------------------------------------------
# Évaluation modèle
# ---------------------------------------------------------------------------

def run_model(model, seed: int) -> tuple[bool, int]:
    """Joue un épisode déterministe. Retourne (won, score)."""
    from train_ppo import DungeonGymEnv, CNN_OBS_SHAPE
    obs_type = "cnn" if model.observation_space.shape == CNN_OBS_SHAPE else "mlp"
    env = DungeonGymEnv(seed=seed, obs_type=obs_type)
    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
    return bool(info["won"]), int(info["score"])


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "min": 0, "max": 0, "median": 0.0}
    s = sorted(values)
    n = len(s)
    return {
        "n":      n,
        "mean":   sum(s) / n,
        "min":    s[0],
        "max":    s[-1],
        "median": s[n // 2],
    }


def histogram(values: list[float], n_bins: int = 5) -> list[tuple]:
    """Retourne une liste de (label_tranche, count, pourcentage)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [(f"{lo}", len(values), 100.0)]
    step = (hi - lo) / n_bins
    bins = [0] * n_bins
    for v in values:
        idx = min(int((v - lo) / step), n_bins - 1)
        bins[idx] += 1
    result = []
    for i, count in enumerate(bins):
        a = lo + i * step
        b = a + step
        label = f"{a:.1f}–{b:.1f}"
        result.append((label, count, count / len(values) * 100))
    return result


def print_comparison(label: str, wins_vals: list, loss_vals: list, n_bins: int = 5) -> None:
    """Affiche stats + histogramme comparé victoires vs échecs pour une métrique."""
    sw = stats(wins_vals)
    sl = stats(loss_vals)
    print(f"\n  {label}")
    print(f"    {'':20s}  {'Victoires':>10}  {'Échecs':>10}")
    print(f"    {'N':20s}  {sw['n']:>10d}  {sl['n']:>10d}")
    print(f"    {'Moyenne':20s}  {sw['mean']:>10.2f}  {sl['mean']:>10.2f}")
    print(f"    {'Médiane':20s}  {sw['median']:>10.2f}  {sl['median']:>10.2f}")
    print(f"    {'Min':20s}  {sw['min']:>10}  {sl['min']:>10}")
    print(f"    {'Max':20s}  {sw['max']:>10}  {sl['max']:>10}")

    # Histogramme sur l'union des valeurs pour des tranches communes
    all_vals = wins_vals + loss_vals
    if not all_vals:
        return
    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        return
    step = (hi - lo) / n_bins
    print(f"\n    Tranche{' ':14s}  {'Victoires':>10}  {'Échecs':>10}  Diff")
    print(f"    {'-'*55}")
    for i in range(n_bins):
        a = lo + i * step
        b = a + step
        label_t = f"{a:.1f}–{b:.1f}"
        wc = sum(1 for v in wins_vals if a <= v < b + (1e-9 if i == n_bins - 1 else 0))
        lc = sum(1 for v in loss_vals if a <= v < b + (1e-9 if i == n_bins - 1 else 0))
        wp = wc / len(wins_vals) * 100 if wins_vals else 0
        lp = lc / len(loss_vals) * 100 if loss_vals else 0
        bar = "^ Victoires" if wp > lp + 5 else ("v Echecs" if lp > wp + 5 else "  =")
        print(f"    {label_t:20s}  {wp:>8.1f}%  {lp:>8.1f}%  {bar}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def _parse_seeds(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse les profils terrain des victoires vs échecs du modèle"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seeds",      required=True,
                        help="Plage (ex: 100-499) ou liste (ex: 0,1,5)")
    parser.add_argument("--n-bins",     type=int, default=5,
                        help="Nombre de tranches pour les histogrammes (défaut : 5)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Erreur : checkpoint introuvable : {args.checkpoint}")
        sys.exit(1)

    seeds = _parse_seeds(args.seeds)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Seeds      : {len(seeds)} seeds ({args.seeds})")
    print(f"\nÉvaluation en cours...", flush=True)

    model = load_model(args.checkpoint)

    wins_profiles: list[dict] = []
    loss_profiles: list[dict] = []

    for i, seed in enumerate(seeds):
        profile = terrain_profile(seed)
        won, score = run_model(model, seed)
        profile["score"] = score
        if won:
            wins_profiles.append(profile)
        else:
            loss_profiles.append(profile)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(seeds)} seeds traités "
                  f"(V:{len(wins_profiles)} E:{len(loss_profiles)})...", flush=True)

    total = len(seeds)
    nw    = len(wins_profiles)
    nl    = len(loss_profiles)
    print(f"\n{'='*60}")
    print(f"RÉSULTATS — {nw}/{total} victoires ({nw/total*100:.1f}%)  "
          f"| {nl} échecs ({nl/total*100:.1f}%)")
    print(f"{'='*60}")

    metrics = [
        ("Coût optimal (Dijkstra)",  "optimal_cost"),
        ("Cases traversées optimal", "optimal_moves"),
        ("Distance Manhattan",       "manhattan"),
        ("Détour rochers",           "rock_detour"),
        ("Cases eau sur chemin opt", "water_on_path"),
        ("Total rochers (grille)",   "total_rocks"),
        ("Total eau (grille)",       "total_water"),
    ]

    print("\n--- Comparaison Victoires vs Echecs ---")
    for label, key in metrics:
        wv = [p[key] for p in wins_profiles]
        lv = [p[key] for p in loss_profiles]
        print_comparison(label, wv, lv, n_bins=args.n_bins)

    # Résumé : top-5 seeds les plus difficiles (échecs avec coût optimal le plus élevé)
    if loss_profiles:
        print(f"\n--- Top-10 echecs (cout optimal le plus eleve) ---")
        top = sorted(loss_profiles, key=lambda p: -p["optimal_cost"])[:10]
        print(f"  {'seed':>6}  {'cout':>5}  {'moves':>6}  {'detour':>7}  "
              f"{'eau_chemin':>10}  {'rochers':>8}  {'eau_total':>9}")
        print(f"  {'-'*60}")
        for p in top:
            print(f"  {p['seed']:>6}  {p['optimal_cost']:>5}  {p['optimal_moves']:>6}  "
                  f"{p['rock_detour']:>7}  {p['water_on_path']:>10}  "
                  f"{p['total_rocks']:>8}  {p['total_water']:>9}")

    # Score moyen par tranche de difficulté
    print(f"\n--- Win rate par tranche de cout optimal ---")
    buckets = [(1, 8), (9, 12), (13, 16), (17, 20), (21, 99)]
    bucket_names = ["Facile (1–8)", "Rochers (9–12)", "Mixte (13–16)",
                    "Complexe (17–20)", "Difficile (21+)"]
    all_profiles = [(p, True) for p in wins_profiles] + [(p, False) for p in loss_profiles]
    print(f"  {'Groupe':22s}  {'Victoires':>10}  {'Total':>7}  {'Win rate':>9}")
    print(f"  {'-'*52}")
    for (lo, hi), name in zip(buckets, bucket_names):
        bucket = [(p, w) for p, w in all_profiles if lo <= p["optimal_cost"] <= hi]
        if not bucket:
            continue
        bw = sum(1 for _, w in bucket if w)
        bt = len(bucket)
        print(f"  {name:22s}  {bw:>10d}  {bt:>7d}  {bw/bt*100:>8.1f}%")


if __name__ == "__main__":
    main()
