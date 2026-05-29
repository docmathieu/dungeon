"""search_seeds.py — Recherche de seeds pédagogiquement intéressants pour le curriculum RL.

Objectif : trouver 20 seeds candidats, répartis en 5 groupes de 4 selon la difficulté
et les propriétés structurelles du terrain, pour que le curriculum enseigne
des compétences progressives au modèle.

Critères de groupement :
    Groupe 1 — Facile      : coût optimal 5–8,  détour rocher faible, peu/pas d'eau
    Groupe 2 — Rochers     : coût optimal 8–12, détour rocher marqué (optimal_moves > manhattan)
    Groupe 3 — Mixte       : coût optimal 12–16, eau et/ou rochers sur le chemin
    Groupe 4 — Complexe    : coût optimal 16–20, eau sur le chemin optimal
    Groupe 5 — Difficile   : coût optimal 21+,   chemin long et tortueux

Métriques calculées pour chaque seed :
    optimal_cost   : coût Dijkstra du chemin optimal (herbe=1, eau=2)
    optimal_moves  : nombre de cases traversées sur le chemin optimal
    manhattan      : |Δx| + |Δy| — distance directe sans obstacles
    rock_detour    : optimal_moves - manhattan (>0 = rochers/bords forcent un détour)
    water_steps    : nombre de cases EAU sur le chemin optimal
    near_border    : True si le personnage ou la sortie est sur le bord de la grille
    char_pos       : position du personnage
    exit_pos       : position de la sortie

Utilisation :
    cd dungeon/claude
    .venv\\Scripts\\python.exe analyze/search_seeds.py

Sortie :
    Affiche les 20 meilleurs candidats (4 par groupe) avec leurs métriques.
    Les seeds avec terrain identique (même char+exit) sont dédoublonnés.
"""

import os
import sys

# Permet d'importer les modules src/ sans modifier PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dungeon_env import DungeonEnv
from pathfinder import PathFinder
from grid import TileType


# ---------------------------------------------------------------------------
# Constantes de la recherche
# ---------------------------------------------------------------------------

SEED_RANGE   = 3000   # nombre de seeds à analyser (0..SEED_RANGE-1)
TARGETS      = 4      # nombre de seeds à retenir par groupe
BORDER_VALS  = {0, 9} # valeurs de coordonnée considérées "bord de grille"

# Bornes de coût optimal pour chaque groupe (inclus)
GROUPS = [
    {"name": "Facile",    "cost_min":  5, "cost_max":  8},
    {"name": "Rochers",   "cost_min":  9, "cost_max": 12},
    {"name": "Mixte",     "cost_min": 13, "cost_max": 16},
    {"name": "Complexe",  "cost_min": 17, "cost_max": 20},
    {"name": "Difficile", "cost_min": 21, "cost_max": 999},
]


# ---------------------------------------------------------------------------
# Analyse d'un terrain
# ---------------------------------------------------------------------------

def analyze_seed(seed: int) -> dict | None:
    """Calcule les métriques d'un terrain pour un seed donné.

    Retourne None si le terrain n'est pas solvable (ne devrait pas arriver
    car DungeonEnv.create_solvable() garantit un chemin, mais on garde la
    protection au cas où).
    """
    env = DungeonEnv(seed=seed)
    obs = env.reset()
    gs  = env._state
    pf  = PathFinder()

    optimal_path = pf.find_shortest_path(gs.grid, gs.char_pos, gs.exit_pos)
    optimal_cost = pf.shortest_cost(gs.grid, gs.char_pos, gs.exit_pos)

    if optimal_path is None or optimal_cost is None:
        return None  # terrain non solvable (ne devrait pas arriver)

    # Nombre de cases EAU traversées sur le chemin optimal
    water_steps = 0
    cx, cy = gs.char_pos
    for direction in optimal_path:
        from directions import DIRECTIONS
        dx, dy = DIRECTIONS[direction]
        cx += dx
        cy += dy
        if gs.grid.get_tile(cx, cy) == TileType.WATER:
            water_steps += 1

    # Distance Manhattan (chemin direct sans obstacles)
    ex, ey = gs.exit_pos
    manhattan = abs(gs.char_pos[0] - ex) + abs(gs.char_pos[1] - ey)

    # Détour causé par les rochers / bords (>0 = obstacles forcent un contournement)
    rock_detour = len(optimal_path) - manhattan

    # Personnage ou sortie sur le bord de la grille
    near_border = (
        gs.char_pos[0] in BORDER_VALS or gs.char_pos[1] in BORDER_VALS or
        gs.exit_pos[0]  in BORDER_VALS or gs.exit_pos[1]  in BORDER_VALS
    )

    return {
        "seed":          seed,
        "char_pos":      gs.char_pos,
        "exit_pos":      gs.exit_pos,
        "optimal_cost":  optimal_cost,
        "optimal_moves": len(optimal_path),
        "manhattan":     manhattan,
        "rock_detour":   rock_detour,
        "water_steps":   water_steps,
        "near_border":   near_border,
    }


# ---------------------------------------------------------------------------
# Sélection de 4 candidats par groupe
# ---------------------------------------------------------------------------

def select_candidates(profiles: list[dict], group: dict, n: int = TARGETS) -> list[dict]:
    """Sélectionne n seeds pour un groupe de difficulté donné.

    Stratégie de tri pour maximiser la diversité au sein du groupe :
      1. Priorité aux seeds avec rock_detour > 0 (obstacles réels à contourner)
      2. Puis ceux avec water_steps > 0 (eau sur le chemin)
      3. Puis par coût optimal croissant (varier la difficulté dans le groupe)

    Les doublons (même char_pos + exit_pos) sont éliminés avant la sélection.
    """
    # Filtrer par plage de coût du groupe
    candidates = [
        p for p in profiles
        if group["cost_min"] <= p["optimal_cost"] <= group["cost_max"]
    ]

    # Dédoublonner : garder un seul seed par (char_pos, exit_pos)
    seen_positions: set[tuple] = set()
    unique: list[dict] = []
    for p in candidates:
        key = (p["char_pos"], p["exit_pos"])
        if key not in seen_positions:
            seen_positions.add(key)
            unique.append(p)

    # Tri : d'abord les terrains avec obstacles (détour > 0), puis eau, puis coût
    unique.sort(key=lambda p: (
        -(p["rock_detour"] > 0),   # -1 si détour rocher (priorité haute)
        -(p["water_steps"] > 0),   # -1 si eau sur chemin
        p["optimal_cost"],          # coût croissant
    ))

    return unique[:n]


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Analyse de {SEED_RANGE} seeds (0..{SEED_RANGE - 1})...\n")

    # Collecter les profils de tous les seeds
    profiles: list[dict] = []
    for seed in range(SEED_RANGE):
        profile = analyze_seed(seed)
        if profile:
            profiles.append(profile)
        if seed % 500 == 499:
            print(f"  {seed + 1}/{SEED_RANGE} seeds analysés...")

    print(f"\n{len(profiles)} seeds solvables trouvés.\n")

    # Sélectionner et afficher les candidats par groupe
    all_selected: list[dict] = []

    for group in GROUPS:
        candidates = select_candidates(profiles, group)
        print(f"=== Groupe {group['name']} "
              f"(coût {group['cost_min']}–{group['cost_max']}) "
              f"— {len(candidates)} seeds ===")
        print(f"{'seed':>6}  {'char':>8}  {'exit':>8}  "
              f"{'cout':>5}  {'moves':>6}  {'manh':>5}  "
              f"{'detour':>7}  {'eau':>4}  {'bord':>5}")
        print("-" * 70)
        for p in candidates:
            print(
                f"  {p['seed']:>4}  {str(p['char_pos']):>8}  {str(p['exit_pos']):>8}  "
                f"{p['optimal_cost']:>5}  {p['optimal_moves']:>6}  {p['manhattan']:>5}  "
                f"{p['rock_detour']:>7}  {p['water_steps']:>4}  "
                f"{'oui' if p['near_border'] else 'non':>5}"
            )
        print()
        all_selected.extend(candidates)

    # Résumé final : liste des seeds à transmettre au curriculum
    seeds_by_group = []
    for group in GROUPS:
        group_seeds = [p["seed"] for p in select_candidates(profiles, group)]
        seeds_by_group.append(group_seeds)
        print(f"Groupe {group['name']:12} : {group_seeds}")

    all_seeds = [s for grp in seeds_by_group for s in grp]
    print(f"\n20 seeds candidats (ordre curriculum) : {all_seeds}")
    print(f"Commande curriculum (stages 4,8,12,16,20) :")
    print(f"  --pool {','.join(str(s) for s in all_seeds)} --stages 4,8,12,16,20")
