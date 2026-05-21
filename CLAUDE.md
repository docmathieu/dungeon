# Dungeon POC — Claude Code Project

## Description
Jeu de grille 10×10 en Python/pygame. Un personnage jaune doit atteindre une sortie jaune en se déplaçant case par case via les touches fléchées du clavier.

Ce POC est la première étape vers un système d'**apprentissage par renforcement** : à terme, des centaines de parties headless (sans UI) tourneront en parallèle via `multiprocessing.Pool`. L'architecture maintient une séparation stricte entre logique de jeu et affichage.

## Deux modes d'exécution
| Mode | Usage | Interaction | Pause |
|------|-------|-------------|-------|
| UI | Jeu interactif pygame | touches fléchées → `apply_move()` direct | aucune (temps réel) |
| Headless | Entraînement RL, tests | `Simulation(state, instruct, queue=None)` | aucune |

## Stack technique
- **Python** : 3.12 (LTS-équivalent, supporté jusqu'en 2028)
- **Graphique** : pygame (SDL2)
- **RL (à venir)** : PyTorch + Stable-Baselines3
- **Tests** : pytest (170 tests, 0 échec)
- **Exécutable** : PyInstaller

## Structure du projet
```
dungeon/claude/
├── CLAUDE.md
├── AGENTS.md
├── SKILLS.md
├── requirements.txt
├── .claude/
│   └── skills/
│       ├── generate-game.md
│       ├── generate-tests.md
│       ├── run-tests.md
│       ├── run-game.md
│       └── build-exe.md
├── .vscode/
│   ├── tasks.json
│   └── extensions.json
├── src/
│   ├── main.py           ← point d'entrée pygame
│   ├── directions.py     ← constante DIRECTIONS partagée
│   ├── grid.py           ← TileType, Grid
│   ├── game_state.py     ← GameState, logique déplacement, scoring, trail
│   ├── pathfinder.py     ← PathFinder (Dijkstra pondéré)
│   ├── simulation.py     ← thread de simulation headless (RL)
│   └── ui.py             ← GameUI, rendu pygame, touches fléchées directes
└── tests/
    ├── conftest.py        ← sys.path setup
    ├── helpers.py         ← FakeGrid partagée
    ├── test_game.py       ← tests Grid, GameState, Simulation, scoring, trail
    └── test_pathfinder.py ← tests PathFinder
```

## Spécification du jeu

### Terrain
- Grille 10×10 cases, chaque case 40×40 pixels, séparateur 1px
- Fond noir
- Herbe (vert) : type par défaut
- Pierre (gris) : 30% des cases, infranchissable
- Eau (bleu) : 20% des cases, franchissable mais coûte 2 déplacements

### Éléments
- Personnage : stick figure jaune, placé aléatoirement sur une case herbe
- Sortie : porte jaune, placée aléatoirement sur une case herbe (≠ personnage)
- Si Dijkstra ne trouve aucun chemin entre personnage et sortie → restart automatique

### Trails (tracés)
- **Jaune** : chemin parcouru par le joueur (affiché en continu)
- **Rouge** (+2px décalage) : chemin optimal calculé par Dijkstra (affiché uniquement en fin de partie)

### Interface
- **Au-dessus du terrain** : champ "Déplacements" (init 0), champ "Note" (init 0), champ "Information" (vide), légende des touches
- **En dessous du terrain** : bouton "restart" uniquement

### Contrôles
- **Flèches clavier ← ↑ → ↓** : déplacent le personnage d'une case immédiatement (`apply_move()` direct)
- **Touche R ou bouton restart** : réinitialise tout, génère un nouveau terrain solvable

### Scoring
- Déplacement réussi : incrémente "Déplacements" du coût de la case (herbe=1, eau=2)
- Choc contre un rocher ou bord de grille : incrémente "Déplacements" de 1, position inchangée (pénalité RL)
- Victoire si personnage == sortie → `Note = round(100 × optimal_cost / player_cost)`, Information="GAGNE"
- Note 100 = chemin optimal, note < 100 = chemin sous-optimal
- Chemin optimal (rouge) affiché uniquement en fin de partie

## Commandes rapides (skills)
| Skill            | Commande VSCode       | Action                          |
|------------------|-----------------------|---------------------------------|
| generate-game    | Ctrl+Shift+P → Task   | Génère src/ (7 fichiers)        |
| generate-tests   | Ctrl+Shift+P → Task   | Génère tests/                   |
| run-tests        | Ctrl+Shift+P → Task   | Lance pytest                    |
| run-game         | Ctrl+Shift+P → Task   | Démarre le jeu                  |
| build-exe        | Ctrl+Shift+P → Task   | Produit dungeon.exe             |

---

## Roadmap Apprentissage par Renforcement

### Décisions clés (2026-05-20)
- **Modèle** : petit réseau MLP (pas un LLM) — ~304 entrées → 128 → 64 → 4 sorties (Q-values)
- **Algorithme** : DQN pour commencer, PPO ensuite
- **Librairie** : PyTorch + Stable-Baselines3
- **Reward** : `score / 100` (0.0 à 1.0, 1.0 = chemin optimal)
- **Reproductibilité** : `random.Random(seed)` est déterministe → un seed suffit à reconstruire une partie
- **Terrains** : pool de ~1000 seeds fixes (court terme), puis curriculum (moyen terme)

### Phases de développement

#### Phase 1 — `DungeonEnv` (interface Gym) ← prochaine étape
Fichier : `src/dungeon_env.py`
```python
env = DungeonEnv(seed=42)          # terrain reproductible
env = DungeonEnv(seed=None)        # terrain aléatoire à chaque reset()
env = DungeonEnv(seed_pool=[...])  # tirage dans un ensemble fixe

obs  = env.reset()                 # → dict {grid, char_pos, exit_pos}
obs, reward, done, info = env.step("RIGHT")  # un pas à la fois
```

#### Phase 2 — Entraînement headless
- Boucle `multiprocessing.Pool` (centaines de parties en parallèle)
- Logs JSON : `{"grid_seed": 42, "state_seed": 42, "moves": [...], "score": 87}`

#### Phase 3 — Visualisation pygame
- Charger un checkpoint `.pt` (PyTorch)
- Rejouer la partie dans l'UI via seed + séquence de mouvements générée par le modèle
- Lancement : `python src/main.py --seed 42 --replay logs/episode_xxx.jsonl`

#### Phase 4 — Amélioration itérative
- Curriculum : herbe seule → roches → eau → full random
- Reward shaping si nécessaire

### Stratégie terrains (détail)
| Option | Description | Quand |
|--------|-------------|-------|
| B — Pool fixe | ~1000 seeds pré-générés, jeu de validation séparé | Court terme |
| C — Curriculum | Difficulté progressive par phases | Moyen terme |
| A — Full random | Nouveau terrain à chaque épisode | Long terme |
