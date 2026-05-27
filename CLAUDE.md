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
│   ├── dungeon_env.py    ← DungeonEnv : interface Gym reset()/step() (Phase 1 RL)
│   ├── model.py          ← DQNetwork MLP 304→128→64→4 (Phase 2 RL)
│   ├── train.py          ← boucle DQN, ReplayBuffer, DQNAgent, logs JSON (Phase 2 RL)
│   └── ui.py             ← GameUI, rendu pygame, touches fléchées directes
└── tests/
    ├── conftest.py        ← sys.path setup
    ├── helpers.py         ← FakeGrid partagée
    ├── test_game.py       ← tests Grid, GameState, Simulation, scoring, trail
    ├── test_pathfinder.py  ← tests PathFinder
    ├── test_dungeon_env.py ← tests DungeonEnv (43 tests)
    └── test_train.py       ← tests encode_obs, ReplayBuffer, DQNetwork, DQNAgent (35 tests)
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
- **Reward** : `score / 100` (0.0 à 1.0, 1.0 = chemin optimal) + reward shaping intermédiaire
- **Reproductibilité** : `random.Random(seed)` est déterministe → un seed suffit à reconstruire une partie
- **Terrains** : pool de ~1000 seeds fixes (court terme), puis curriculum (moyen terme)

### Phases de développement

#### Phase 1 — `DungeonEnv` (interface Gym) ✅ implémentée
Fichier : `src/dungeon_env.py`
```python
env = DungeonEnv(seed=42)          # terrain reproductible
env = DungeonEnv(seed=None)        # terrain aléatoire à chaque reset()
env = DungeonEnv(seed_pool=[...])  # tirage dans un ensemble fixe

obs  = env.reset()                 # → dict {grid, char_pos, exit_pos}
obs, reward, done, info = env.step("RIGHT")  # un pas à la fois
# reward : victoire → score/100.0 | déplacement normal → -0.01 | choc → -0.05
```

#### Phase 2 — Entraînement headless ✅ implémentée + analysée
Fichiers : `src/model.py`, `src/train.py`

```bash
python src/train.py --episodes 3000 --seed 42
```
- `DQNetwork` MLP : 304 → 128 → 64 → 4 sorties (Q-values)
- `DQNAgent` : epsilon-greedy (ε 1.0→0.05), réseau cible, replay buffer 10 000
- Reward shaping : `REWARD_STEP=-0.01`, `REWARD_BUMP=-0.05`
- Logs JSON dans `logs/yyyymmdd_hhmm_{label}_ep{N}.jsonl` : `{"episode", "score", "moves", "epsilon", "reward"}`
- Checkpoints dans `models/yyyymmdd_hhmm_{label}_ep{N}/ep<N>.pt` + `final.pt`
- `{label}` = `seed42` / `pool10` / `random` selon le mode de seed

**Résultats run seed=42, 3 000 épisodes (2026-05-21) :**
| Bloc | Victoires | Score moy | Moves moy |
|------|-----------|-----------|-----------|
| ep 1–500 | 23% | 23.8 | 53.6 |
| ep 501–1000 | 36% | 62.6 | 25.2 |
| ep 1001–1500 | **99.6%** | **93.4** | **9.6** |
| ep 1501–2000 | **98.2%** | **89.8** | **10.0** |
| ep 2001–2500 | 35.8% | 54.3 | 27.7 |
| ep 2501–3000 | 3.6% | 17.4 | 53.3 |

**Diagnostic : catastrophic forgetting** — l'agent maîtrise parfaitement seed=42 entre ep 1000–2000
puis désapprend. Les poids encodant la bonne politique sont écrasés par les nouvelles mises à jour.
**Meilleur checkpoint exploitable : `models/dqn_ep1500.pt` ou `dqn_ep2000.pt`.**

**Expériences seed_pool (2026-05-22) :**

Fix implémenté dans `train.py` (non commité) :
```python
# Decay adaptatif — epsilon atteint EPSILON_END à la moitié des épisodes
eps_decay = (EPSILON_END / EPSILON_START) ** (2.0 / episodes)
```
- seed=42 seul, 3 000 ep → decay=0.9980, minimum à ep 1500 ✅ (reproduit le pattern d'hier)
- pool 10 seeds, 10 000 ep → decay=0.9991, minimum à ep 5000

**Résultats seed_pool (2026-05-22) :**
| Run | Pool | Ep | Win rate exploitation | Diagnostic |
|-----|------|----|-----------------------|-----------|
| pool20 v1 | 20 seeds | 3 000 | 7–8% | eps trop rapide (0.995 fixe) |
| pool20 v2 | 20 seeds | 3 000 | ~20% stable | decay adaptatif, plus de crash |
| pool10 10k | 10 seeds | 10 000 | 8–24% puis déclin | gradients conflictuels, LR trop fort |

**Diagnostic multi-seeds :** le réseau MLP génère des **gradients conflictuels** quand les terrains ont des politiques optimales différentes. La politique "moyenne" est pire qu'aléatoire en exploitation. Le decay adaptatif améliore la stabilité mais ne suffit pas.

**Bug corrigé :** `ε=` → `eps=` dans le print (UnicodeEncodeError cp1252 Windows).

**Prochaine action Phase 2 — Réduire le learning rate :**
- Hypothèse : `LR=1e-3` trop agressif pour la généralisation multi-seeds
- Tester `LR=3e-4` avec pool 10 seeds, 10 000 épisodes
- Si stable → scaler à 100 seeds

**À commiter (prochaine session) :**
- Fix `eps=` (UnicodeEncodeError)
- Decay adaptatif `eps_decay = (EPSILON_END/EPSILON_START)^(2/episodes)`
- 2 nouveaux tests (`test_adaptive_decay_*`, `test_seed_pool_runs_without_error`)

#### Phase 3 — Visualisation pygame *(après stabilisation Phase 2)*
- Charger un checkpoint `.pt` (PyTorch)
- Rejouer la partie dans l'UI via seed + séquence de mouvements générée par le modèle
- Lancement : `python src/main.py --seed 42 --replay logs/episode_xxx.jsonl`

#### Phase 4 — Amélioration itérative
- Réduire le learning rate (1e-3 → 3e-4) ← prochaine étape concrète
- Curriculum : herbe seule → roches → eau → full random
- Augmenter BUFFER_SIZE si instabilité persiste avec seed_pool

### Stratégie terrains (détail)
| Option | Description | Quand |
|--------|-------------|-------|
| B — Pool fixe | pool 10–100 seeds, decay adaptatif | ← En cours |
| C — Curriculum | Difficulté progressive par phases | Moyen terme |
| A — Full random | Nouveau terrain à chaque épisode | Long terme |
