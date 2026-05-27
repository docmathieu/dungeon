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
│   ├── dungeon_env.py    ← DungeonEnv : interface Gym reset()/step() (Phase 1 RL)
│   ├── model.py          ← DQNetwork MLP 304→128→64→4 (Phase 2 RL)
│   ├── train.py          ← boucle DQN, ReplayBuffer, DQNAgent, logs JSON (Phase 2 RL)
│   ├── curriculum.py     ← curriculum progressif par étapes de seeds (Phase 2 RL)
│   └── ui.py             ← GameUI, rendu pygame, touches fléchées directes
└── tests/
    ├── conftest.py        ← sys.path setup
    ├── helpers.py         ← FakeGrid partagée
    ├── test_game.py       ← tests Grid, GameState, scoring, trail
    ├── test_pathfinder.py  ← tests PathFinder
    ├── test_dungeon_env.py ← tests DungeonEnv (43 tests)
    ├── test_train.py       ← tests encode_obs, ReplayBuffer, DQNetwork, DQNAgent
    └── test_curriculum.py  ← tests _win_rate, _train_stage, run_curriculum
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
- Logs JSON dans `logs/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}].jsonl` : `{"episode", "score", "moves", "epsilon", "reward"}`
- Checkpoints dans `models/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}]/ep<N>.pt` + `final.pt`
- `{label}` = `seed42` / `pool10` / `random` selon le mode de seed
- `_from_{timestamp}` présent uniquement si `--pretrained` est utilisé

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

**Résultats Phase 2 (2026-05-27) :**
| Run | LR | Pretrained | ep 500 | ep 1000 | ep 1500 | ep 2000 |
|-----|----|-----------|--------|---------|---------|---------|
| seed42 | 3e-4 | — | 16 | 92 | 52 | 100 |
| pool10 | 1e-3 | — | 0 | 0 | 0 | 0 |
| pool10 | 3e-4 | — | 15 | 100 | 0 | 0 |
| pool10 | 3e-4 | seed42 | 0 | 0 | 0 | 0 |

**Diagnostic :** gradients conflictuels multi-seeds résistants au transfer learning.
**Prochaine action : curriculum progressif** (fichier `src/curriculum.py`).

**Curriculum progressif ✅ implémenté + run effectué (2026-05-27)**
Fichier : `src/curriculum.py`
```bash
python src/curriculum.py --pool 0,1,2,3,4,5,6,7,8,9 \
                         --stages 1,3,6,10 \
                         --max-episodes-per-stage 2000 \
                         --win-rate-threshold 0.8 \
                         --lr 3e-4
```
- Élargit le pool de seeds par étapes (1 → 3 → 6 → 10)
- Passe à l'étape suivante si win rate ≥ seuil sur les 100 derniers épisodes
- Fallback : `--max-episodes-per-stage` si le seuil n'est jamais atteint
- Transfer learning automatique entre étapes via `_from_` dans le nom des fichiers

**Résultats curriculum (2026-05-27) :**
| Stage | Pool | Episodes | Win rate max | Win rate final | Notes |
|-------|------|----------|-------------|----------------|-------|
| 1 | seed0 | 598 | **80%** | 80% ep499-598 | Mastery atteinte ! Arrêt précoce |
| 2 | pool3 | 2000 | 35% | 9% | Transfer aide (30% dès ep1), pas de mastery |
| 3 | pool6 | 2000 | 58% | 12% | Pic ep901-1000, catastrophic forgetting ep1200 |
| 4 | pool10 | 2000 | 17% | 3% | Forgetting immédiat, pool trop grand |

**Diagnostic curriculum :** le curriculum améliore le démarrage (Stage 3 part à 39% vs 0% sans),
mais le catastrophic forgetting frappe toujours à mi-parcours pour les pools > 1 seed.
Le réseau MLP 304→128→64→4 semble insuffisant pour mémoriser plusieurs politiques à la fois.

**Pistes suivantes :**
- Augmenter la capacité du réseau (plus de neurones / couches)
- Experience replay prioritaire (PER) pour équilibrer les seeds
- Réduire le learning rate pour les stages multi-seeds (ex. 1e-4)

#### Phase 3 — Visualisation pygame *(après stabilisation Phase 2)*
- Charger un checkpoint `.pt` (PyTorch)
- Rejouer la partie dans l'UI via seed + séquence de mouvements générée par le modèle
- Lancement : `python src/main.py --seed 42 --replay logs/episode_xxx.jsonl`

#### Phase 4 — Amélioration itérative
- ~~Curriculum progressif seeds (1→3→6→10)~~ ✅ effectué
- Augmenter capacité réseau si forgetting persiste
- Prioritized Experience Replay (PER)

### Stratégie terrains (détail)
| Option | Description | Quand |
|--------|-------------|-------|
| B — Pool fixe | pool 10–100 seeds, decay adaptatif | Référence |
| C — Curriculum | Difficulté progressive par étapes | ← En cours |
| A — Full random | Nouveau terrain à chaque épisode | Long terme |
