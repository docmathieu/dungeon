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
- **Modèle** : réseau MLP (pas un LLM) — 304 entrées → 256 → 128 → 64 → 4 sorties (Q-values)
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
- `DQNetwork` MLP : 304 → 256 → 128 → 64 → 4 sorties (Q-values)
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
                         --lr 3e-4,1e-4
```
- Élargit le pool de seeds par étapes (1 → 3 → 6 → 10)
- Passe à l'étape suivante si win rate ≥ seuil sur les 100 derniers épisodes
- Fallback : `--max-episodes-per-stage` si le seuil n'est jamais atteint
- Transfer learning automatique entre étapes via `_from_` dans le nom des fichiers
- `--lr` accepte une liste : `3e-4,1e-4` → lr=3e-4 stage 1, lr=1e-4 stages suivants (dernière valeur répétée)

**Résultats curriculum (2026-05-27) :**
| Stage | Pool | Episodes | Win rate max | Win rate final | Notes |
|-------|------|----------|-------------|----------------|-------|
| 1 | seed0 | 598 | **80%** | 80% ep499-598 | Mastery atteinte ! Arrêt précoce |
| 2 | pool3 | 2000 | 35% | 9% | Transfer aide (30% dès ep1), pas de mastery |
| 3 | pool6 | 2000 | 58% | 12% | Pic ep901-1000, catastrophic forgetting ep1200 |
| 4 | pool10 | 2000 | 17% | 3% | Forgetting immédiat, pool trop grand |

**Diagnostic curriculum :** le curriculum améliore le démarrage (Stage 3 part à 39% vs 0% sans),
mais le catastrophic forgetting frappe toujours à mi-parcours pour les pools > 1 seed.
Le réseau MLP 304→128→64→4 s'est révélé insuffisant pour mémoriser plusieurs politiques à la fois.

**Architecture réseau augmentée ✅ + run effectué (2026-05-27)**
`model.py` mis à jour : 304→256→128→64→4 (3 couches cachées, ~112k paramètres vs ~51k avant).
Run curriculum `--lr 3e-4,1e-4` avec la nouvelle architecture :

| Stage | Pool | Ep | Win rate max | Win rate final | vs run précédent |
|-------|------|----|-------------|----------------|-----------------|
| 1 | seed0 | **413** | **80%** | 80% | ✅ plus rapide (était 598) |
| 2 | pool3 | 2000 | **43%** | 9% | +8 pts max (était 35%) |
| 3 | pool6 | 2000 | 24% | **22%** | +10 pts final (était 12%) |
| 4 | pool10 | 2000 | **32%** | 9% | +15 pts max (était 17%) |

Progrès notables : Stage 1 converge 31% plus vite. Stage 4 atteint 32% de win rate (vs 17%).
Stage 3 est plus stable (pas de cliff brutal — 22% en fin vs 12%).
Catastrophic forgetting reste présent mais atténué par lr=1e-4 et la capacité réseau accrue.

#### Phase 3 — Visualisation pygame *(après stabilisation Phase 2)*
- Charger un checkpoint `.pt` (PyTorch)
- Rejouer la partie dans l'UI via seed + séquence de mouvements générée par le modèle
- Lancement : `python src/main.py --seed 42 --replay logs/episode_xxx.jsonl`

**Replay stratifié ✅ implémenté + run effectué (2026-05-27)**
`StratifiedReplayBuffer(capacity, n_seeds)` : un sous-buffer par seed, batch toujours équilibré.
Utilisé automatiquement par `_train_stage` quand `len(stage_pool) > 1`.

Résultats comparés (sans vs avec stratified, mêmes hyperparamètres) :
| Stage | Win rate max (sans) | Win rate max (avec) | Win rate final (sans) | Win rate final (avec) |
|-------|--------------------|--------------------|----------------------|----------------------|
| 1 seed | 80% | 80% | 80% | 80% |
| pool3 | 43% | **55%** ep1-200 | 9% | 0% |
| pool6 | 24% | 21% | **22%** | 13% |
| pool10 | 32% | 21% | 9% | **21%** |

**Diagnostic :** le buffer stratifié améliore le démarrage (pool3 : 55% vs 37% sur ep1-200)
et la fin de pool10 (42% vs 24% sur les 200 derniers épisodes). Mais le catastrophic forgetting
reste dominant : le buffer équilibré ne peut pas résoudre les **gradients conflictuels** (deux
seeds peuvent exiger des actions opposées dans des états similaires).
La racine du problème n'est pas le déséquilibre du buffer mais l'architecture MLP partagée.

**Pistes restantes :**
- Augmenter `--max-episodes-per-stage` pour laisser le buffer stratifié converger (pool10 était en progression en fin de run)
- Elastic Weight Consolidation (EWC) pour protéger les poids déjà appris

#### Phase 4 — Amélioration itérative
- ~~Curriculum progressif seeds (1→3→6→10)~~ ✅ effectué
- ~~Augmenter capacité réseau~~ ✅ 304→256→128→64→4
- ~~Replay stratifié par seed~~ ✅ `StratifiedReplayBuffer` implémenté et testé

### Stratégie terrains (détail)
| Option | Description | Quand |
|--------|-------------|-------|
| B — Pool fixe | pool 10–100 seeds, decay adaptatif | Référence |
| C — Curriculum | Difficulté progressive par étapes | ← En cours |
| A — Full random | Nouveau terrain à chaque épisode | Long terme |
