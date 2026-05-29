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
- **Tests** : pytest (221 tests, 0 échec)
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
│   ├── exploit.py        ← load_net(), run_one_episode() — mode exploitation IA (Phase 3 RL)
│   └── ui.py             ← GameUI, rendu pygame, boutons [▶ IA] / [↺ IA]
└── tests/
    ├── conftest.py        ← sys.path setup
    ├── helpers.py         ← FakeGrid partagée
    ├── test_game.py       ← tests Grid, GameState, scoring, trail
    ├── test_pathfinder.py  ← tests PathFinder
    ├── test_dungeon_env.py ← tests DungeonEnv
    ├── test_train.py       ← tests encode_obs, ReplayBuffer, DQNetwork, DQNAgent
    ├── test_curriculum.py  ← tests _win_rate, _train_stage, run_curriculum
    └── test_exploit.py     ← tests load_net, run_one_episode
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
- Logs JSON dans `logs/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}].jsonl` : première ligne `{"type":"meta", "command", "architecture", "hyperparams"}`, puis une ligne `{"episode", "score", "moves", "epsilon", "reward"}` par épisode
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
- `--pretrained` : checkpoint `.pt` de départ (optionnel) — permet de repartir d'un run existant sans relancer depuis seed0
- `--architecture {film,taskcond}` : choix du réseau (défaut : `film`) — disponible aussi dans `train.py`

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

#### Phase 3 — Visualisation pygame ✅ version light implémentée (2026-05-29)
Fichiers : `src/exploit.py`, `src/ui.py` (boutons `[▶ IA]` / `[↺ IA]`)

**Mode exploitation IA dans l'UI :**
- `[▶ IA]` : file picker → charge un checkpoint `.pt` → joue un épisode complet (epsilon=0, boucle jusqu'à victoire ou MAX_STEPS) → affiche le tracé cyan sur le terrain courant
- `[↺ IA]` : rejoue le modèle déjà chargé sur le terrain affiché (utile après restart ou changement de seed)
- Tracé cyan effacé automatiquement au chargement d'un nouveau terrain
- Détection automatique d'architecture (FiLMDQNetwork / DQNetwork) via les clés du state_dict

**Prochaine étape (visualisation avancée) :**
- Charger plusieurs checkpoints d'une même run (ep500, ep1000…) et animer la progression de l'IA
- Affichage multi-trails avec dégradé alpha par étape de curriculum

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
- ~~Augmenter `--max-episodes-per-stage`~~ ✅ testé (5000 ep) — aide pool6 (55%) mais pool10 régresse
- ~~Elastic Weight Consolidation (EWC)~~ ❌ déconseillé — la recherche montre qu'EWC échoue sur les grid worlds (Stanford CS224R)

**Sessions du 2026-05-28 — résumé complet**

**Modifications architecturales implémentées :**
- **Task-conditioning** (entrée) : `encode_obs(obs, seed_idx=0)` → one-hot 10 bits ajouté aux indices 304–313. `INPUT_DIM` : 304 → 314. Constante `N_SEEDS_DIM = 10`.
- **FiLM conditioning** : `FiLMLayer` + `FiLMDQNetwork` dans `model.py`. Chaque couche cachée modulée par `gamma(seed)*x + beta(seed)`. `DQNAgent` utilise `FiLMDQNetwork` par défaut.
- **310 tests, 0 échec.**

**Tableau comparatif complet (stages 1→3→6→10, 5000 ep/stage, win rate = final 100 épisodes) :**

| Run | Architecture | lr | pool3 | pool6 | pool10 |
|-----|--------------|----|-------|-------|--------|
| A | MLP baseline | 3e-4,1e-4 | 30% | 55% | 5% |
| B | Task-cond input | 3e-4,1e-4 | 0% | **68%** | 7% |
| C | Task-cond input | 3e-4,3e-4,1e-4 | 0% | 19% | 1% |
| D | Task-cond input (stages 1,6,10) | 3e-4,1e-4 | — | 17% | 4% |
| **E** | **FiLM** | **3e-4,1e-4** | **1%** | **51%** | **16%** |

**Analyse complémentaire (2026-05-29) — win rate max ET final par chaîne complète :**

L'architecture réelle de chaque run a été vérifiée via les poids des checkpoints (INPUT_DIM et présence de couches FiLM).

| Architecture | Run | pool6 max | pool6 final | pool10 max | pool10 final |
|-------------|-----|-----------|-------------|-----------|--------------|
| MLP | 27/1534 (2000ep) | 23% | 13% | 24% | 21% |
| MLP | 28/0835 | 71% | 55% | 33% | 5% |
| **Task-cond** | **28/1039** | **69%** | **68%** | **59%** | 7% |
| Task-cond | 28/1411 | 44% | 19% | 51% | 1% |
| FiLM | 28/1620 | 55% | 51% | 34% | **16%** |

**Enseignements clés (corrigés) :**
- Le tableau initial ne mesurait que le win rate **final** — FiLM (16%) semblait meilleur, mais en win rate **max**, task-cond `1039` atteint **59%** sur pool10 et **69%** sur pool6.
- FiLM est le plus **stable** en fin de run (16% final vs 7% pour task-cond), mais task-cond a un pic plus élevé.
- La variance inter-runs est aussi grande que la variance inter-architectures (2 runs par groupe insuffisant pour conclure statistiquement).
- Pool3 échoue systématiquement (0–1%) avec task-conditioning ou FiLM — mais ces poids "ratés" servent de fondation utile pour pool6 (sauter pool3 = pool6 tombe à 17%).
- Le catastrophic forgetting frappe systématiquement entre ep 2500–4000 dans tous les runs.
- Meilleur checkpoint pool3 task-cond : `models/20260528_1039_pool3_ep5000_from_20260528_1037/final.pt`

#### Phase 4 — Amélioration itérative
- ~~Curriculum progressif seeds (1→3→6→10)~~ ✅ effectué
- ~~Augmenter capacité réseau~~ ✅ 304→256→128→64→4
- ~~Replay stratifié par seed~~ ✅ `StratifiedReplayBuffer` implémenté et testé
- ~~Task-conditioning (entrée)~~ ✅ `encode_obs` 304→314, seed one-hot bits 304–313
- ~~FiLM conditioning~~ ✅ `FiLMLayer`, `FiLMDQNetwork`, `DQNAgent` mis à jour
- ~~`--pretrained` CLI curriculum~~ ✅ ajouté à `curriculum.py`
- ~~`--architecture {film,taskcond}` CLI~~ ✅ ajouté à `train.py` et `curriculum.py`
- ~~Métadonnées dans les logs~~ ✅ `_log_meta()` : ligne `{"type":"meta"}` en tête de chaque fichier

**Session du 2026-05-29 :**
- **332 tests, 0 échec.**
- Analyse rétrospective des runs 2026-05-28 : win rate max vs final par architecture (voir tableau ci-dessus).
- Décision : prochain run en **task-cond** (meilleur pic pool10 : 59%, pool6 très stable à 68% final, 2× plus rapide que FiLM).

**Session du 2026-05-29 (suite) — Sélection des seeds pédagogiques**

Analyse de 3 000 seeds (0–2999) via `analyze/search_seeds.py` : coût Dijkstra, détour rochers,
cases eau, position bord. Sélection manuelle de 10 seeds répartis en 5 groupes de difficulté
après inspection visuelle dans l'UI (champ de saisie de seed ajouté au HUD).

**Pool retenu (`N_SEEDS_DIM=10`, inchangé) :**

| Index | Seed | Groupe | Caractéristique principale |
|-------|------|--------|---------------------------|
| 0 | 1619 | Facile (coût 5–8) | chemin court |
| 1 | 1240 | Facile (coût 5–8) | chemin court |
| 2 | 113 | Rochers (coût 9–12) | détour rochers élevé |
| 3 | 173 | Rochers (coût 9–12) | détour rochers élevé |
| 4 | 57 | Mixte (coût 13–16) | eau et/ou rochers |
| 5 | 61 | Mixte (coût 13–16) | eau et/ou rochers |
| 6 | 87 | Complexe (coût 17–20) | eau sur chemin optimal |
| 7 | 278 | Complexe (coût 17–20) | eau sur chemin optimal |
| 8 | 88 | Difficile (coût 21+) | chemin long et tortueux |
| 9 | 361 | Difficile (coût 21+) | chemin long et tortueux |

**⏭️ Prochaine étape :**

**Objectif : curriculum progressif sur les 10 seeds pédagogiques.**

Curriculum en 5 étapes (2 seeds par groupe), architecture task-cond :
```bash
.venv\Scripts\python.exe src/curriculum.py `
    --pool 1619,1240,113,173,57,61,87,278,88,361 `
    --stages 2,4,6,8,10 `
    --max-episodes-per-stage 5000 `
    --win-rate-threshold 0.8 `
    --lr 3e-4,1e-4 `
    --architecture taskcond
```
Durée estimée : ~2h30. Le curriculum part des seeds faciles (groupe 1) et ajoute progressivement
la difficulté, ce qui devrait éviter le catastrophic forgetting observé sur des pools homogènes.

### Stratégie terrains (détail)
| Option | Description | Quand |
|--------|-------------|-------|
| B — Pool fixe | pool 10–100 seeds, decay adaptatif | Référence |
| C — Curriculum | Difficulté progressive par étapes | ← En cours |
| A — Full random | Nouveau terrain à chaque épisode | Long terme |
