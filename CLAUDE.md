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
- **Tests** : pytest (249 tests, 0 échec)
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
│   ├── exploit.py        ← load_net(), run_one_episode(), scan_run_dir() (Phase 3 RL)
│   └── ui.py             ← GameUI, rendu pygame, [IA simple model] [IA multi model] [IA restart]
├── tools/
│   └── migrate_models.py ← migration dossiers existants vers structure *_run/
└── tests/
    ├── conftest.py        ← sys.path setup
    ├── helpers.py         ← FakeGrid partagée
    ├── test_game.py       ← tests Grid, GameState, scoring, trail
    ├── test_pathfinder.py  ← tests PathFinder
    ├── test_dungeon_env.py ← tests DungeonEnv
    ├── test_train.py       ← tests encode_obs, ReplayBuffer, DQNetwork, DQNAgent
    ├── test_curriculum.py  ← tests _win_rate, _train_stage, run_curriculum
    └── test_exploit.py     ← tests load_net, run_one_episode, scan_run_dir
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

#### Phase 3 — Visualisation pygame ✅ implémentée (2026-05-29)
Fichiers : `src/exploit.py`, `src/ui.py`, `tools/migrate_models.py`

**Structure de dossiers des runs (depuis cette session) :**
```
models/
└── 20260529_1258_run/          ← créé par curriculum.py, sélectionnable depuis l'UI
    ├── 20260529_1258_pool2_ep5000/
    │   ├── ep500.pt … final.pt
    └── 20260529_1301_pool4_ep5000_from_1258/
        └── ep500.pt … final.pt
logs/
└── 20260529_1258_run/
    ├── 20260529_1258_pool2_ep5000.jsonl
    └── …
```

**Migration des runs existants :**
```bash
python tools/migrate_models.py --dry-run   # prévisualisation
python tools/migrate_models.py             # migration réelle
```

**Boutons IA dans l'UI (ligne 2 du HUD bas) :**
- `[IA simple model]` : file picker `.pt` → joue un épisode complet (epsilon=0) → tracé orange + chemin optimal rouge
- `[IA multi model]` : directory picker `*_run/` → charge TOUS les checkpoints en thread de fond → animation 200ms/trail avec barre de progression → chemin optimal rouge en fin d'animation
- `[IA restart]` : relance l'animation (trails présents) ou recalcule les trails sans relire le disque (nets en cache `_ai_nets_cache`), ou reload complet si cache vide ; réinitialise les stats avant chaque nouveau calcul

**Ligne 3 du HUD bas — statistiques IA :**
Affichée sous les boutons IA : `Trail X/N   |   Victoires : Y/N   |   Note moy : ZZ`
- **Trail X/N** : progression de l'animation (mode multi uniquement)
- **Victoires : Y/N** : nombre d'épisodes gagnés / total chargés
- **Note moy : ZZ** : `scores_sum / total_episodes` — les échecs (score 0) sont inclus dans la moyenne, ce qui reflète la performance globale réelle du modèle
- Les stats se remettent à zéro à chaque nouveau terrain (reset) et à chaque clic sur IA restart
- Pendant le chargement multi, les stats se mettent à jour incrémentalement

**Couleur unique IA : orange `(255, 140, 0)`**
Alpha par checkpoint : 50 (premier d'un stage) → 220 (dernier), dégradé linéaire.

**`run_one_episode_info(net, seed, seed_idx=0)` dans `exploit.py` :**
Variante de `run_one_episode` retournant `(trail, won, score)`.
`run_one_episode` délègue désormais à `run_one_episode_info` pour éviter la duplication.

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
- **249 tests, 0 échec.**
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

**⏭️ Prochaine session — Généralisation par élargissement du pool**

### Objectif
Démontrer que le réseau peut transférer sa connaissance à des seeds **jamais vus pendant l'entraînement**.
Metric clé : win rate sur seeds hors-training.

### Décision architecturale
Le task-conditioning (one-hot seed, 10 bits) apprend au réseau à tricher ("je suis sur le seed 3 → tourne à droite"). Sur un seed inconnu, ce signal n'a plus de sens. Pour la généralisation, le réseau doit décider **uniquement depuis la grille visible**.

→ Nouveau réseau `ObsDQNetwork` (304→256→128→64→4, sans one-hot seed) + option `--architecture obs` dans `train.py` et `curriculum.py`.

### Étape 0 — Baseline généralisation (avant tout entraînement)
Créer `src/evaluate.py` : prend un checkpoint, teste sur N seeds, retourne win rate + score moyen.
```bash
python src/evaluate.py --checkpoint models/.../final.pt --seeds 100-299
```
Tester le meilleur checkpoint existant (run 28/1039, 59% pool10) sur seeds 100–299.
Ce chiffre devient la **baseline de généralisation**.

### Étape 1 — Entraînement pool100 sans task-conditioning
```bash
python src/train.py --episodes 20000 --seed-pool 0-99 --architecture obs --lr 3e-4
```
- 100 seeds, tirage aléatoire à chaque épisode (la diversité naturelle est le curriculum)
- Checkpoints toutes les 2000 épisodes
- Évaluation après chaque checkpoint sur seeds 100–199 (jamais vus)

### Étape 2 — Décision selon les résultats
| Win rate sur seeds inconnus | Action suivante |
|-----------------------------|-----------------|
| > 40% | ✅ Généralisation amorcée → passer à pool500 |
| 20–40% | 🔄 Ajouter curriculum léger (50→100) + plus d'épisodes |
| < 20% | 🔍 Diagnostic : le réseau mémorise plutôt qu'il ne comprend |

### Étape 3 — Montée en charge (si étape 2 positive)
```
pool100 → pool500 → pool2000 → seed=None (full random)
```
À chaque palier : transfer learning depuis le checkpoint précédent, évaluation sur 200 seeds hors-training.

### Métriques de succès de la session
| Metric | Cible |
|--------|-------|
| Win rate seeds entraînement | > 60% |
| Win rate seeds **inconnus** | > 40% |
| Score moyen victoires inconnus | > 50 |

### Nouveaux fichiers à créer
- `src/evaluate.py` — test d'un checkpoint sur N seeds, rapport win rate / score / distribution
- Ajout `ObsDQNetwork` dans `model.py` (INPUT_DIM = OBS_DIM = 304)
- Ajout `--architecture obs` dans `train.py` et `curriculum.py`

### Stratégie terrains (détail)
| Option | Description | Statut |
|--------|-------------|--------|
| C — Curriculum 10 seeds | Difficulté progressive par étapes | ✅ Testé (59% max pool10) |
| B — Pool fixe large | pool 100–2000 seeds, sans task-cond | ← **Prochaine session** |
| A — Full random | Nouveau terrain à chaque épisode | Long terme |

---

## Session 2026-06-02 — Support PPO UI + évaluation pool100

### Tests : 331 tests, 0 échec

### Nouveaux fichiers
- `tests/test_evaluate.py` — 21 tests (TestParseSeeds, TestEvaluateDQN, TestEvaluateNEpisodes, TestEvaluatePPO)

### Modifications
- `exploit.py` : `load_ppo(path)`, `load_model(path)`, `run_one_episode_info_ppo(model, seed)`,
  dispatch auto DQN/PPO dans `run_one_episode_info`, `scan_run_dir` supporte `.pt` + `.zip`
- `ui.py` : file picker `.pt`+`.zip`, thread multi incrémental, pygame.time.get_ticks() hors thread,
  `_ai_optimal_path` corrigé dans `_restart_ai_anim`
- `analyze/evaluate.py` : `--n-episodes N` (épisodes/seed), `--stochastic` (politique stochastique PPO),
  utilise `load_model` de exploit.py, dict retour enrichi (`n_seeds`, `n_episodes`, `total_episodes`)

### Entraînement PPO pool100 ✅
Run : `20260602_1028_run`, from scratch, 2M timesteps, seeds 0–99, ~38 minutes

**Courbe online :**
| Phase | Timesteps | Win rate online |
|-------|-----------|-----------------|
| Montée | 0–200k | 33% → 63% |
| Consolidation | 200k–800k | 55–**77%** (pic ts 400k) |
| Plateau stable | 800k–2M | 45–70%, moyenne ~57% |
| Final | 2M | 56% |

Pas de catastrophic forgetting sur toute la durée (plancher ~45% vs 0% en DQN).

### Résultats d'évaluation — PPO pool100 vs PPO pool10

| Métrique | Pool10 (500k ts) | Pool100 (2M ts) |
|---|---|---|
| Training — déterministe | 30% (3/10) | **35%** (35/100) |
| Training — stochastique ×5 | 54% (27/50) | — |
| Inconnus 100–299 — déterministe | **4%** (8/200) | 3% (6/200) |
| Inconnus 100–299 — stochastique ×3 | — | **13.8%** (83/600) |
| Score moy wins (inconnus det) | 95.1 | 94.5 |
| Score moy wins (inconnus stoch) | — | 41.5 |

**Observations clés :**
- Pool100 est meilleur sur ses seeds d'entraînement (35% vs 30%) mais pas en généralisation déterministe (3% vs 4%)
- Mode stochastique ×3 → **13.8%** sur seeds inconnus (4.6× mieux que déterministe) : la randomité débloque les situations bloquées
- Quand PPO gagne en stochastique sur seeds inconnus, score moy = 41.5 (sous-optimal) : il trouve la sortie par tâtonnement, pas par stratégie
- Clusters de défaites consécutives (seeds 70–91, 100–155…) → configurations structurellement similaires non résolues

### Conclusion architecturale — Limite du MLP pour la généralisation

**Le problème n'est pas DQN vs PPO, ni la taille du pool. C'est l'architecture MLP.**

Le MLP reçoit la grille comme **liste plate de 300 features indépendantes** (one-hot 100 cases × 3 types).
C'est déjà la grille 10×10×3 aplatie — le réseau voit l'ensemble du terrain.

**Mais le MLP traite chaque position avec des poids différents :** apprendre "rocher en (2,3) → tourner"
et "rocher en (7,8) → tourner" nécessite des exemples séparés pour chaque position.
Il faut des milliers de seeds différents pour couvrir l'espace des configurations.

**CNN = partage de poids spatial :** un seul filtre 3×3 détecte "obstacle à droite" partout sur la grille.
Vu une fois, généralisé partout. C'est l'inductive bias manquant au MLP.

| | MLP (actuel) | CNN (prochaine étape) |
|---|---|---|
| Input | 300 floats (grille aplatie) | Tenseur 10×10×3 |
| "Obstacle à droite" | 100 règles (une par position) | 1 filtre partagé |
| Généralisation spatiale | Nécessite beaucoup de seeds | Automatique |
| Changement requis | — | Architecture réseau uniquement |

### Roadmap RL — état actuel

1. ~~DungeonEnv (interface Gym)~~ ✅
2. ~~Entraînement DQN~~ ✅ (limite : catastrophic forgetting multi-seeds)
3. ~~Task-conditioning / FiLM / ObsDQNetwork~~ ✅ (limite : même problème)
4. ~~Visualisation pygame~~ ✅ (DQN + PPO)
5. ~~PPO Stable-Baselines3~~ ✅ (résout catastrophic forgetting, 89% pool10)
6. ~~UI PPO (.zip)~~ ✅ (load_model, scan_run_dir, animation incrémentale)
7. ~~evaluate.py (--n-episodes, --stochastic)~~ ✅
8. ~~PPO MLP pool100 2M ts~~ ✅ (35% training, 3% inconnus det, 13.8% inconnus stoch)
9. ~~PPO CNN pool100 2M ts~~ ✅ (76% training, 3% inconnus det, 10.7% inconnus stoch)
10. **Full random seeds (seed=None)** ← prochaine étape : diversité maximale

---

## Session 2026-06-02 (fin) — CNN + PPO pool100

### Tests : 360 tests, 0 échec

### Architecture CNN implémentée

**Nouveaux fichiers / modifications :**
- `src/train_ppo.py` :
  - `encode_obs_cnn(obs_dict)` → tenseur `(10,10,5)` : canaux herbe/roche/eau/personnage/sortie
  - `DungeonCnnExtractor(BaseFeaturesExtractor)` : Conv2D(5→16,3×3)→ReLU→Conv2D(16→32,3×3)→ReLU→Flatten→Linear(1152→128)→ReLU
  - `DungeonGymEnv(obs_type='mlp'|'cnn')` — rétrocompatible (défaut='mlp')
  - `train(architecture='mlp'|'cnn')` + `--architecture` CLI
  - Constantes : `CNN_OBS_SHAPE=(10,10,5)`, `CNN_FEATURES_DIM=128`, `CNN_NET_ARCH=[64]`
- `src/exploit.py` : `run_one_episode_info_ppo` — auto-détecte CNN via `model.observation_space.shape`
- `analyze/evaluate.py` : `_run_ppo_episode` — idem (fix : créait toujours env MLP)
- `tests/test_train_ppo.py` : +17 tests (TestEncodeObsCnn, TestDungeonGymEnvCnn, TestDungeonCnnExtractor, TestTrainCnn)
- `tests/test_exploit.py` : +3 tests (TestRunOneEpisodeInfoPPOCnn)
- `tests/test_evaluate.py` : +3 tests (TestEvaluatePPOCnn)

### Résultats PPO CNN pool100 — comparatif complet

| Métrique | MLP pool100 | CNN pool100 | Δ |
|---|---|---|---|
| Win rate online final | 56% | **88%** | +32 pts |
| Training 0–99 — déterministe | 35% | **76%** | +41 pts |
| Training 0–99 — score moy wins | 94.1 | **96.3** | +2 pts |
| Inconnus 100–299 — déterministe | 3% | **3%** | = |
| Inconnus 100–299 — stochastique ×3 | **13.8%** | 10.7% | -3 pts |

### Analyse

Le CNN améliore massivement la mémorisation des seeds vus (+41 pts) mais ne progresse pas
sur les seeds inconnus (3% = 3%). La généralisation stochastique CNN (10.7%) est même légèrement
en dessous du MLP (13.8%).

**Diagnostic :** le problème n'est pas l'invariance spatiale (que CNN résout) mais la
**planification globale** : naviguer vers une cible à travers des obstacles requiert un
raisonnement sur le chemin complet, que ni MLP ni CNN ne résolvent nativement.

### Pistes pour la généralisation

| Approche | Ce qu'elle apporte | Complexité | Statut |
|---|---|---|---|
| **Full random seeds (seed=None)** | Diversité max, force vraie stratégie | Faible | ✅ fait |
| Attention / Transformer | Raisonnement global sur la grille | Élevée | — |
| Model-based RL | Planification explicite | Très élevée | — |

---

## Session 2026-06-02 — CNN full-random : percée majeure

### Résultats (run CNN seed=None, 2M timesteps)

| Métrique | MLP pool100 | CNN pool100 | **CNN full-random** |
|---|---|---|---|
| Win rate online final | 56% | 88% | 34% |
| Seeds 0–99, déterministe | 35% | 76% | **8%** |
| Inconnus 100–299, déterministe | 3% | 3% | **10.5%** (+3.5×) |
| Inconnus 100–299, stochastique ×3 | 13.8% | 10.7% | **32.2%** (+2.3×) |
| Score moy wins (inconnus stoch) | 41.5 | 46.7 | 50.4 |

### Analyse

**La vraie percée :** full-random généralise 3× mieux que les modèles pool fixe.
10.5% déterministe et 32.2% stochastique sur seeds jamais vus (vs 3% et ~11% pour pool fixe).

**Observation clé :** seeds 0–99 (8%) ≈ seeds 100–299 (10.5%) — le modèle ne distingue plus
"vu" vs "non vu". C'est de la vraie généralisation. Pour CNN pool100, l'écart était 76% vs 3%.

**Coût :** seulement 8% déterministe sur seed fixe (vs 76% CNN pool100). Normal : il navigue
au lieu de mémoriser.

**Conclusion définitive :** ce n'était pas une question d'architecture (MLP vs CNN) — c'était
une question de **diversité des données d'entraînement**. Full-random force une vraie stratégie
de navigation. Pool fixe → mémorisation.

### Roadmap RL — état mis à jour
1. DungeonEnv ✅
2. Entraînement DQN ✅
3. Task-conditioning / FiLM / ObsDQNetwork ✅
4. Visualisation pygame ✅
5. PPO Stable-Baselines3 ✅ — percée : 89% max, pas de catastrophic forgetting
6. UI PPO (.zip) ✅
7. evaluate.py (--n-episodes, --stochastic) ✅
8. PPO MLP pool100 2M ts ✅ — 35% training, 3% inconnus det
9. PPO CNN pool100 2M ts ✅ — 76% training, 3% inconnus det
10. **PPO CNN full-random 2M ts ✅** — 8% training, **10.5% inconnus det, 32.2% stoch** ← percée
11. **Continuer entraînement CNN full-random → objectif >90% win rate** ← prochaine session

---

## Session 2026-06-02 (UI) — Corrections bouton "IA restart"

### Bug : animation jouée deux fois
Pendant `_rerun_from_cache`, les trails s'ajoutaient à `_ai_trails` incrémentalement.
La boucle `run()` avançait `_anim_idx` dès le premier trail (condition `_anim_idx == -1`).
En fin de thread : `_anim_idx = -1` était réassigné + `_loading_progress = None`
→ la boucle relançait l'animation depuis le début → **double passage**.

**Fix** (`run()`) : ajout de `and self._loading_progress is None` dans la condition d'avancement.
Le chargement se fait silencieusement, l'animation joue une seule fois proprement après.

### Loader "Calcul..." animé sur le bouton
Pendant le chargement, le bouton "IA restart" affiche un texte cyclique `"Calcul."` / `"Calcul.."` /
`"Calcul..."` / `"Calcul"` (rotation toutes les 400ms via `pygame.time.get_ticks() // 400 % 4`).
Revient à `"IA restart"` dès que `_loading_progress is None`. Bouton GREY pendant loading.

---

## Session 2026-06-03 — Sprites PNG + refonte UI + continuation entraînement

### Tests : 362 tests, 0 échec

### Nouveaux fichiers
- `src/run_utils.py` — `_now()` et `_pretrained_label()` factorisés (partagés par `train.py` et `train_ppo.py`)
- `assets/tiles/grass.png`, `rock.png`, `water.png` — tuiles terrain PNG
- `assets/tiles/player.png` — guerrier pixel art 80×80 (généré avec Pillow : heaume, plastron, jambes)
- `assets/tiles/castle.png` — porte de sortie PNG
- `analyze/rename_run_from.py` — renomme un run PPO en ajoutant `_from_<ts>`

### Modifications

**`src/run_utils.py`** (nouveau) :
- `_now()` et `_pretrained_label()` extraits de `train.py` où ils étaient dupliqués dans `train_ppo.py`
- `train.py` et `train_ppo.py` importent depuis `run_utils` — `curriculum.py` inchangé

**`src/train_ppo.py`** :
- Correction : `_from_<ts>` ajouté au nom du run quand `--pretrained` est fourni
- `_pretrained_ts()` et `_now()` locales supprimées (→ `run_utils`)

**`src/ui.py`** — refonte complète :

*Sprites PNG :*
- `_ASSETS_DIR` = `assets/tiles/` (relatif à `ui.py`)
- `_load_tile_surfaces()` : grass/rock/water.png → dict `{TileType → Surface}`, fallback couleur si absent
- `_load_sprite(filename)` : charge player.png ou castle.png, redimensionné à `TILE_PX×TILE_PX`
- `_draw_grid()` : blit PNG → `draw.rect` coloré (fallback)
- `_draw_character()` : blit `player.png` (fallback : stick figure jaune)
- `_draw_castle()` : renommée depuis `_draw_exit()`, blit `castle.png` (fallback : porte jaune)

*Paramètres visuels :*
- `TILE_PX = 80` (cases doublées), `SEP_PX = 0` (plus de séparateur)
- `TRAIL_WIDTH = 3`, `TRAIL_OFFSET = 5`
- Trails : rouge `(−5,−5)` gauche / jaune `(0,0)` centre / orange `(+5,+5)` droite
- Surface multi-trails : colorkey + `set_alpha` remplace SRCALPHA (garantit TRAIL_WIDTH effectif)

*Correction compteur victoires :*
- `won` et `score` stockés dans chaque dict de trail
- Stats affichées synchronisées avec l'animation (`_ai_trails[:shown]`), plus figées en fin de chargement

*Refonte HUD :*
- `HUD_TOP_H = 108`, `HUD_BOT_H = 0` (HUD bas supprimé)
- Ligne 1 : Déplacements / Note / Information + touches fléchées (gris, aligné droite)
- Ligne 2 : `[Génération terrain]` + `Seed:[input]` + `Trail X/N | Victoires Y/N | Note moy ZZ`
- Ligne 3 : `[IA simple model]` `[IA multi model]` `[IA restart]`
- Ligne 4 : barre de chargement 8px (espace toujours réservé, dessinée seulement si loading)
- Tous les textes en blanc sauf touches fléchées (gris)
- "IA restart" efface les trails orange+rouge avant de recalculer

### Entraînement CNN full-random — run terminé (2026-06-03)
- Commande : `--pretrained 20260602_1556_run/final.zip --architecture cnn --lr 1e-4 --n-envs 8 --timesteps 5000000`
- Dossier : `models/20260603_0811_run/20260603_0811_ppo_random_cnn_ts5000000_from_20260602_1556/`
- ⚠️ Bug découvert et corrigé : `LogCallback` loggait `LEARNING_RATE` (constante 3e-4) au lieu du lr effectif. La ligne `meta` de ce run affiche `lr=0.0003` mais l'entraînement s'est bien déroulé à `lr=1e-4`. Les résultats restent valides.
- Baseline avant ce run : **30.8% stoch ×3** seeds 100–299

**Résultats (seeds 100–299 inconnus) :**
| Mode | Win rate | Score moy wins |
|---|---|---|
| Déterministe | **56%** | 93.7 |
| Stochastique ×3 | **73.2%** | 78.1 |

Win rate online en fin de run : ~68% (plateau entre 3M–5M ts, +5 pts sur les 2 derniers millions).

### Roadmap RL — état mis à jour
1. DungeonEnv ✅
2. Entraînement DQN ✅
3. Task-conditioning / FiLM / ObsDQNetwork ✅
4. Visualisation pygame ✅ — sprites PNG, HUD redesigné
5. PPO Stable-Baselines3 ✅ — percée : 89% max, pas de catastrophic forgetting
6. UI PPO (.zip) ✅
7. evaluate.py ✅
8. PPO MLP pool100 2M ts ✅ — 35% training, 3% inconnus det
9. PPO CNN pool100 2M ts ✅ — 76% training, 3% inconnus det
10. PPO CNN full-random 2M ts ✅ — baseline 30.8% stoch seeds inconnus
11. PPO CNN full-random +5M ts ✅ — **56% det, 73.2% stoch** seeds inconnus (run 20260603_0811, lr=1e-4)
12. **Continuer entraînement ou nouvelle piste** ← prochaine session
