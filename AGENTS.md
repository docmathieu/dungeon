# AGENTS — Dungeon POC

Ce fichier décrit les agents Claude Code disponibles dans ce projet.
Chaque agent est déclenché via le skill correspondant dans Claude Code (VSCode ou CLI).

---

## Agent : generate-game
**Skill** : `/generate-game`
**Fichiers produits** : `src/` (7 fichiers)

Génère le code source complet du jeu selon la spécification dans CLAUDE.md.

Structure de fichiers produite :
```
src/
├── directions.py     ← constante DIRECTIONS partagée (LEFT/RIGHT/UP/DOWN → dx,dy)
├── grid.py           ← TileType (GRASS/ROCK/WATER), Grid (30% ROCK, 20% WATER)
├── pathfinder.py     ← PathFinder : Dijkstra pondéré (herbe=1, eau=2)
├── game_state.py     ← GameState : déplacements, score 0-100, trail, optimal_path
├── simulation.py     ← Simulation (thread, modes UI et headless)
├── ui.py             ← GameUI : rendu pygame, trails jaune/rouge, HUD
└── main.py           ← point d'entrée : python src/main.py
```

Fonctionnalités couvertes :
- Grille pygame 10×10, cases 40px, séparateurs 1px, fond noir
- Herbe/Pierre/Eau avec répartition aléatoire et seed reproductible
- Personnage et sortie placés aléatoirement sur cases herbe
- Restart automatique si Dijkstra ne trouve aucun chemin (`create_solvable`)
- Trail jaune = chemin du joueur ; trail rouge (+2px) = chemin optimal, visible en fin de partie
- Score 0-100 : `round(100 × optimal_cost / player_cost)`, 100 = chemin optimal
- Choc contre un rocher ou bord de grille : `move_count += 1`, position inchangée, trail inchangé
- Interface : labels déplacements/note/information, légende touches, bouton restart uniquement
- Contrôle : touches fléchées → `apply_move()` direct (pas de thread, pas de queue dans l'UI)

Contraintes imposées à l'agent :
- `directions.py`, `grid.py`, `game_state.py`, `pathfinder.py`, `simulation.py` : aucun import pygame
- `directions.DIRECTIONS` est la seule définition du mapping direction→delta ; ne pas redéfinir localement
- `GameState` appelle `PathFinder().find_shortest_path()` **une seule fois** à l'init pour dériver `_optimal_cost` ET `optimal_path`
- Point d'entrée unique : `python src/main.py`

---

## Agent : generate-tests
**Skill** : `/generate-tests`
**Fichiers produits** : `tests/helpers.py`, `tests/test_game.py`, `tests/test_pathfinder.py`
**Prérequis** : `src/` doit exister

Génère les tests unitaires pytest couvrant :

**`tests/helpers.py`** — utilitaires partagés
- `FakeGrid` : grille tout-herbe 10×10 avec overrides par cellule (ne pas redéfinir dans les fichiers de test)

**`tests/test_game.py`** — logique de jeu
- `TileType` : valeurs enum, membres distincts
- `Grid` : proportions tuiles, seed déterministe, index, accesseurs, passabilité, coût
- `GameState` : init, déplacements (4 directions, bordures, roches, eau), accumulation coût
- Condition de victoire : score 0-100, formule exacte, eau prise en compte
- `trail` : initialisé avec char_pos, alimenté à chaque déplacement réussi, inchangé si bloqué
- `optimal_path` : positions du chemin optimal, None si inaccessible, cohérent avec `_optimal_cost`
- `is_solvable()` et `create_solvable()` : retry jusqu'à obtenir un terrain jouable

**`tests/test_pathfinder.py`** — PathFinder Dijkstra
- start == end, chemins de base (4 directions, coin à coin)
- Coût eau (2) vs herbe (1), préférence du détour herbe sur corridor eau
- Rochers bloquants, aucun chemin (None), couloir unique
- Cohérence coût rapporté vs chemin simulé

Contraintes :
- `pytest` uniquement, aucun import pygame
- `from helpers import FakeGrid` — ne jamais redéfinir FakeGrid dans un fichier de test
- `from directions import DIRECTIONS` — ne jamais redéfinir un mapping direction→delta local
- `make_state()` dans test_game.py doit recalculer `_optimal_cost`, `optimal_path` ET `trail` après avoir forcé `char_pos`/`exit_pos`

---

## Agent : run-tests
**Skill** : `/run-tests`
**Prérequis** : tests générés, dépendances installées (`.venv`)

Lance `.venv\Scripts\python.exe -m pytest tests/ -v` et rapporte :
- Nombre de tests passés / échoués (cible : 170+)
- En cas d'échec : résumé des assertions qui ont échoué et fichier concerné

---

## Agent : run-game
**Skill** : `/run-game`
**Prérequis** : `src/main.py` doit exister, `pygame` installé dans `.venv`

Démarre le jeu via `.venv\Scripts\python.exe src/main.py`.

---

## Agent : build-exe
**Skill** : `/build-exe`
**Produit** : `dist/dungeon.exe`
**Prérequis** : `src/main.py` doit exister, `pyinstaller` installé dans `.venv`

Génère un exécutable Windows autonome via PyInstaller :
- `--onefile` : fichier unique
- `--noconsole` : pas de fenêtre console
- `--name dungeon`
- Point d'entrée : `src/main.py`
- Inclut tous les modules `src/` dans le bundle

---

## Agent : dungeon-env ✅ *(Phase 1 RL)*
**Skill** : `/dungeon-env`
**Fichier produit** : `src/dungeon_env.py`
**Tests** : `tests/test_dungeon_env.py` (43 tests)

Interface Gym (`reset` / `step`) wrappant `GameState` pour l'entraînement RL :

```python
env = DungeonEnv(seed=42)           # terrain reproductible
env = DungeonEnv(seed=None)         # terrain aléatoire à chaque reset()
env = DungeonEnv(seed_pool=[...])   # tirage dans un ensemble fixe

obs              = env.reset()      # → dict {grid, char_pos, exit_pos}
obs, reward, done, info = env.step("RIGHT")
# reward : victoire → score/100.0 (1.0=optimal)
#          déplacement normal → -0.01
#          choc mur/bord      → -0.05
# done   = True si victoire ou steps >= max_steps (défaut 100)
```

Observation :
- `grid` : list[int] de 100 valeurs (0=HERBE, 1=ROCHE, 2=EAU), ordre ligne-major
- `char_pos` / `exit_pos` : tuple (x, y)

Contraintes :
- Aucun import pygame
- Pas de dépendance numpy (encodage en int Python natif — la couche d'entraînement convertit en tenseur)

---

## Agents à venir — Roadmap RL

### Agent : train-rl ✅ *(Phase 2)*
**Skill** : `/train-rl`
**Fichiers produits** : `src/model.py`, `src/train.py`, `models/`, `logs/`
**Tests** : `tests/test_train.py`

Boucle d'entraînement DQN headless.

```bash
python src/train.py --episodes 5000 --seed-pool 0,1,2,... --architecture obs
```

Architectures disponibles (`--architecture`) :
- `obs` — `ObsDQNetwork` (304→256→128→64→4) : grille seule, sans seed one-hot
  ⚠️ Ne résout pas le catastrophic forgetting multi-seeds — pire que task-cond car aucune séparation possible
- `taskcond` — `DQNetwork` (314→256→128→64→4) : grille + seed one-hot concaténé
- `film` — `FiLMDQNetwork` (obs 304 + task 10) : grille + FiLM conditioning par couche (défaut)

Composants :
- `encode_obs_pure(obs)` — one-hot grille (300) + positions normalisées (4) = **304 floats** (sans seed)
- `encode_obs(obs, seed_idx)` — idem + seed one-hot 10 bits = **314 floats**
- `_encoder_for(arch)` — retourne l'encodeur adapté à l'architecture
- `DQNAgent.arch` + `DQNAgent._encode` — encodeur sélectionné à l'init, utilisé dans `_run_episode`
- `ReplayBuffer` — buffer circulaire FIFO, capacité 10 000
- `StratifiedReplayBuffer(capacity, n_seeds)` — un sous-buffer par seed, batch équilibré
- Logs JSON : première ligne `{"type":"meta"}`, puis `{"episode", "score", "moves", "epsilon", "reward"}`
- **Nommage CLI** : sorties dans `logs/{ts}_run/{run}.jsonl` et `models/{ts}_run/{run}/` (cohérence curriculum)
- `{label}` = `seed42` / `pool10` / `random` — checkpoints `ep<N>.pt` tous les 500 épisodes + `final.pt`
- `--pretrained path/final.pt` : repart des poids d'un run précédent (transfer learning)

### Agent : curriculum ✅ *(Phase 2 — curriculum)*
**Skill** : `/curriculum`
**Fichiers produits** : `src/curriculum.py`, `models/`, `logs/`
**Tests** : `tests/test_curriculum.py` (22 tests)

Entraînement progressif par curriculum : élargit le pool de seeds par étapes, en passant à l'étape
suivante quand l'agent atteint le win rate cible ou quand `--max-episodes-per-stage` est épuisé.

```bash
python src/curriculum.py --pool 0,1,2,3,4,5,6,7,8,9 \
                         --stages 1,3,6,10 \
                         --max-episodes-per-stage 2000 \
                         --win-rate-threshold 0.8 \
                         --lr 3e-4,1e-4
```

Composants :
- `_pad_lr(lrs, n)` — normalise la liste de lr à n éléments (répète la dernière valeur ou tronque)
- `_win_rate(scores, window=100)` — taux de victoire sur les N derniers épisodes (score > 0 = victoire)
- `_train_stage(...)` — entraîne jusqu'à maîtrise (win rate ≥ seuil sur 100 ep) ou max_episodes
- `run_curriculum(pool, stages, lr, ...)` — orchestre les étapes, route `lr[stage_idx]` à chaque stage
- Chaque étape produit : `logs/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}].jsonl` + `models/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}]/final.pt`
- Transfer learning automatique : chaque étape repart des poids de l'étape précédente (`_from_` dans le nom)

Paramètres CLI :
- `--pool` (obligatoire) : seeds disponibles, ex. `0,1,2,3,4,5,6,7,8,9`
- `--stages` : seeds par étape (défaut `1,3,6,10`)
- `--max-episodes-per-stage` : plafond épisodes par étape / timeout (défaut `2000`)
- `--win-rate-threshold` : taux de victoire cible pour progresser (défaut `0.8`)
- `--lr` : learning rate(s) par étape, ex. `3e-4,1e-4` — liste de mêmes longueur que `stages` ou plus courte (dernière valeur répétée) (défaut `1e-3`)

---

## Conclusions expérimentales DQN (2026-06-01) — Limites atteintes

### Catastrophic forgetting — diagnostic définitif

Après 10+ runs couvrant toutes les architectures (MLP, task-cond, FiLM, obs) et stratégies
(curriculum, stratified replay, transfer learning, pool100) :

**Le catastrophic forgetting n'est pas un problème d'architecture — c'est un problème d'algorithme.**

DQN avec gradient descent classique écrase les poids précédents à chaque mise à jour.
Quand deux seeds demandent des actions opposées dans des états similaires, aucun réseau
MLP ne peut maintenir les deux politiques simultanément sans mécanisme explicite.

| Architecture | Séparation politiques | Généralisation | Catastrophic forgetting |
|---|---|---|---|
| `taskcond` | ✅ via one-hot seed | ❌ (triche par seed) | ✅ présent pool3+ |
| `film` | ✅ via FiLM layers | ❌ (triche par seed) | ✅ présent pool3+ |
| `obs` | ❌ aucune | ❌ (pas mieux) | ✅ présent pool2+ |

**Résultat expérimental `obs` curriculum (2026-06-01) :**
- Stage 1 (seed0, pretrained) : 80% win rate atteint en 172 épisodes ✅
- Stage 2 (pool3) : pic 23% à ep200, effondrement à 0% dès ep1000 — même pattern que FiLM/task-cond

**Résultat expérimental `obs` pool100 (2026-06-01) :**
- 20 000 épisodes, plafond 4% sur seeds inconnus ET seeds vus → le réseau n'apprend rien
- Cause : ~200 épisodes/seed, insuffisant pour converger

### Ce qui fonctionne
- **Seed unique** : toutes les architectures convergent en <2000 épisodes
- **Pool ≤ 3 seeds** : convergence partielle possible (23–55% selon l'archi)
- **Meilleur checkpoint exploitable** : task-cond run 28/1039, pool10, ~59% win rate max

### Pistes restantes (non testées avec DQN)
- **PPO** (Proximal Policy Optimization) — algorithme on-policy, mises à jour bornées,
  moins susceptible d'écraser les politiques précédentes
- **Accepter la limite** : pool fixe de seeds connus avec task-cond/FiLM,
  sans prétendre à la généralisation

---

### Agent : train-ppo ✅ *(Phase 2 — PPO)*
**Fichier** : `src/train_ppo.py`
**Tests** : `tests/test_train_ppo.py` (33 tests)

Entraînement PPO via Stable-Baselines3 — résout le catastrophic forgetting multi-seeds.

```bash
python src/train_ppo.py --timesteps 500000 --seed-pool 0,1,2,3,4,5,6,7,8,9
python src/train_ppo.py --timesteps 200000 --seed 0
python src/train_ppo.py --timesteps 500000 --seed-pool 0-99 --n-envs 4
python src/train_ppo.py --timesteps 500000 --seed-pool 0,...,9 --pretrained models/.../final.zip
```

**`DungeonGymEnv(gym.Env)`** — wrapper gymnasium/SB3 :
- `observation_space = Box(0, 1, shape=(304,))` — même encodage que `encode_obs_pure`
- `action_space = Discrete(4)` — LEFT/RIGHT/UP/DOWN
- `reset()` → `(np.ndarray 304, info_dict)`
- `step(action_int)` → `(obs, reward, terminated, truncated, info)`
- `info` contient : `score`, `moves`, `steps`, `won`, `seed`

**`LogCallback`** — log JSON par épisode terminé :
```json
{"episode": 42, "timestep": 3800, "seed": 0, "won": true, "score": 100, "moves": 8, "reward": 0.93}
```
Affichage toutes les 10 000 timesteps : `ts X/N  ep=N  wr=X%  t=Xs`

**Hyperparamètres PPO** (défauts) :
- `N_STEPS=2048`, `BATCH_SIZE=64`, `N_EPOCHS=10`
- `GAMMA=0.99`, `GAE_LAMBDA=0.95`, `CLIP_RANGE=0.2`
- `NET_ARCH=[256, 128, 64]`, `LR=3e-4`
- Checkpoints `.zip` tous les 50 000 timesteps + `final.zip`

**Sorties** : `logs/{ts}_run/{ts}_ppo_{label}_ts{N}.jsonl` et `models/{ts}_run/{ts}_ppo_{label}_ts{N}/`

**Résultats run pool10, 500k timesteps (2026-06-01) :**

| Métrique | DQN (meilleur) | PPO |
|----------|---------------|-----|
| Win rate max online | 59% (task-cond) | **89%** |
| Win rate plancher | 0% (effondrement) | **30%** |
| Catastrophic forgetting | ✅ oui | **❌ non** |
| Eval déterministe pool10 | — | 30–50% |

⚠️ Écart online vs eval déterministe : PPO utilise une politique stochastique à l'entraînement.
`deterministic=True` dans l'éval donne des scores plus bas — prévoir plusieurs épisodes/seed.

---

### ⏭️ Prochaines étapes PPO (session suivante)

**1. ~~Adapter l'UI pour charger les modèles PPO (.zip)~~ ✅ (2026-06-02)**
- `load_ppo`, `load_model`, `run_one_episode_info_ppo` ajoutés à `exploit.py`
- `scan_run_dir` supporte `.pt` et `.zip`
- File picker UI accepte `.pt` et `.zip` ; animation multi-model supporte les deux formats

**2. Améliorer l'évaluation**
- Tester avec plusieurs épisodes par seed (politique stochastique vs déterministe)
- Évaluer sur seeds 100–299 (généralisation hors-training)

**3. Entraîner pool100 avec PPO**
```bash
python src/train_ppo.py --timesteps 2000000 --seed-pool 0-99
```

---

### Agent : replay-model *(Phase 3 — visualisation pygame)* ✅
Fichiers : `src/exploit.py`, `src/ui.py`
Tests : `tests/test_exploit.py` (55 tests)

**Chargement de modèles :**

`load_net(path)` — charge un checkpoint `.pt` DQN, détecte l'architecture automatiquement :
- Clés contenant `film` → `FiLMDQNetwork`
- `net.0.weight.shape[1] == OBS_DIM (304)` → `ObsDQNetwork`
- Sinon → `DQNetwork`

`load_ppo(path)` — charge un checkpoint `.zip` PPO Stable-Baselines3 (`PPO.load(path)`).

`load_model(path)` — dispatcher selon l'extension :
- `.pt` → `load_net`
- `.zip` → `load_ppo`

**Exécution d'épisodes :**

`run_one_episode_info(model, seed, seed_idx=0)` → `(trail, won, score)`
- Dispatch automatique : si `model` possède `predict()` (SB3 PPO) → `run_one_episode_info_ppo`
- Sinon (DQN) : utilise `encode_obs_pure` pour `ObsDQNetwork`, `encode_obs` pour les autres

`run_one_episode_info_ppo(model, seed)` → `(trail, won, score)`
- Crée un `DungeonGymEnv(seed=seed)`, joue en mode déterministe (`deterministic=True`)

`run_one_episode(model, seed, seed_idx=0)` → `trail` (délègue à `run_one_episode_info`)

**Exploration de runs :**

`scan_run_dir(run_dir)` → liste ordonnée de checkpoints `{stage_idx, color, pt_path}`
- Supporte les checkpoints DQN (`.pt`) et PPO (`.zip`) dans le même run
- Tri numérique : `ep{N}.pt` et `ppo_{N}_steps.zip` → N, `final.*` → +∞

**Boutons IA dans l'UI :**
- `[IA simple model]` : file picker acceptant `.pt` et `.zip` → joue un épisode, affiche trail orange + chemin rouge
- `[IA multi model]` : directory picker `*_run/` → charge TOUS les checkpoints (DQN et/ou PPO) en thread de fond → animation incrémentale 200ms/trail + barre de progression
- `[IA restart]` : rejoue le(s) modèle(s) chargé(s) sur le terrain courant ; met à jour trail ET chemin optimal

---

## Scripts d'analyse (`analyze/`)

Scripts utilitaires pour comprendre les seeds et le comportement RL.
À lancer manuellement depuis la racine du projet — pas de skill associé.

### `analyze/evaluate.py` *(ajouté 2026-06-01, amélioré 2026-06-02)*
**Objectif** : évaluer la performance d'un checkpoint sur un ensemble de seeds (connus ou inconnus).
**Tests** : `tests/test_evaluate.py` (21 tests)

**Usage** :
```bash
.venv\Scripts\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 0-9
.venv\Scripts\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 100-299
.venv\Scripts\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 0-9 --n-episodes 5
.venv\Scripts\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 0-9 --stochastic --n-episodes 10
.venv\Scripts\python.exe analyze/evaluate.py --checkpoint models/.../final.pt  --seeds 100-299 --verbose
```

**Options** :
- `--n-episodes N` : épisodes par seed (défaut 1 ; utile en stochastique pour mesurer la variance)
- `--stochastic` : politique stochastique PPO (`deterministic=False`), ignoré pour DQN
- `--verbose` : détail par seed (win/loss + score, ou win rate + score moy si n-episodes > 1)

**Sorties** :
- `Victoires : N/total (X%)[épisodes total]` — win rate
- `Score moyen : X` — tous épisodes (échecs = 0)
- `Score moyen wins : X` — victoires uniquement

**Détection automatique** via `load_model` : `.pt` → DQN (FiLM / ObsDQNetwork / DQNetwork), `.zip` → PPO SB3.

**Résultats de référence :**
- Baseline task-cond DQN (run 28/1039, seeds 100–299) : **1.0%** déterministe
- PPO pool10 500k ts (seeds 100–299) : **4.0%** déterministe, **~14%** stochastique (estimé)
- PPO pool100 2M ts (seeds 0–99 vus) : **35%** déterministe
- PPO pool100 2M ts (seeds 100–299 inconnus) : **3%** déterministe, **13.8%** stochastique ×3

**⚠️ Stochastique >> déterministe pour PPO :** le mode greedy se coince dans des boucles sur les
configurations inconnues. La stochasticité débride la politique (+4.6× sur seeds inconnus).
Contrepartie : score moy wins chute (94.5 → 41.5) — la politique tâtonne plutôt que de naviguer.

---

## Conclusion architecturale — Limite du MLP (2026-06-02)

### Diagnostic définitif

La généralisation plafonne (~4% déterministe, ~14% stochastique sur seeds inconnus) **quel que soit
l'algorithme** (DQN ou PPO) et **quel que soit le pool** (10 ou 100 seeds).

**Cause :** le MLP reçoit la grille comme une liste plate de 300 features indépendantes.
C'est déjà le 10×10×3 aplati — le réseau voit tout. Mais chaque position utilise des poids différents :
apprendre "obstacle en (2,3) → tourner" ne transfère pas à "obstacle en (7,8) → tourner".

### Solution : CNN (prochaine étape)

Un filtre convolutionnel 3×3 s'applique à toutes les positions de la grille avec les mêmes poids :
vu une fois en (2,3), généralisé en (7,8) automatiquement. L'inductive bias spatial manquant au MLP.

| | MLP (actuel) | CNN (prochaine étape) |
|---|---|---|
| Input | 300 floats (grille aplatie) | Tenseur 10×10×3 |
| "Obstacle à droite" | 100 règles (une par position) | 1 filtre partagé |
| Généralisation | Nécessite beaucoup de seeds | Automatique |

**Architecture cible : CNN + PPO** (PPO résout le catastrophic forgetting, CNN résout la généralisation).

### `analyze/search_seeds.py`
**Objectif** : trouver 20 seeds pédagogiquement intéressants pour le curriculum RL.

**Contexte** : les seeds arbitraires (0..9) présentent des problèmes identifiés —
doublons (seeds 3 et 4 identiques), déséquilibre de difficulté (coût 5 à 20),
signal de gradient dominé par les seeds faciles (seed=5, UP×5).

**Métriques calculées** :
- `optimal_cost` : coût Dijkstra (herbe=1, eau=2)
- `optimal_moves` : nombre de moves sur le chemin optimal
- `rock_detour` : optimal_moves − manhattan (>0 = rochers forcent un contournement)
- `water_steps` : cases EAU traversées sur le chemin optimal
- `near_border` : personnage ou sortie sur le bord de la grille

**Groupes cibles** (4 seeds chacun) :

| Groupe | Coût | Priorité de sélection |
|--------|------|-----------------------|
| Facile | 5–8 | rock_detour=0, chemin court |
| Rochers | 9–12 | rock_detour > 0 |
| Mixte | 13–16 | eau et/ou rochers |
| Complexe | 17–20 | water_steps > 0 |
| Difficile | 21+ | chemin long |

**Utilisation** :
```bash
.venv\Scripts\python.exe analyze/search_seeds.py
```
