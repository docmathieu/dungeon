# Dungeon RL — Navigation autonome par apprentissage par renforcement

Jeu de grille 10×10 en Python/pygame servant de terrain d'expérimentation pour l'**apprentissage par renforcement** (Reinforcement Learning). Un personnage doit rejoindre une sortie en évitant les obstacles. L'objectif est d'entraîner un agent IA capable de trouver seul le chemin optimal — y compris sur des terrains qu'il n'a jamais vus.

---

## Table des matières

1. [Librairies utilisées](#librairies-utilisées)
2. [Technologies IA](#technologies-ia)
3. [Scripts et commandes](#scripts-et-commandes)
4. [Interface graphique](#interface-graphique)

---

## Librairies utilisées

| Librairie | Version | Rôle |
|---|---|---|
| **pygame** | 2.6.1 | Affichage de la grille, sprites PNG, gestion des événements clavier/souris |
| **torch** | 2.12.0+cpu | Calcul tensoriel, réseaux de neurones DQN (MLP, FiLM, CNN) |
| **stable-baselines3** | ≥ 2.3.0 | Implémentation de l'algorithme PPO, gestion des environnements vectorisés |
| **gymnasium** | ≥ 0.29.0 | Interface standard `reset()` / `step()` pour les environnements RL |
| **numpy** | ≥ 1.24.0 | Encodage des observations, calculs vectoriels |
| **pytest** | 8.3.5 | Tests unitaires (362 tests, 0 échec) |
| **pytest-cov** | 6.1.0 | Mesure de la couverture de code |
| **pyinstaller** | 6.13.0 | Compilation en exécutable Windows autonome (`dungeon.exe`) |

---

## Technologies IA

### Algorithmes explorés

#### DQN — Deep Q-Network *(abandonné pour le multi-seeds)*

DQN est un algorithme **off-policy** : l'agent mémorise ses expériences dans un replay buffer et apprend en rejouant des transitions passées. À chaque mise à jour, les poids du réseau peuvent être fortement modifiés.

**Problème rencontré :** quand plusieurs terrains différents sont utilisés, deux seeds peuvent exiger des actions opposées dans des états visuellement similaires. DQN écrase les poids appris pour un seed en apprenant un autre → **catastrophic forgetting** systématique au-delà de 3 seeds.

**Résultat :** excellent sur seed unique (score 100 en <2000 épisodes), inutilisable en généralisation.

---

#### PPO — Proximal Policy Optimization *(retenu)*

PPO est un algorithme **on-policy** avec une contrainte de stabilité : le paramètre `clip_range` limite l'écart entre l'ancienne et la nouvelle politique à chaque mise à jour. L'agent ne peut pas "désapprendre" brusquement.

**Pourquoi PPO résout le catastrophic forgetting :** les mises à jour bornées préservent les politiques déjà acquises tout en permettant l'amélioration progressive. Sur un pool de 10 seeds, PPO atteint 89% de win rate là où DQN s'effondre à 0%.

---

### Architectures de réseaux

#### MLP — Multi-Layer Perceptron *(baseline)*

Réseau dense entièrement connecté. La grille 10×10 est aplatie en un vecteur de 300 valeurs (one-hot par case × 3 types), auxquelles s'ajoutent 4 positions normalisées → **304 entrées**.

```
Entrée (304) → Dense(256) → ReLU → Dense(128) → ReLU → Dense(64) → ReLU → Sortie (4 Q-values)
```

**Limite :** chaque position de la grille utilise des poids différents. Apprendre "obstacle en (2,3) → tourner" ne transfère pas à "obstacle en (7,8) → tourner". Nécessite des milliers de seeds différents pour couvrir l'espace des configurations.

---

#### Task-conditioning et FiLM *(expérimentaux, non retenus)*

- **Task-conditioning** : le seed est encodé en one-hot (10 bits) et concaténé à l'observation → 314 entrées. Le réseau "triche" en mémorisant une politique par seed plutôt qu'en apprenant à naviguer.
- **FiLM (Feature-wise Linear Modulation)** : le seed module chaque couche cachée via des paramètres `gamma` et `beta` appris. Plus stable que le task-conditioning mais même limitation fondamentale.

Ces deux approches améliorent les performances sur seeds connus mais ne généralisent pas aux seeds inconnus.

---

#### CNN — Convolutional Neural Network *(retenu)*

La grille est encodée en tenseur spatial **10×10×5** (5 canaux : herbe, roche, eau, personnage, sortie). Des filtres convolutionnels glissent sur toute la grille avec les **mêmes poids partout**.

```
Entrée (10×10×5)
  → Conv2D(5→16 filtres, noyau 3×3) → ReLU  [sortie : 8×8×16]
  → Conv2D(16→32 filtres, noyau 3×3) → ReLU  [sortie : 6×6×32]
  → Flatten (1 152 valeurs)
  → Dense(128) → ReLU
  → Dense(64) → ReLU
  → Sortie (4 Q-values)
```

**Avantage :** un filtre "obstacle à droite" apprend à reconnaître un obstacle **en n'importe quelle position** de la grille. Vu une fois en (2,3), généralisé automatiquement en (7,8).

**Résultat CNN + PPO + full-random (état actuel) :**

| Mode évaluation | Seeds entraînement | Seeds inconnus |
|---|---|---|
| Déterministe | ~8% | **73.0%** |
| Stochastique ×3 | — | **85.8%** |

Le modèle généralise sur des terrains jamais vus, avec un score moyen de 95/100 sur les victoires en déterministe (chemin quasi-optimal).

---

### Stratégie d'entraînement retenue

| Approche | Résultat | Statut |
|---|---|---|
| Seed unique | Convergence rapide (<2000 ep) | ✅ fonctionne |
| Pool fixe de seeds + curriculum | 59% max, forgetting au-delà de pool3 | ✅ testé, limité |
| Pool fixe + PPO | 76% training, 3% inconnus | ✅ testé |
| **Full random (seed=None) + PPO + CNN** | **73.0% inconnus déterministe** | ✅ **retenu** |

La clé de la généralisation n'était ni l'architecture ni l'algorithme seuls, mais la **diversité maximale des terrains** combinée au CNN. En voyant un terrain différent à chaque épisode, l'agent ne peut pas mémoriser — il doit apprendre une vraie stratégie de navigation.

---

## Scripts et commandes

> Tous les scripts s'exécutent depuis la racine du projet avec `.venv\Scripts\python.exe`.
> **Python requis : 3.12** (testé et validé sur Python 3.12.x).

---

### Démarrer le jeu (UI interactive)

```bash
.venv\Scripts\python.exe src\main.py
```

---

### Entraînement PPO — `src/train_ppo.py`

Entraînement via Stable-Baselines3. Produit des checkpoints `.zip` et des logs `.jsonl`. Recommandé pour la généralisation multi-seeds.

```bash
.venv\Scripts\python.exe src\train_ppo.py [options]
```

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--timesteps N` | int | 500 000 | Nombre total de pas d'environnement |
| `--seed N` | int | None | Seed fixe — même terrain à chaque épisode |
| `--seed-pool 0-99` | str | None | Pool de seeds. Accepte `0,1,2` ou plage `0-99`. Sans ce paramètre ni `--seed` : full random (seed=None, terrain différent à chaque épisode) |
| `--lr 1e-4` | float | 3e-4 | Learning rate. ⚠️ Ne pas augmenter en cours d'entraînement pour éviter le désapprentissage |
| `--n-envs N` | int | 1 | Nombre d'environnements parallèles (accélère l'entraînement, recommandé : 8) |
| `--pretrained path` | Path | None | Checkpoint `.zip` SB3 de départ (transfer learning) |
| `--architecture` | str | `mlp` | Architecture réseau : `mlp` (304 entrées) ou `cnn` (tenseur 10×10×5, recommandé) |

#### Commande utilisée en production

La configuration qui a donné les meilleurs résultats de généralisation est **PPO + CNN + seeds aléatoires** :

```bash
# Premier run — from scratch
.venv\Scripts\python.exe src\train_ppo.py ^
    --timesteps 5000000 ^
    --architecture cnn ^
    --n-envs 8 ^
    --lr 0.0001

# Runs suivants — continuation depuis le dernier checkpoint
.venv\Scripts\python.exe src\train_ppo.py ^
    --timesteps 5000000 ^
    --architecture cnn ^
    --pretrained "models\{ts}_run\{run}\final.zip" ^
    --n-envs 8 ^
    --lr 0.0001
```

**Pourquoi cette configuration ?**

- **Pas de `--seed` ni `--seed-pool`** (full random) : un terrain complètement nouveau est généré à chaque épisode. L'agent ne peut donc pas mémoriser les solutions — il est forcé d'apprendre une vraie stratégie de navigation générale. Avec un pool fixe, il "apprend par cœur" et ne généralise pas aux terrains inconnus.
- **`--architecture cnn`** : le CNN partage ses filtres sur toute la grille, ce qui lui permet de reconnaître les obstacles et les chemins quelle que soit leur position.
- **`--n-envs 8`** : 8 environnements tournent en parallèle, ce qui multiplie par ~8 la vitesse de collecte d'expériences.
- **`--lr 0.0001`** : un learning rate faible (3× plus petit que le défaut) assure des mises à jour stables et évite d'écraser ce qui a déjà été appris lors des runs précédents.

**Il faut enchaîner plusieurs runs** : chaque run de 5M timesteps prend environ 40–60 minutes et améliore les performances de ~10 pts. L'entraînement converge progressivement par accumulation :

| Run | Timesteps cumulés | Win rate déterministe (seeds inconnus) |
|---|---|---|
| Run 1 (from scratch) | 5M | 10.5% |
| Run 2 | 10M | 56.0% |
| Run 3 | 15M | 66.5% |
| Run 4 | 20M | **73.0%** |

---

### Entraînement DQN — `src/train.py`

Entraînement headless par Q-learning. Produit des checkpoints `.pt` et des logs `.jsonl`.
⚠️ Limité au seed unique ou petit pool (≤3 seeds) — catastrophic forgetting au-delà.

```bash
.venv\Scripts\python.exe src\train.py [options]
```

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--episodes N` | int | 3000 | Nombre d'épisodes d'entraînement |
| `--seed N` | int | None | Seed fixe — même terrain à chaque épisode |
| `--seed-pool 0,1,2` | str | None | Pool de seeds tirés aléatoirement. Accepte `0,1,2` ou plage `0-9` |
| `--lr 3e-4` | float | 1e-3 | Learning rate de l'optimiseur Adam |
| `--pretrained path` | Path | None | Checkpoint `.pt` de départ (transfer learning) |
| `--architecture` | str | `film` | Architecture réseau : `obs` (304 entrées), `taskcond` (314 entrées), `film` (FiLM conditioning) |

**Exemple :**
```bash
.venv\Scripts\python.exe src\train.py --episodes 3000 --seed 42 --architecture film
```

**Sorties :**
- `logs/{ts}_run/{ts}_{label}_ep{N}.jsonl` — log JSON (une ligne par épisode + ligne meta)
- `models/{ts}_run/{ts}_{label}_ep{N}/ep<N>.pt` — checkpoints tous les 500 épisodes
- `models/{ts}_run/{ts}_{label}_ep{N}/final.pt` — poids finaux

---

### Curriculum progressif DQN — `src/curriculum.py`

Élargit progressivement le pool de seeds par étapes. Passe à l'étape suivante quand le win rate cible est atteint.

```bash
.venv\Scripts\python.exe src\curriculum.py [options]
```

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--pool 0,1,...,9` | str | *obligatoire* | Ensemble de seeds disponibles |
| `--stages 1,3,6,10` | str | `1,3,6,10` | Nombre de seeds par étape (progression) |
| `--max-episodes-per-stage N` | int | 2000 | Plafond d'épisodes par étape si le seuil n'est pas atteint |
| `--win-rate-threshold 0.8` | float | 0.8 | Taux de victoire cible (sur 100 derniers épisodes) pour avancer |
| `--lr 3e-4,1e-4` | str | `1e-3` | Learning rate par étape (liste séparée par virgules, dernière valeur répétée) |
| `--pretrained path` | str | None | Checkpoint `.pt` de départ |
| `--architecture` | str | `film` | Architecture réseau : `obs`, `taskcond`, `film` |

**Exemple :**
```bash
.venv\Scripts\python.exe src\curriculum.py ^
    --pool 0,1,2,3,4,5,6,7,8,9 ^
    --stages 1,3,6,10 ^
    --max-episodes-per-stage 2000 ^
    --win-rate-threshold 0.8 ^
    --lr 3e-4,1e-4 ^
    --architecture film
```

---

### Structure des fichiers produits

```
logs/
└── {ts}_run/
    └── {ts}_{label}_ts{N}[_from_{ts}].jsonl   ← log JSON, une ligne par épisode + ligne meta

models/
└── {ts}_run/
    └── {ts}_{label}_ts{N}[_from_{ts}]/
        ├── ppo_50000_steps.zip                 ← checkpoint PPO toutes les 50k steps
        ├── ppo_100000_steps.zip
        ├── ...
        └── final.zip                           ← poids finaux
```

Le suffixe `_from_{ts}` est automatiquement ajouté quand `--pretrained` est utilisé, permettant de tracer la chaîne de transfer learning.

---

### Évaluation d'un checkpoint — `analyze/evaluate.py`

Mesure les performances d'un checkpoint sur un ensemble de seeds (connus ou inconnus).

```bash
.venv\Scripts\python.exe analyze\evaluate.py [options]
```

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `--checkpoint path` | Path | ✅ | Chemin vers le checkpoint `.pt` (DQN) ou `.zip` (PPO) |
| `--seeds 100-299` | str | ✅ | Seeds à évaluer. Accepte plage `100-299` ou liste `0,1,2` |
| `--n-episodes N` | int | Non (1) | Nombre d'épisodes par seed. Utile en mode stochastique pour réduire la variance |
| `--stochastic` | flag | Non | Mode stochastique PPO (`deterministic=False`). Ignoré pour DQN |
| `--verbose` | flag | Non | Affiche le détail par seed (win/loss + score) |

**Mode déterministe vs stochastique :**
- **Déterministe** : l'agent choisit toujours l'action de probabilité maximale. Mesure ce que le modèle *sait* faire.
- **Stochastique** : l'agent tire aléatoirement selon ses probabilités. Peut débloquer des situations difficiles mais les scores sont plus bas.

**Exemples :**
```bash
# Évaluation déterministe sur 200 seeds inconnus
.venv\Scripts\python.exe analyze\evaluate.py ^
    --checkpoint "models\{ts}_run\{run}\final.zip" ^
    --seeds 100-299

# Évaluation stochastique, 3 épisodes par seed
.venv\Scripts\python.exe analyze\evaluate.py ^
    --checkpoint "models\{ts}_run\{run}\final.zip" ^
    --seeds 100-299 ^
    --n-episodes 3 ^
    --stochastic

# Évaluation sur seeds d'entraînement avec détail par seed
.venv\Scripts\python.exe analyze\evaluate.py ^
    --checkpoint "models\{ts}_run\{run}\final.zip" ^
    --seeds 0-99 ^
    --verbose
```

---

### Analyse des runs — `analyze/analyze_runs.py`

Calcule le win rate max et final (fenêtre glissante 100 épisodes) pour chaque run DQN dans `logs/`. Permet de comparer les architectures et stratégies.

```bash
.venv\Scripts\python.exe analyze\analyze_runs.py
```

Aucun paramètre. Affiche un tableau comparatif de tous les runs détectés.

---

### Analyse des seeds — `analyze/search_seeds.py`

Scanne les seeds 0–2999 et calcule pour chacun le coût Dijkstra, le détour rochers, les cases eau traversées. Utile pour sélectionner des seeds pédagogiquement intéressants pour un curriculum.

```bash
.venv\Scripts\python.exe analyze\search_seeds.py
```

Aucun paramètre. Affiche 20 seeds candidats répartis en 5 groupes de difficulté (facile → difficile).

---

### Tests unitaires

```bash
.venv\Scripts\python.exe -m pytest tests\ -v
```

362 tests couvrant : `Grid`, `GameState`, `PathFinder`, `DungeonEnv`, `DQNAgent`, `ReplayBuffer`, `curriculum`, `exploit`, `evaluate`, `train_ppo` (MLP et CNN).

---

### Produire un exécutable Windows

```bash
.venv\Scripts\python.exe -m PyInstaller src\main.py --onefile --noconsole --name dungeon
```

Produit `dist\dungeon.exe` — autonome, ne nécessite pas Python installé.

---

## Interface graphique

Lancer avec `.venv\Scripts\python.exe src\main.py`.

```
┌──────────────────────────────────────────────────────────────────────────────┐  ← HUD (96px)
│ [Génération terrain] Seed:[____] Déplacements: 0  Note: 0  Info:  ← ↑ → ↓ | R restart │  Ligne 1
│ [IA simple model] [Start SM déterministe] [Start SM stochastique]  stats SM  │  Ligne 2
│ [IA multi model]  [Start MM déterministe] [Start MM stochastique]  stats MM  │  Ligne 3
│ ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░            │  Ligne 4 — barre
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                      Grille 10×10  (800×800 px)                              │  ← Terrain
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Terrain de jeu

| Élément | Apparence | Description |
|---|---|---|
| **Herbe** | Tuile verte (grass.png) | Case franchissable, coût 1 déplacement |
| **Roche** | Tuile grise (rock.png) | Obstacle infranchissable. Tenter d'y entrer coûte 1 déplacement sans bouger |
| **Eau** | Tuile bleue (water.png) | Case franchissable, coût 2 déplacements |
| **Personnage** | Guerrier pixel art (player.png) | Position courante du joueur |
| **Sortie** | Porte (castle.png) | Objectif à atteindre |
| **Trail bleu foncé** | Trait centré | Chemin parcouru par le joueur au clavier (affiché en continu) |
| **Trail jaune** | Trait décalé à gauche (−5px) | Chemin optimal Dijkstra (affiché uniquement en fin de partie) |
| **Trail IA simple** | Trait décalé à droite (+5px), orange `(255,140,0)` | Chemin joué par le modèle simple |
| **Trail IA multi** | Trait décalé à droite (+5px), dégradé orange→rouge | Checkpoints animés : premier=orange, dernier=rouge |

### HUD — Ligne 1 : terrain, seed et statistiques de jeu

| Élément | Interaction | Description |
|---|---|---|
| **[Génération terrain]** | Clic | Génère un nouveau terrain aléatoire solvable |
| **Seed : [champ]** | Clic + saisie + Entrée | Génère le terrain correspondant au seed saisi (reproductible) |
| **Déplacements** | Lecture seule | Nombre de déplacements accumulés (herbe=1, eau=2, choc mur=1) |
| **Note** | Lecture seule | Score final 0–100. `round(100 × coût_optimal / coût_joueur)`. 100 = chemin optimal |
| **Info** | Lecture seule | Vide pendant la partie, affiche **GAGNE** à l'arrivée |
| Touches (gris) | — | Rappel des contrôles clavier `← ↑ → ↓ \| R restart` |

### HUD — Ligne 2 : modèle IA simple

| Élément | État | Description |
|---|---|---|
| **[IA simple model]** | Blanc = non chargé / Cyan = chargé | Ouvre un sélecteur de fichier `.pt` (DQN) ou `.zip` (PPO). Charge le modèle sans jouer d'épisode. |
| **[Start SM déterministe]** | Grisé si non chargé / Cyan si prêt | Lance un épisode déterministe (argmax). Affiche le trail orange. |
| **[Start SM stochastique]** | Grisé si non chargé / Cyan si prêt | Lance un épisode stochastique (PPO uniquement). Résultat différent à chaque clic. |
| **Victoires / Note moy** | Lecture seule | Résultat du dernier épisode joué |

### HUD — Ligne 3 : modèles IA multi

| Élément | État | Description |
|---|---|---|
| **[IA multi model]** | Blanc = non chargé / Cyan = chargé | Ouvre un sélecteur de dossier `*_run/`. Charge tous les modèles en fond (barre de progression). Bouton accessible après chargement complet. |
| **[Start MM déterministe]** | Grisé si non chargé / Cyan si prêt | Lance les épisodes déterministes sur le terrain courant. Animation 200ms/trail, dégradé orange→rouge. |
| **[Start MM stochastique]** | Grisé si non chargé / Cyan si prêt | Lance les épisodes stochastiques. Recalcul systématique (résultats variables). |
| **Trail X/N / Victoires / Note moy** | Lecture seule | Progression de l'animation et statistiques cumulées |

> **Exclusivité SM / MM :** charger un modèle simple (SM) efface le mode multi (MM) et vice versa. Un seul bouton peut être cyan à la fois.

### HUD — Ligne 4 : barre de chargement

Visible pendant le chargement des modèles multi (`[IA multi model]`) et pendant le calcul des épisodes (`[Start MM ...]`).

### Contrôles clavier

| Touche | Action |
|---|---|
| **← ↑ → ↓** | Déplace le personnage d'une case dans la direction indiquée |
| **R** | Génère un nouveau terrain aléatoire (équivalent au bouton [Génération terrain]) |
