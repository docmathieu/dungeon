# Guide de démarrage — Dungeon RL

Ce fichier est le point d'entrée pour tout développeur. Il couvre l'installation, le lancement du jeu,
l'entraînement d'un modèle RL, l'évaluation et les outils d'analyse.

---

## 0. Télécharger le modèle final (optionnel)

Le modèle entraîné (Run P8 — 85% win rate sur seeds inconnus) est disponible en tant qu'asset
de la release GitHub. Les binaires ne sont pas versionnés dans git (`models/` est dans `.gitignore`).

**Téléchargement :**

```bash
# Via gh CLI
gh release download v1.0 --pattern "model-ppo-cnn-30M-final.zip" --dir models/

# Ou manuellement : https://github.com/docmathieu/dungeon/releases/tag/v1.0
```

Une fois téléchargé, le fichier `models/model-ppo-cnn-30M-final.zip` peut être :
- Chargé dans l'UI via **[IA simple model]**
- Utilisé directement dans les scripts d'évaluation (voir section 6)

> Pour publier un nouveau modèle comme release :
> ```bash
> gh release create v1.1 "models/.../final.zip#nom-descriptif.zip" --title "..." --notes "..."
> ```

---

## 1. Prérequis et installation

**Python requis : 3.12** (testé et validé sur 3.12.x).

```bash
# Créer l'environnement virtuel
python -m venv .venv

# Activer (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Installer les dépendances
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> Tous les scripts s'exécutent avec `.venv\Scripts\python.exe` — ne jamais utiliser `python` ou `python3` nu.

---

## 2. Lancer le jeu (interface UI)

```bash
.venv\Scripts\python.exe src\main.py
```

### Ce que vous voyez

- **Grille 10×10** : herbe (verte), rochers (gris, infranchissables), eau (bleue, coût ×2)
- **Personnage** (guerrier pixel art) à déplacer avec les flèches ←↑→↓
- **Sortie** (porte) à atteindre
- **HUD** en haut : déplacements, note, boutons IA

### Actions essentielles dans l'UI

| Action | Commande |
|--------|----------|
| Déplacer le personnage | ←↑→↓ |
| Nouveau terrain | Touche **R** ou bouton **[Génération terrain]** |
| Terrain reproductible | Cliquer **Seed:[champ]**, taper un nombre, **Entrée** |
| Charger un modèle IA simple | **[IA simple model]** → sélectionner un `.pt` ou `.zip` |
| Charger un run complet | **[IA multi model]** → sélectionner un dossier `*_run/` |
| Jouer un épisode IA | **[Start SM déterministe]** ou **[Start SM stochastique]** |

> Référence complète de l'interface → [05-interface.md](05-interface.md)

---

## 3. Entraîner un modèle PPO (recommandé)

PPO (Proximal Policy Optimization) via Stable-Baselines3. C'est l'algorithme retenu — il résout
le catastrophic forgetting que DQN rencontre dès 3+ seeds différents.

### Commande de référence (configuration production)

```bash
# Premier run — from scratch
.venv\Scripts\python.exe src\train_ppo.py ^
    --timesteps 5000000 ^
    --architecture cnn ^
    --n-envs 8 ^
    --lr 0.0001

# Runs suivants — transfer learning depuis le dernier checkpoint
.venv\Scripts\python.exe src\train_ppo.py ^
    --timesteps 5000000 ^
    --architecture cnn ^
    --pretrained "models\{ts}_run\{run}\final.zip" ^
    --n-envs 8 ^
    --lr 0.0001
```

**Pourquoi cette configuration ?**
- **Pas de `--seed` ni `--seed-pool`** : full random — un terrain différent à chaque épisode.
  L'agent est forcé d'apprendre une stratégie générale, pas de mémoriser des solutions.
- **`--architecture cnn`** : partage des filtres sur toute la grille → détecte les obstacles
  quelle que soit leur position.
- **`--n-envs 8`** : 8 environnements parallèles → 8× plus de données par seconde.
- **`--lr 0.0001`** : learning rate faible pour des mises à jour stables sur les runs successifs.

### Tous les paramètres `train_ppo.py`

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--timesteps N` | int | 500 000 | Nombre total de pas d'environnement |
| `--seed N` | int | None | Seed fixe — même terrain à chaque épisode |
| `--seed-pool 0-99` | str | None | Pool de seeds. Accepte `0,1,2` ou plage `0-99`. Sans ce paramètre ni `--seed` : full random |
| `--lr 1e-4` | float | 3e-4 | Learning rate. Ne pas augmenter en cours d'entraînement |
| `--n-envs N` | int | 1 | Nombre d'environnements parallèles (recommandé : 8) |
| `--pretrained path` | Path | None | Checkpoint `.zip` SB3 de départ (transfer learning) |
| `--architecture` | str | `mlp` | Architecture : `mlp` (304 entrées) ou `cnn` (tenseur 10×10×5, recommandé) |

### Sorties produites

```
logs/{ts}_run/{ts}_{label}_ts{N}[_from_{ts}].jsonl   ← log JSON (1 ligne/épisode + ligne meta)
models/{ts}_run/{ts}_{label}_ts{N}[_from_{ts}]/
    ppo_50000_steps.zip    ← checkpoint toutes les 50k steps
    ...
    final.zip              ← poids finaux
```

Le suffixe `_from_{ts}` est automatiquement ajouté quand `--pretrained` est utilisé.

### Progression attendue (CNN full-random, 5M ts/run)

| Runs cumulés | Timesteps | Win rate seeds inconnus (det) |
|---|---|---|
| 1 | 5M | ~10% |
| 2 | 10M | ~56% |
| 4 | 20M | ~73% |
| 6 | 30M | **~85%** ← objectif atteint |
| 8 | 40M | ~77–81% (plateau) |

---

## 4. Entraîner un modèle DQN

DQN (Deep Q-Network) — algorithme maison, pertinent pour seed unique ou très petit pool.

```bash
.venv\Scripts\python.exe src\train.py --episodes 3000 --seed 42 --architecture film
```

### Tous les paramètres `train.py`

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--episodes N` | int | 3000 | Nombre d'épisodes |
| `--seed N` | int | None | Seed fixe |
| `--seed-pool 0,1,2` | str | None | Pool de seeds (plage ou liste) |
| `--lr 3e-4` | float | 1e-3 | Learning rate Adam |
| `--pretrained path` | Path | None | Checkpoint `.pt` de départ |
| `--architecture` | str | `film` | `obs` (304 entrées), `taskcond` (314), `film` (FiLM conditioning) |

> **Limite connue :** catastrophic forgetting systématique au-delà de 3 seeds différents.
> Pour la généralisation, préférer PPO.

### Sorties

```
logs/{ts}_run/{ts}_{label}_ep{N}.jsonl
models/{ts}_run/{ts}_{label}_ep{N}/ep<N>.pt   ← checkpoint tous les 500 épisodes
models/{ts}_run/{ts}_{label}_ep{N}/final.pt
```

---

## 5. Curriculum progressif DQN

Élargit le pool de seeds par étapes. Utile pour DQN multi-seeds malgré ses limites.

```bash
.venv\Scripts\python.exe src\curriculum.py ^
    --pool 0,1,2,3,4,5,6,7,8,9 ^
    --stages 1,3,6,10 ^
    --max-episodes-per-stage 2000 ^
    --win-rate-threshold 0.8 ^
    --lr 3e-4,1e-4 ^
    --architecture film
```

### Tous les paramètres `curriculum.py`

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--pool 0,1,...,9` | str | *obligatoire* | Ensemble de seeds disponibles |
| `--stages 1,3,6,10` | str | `1,3,6,10` | Taille du pool à chaque étape |
| `--max-episodes-per-stage N` | int | 2000 | Plafond si le seuil n'est pas atteint |
| `--win-rate-threshold 0.8` | float | 0.8 | Win rate cible (100 derniers épisodes) pour avancer |
| `--lr 3e-4,1e-4` | str | `1e-3` | LR par étape (dernière valeur répétée) |
| `--pretrained path` | str | None | Checkpoint `.pt` de départ |
| `--architecture` | str | `film` | Architecture réseau |

---

## 6. Évaluer un checkpoint

Mesure les performances sur N seeds (connus ou inconnus).

```bash
# Évaluation déterministe — 400 seeds inconnus
.venv\Scripts\python.exe analyze\evaluate.py ^
    --checkpoint "models\{ts}_run\{run}\final.zip" ^
    --seeds 100-499

# Évaluation stochastique — 3 épisodes par seed
.venv\Scripts\python.exe analyze\evaluate.py ^
    --checkpoint "models\{ts}_run\{run}\final.zip" ^
    --seeds 100-499 ^
    --n-episodes 3 ^
    --stochastic

# Seeds d'entraînement avec détail par seed
.venv\Scripts\python.exe analyze\evaluate.py ^
    --checkpoint "models\{ts}_run\{run}\final.zip" ^
    --seeds 0-99 ^
    --verbose
```

### Tous les paramètres `evaluate.py`

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `--checkpoint path` | Path | ✅ | Chemin vers `.pt` (DQN) ou `.zip` (PPO) |
| `--seeds 100-299` | str | ✅ | Seeds à évaluer. Plage `100-299` ou liste `0,1,2` |
| `--n-episodes N` | int | Non (1) | Épisodes par seed (utile en stochastique) |
| `--stochastic` | flag | Non | Mode stochastique PPO (`deterministic=False`) |
| `--verbose` | flag | Non | Détail par seed |

**Déterministe vs stochastique :**
- **Déterministe** : toujours l'action de probabilité maximale — mesure ce que le modèle *sait* faire.
- **Stochastique** : tirage selon les probabilités — peut débloquer des situations, scores plus bas.

---

## 7. Analyser les échecs

Croise résultats (victoire/défaite) et métriques terrain pour identifier les configurations difficiles.

```bash
.venv\Scripts\python.exe analyze\analyze_failures.py ^
    --checkpoint "models\{ts}_run\{run}\final.zip" ^
    --seeds 100-499
```

Affiche : win rate par groupe de difficulté (coût Dijkstra), comparaison victoires vs échecs
sur les métriques (coût optimal, détour rochers, eau sur le chemin).

---

## 8. Analyser les runs DQN

Compare le win rate max et final de tous les runs DQN dans `logs/`.

```bash
.venv\Scripts\python.exe analyze\analyze_runs.py
```

---

## 9. Chercher des seeds pédagogiques

Scanne 3 000 seeds et les classe par difficulté (coût Dijkstra, détour, eau).

```bash
.venv\Scripts\python.exe analyze\search_seeds.py
```

---

## 10. Lancer les tests

```bash
.venv\Scripts\python.exe -m pytest tests\ -v
```

362 tests couvrant : `Grid`, `GameState`, `PathFinder`, `DungeonEnv`, `DQNAgent`, `ReplayBuffer`,
`curriculum`, `exploit`, `evaluate`, `train_ppo` (MLP et CNN).

---

## 11. Produire un exécutable Windows

```bash
.venv\Scripts\python.exe -m PyInstaller src\main.py --onefile --noconsole --name dungeon
```

Produit `dist\dungeon.exe` — autonome, ne nécessite pas Python installé.

---

## 12. Migrer les anciens runs vers la nouvelle structure

```bash
# Prévisualisation (aucun changement)
.venv\Scripts\python.exe tools\migrate_models.py --dry-run

# Migration réelle
.venv\Scripts\python.exe tools\migrate_models.py
```

Déplace les dossiers existants vers la structure `{ts}_run/`.
