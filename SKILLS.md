# SKILLS — Dungeon POC

Liste des skills Claude Code personnalisés disponibles dans ce projet.
Ils se déclenchent avec `/nom-du-skill` depuis l'interface Claude Code (CLI ou extension VSCode).

---

## /generate-game
**Fichier** : `.claude/skills/generate-game.md`
**Résultat** : crée `src/game.py`

Génère le jeu dungeon complet en Python/pygame selon la spécification CLAUDE.md.
Utilise la skill comme un agent de code complet : lit la spec, produit le fichier, vérifie la syntaxe.

**Déclencheurs** :
- L'utilisateur tape `/generate-game`
- `src/game.py` est absent ou incomplet

---

## /generate-tests
**Fichier** : `.claude/skills/generate-tests.md`
**Résultat** : crée `tests/test_game.py`

Génère les tests unitaires pytest pour `src/game.py`.
Teste la logique de jeu de façon isolée (sans affichage pygame).

**Déclencheurs** :
- L'utilisateur tape `/generate-tests`
- `tests/test_game.py` est absent

---

## /run-tests
**Fichier** : `.claude/skills/run-tests.md`
**Résultat** : rapport pytest dans le terminal

Lance les tests unitaires et affiche le résultat.

**Déclencheurs** :
- L'utilisateur tape `/run-tests`
- L'utilisateur demande de vérifier que le code fonctionne

---

## /run-game
**Fichier** : `.claude/skills/run-game.md`
**Résultat** : fenêtre pygame du jeu

Démarre le jeu dungeon.

**Déclencheurs** :
- L'utilisateur tape `/run-game`
- L'utilisateur veut tester visuellement le jeu

---

## /build-exe
**Fichier** : `.claude/skills/build-exe.md`
**Résultat** : `dist/dungeon.exe`

Compile le jeu en exécutable Windows autonome (PyInstaller).

**Déclencheurs** :
- L'utilisateur tape `/build-exe`
- L'utilisateur veut distribuer le jeu sans Python installé

---

## PPO Stable-Baselines3 (2026-06-01) ✅

`src/train_ppo.py` — entraînement PPO via SB3. Résout le catastrophic forgetting.

```bash
.venv\Scripts\python.exe src/train_ppo.py --timesteps 500000 --seed-pool 0,1,2,3,4,5,6,7,8,9
.venv\Scripts\python.exe src/train_ppo.py --timesteps 200000 --seed 0
```

**Résultats vs DQN (pool10) :**
- Win rate max online : **89%** (PPO) vs 59% (DQN task-cond)
- Pas de catastrophic forgetting : plancher **30%** vs 0% (DQN)
- Eval déterministe : 30–50% (10 seeds)

**Logs** : `{"episode", "timestep", "seed", "won", "score", "moves", "reward"}` — même format que DQN.
**Checkpoints** : `.zip` (SB3) tous les 50k timesteps + `final.zip`

**`analyze/evaluate.py`** supporte les deux formats :
- `.pt` → DQN (inchangé)
- `.zip` → PPO SB3 (`deterministic=True`)

---

## Visualisation UI des modèles IA (2026-06-02) ✅

`src/exploit.py` — chargement et exécution de modèles DQN et PPO :

| Fonction | Description |
|----------|-------------|
| `load_net(path)` | Charge `.pt` DQN (détection archi automatique) |
| `load_ppo(path)` | Charge `.zip` PPO SB3 |
| `load_model(path)` | Dispatcher : `.pt` → DQN, `.zip` → PPO |
| `run_one_episode_info(model, seed)` | Trail + résultat — dispatch auto DQN/PPO |
| `run_one_episode_info_ppo(model, seed)` | Épisode PPO déterministe via `DungeonGymEnv` |
| `scan_run_dir(run_dir)` | Checkpoints ordonnés dans un `*_run/` (`.pt` et `.zip`) |

**Boutons IA dans l'UI :**
- `[IA simple model]` — file picker `.pt`/`.zip` → un épisode, trail orange + chemin rouge
- `[IA multi model]` — directory picker `*_run/` → tous les checkpoints en thread de fond, animation incrémentale
- `[IA restart]` — rejoue sur le terrain courant, met à jour trail ET chemin optimal

---

## Bilan expérimental DQN (2026-06-01)

Le catastrophic forgetting sur multi-seeds n'est **pas un problème d'architecture** mais d'**algorithme**.
Toutes les variantes DQN (MLP/task-cond/FiLM/obs) échouent à maintenir plusieurs politiques au-delà de pool3.

| Expérience | Résultat clé |
|---|---|
| FiLM curriculum pool10 | 16% win rate final (meilleur multi-seeds) |
| Task-cond curriculum pool10 | 59% win rate **max**, 7% final |
| ObsDQNetwork pool100 20k ep | 4% win rate max seeds inconnus = seeds vus → pas d'apprentissage |
| ObsDQNetwork seed unique | Score 100 dès ep1500 ✅ |
| ObsDQNetwork curriculum pool3 | 23% max ep200, 0% à ep1000 — même pattern |

---

## Bilan expérimental PPO (2026-06-02)

PPO résout le catastrophic forgetting mais la généralisation reste limitée par l'architecture MLP.

| Expérience | Résultat clé |
|---|---|
| PPO pool10 500k ts — online | **89%** wr max, plancher 30%, pas de catastrophic forgetting ✅ |
| PPO pool10 500k ts — det seeds 0–9 | 30% (3/10), score wins 100 |
| PPO pool10 500k ts — stoch ×5 seeds 0–9 | **54%** (27/50) → stochastique >> déterministe |
| PPO pool10 500k ts — det seeds 100–299 | **4%** baseline généralisation |
| PPO pool100 2M ts — online | 56% final, stable 45–77%, pas de catastrophic forgetting ✅ |
| PPO pool100 2M ts — det seeds 0–99 | **35%** wr, score wins 94.1 |
| PPO pool100 2M ts — det seeds 100–299 | **3%** généralisation (≈ pool10) |
| PPO pool100 2M ts — stoch ×3 seeds 100–299 | **13.8%** (+4.6× vs déterministe) |

**Conclusion :** augmenter le pool (10→100 seeds) n'améliore pas la généralisation déterministe.
La limite est **architecturale** : le MLP ne capte pas les invariances spatiales de la grille.

---

## Limite MLP et prochaine étape : CNN + PPO (2026-06-02)

**Pourquoi le MLP ne généralise pas :**
- Le MLP reçoit 300 features indépendantes (grille 10×10×3 aplatie)
- Chaque position utilise des poids distincts → apprendre "obstacle en (2,3)" ≠ "obstacle en (7,8)"
- Nécessite des milliers de grilles différentes pour couvrir l'espace des configurations

**Ce qu'apporte CNN :**
- Un filtre convolutionnel 3×3 s'applique à toutes les positions avec les mêmes poids
- Invariance spatiale par construction : vu en (2,3), généralisé en (7,8)
- Même architecture input/output, mais traitement spatial au lieu de liste plate

**Architecture cible : CNN + PPO**
- PPO : résout catastrophic forgetting (✅ déjà prouvé)
- CNN : résout généralisation spatiale (← prochaine implémentation)

---

## Scripts d'analyse (`analyze/`)

Scripts Python utilitaires pour l'étude des seeds et du comportement RL.
Ne font pas partie du jeu ni de l'entraînement — à lancer manuellement depuis la racine du projet.

### `analyze/evaluate.py` *(ajouté 2026-06-01, amélioré 2026-06-02)*
**Résultat** : win rate + score moyen d'un checkpoint sur N seeds (DQN ou PPO)

```bash
.venv\Scripts\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 0-9
.venv\Scripts\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 100-299
.venv\Scripts\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 0-9 --n-episodes 5
.venv\Scripts\python.exe analyze/evaluate.py --checkpoint models/.../final.zip --seeds 0-9 --stochastic --n-episodes 10
```

Options clés :
- `--n-episodes N` : répète chaque seed N fois (utile en stochastique pour mesurer la variance)
- `--stochastic` : politique stochastique PPO — recommandé pour évaluer la vraie capacité du modèle
- `--verbose` : détail par seed

**Résultats PPO pool100 (2026-06-02) :**
- Seeds 0–99 (entraînement) déterministe : **35%** wr, score wins 94.1
- Seeds 100–299 (inconnus) déterministe : **3%** wr, score wins 94.5
- Seeds 100–299 (inconnus) stochastique ×3 : **13.8%** wr, score wins 41.5

---

### `analyze/analyze_runs.py`
**Résultat** : tableau comparatif win rate max / final par chaîne de runs

Pour chaque chaîne curriculum (pool3 → pool6 → pool10), calcule le win rate
**max** (meilleure fenêtre glissante 100 ep) et **final** (100 derniers épisodes).
Permet de comparer les architectures MLP / Task-cond / FiLM sur les mêmes métriques.

**Utilisation** :
```bash
.venv\Scripts\python.exe analyze/analyze_runs.py
```

---

### `analyze/search_seeds.py`
**Résultat** : liste de 20 seeds candidats répartis en 5 groupes de difficulté

Scanne les seeds 0..2999, calcule pour chacun :
- `optimal_cost` : coût Dijkstra (herbe=1, eau=2)
- `optimal_moves` : nombre de moves sur le chemin optimal
- `rock_detour` : optimal_moves − manhattan (>0 = rochers forcent un détour)
- `water_steps` : cases EAU sur le chemin optimal
- `near_border` : personnage ou sortie sur le bord de la grille

Sélectionne 4 seeds par groupe sans doublons (même char+exit) :

| Groupe | Coût optimal | Critère prioritaire |
|--------|-------------|---------------------|
| Facile | 5–8 | chemin court, peu d'obstacles |
| Rochers | 9–12 | rock_detour élevé |
| Mixte | 13–16 | eau et/ou rochers sur le chemin |
| Complexe | 17–20 | eau sur le chemin optimal |
| Difficile | 21+ | chemin long et tortueux |

**Utilisation** :
```bash
.venv\Scripts\python.exe analyze/search_seeds.py
```
