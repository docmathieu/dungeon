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

**Piste suivante sérieuse : PPO** (algorithme on-policy, mises à jour stables).

---

## Scripts d'analyse (`analyze/`)

Scripts Python utilitaires pour l'étude des seeds et du comportement RL.
Ne font pas partie du jeu ni de l'entraînement — à lancer manuellement depuis la racine du projet.

### `analyze/evaluate.py` *(ajouté 2026-06-01)*
**Résultat** : win rate + score moyen d'un checkpoint sur N seeds

Évalue la performance d'un modèle entraîné, notamment sur des seeds jamais vus (mesure de généralisation).
Détecte automatiquement l'architecture (ObsDQNetwork / FiLMDQNetwork / DQNetwork).

```bash
.venv\Scripts\python.exe analyze/evaluate.py --checkpoint models/.../final.pt --seeds 100-299
.venv\Scripts\python.exe analyze/evaluate.py --checkpoint models/.../final.pt --seeds 0-99 --verbose
```

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
