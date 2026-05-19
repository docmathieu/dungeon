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
