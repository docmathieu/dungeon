# AGENTS — Dungeon POC

Ce fichier décrit les agents Claude Code disponibles dans ce projet.
Chaque agent est déclenché via le skill correspondant dans Claude Code (VSCode ou CLI).

---

## Agent : generate-game
**Skill** : `/generate-game`
**Fichier produit** : `src/game.py`

Génère le code source complet du jeu selon la spécification dans CLAUDE.md.
- Grille pygame 10×10, cases 10px, séparateurs 1px, fond noir
- Cases herbe/pierre/eau avec répartition aléatoire
- Personnage et sortie jaunes positionnés sur cases herbe
- Interface complète (labels, input "instruct", boutons "start" et "restart")
- Simulation avec pause 0.5s, compteur déplacements, détection victoire

Structure de fichiers produite :
```
src/
├── main.py          ← point d'entrée, lance GameUI
├── grid.py          ← TileType (enum), Grid
├── game_state.py    ← GameState, logique de déplacement
├── simulation.py    ← Simulation (thread)
└── ui.py            ← GameUI, rendu pygame, event loop
```

Contraintes imposées à l'agent :
- Logique de jeu (`grid.py`, `game_state.py`) sans aucun import pygame — testable en isolation
- Aucune dépendance hors `pygame` et bibliothèque standard Python
- `main.py` doit être le seul point d'entrée : `python src/main.py`

---

## Agent : generate-tests
**Skill** : `/generate-tests`
**Fichier produit** : `tests/test_game.py`
**Prérequis** : `src/game.py` doit exister

Génère les tests unitaires pytest couvrant :
- Génération du terrain (proportions herbe/pierre/eau)
- Placement du personnage et de la sortie (toujours sur herbe, positions distinctes)
- Règles de déplacement (pierre bloque, eau coûte 2, bords bloquent)
- Compteur de déplacements
- Détection victoire (personnage sur case sortie)
- Restart (réinitialisation complète de l'état)

Contraintes :
- `pytest` uniquement, aucun mock pygame (tester la logique pure)
- Couverture minimale : 80% des branches logiques

---

## Agent : run-tests
**Skill** : `/run-tests`
**Prérequis** : `tests/test_game.py` doit exister, dépendances installées

Lance `pytest tests/ -v` et rapporte :
- Nombre de tests passés / échoués
- Couverture si `pytest-cov` est installé
- En cas d'échec : résumé des assertions qui ont échoué

---

## Agent : run-game
**Skill** : `/run-game`
**Prérequis** : `src/game.py` doit exister, `pygame` installé

Démarre le jeu via `python src/game.py` dans l'environnement virtuel `.venv`.

---

## Agent : build-exe
**Skill** : `/build-exe`
**Produit** : `dist/dungeon.exe`
**Prérequis** : `src/game.py` doit exister, `pyinstaller` installé

Génère un exécutable Windows autonome via PyInstaller :
- `--onefile` : fichier unique
- `--noconsole` : pas de fenêtre console
- `--name dungeon`
- Inclut les assets pygame nécessaires
