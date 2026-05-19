# Dungeon POC — Claude Code Project

## Description
Jeu de grille 10×10 en Python/pygame. Un personnage jaune doit atteindre une sortie jaune en suivant une séquence de déplacements entrée par l'utilisateur.

Ce POC est la première étape vers un système d'**apprentissage par renforcement** : à terme, des centaines de parties headless (sans UI) tourneront en parallèle via `multiprocessing.Pool`. L'architecture doit donc maintenir une séparation stricte entre logique de jeu et affichage.

## Deux modes d'exécution
| Mode | Usage | Queue | Pause |
|------|-------|-------|-------|
| UI | Jeu interactif pygame | `queue.Queue` fournie | 0.5s |
| Headless | Entraînement RL, tests | `None` | Aucune |

## Stack technique
- **Python** : 3.12 (LTS-équivalent, supporté jusqu'en 2028)
- **Graphique** : pygame (SDL2, performant pour la manipulation de pixels, adapté aux threads futurs)
- **Tests** : pytest
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
│   ├── main.py          ← point d'entrée
│   ├── grid.py          ← TileType, Grid
│   ├── game_state.py    ← GameState, logique déplacement
│   ├── simulation.py    ← thread de simulation
│   └── ui.py            ← GameUI, rendu pygame
└── tests/
    └── test_game.py     ← généré par /generate-tests
```

## Spécification du jeu

### Terrain
- Grille 10×10 cases, chaque case 10×10 pixels
- Trait de 1px entre chaque case
- Fond noir
- Herbe (vert) : type par défaut
- Pierre (gris) : 30% des cases, infranchissable
- Eau (bleu) : 20% des cases, franchissable mais coûte 2 déplacements

### Éléments
- Personnage : dessin simple jaune, placé aléatoirement sur une case herbe
- Sortie : porte simple jaune, placée aléatoirement sur une case herbe (≠ personnage)

### Interface
- **Au-dessus du terrain** : champ "déplacements" (init 0), champ "note" (init 0), champ "Information" (vide)
- **En dessous du terrain** : bouton "restart", champ input "instruct", bouton "start"

### Saisie
- Flèches du clavier dans le champ "instruct" : ← ↑ → ↓
- "start" ou Entrée depuis "instruct" : déclenche la simulation
- "restart" : réinitialise tout et génère un nouveau terrain

### Simulation
- Déplacement séquentiel selon la séquence "instruct"
- Pause 0.5s entre chaque déplacement
- Mise à jour du champ "déplacements" à chaque pas
- Victoire si personnage == sortie → note=1, Information="GAGNE", séquence stoppée

## Commandes rapides (skills)
| Skill            | Commande VSCode       | Action                        |
|------------------|-----------------------|-------------------------------|
| generate-game    | Ctrl+Shift+P → Task   | Génère src/ (5 fichiers)      |
| generate-tests   | Ctrl+Shift+P → Task   | Génère tests/test_game.py     |
| run-tests        | Ctrl+Shift+P → Task   | Lance pytest                  |
| run-game         | Ctrl+Shift+P → Task   | Démarre le jeu                |
| build-exe        | Ctrl+Shift+P → Task   | Produit dungeon.exe           |
