# Dungeon RL

Jeu de grille 10×10 en Python/pygame servant de terrain d'expérimentation pour l'**apprentissage
par renforcement** (Reinforcement Learning). Un personnage doit rejoindre une sortie en évitant
les obstacles. L'objectif : entraîner un agent IA capable de trouver seul le chemin optimal —
y compris sur des terrains qu'il n'a **jamais vus pendant l'entraînement**.

**Stack :** Python 3.12 · pygame · PyTorch · Stable-Baselines3 · gymnasium · pytest

---

## Résultats

| Modèle | Algorithme | Timesteps | Seeds inconnus det | Seeds inconnus stoch |
|--------|-----------|-----------|-------------------|---------------------|
| CNN full-random | PPO | 30M | **85%** | **90%** |
| CNN full-random | PPO | 40M (+ Dijkstra shaping) | 77% | 89% |
| CNN pool100 | PPO | 2M | 3% | ~11% |

> Le modèle gagne 85% des parties sur des terrains jamais vus, avec un score moyen de 95/100
> (chemin quasi-optimal). La clé : **diversité maximale des terrains** (full-random) + CNN.

---

## Modèle pré-entraîné

Le modèle final (Run P8, 85% det) est disponible dans les [releases GitHub](https://github.com/docmathieu/dungeon/releases/tag/v1.0) — les binaires ne sont pas versionnés dans git.

```bash
gh release download v1.0 --pattern "model-ppo-cnn-30M-final.zip" --dir models/
```

---

## Par où commencer ?

> **[docs/start.md](docs/start.md)** — Installation, lancement du jeu, entraînement, évaluation.
> C'est le point d'entrée pour tout développeur.

---

## Documentation complète

| Fichier | Contenu |
|---------|---------|
| **[docs/start.md](docs/start.md)** ⭐ | Guide complet : setup, UI, entraînement PPO/DQN, évaluation, tests, exe |
| [docs/01-runs.md](docs/01-runs.md) | Historique de tous les runs (DQN + PPO) avec dates, configs, résultats |
| [docs/02-architecture.md](docs/02-architecture.md) | Architecture réseau finale : comptage neurones, parallèle biologique, schéma |
| [docs/03-librairies.md](docs/03-librairies.md) | Librairies utilisées : rôles et focus code par librairie |
| [docs/04-technologies-ia.md](docs/04-technologies-ia.md) | Algorithmes DQN/PPO, architectures MLP/CNN/FiLM, stratégie full-random |
| [docs/05-interface.md](docs/05-interface.md) | Référence complète de l'interface pygame : HUD, boutons, trails, terrain |

---

## Démarrage rapide

```bash
# Installation
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Lancer le jeu
.venv\Scripts\python.exe src\main.py

# Entraîner un modèle (PPO CNN full-random — configuration production)
.venv\Scripts\python.exe src\train_ppo.py --timesteps 5000000 --architecture cnn --n-envs 8 --lr 0.0001

# Évaluer sur seeds inconnus
.venv\Scripts\python.exe analyze\evaluate.py --checkpoint "models\{run}\final.zip" --seeds 100-499

# Tests
.venv\Scripts\python.exe -m pytest tests\ -v
```
