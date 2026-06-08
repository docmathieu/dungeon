# Technologies IA — Dungeon RL

---

## Algorithmes explorés

### DQN — Deep Q-Network (abandonné pour le multi-seeds)

DQN est un algorithme **off-policy** : l'agent mémorise ses expériences dans un replay buffer
et apprend en rejouant des transitions passées. À chaque mise à jour, les poids du réseau
peuvent être fortement modifiés.

**Mécanismes clés :**
- **Replay buffer** : stocke les transitions `(état, action, reward, état suivant)` — brise les corrélations temporelles
- **Réseau cible** : copie du réseau principal, mise à jour toutes les N itérations — stabilise les valeurs cibles
- **ε-greedy** : exploration aléatoire avec ε décroissant (1.0 → 0.05 au fil des épisodes)

**Problème rencontré :** quand plusieurs terrains différents sont utilisés, deux seeds peuvent
exiger des actions opposées dans des états visuellement similaires. DQN écrase les poids appris
pour un seed en apprenant un autre → **catastrophic forgetting** systématique au-delà de 3 seeds.

**Résultat :** excellent sur seed unique (99.6% en <2000 épisodes), inutilisable en généralisation.

---

### PPO — Proximal Policy Optimization (retenu)

PPO est un algorithme **on-policy** avec une contrainte de stabilité : le paramètre `clip_range`
limite l'écart entre l'ancienne et la nouvelle politique à chaque mise à jour.

**Mécanismes clés :**
- **Clip range** : le ratio de probabilités `π_new / π_old` est clippé entre `[1-ε, 1+ε]` (ε=0.2).
  L'agent ne peut pas "désapprendre" brusquement en un seul batch.
- **GAE (Generalized Advantage Estimation)** : estimation lissée de l'avantage `A(s,a)` — réduit
  la variance des gradients sans biaiser l'estimation.
- **Actor-Critic** : deux têtes partagent le même extracteur de features. L'acteur apprend *quoi
  faire*, le critique apprend *si c'est une bonne situation* — se guident mutuellement.
- **Environnements vectorisés** (`n_envs=8`) : 8 parties en parallèle → 8× plus de données
  par seconde sur CPU.

**Pourquoi PPO résout le catastrophic forgetting :** les mises à jour bornées préservent les
politiques déjà acquises tout en permettant l'amélioration progressive. Sur un pool de 10 seeds,
PPO atteint 89% de win rate là où DQN s'effondre à 0%.

---

## Architectures de réseaux

### MLP — Multi-Layer Perceptron (baseline)

La grille 10×10 est aplatie en un vecteur de 300 valeurs (one-hot par case × 3 types),
auxquelles s'ajoutent 4 positions normalisées → **304 entrées**.

```
Entrée (304) → Dense(256) → ReLU → Dense(128) → ReLU → Dense(64) → ReLU → Sortie (4)
```

**Limite :** chaque position de la grille utilise des poids différents. Apprendre "obstacle en (2,3)
→ tourner" ne transfère pas à "obstacle en (7,8) → tourner". Nécessite des milliers de seeds
différents pour couvrir l'espace des configurations.

---

### Task-conditioning et FiLM (expérimentaux, non retenus)

- **Task-conditioning** : le seed est encodé en one-hot (10 bits) et concaténé à l'observation
  → 314 entrées. Le réseau "triche" en mémorisant une politique par seed plutôt qu'en apprenant
  à naviguer.
- **FiLM (Feature-wise Linear Modulation)** : le seed module chaque couche cachée via des
  paramètres `gamma` et `beta` appris. Plus stable que le task-conditioning (16% final vs 7%)
  mais même limitation fondamentale.

Ces deux approches améliorent les performances sur seeds connus mais ne généralisent pas aux
seeds inconnus — le réseau "triche" en s'appuyant sur l'identité du seed plutôt que sur la grille.

---

### CNN — Convolutional Neural Network (retenu)

La grille est encodée en tenseur spatial **10×10×5** (5 canaux : herbe, roche, eau, personnage,
sortie). Des filtres convolutionnels glissent sur toute la grille avec les **mêmes poids partout**.

```
Entrée (10×10×5)
  → Conv2D(5→16 filtres, noyau 3×3) → ReLU  [sortie : 8×8×16 = 1 024 neurones]
  → Conv2D(16→32 filtres, noyau 3×3) → ReLU  [sortie : 6×6×32 = 1 152 neurones]
  → Flatten (1 152 valeurs)
  → Dense(128) → ReLU
  → [Acteur] Dense(64) → ReLU → Dense(4)   ← LEFT/RIGHT/UP/DOWN
  → [Critique] Dense(64) → ReLU → Dense(1)  ← valeur d'état V
```

**Avantage :** un filtre "obstacle à droite" apprend à reconnaître un obstacle **en n'importe
quelle position** de la grille. Vu une fois en (2,3), généralisé automatiquement en (7,8).

---

## Stratégie d'entraînement retenue : full-random

| Approche | Win rate seeds inconnus | Statut |
|----------|------------------------|--------|
| DQN seed unique | N/A (pas de généralisation) | ✅ fonctionne, limité |
| DQN pool fixe + curriculum | ~3% | ✅ testé, limité |
| PPO MLP pool100 | 3% | ✅ testé |
| PPO CNN pool100 | 3% | ✅ testé |
| **PPO CNN full-random** | **85% déterministe** | ✅ **retenu** |

La clé de la généralisation n'était **ni l'architecture ni l'algorithme seuls**, mais la
**diversité maximale des terrains** combinée au CNN.

En voyant un terrain différent à chaque épisode (`seed=None`), l'agent ne peut pas mémoriser :
il doit apprendre une vraie stratégie de navigation. Pour CNN pool100, l'écart "vu vs non vu"
était 76% vs 3% — mémorisation pure. En full-random, l'écart disparaît : 8% vs 10% — généralisation réelle.

---

## Reward shaping Dijkstra

Introduit au Run P10 (`dungeon_env.py`) pour guider l'agent sur les terrains difficiles :

```
déplacement normal → REWARD_STEP (-0.01) + (dijkstra_avant - dijkstra_après) × 0.10
choc mur/bord     → REWARD_BUMP (-0.05)
victoire          → score / 100.0
```

- Pas optimal sur herbe : `−0.01 + 1×0.10 = +0.09`
- Pas optimal sur eau : `−0.01 + 2×0.10 = +0.19`
- Pas s'éloignant de la sortie : `−0.01 + valeur_négative` (plus pénalisant)
- Dijkstra intègre les obstacles → ne pénalise pas les contournements nécessaires

**Résultat :** entraînement plus stable (92% win rate constant vs 82–97% fluctuant avant),
mais généralisation identique. Le plateau à ~81–85% déterministe reste structurel.

---

## Limite actuelle et pistes

Le blocage résiduel (~15% d'échecs) correspond aux terrains à chemin long (coût Dijkstra >16).
Le CNN réagit à l'état courant sans mémoire entre les pas : il ne peut pas planifier sur 20+ mouvements.

| Piste | Ce qu'elle apporte | Complexité |
|-------|-------------------|-----------|
| Plus de timesteps (>40M) | +3 pts/run (tendance log) | Faible |
| Architecture Attention/Transformer | Raisonnement global sur la grille | Élevée |
| Model-based RL | Planification explicite (simuler le futur) | Très élevée |
| Curriculum hard seeds | Forcer l'apprentissage des cas difficiles | Moyenne |
