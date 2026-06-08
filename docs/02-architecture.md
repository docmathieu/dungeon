# Architecture du réseau — Dungeon RL

---

## Architecture finale utilisée (PPO + CNN)

### Schéma de flux

```
ENTRÉE
  Grille 10×10 × 5 canaux = 500 cellules
  Canal 0 = HERBE  | Canal 1 = ROCHE  | Canal 2 = EAU
  Canal 3 = PERSONNAGE              | Canal 4 = SORTIE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EXTRACTEUR CNN  (DungeonCnnExtractor)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Conv2D(5→16 filtres, 3×3)  →  8×8×16  =  1 024 neurones
  Conv2D(16→32 filtres, 3×3) →  6×6×32  =  1 152 neurones
  Flatten                    →           =  1 152 valeurs
  Linear(1152 → 128) + ReLU  →           =    128 neurones

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  POLITIQUE PPO  (partagée puis bifurquée)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ACTEUR (décision)           CRITIQUE (évaluation)
  Linear(128→64) + ReLU       Linear(128→64) + ReLU
  = 64 neurones               = 64 neurones
  Linear(64→4)                Linear(64→1)
  = 4 sorties                 = 1 sortie
  (LEFT/RIGHT/UP/DOWN)        (valeur de l'état V)

  Total neurones distincts    :   2 937
  Total paramètres (poids)    :  ~170 000
```

### Décompte des paramètres

| Couche | Calcul | Paramètres |
|--------|--------|-----------|
| Conv1 (5→16, 3×3) | 5×3×3×16 + 16 | 736 |
| Conv2 (16→32, 3×3) | 16×3×3×32 + 32 | 4 640 |
| Linear CNN (1152→128) | 1152×128 + 128 | 147 584 |
| Acteur (128→64) | 128×64 + 64 | 8 256 |
| Acteur tête (64→4) | 64×4 + 4 | 260 |
| Critique (128→64) | 128×64 + 64 | 8 256 |
| Critique tête (64→1) | 64×1 + 1 | 65 |
| **Total** | | **~169 797** |

---

## Parallèle biologique

| Système | Neurones | Connexions | Capacité |
|---------|----------|-----------|---------|
| *C. elegans* (ver rond) | 302 | ~7 000 synapses | Comportements stéréotypés |
| **Notre réseau CNN** | **~2 937** | **~170 000 poids** | **Navigation grille 10×10** |
| Abeille | ~1 million | ~1 milliard | Navigation, mémoire spatiale |
| Souris | 70 millions | ~100 milliards | Apprentissage complexe |
| Humain | 86 milliards | ~100 000 milliards | Raisonnement abstrait |

**Notre réseau se situe entre le ver et l'abeille en neurones**, mais réussit à 85% une tâche
de navigation spatiale avec obstacles que le ver ne peut pas accomplir.

### Analogies fonctionnelles

| Composant réseau | Analogie biologique | Rôle |
|-----------------|--------------------|----|
| Couches Conv2D | Cellules ganglionnaires de la rétine (V1) | Détection de patterns locaux (bords, obstacles) |
| Linear CNN (1152→128) | Compression cortex visuel → cortex préfrontal | Abstraction des features visuelles |
| Acteur (128→64→4) | Striatum (ganglions de la base) | Décision : quelle action effectuer |
| Critique (128→64→1) | Cortex orbitofrontal | Évaluation : est-ce une bonne situation ? |

**Ce que notre réseau ne peut pas faire** (contrairement à une abeille) : planifier explicitement
un trajet. Il n'a pas de mémoire entre les pas — il réagit uniquement à l'état visible à l'instant T.
Les 11.8% d'échec sur terrains difficiles (chemin >20 moves) en sont la preuve directe.

---

## Schéma global de l'architecture (fichiers + librairies)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DUNGEON RL — Vue d'ensemble                      │
└─────────────────────────────────────────────────────────────────────────┘

 TERRAIN (src/grid.py)                   ÉTAT DU JEU (src/game_state.py)
 ┌──────────────────────┐               ┌───────────────────────────────┐
 │ Grid 10×10           │               │ GameState                     │
 │ TileType: GRASS/ROCK/│──────────────▶│ char_pos, exit_pos            │
 │          WATER       │               │ move_count, score, won        │
 │ Générée via seed RNG │               │ apply_move()                  │
 └──────────────────────┘               └───────────────┬───────────────┘
                                                        │
 PATHFINDER (src/pathfinder.py)                         │
 ┌──────────────────────┐                               │
 │ PathFinder (Dijkstra)│◀──────────────────────────────┤
 │ Coût eau=2, roche=∞  │  reward shaping : Δcoût×0.10  │
 │ shortest_cost()      │                               │
 └──────────────────────┘                               │
                                                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    INTERFACE GYM (src/dungeon_env.py)                 │
│                    library: gymnasium                                  │
│                                                                       │
│  DungeonEnv.reset()  →  obs_dict {grid, char_pos, exit_pos}          │
│  DungeonEnv.step(action) →  (obs, reward, done, info)                │
│                                                                       │
│  Reward :  victoire → score/100  |  pas → -0.01 + Δdijkstra×0.10    │
│            choc → -0.05                                               │
└─────────────────────────────────┬─────────────────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
       ┌────────────────────────┐   ┌──────────────────────────────┐
       │  ENCODAGE MLP          │   │  ENCODAGE CNN                │
       │  (src/train.py)        │   │  (src/train_ppo.py)          │
       │  library: numpy        │   │  library: numpy              │
       │                        │   │                              │
       │  encode_obs()          │   │  encode_obs_cnn()            │
       │  100 cases × 3 one-hot │   │  tenseur 10×10×5             │
       │  + 4 coords norm.      │   │  5 canaux binaires           │
       │  = 304 floats          │   │  (herbe/roche/eau/perso/exit)│
       └────────────┬───────────┘   └──────────────┬───────────────┘
                    │                              │
                    ▼                              ▼
       ┌────────────────────────┐   ┌──────────────────────────────┐
       │  DQN (src/train.py)    │   │  PPO (src/train_ppo.py)      │
       │  library: torch        │   │  library: stable_baselines3  │
       │                        │   │  + torch                     │
       │  ObsDQNetwork          │   │                              │
       │  304→256→128→64→4      │   │  DungeonCnnExtractor         │
       │                        │   │  Conv(5→16)→Conv(16→32)      │
       │  DQNAgent              │   │  →Flatten→Linear(→128)       │
       │  ε-greedy              │   │  ↓                           │
       │  ReplayBuffer          │   │  PPO MlpPolicy               │
       │  réseau cible          │   │  Acteur: 128→64→4            │
       │                        │   │  Critique: 128→64→1          │
       │  ❌ Catastrophic       │   │                              │
       │  forgetting multi-seed │   │  ✅ Clip range PPO           │
       └────────────────────────┘   │  ✅ GAE, n_envs=8 parallèle  │
                                    │  ✅ 85% det seeds inconnus   │
                                    └──────────────┬───────────────┘
                                                   │  models/*.zip
                                                   ▼
                     ┌─────────────────────────────────────────────┐
                     │       VISUALISATION (src/ui.py)             │
                     │       library: pygame                       │
                     │                                             │
                     │  Sprites PNG (assets/tiles/)                │
                     │  HUD 108px : stats, boutons IA, loading bar │
                     │  Trails : bleu / orange / orange→rouge /    │
                     │           jaune (chemin optimal)            │
                     │                                             │
                     │  exploit.py : load_model()                  │
                     │  run_one_episode_info() — DQN + PPO auto    │
                     │  scan_run_dir() — détecte .pt et .zip       │
                     └─────────────────────────────────────────────┘
```

---

## Architectures DQN expérimentées (src/model.py)

Trois architectures disponibles, toutes avec l'observation 304 floats en entrée :

```
1. ObsDQNetwork (généralisation)
   304 → Linear(256) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Linear(4)
   Aucun signal de seed — décide uniquement depuis la grille visible.

2. DQNetwork (task-conditioning)
   314 → Linear(256) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Linear(4)
   Seed one-hot (10 bits) concaténé à l'entrée.

3. FiLMDQNetwork (conditioning avancé)
   obs(304) → FC1(256) → FiLM(task) → ReLU
           → FC2(128) → FiLM(task) → ReLU
           → FC3(64)  → FiLM(task) → ReLU → FC4(4)
   Chaque couche modulée par gamma(seed)*x + beta(seed) appris.
```

**Conclusion :** aucune de ces architectures DQN ne résout la généralisation multi-seeds.
Le vrai levier était la **diversité des terrains** (full-random) combinée à **PPO** et au **CNN**.
