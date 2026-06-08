# Librairies utilisées — Dungeon RL

---

## Tableau récapitulatif

| Librairie | Version | Rôle dans le projet |
|-----------|---------|---------------------|
| **pygame** | 2.6.1 | Moteur graphique — rendu grille, sprites PNG, HUD, gestion événements clavier/souris |
| **torch** (PyTorch) | 2.12.0+cpu | Réseaux de neurones DQN — couches Linear, Conv2D, ReLU, backward pass automatique |
| **stable-baselines3** | ≥2.3.0 | Algorithme PPO clé en main — actor-critic, GAE, clipping PPO, environnements vectorisés |
| **gymnasium** | ≥0.29.0 | Standard d'interface RL — contrat reset()/step(), spaces.Box/Discrete |
| **numpy** | ≥1.24.0 | Encodage rapide des observations (tableaux float32, opérations vectorielles) |
| **pytest** | 8.3.5 | Suite de tests — 362 tests, couverture de tous les modules |
| **pytest-cov** | 6.1.0 | Mesure de la couverture de code |
| **pyinstaller** | 6.13.0 | Packaging en `.exe` standalone sans Python installé |

---

## Focus code par librairie

### pygame — `src/ui.py`

```python
# Rendu d'un sprite PNG sur la grille (src/ui.py)
self._screen.blit(surf, (px, py))
```

`blit` copie un `Surface` (image en mémoire GPU) pixel par pixel dans le framebuffer à la position
`(px, py)`. C'est l'opération fondamentale du rendu : 100 cases × blit par frame = l'image complète.
pygame gère aussi la boucle d'événements (`pygame.event.get()`), la fréquence d'images (`clock.tick()`),
et les polices de texte pour le HUD.

---

### torch — `src/train_ppo.py:117`

```python
self.cnn = nn.Sequential(
    nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.Flatten(),
)
```

`nn.Conv2d(5, 16, 3)` crée 16 filtres de 3×3 pixels qui glissent sur la grille 10×10.
Chaque filtre apprend à détecter un pattern local (ex : "mur à droite de ma position").
PyTorch calcule automatiquement les gradients via autograd (`backward()`) pour mettre à jour
ces filtres à chaque batch. Sans PyTorch, il faudrait implémenter la rétropropagation à la main.

---

### stable-baselines3 — `src/train_ppo.py:349`

```python
model = PPO(
    "MlpPolicy", env,
    learning_rate=lr, n_steps=N_STEPS, batch_size=BATCH_SIZE,
    clip_range=CLIP_RANGE, policy_kwargs=policy_kwargs,
)
```

`clip_range=0.2` est le paramètre central de PPO : il interdit à la nouvelle politique de
s'éloigner de plus de 20% de l'ancienne à chaque update (ratio de probabilités clippé entre
`[1-ε, 1+ε]`). C'est ce mécanisme qui évite le catastrophic forgetting que DQN subissait —
les poids ne peuvent pas être écrasés brutalement en une seule mise à jour.

---

### gymnasium — `src/train_ppo.py:165`

```python
self.observation_space = spaces.Box(
    low=0.0, high=1.0, shape=self.OBS_SHAPE_CNN, dtype=np.float32
)
self.action_space = spaces.Discrete(len(ACTIONS))
```

`spaces.Box` déclare formellement la forme et les bornes de l'observation — SB3 l'utilise
pour valider les données et dimensionner automatiquement le réseau. `spaces.Discrete(4)`
dit à PPO qu'il y a exactement 4 actions possibles → sortie acteur à 4 neurones.
Le contrat `reset()` / `step()` de gymnasium est universel : n'importe quel algorithme RL
compatible peut s'y connecter sans modifier l'environnement.

---

### numpy — `src/train_ppo.py:83`

```python
grid = np.zeros(CNN_OBS_SHAPE, dtype=np.float32)
for i, tile in enumerate(obs_dict["grid"]):
    row, col = i // 10, i % 10
    grid[row, col, tile] = 1.0
```

La grille est convertie en tenseur 10×10×5 : pour chaque case, le canal correspondant au type
de tile est mis à 1.0 (one-hot sur l'axe canal). NumPy alloue le tableau en mémoire contiguë
en C — PyTorch peut le consommer directement sans copie (`torch.from_numpy()`). Sans numpy,
l'encodage vectoriel serait 10× plus lent avec des listes Python pures.

---

### pytest — `tests/test_train_ppo.py`

```python
def test_encode_obs_cnn_player_channel(self, obs_grass):
    grid = encode_obs_cnn(obs_grass)
    cx, cy = obs_grass["char_pos"]
    assert grid[cy, cx, 3] == 1.0
```

Vérifie qu'après encodage, le canal 3 (personnage) est bien activé à la bonne position.
362 tests unitaires — un test par comportement attendu — garantissent qu'aucune modification
du code ne casse silencieusement un autre module. Le mode `-v` affiche chaque test individuellement,
`--cov` génère un rapport de couverture HTML.

---

### pyinstaller — `.vscode/tasks.json`

```bash
pyinstaller --onefile --noconsole src/main.py --name dungeon
```

`--onefile` compresse Python + toutes les librairies (pygame, numpy, torch ~200 Mo) dans un
seul `.exe`. `--noconsole` supprime la fenêtre console noire sous Windows. Sans pyinstaller,
l'utilisateur devrait installer Python 3.12 + tous les packages pour lancer le jeu.
L'exécutable final est autonome : il extrait ses dépendances dans un dossier temp au premier lancement.
