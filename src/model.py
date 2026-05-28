"""model.py — réseaux MLP pour l'algorithme DQN.

Deux architectures disponibles :

1. DQNetwork (baseline)
   INPUT(314) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Linear(4)
   Le seed one-hot est concaténé à l'entrée — les couches cachées restent partagées.

2. FiLMDQNetwork (recommandé pour multi-seeds)
   obs(304) → FC1(256) → FiLM(task) → ReLU
           → FC2(128) → FiLM(task) → ReLU
           → FC3(64)  → FiLM(task) → ReLU
           → FC4(4)
   Le seed one-hot module chaque couche cachée via gamma*x+beta appris,
   créant des pathways virtuellement séparés par seed sans multiplier les paramètres.

Entrée commune (314 floats) :
    - grille 10×10 en one-hot : 100 cases × 3 types = 300 floats
    - char_pos normalisé      : (x/9, y/9)          = 2 floats
    - exit_pos normalisé      : (x/9, y/9)           = 2 floats
    - seed one-hot            : 10 bits (task-conditioning) = 10 floats

Sortie (4 floats) :
    Q-values pour les 4 actions (LEFT, RIGHT, UP, DOWN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_DIM  = 314   # 100×3 one-hot + 2 char_pos + 2 exit_pos + 10 seed one-hot
OBS_DIM    = 304   # partie observation pure (sans seed one-hot)
TASK_DIM   = 10    # dimension du seed one-hot
HIDDEN1    = 256
HIDDEN2    = 128
HIDDEN3    = 64
OUTPUT_DIM = 4     # LEFT, RIGHT, UP, DOWN


class DQNetwork(nn.Module):
    """Réseau MLP baseline — seed one-hot concaténé en entrée."""

    def __init__(
        self,
        input_dim:  int = INPUT_DIM,
        hidden1:    int = HIDDEN1,
        hidden2:    int = HIDDEN2,
        hidden3:    int = HIDDEN3,
        output_dim: int = OUTPUT_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : tensor de forme (batch_size, input_dim) ou (input_dim,)."""
        return self.net(x)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation.

    Module les activations x par : gamma(task_emb) * x + beta(task_emb)
    où gamma et beta sont appris depuis le seed one-hot.
    Chaque seed apprend ainsi un scaling et un biais distincts par couche.
    """

    def __init__(self, feature_dim: int, task_dim: int = TASK_DIM):
        super().__init__()
        self.gamma = nn.Linear(task_dim, feature_dim)
        self.beta  = nn.Linear(task_dim, feature_dim)

    def forward(self, x: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        """x : (*, feature_dim) — task_emb : (*, task_dim)."""
        return self.gamma(task_emb) * x + self.beta(task_emb)


class FiLMDQNetwork(nn.Module):
    """DQNetwork avec FiLM conditioning par seed (recommandé pour multi-seeds).

    Reçoit un vecteur de 314 floats (format identique à DQNetwork) :
    les 304 premiers = observation, les 10 derniers = seed one-hot.
    Le seed one-hot module chaque couche cachée via FiLM, créant des
    pathways virtuellement séparés sans gradients conflictuels.
    """

    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        task_dim:   int = TASK_DIM,
        hidden1:    int = HIDDEN1,
        hidden2:    int = HIDDEN2,
        hidden3:    int = HIDDEN3,
        output_dim: int = OUTPUT_DIM,
    ):
        super().__init__()
        self._obs_dim = obs_dim

        self.fc1   = nn.Linear(obs_dim, hidden1)
        self.film1 = FiLMLayer(hidden1, task_dim)
        self.fc2   = nn.Linear(hidden1, hidden2)
        self.film2 = FiLMLayer(hidden2, task_dim)
        self.fc3   = nn.Linear(hidden2, hidden3)
        self.film3 = FiLMLayer(hidden3, task_dim)
        self.fc4   = nn.Linear(hidden3, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : tensor de forme (batch_size, 314) ou (314,)."""
        obs      = x[..., :self._obs_dim]   # (*, 304) — observation
        task_emb = x[..., self._obs_dim:]   # (*, 10)  — seed one-hot

        h = F.relu(self.film1(self.fc1(obs), task_emb))
        h = F.relu(self.film2(self.fc2(h),   task_emb))
        h = F.relu(self.film3(self.fc3(h),   task_emb))
        return self.fc4(h)
