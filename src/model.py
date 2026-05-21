"""model.py — DQNetwork : réseau MLP pour l'algorithme DQN.

Architecture :
    INPUT(304) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(4)

Entrée (304 floats) :
    - grille 10×10 en one-hot : 100 cases × 3 types = 300 floats
    - char_pos normalisé      : (x/9, y/9)          = 2 floats
    - exit_pos normalisé      : (x/9, y/9)           = 2 floats

Sortie (4 floats) :
    Q-values pour les 4 actions (LEFT, RIGHT, UP, DOWN)
"""

import torch
import torch.nn as nn


INPUT_DIM  = 304   # 100×3 one-hot + 2 char_pos + 2 exit_pos
HIDDEN1    = 128
HIDDEN2    = 64
OUTPUT_DIM = 4     # LEFT, RIGHT, UP, DOWN


class DQNetwork(nn.Module):
    """Réseau MLP approximant les Q-values pour les 4 actions."""

    def __init__(
        self,
        input_dim:  int = INPUT_DIM,
        hidden1:    int = HIDDEN1,
        hidden2:    int = HIDDEN2,
        output_dim: int = OUTPUT_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : tensor de forme (batch_size, input_dim) ou (input_dim,)."""
        return self.net(x)
