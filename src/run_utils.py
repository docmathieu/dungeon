"""run_utils.py — utilitaires de nommage partagés entre train.py et train_ppo.py."""

import time
from pathlib import Path


def _now() -> str:
    """Retourne le timestamp courant au format yyyymmdd_hhmm."""
    return time.strftime("%Y%m%d_%H%M")


def _pretrained_label(pretrained: "Path | None") -> str:
    """Extrait le timestamp du dossier source pour le suffixe _from_.

    DQN : Path('models/20260527_1222_seed42_ep2000/final.pt')  → '20260527_1222'
    PPO : Path('models/.../20260602_1556_ppo_random_cnn_ts2M/final.zip') → '20260602_1556'
    None → ''
    """
    if pretrained is None:
        return ""
    return Path(pretrained).parent.name[:13]
