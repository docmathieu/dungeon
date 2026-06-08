"""DungeonEnv — interface Gym wrapping GameState pour l'entraînement RL.

Trois modes de seed :
    DungeonEnv(seed=42)           → même terrain à chaque reset()
    DungeonEnv(seed=None)         → terrain aléatoire à chaque reset()
    DungeonEnv(seed_pool=[...])   → tirage aléatoire dans un pool fixe

Observation retournée par reset() et step() :
    {
        "grid":     list[int] de 100 valeurs (0=HERBE, 1=ROCHE, 2=EAU),
                    ordre ligne-major (y extérieur, x intérieur)
        "char_pos": tuple[int, int]   position (x, y) du personnage
        "exit_pos": tuple[int, int]   position (x, y) de la sortie
    }

Reward :
    victoire  → state.score / 100.0  (1.0 = chemin optimal)
    autres    → 0.0
"""

import random

from grid import Grid, TileType
from game_state import GameState
from pathfinder import PathFinder


# Encodage entier des types de cases (utilisé dans l'observation)
_TILE_ENC: dict[TileType, int] = {
    TileType.GRASS: 0,
    TileType.ROCK:  1,
    TileType.WATER: 2,
}

# Les quatre actions valides
ACTIONS: tuple[str, ...] = ("LEFT", "RIGHT", "UP", "DOWN")

# Reward shaping — signaux intermédiaires à chaque pas
REWARD_STEP            = -0.01   # déplacement normal (herbe ou eau) sans victoire
REWARD_BUMP            = -0.05   # choc contre un mur ou un bord de grille
REWARD_PROGRESS_SCALE  =  0.10   # facteur appliqué à la réduction de coût Dijkstra


MAX_STEPS: int = 100   # nombre maximum de pas par épisode (module-level alias)


class DungeonEnv:
    """Interface Gym pour le jeu Dungeon (Phase 1 RL)."""

    MAX_STEPS: int = MAX_STEPS

    def __init__(
        self,
        seed: int | None = None,
        seed_pool: list[int] | None = None,
        max_steps: int = MAX_STEPS,
    ):
        if seed_pool is not None and len(seed_pool) == 0:
            raise ValueError("seed_pool ne doit pas être vide")
        self._seed            = seed
        self._seed_pool       = seed_pool
        self._max_steps       = max_steps
        self._state: GameState | None = None
        self._steps: int      = 0
        self._rng             = random.Random()   # tirage seed_pool et seeds aléatoires
        self._current_seed_idx: int = 0           # index dans seed_pool du dernier reset()
        self._effective_seed: int | None = None   # seed effectif du dernier épisode
        self._pf              = PathFinder()       # réutilisé à chaque step (pas de reset)
        self._dijkstra_cost: float = 0.0           # coût Dijkstra depuis la position courante

    # ------------------------------------------------------------------
    def reset(self) -> dict:
        """Démarre un nouvel épisode. Retourne l'observation initiale."""
        self._state = GameState.create_solvable(seed=self._pick_seed())
        self._steps = 0
        self._dijkstra_cost = float(
            self._pf.shortest_cost(
                self._state.grid, self._state.char_pos, self._state.exit_pos
            ) or 0
        )
        return self._observe()

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """Applique une action et retourne (obs, reward, done, info).

        action — "LEFT" / "RIGHT" / "UP" / "DOWN" (insensible à la casse).
        Lève RuntimeError si reset() n'a pas encore été appelé.

        Reward shaping :
            victoire            → score / 100.0  (1.0 = chemin optimal)
            choc mur / bord     → REWARD_BUMP  (-0.05)
            déplacement normal  → REWARD_STEP (-0.01)
                                  + (dijkstra_avant - dijkstra_après) * REWARD_PROGRESS_SCALE
                                  (positif si l'agent se rapproche de la sortie,
                                   négatif s'il s'éloigne — Dijkstra intègre les détours
                                   obligatoires donc ne pénalise pas les contournements)
        """
        if self._state is None:
            raise RuntimeError("Appeler reset() avant step()")

        pos_before   = self._state.char_pos
        cost_before  = self._dijkstra_cost
        self._state.apply_move(action)
        self._steps += 1

        won  = self._state.won
        done = won or self._steps >= self._max_steps

        if won:
            reward = self._state.score / 100.0       # victoire
        elif self._state.char_pos == pos_before:
            reward = REWARD_BUMP                      # choc : position inchangée, Dijkstra inchangé
        else:
            cost_after = float(
                self._pf.shortest_cost(
                    self._state.grid, self._state.char_pos, self._state.exit_pos
                ) or 0
            )
            self._dijkstra_cost = cost_after
            progress = cost_before - cost_after
            reward = REWARD_STEP + progress * REWARD_PROGRESS_SCALE

        info = {
            "score": self._state.score,
            "moves": self._state.move_count,
            "steps": self._steps,
            "won":   won,
        }
        return self._observe(), reward, done, info

    # ------------------------------------------------------------------
    @property
    def action_space(self) -> tuple[str, ...]:
        """Les quatre actions valides."""
        return ACTIONS

    # ------------------------------------------------------------------
    @property
    def current_seed_idx(self) -> int:
        """Index dans seed_pool du seed utilisé pour l'épisode courant (0 si pas de pool)."""
        return self._current_seed_idx

    @property
    def current_seed(self) -> int | None:
        """Valeur du seed utilisé pour l'épisode courant.

        Retourne la valeur effective du seed (ex. 42), pas son index dans le pool.
        En mode full-random (seed=None, pas de pool), retourne le seed aléatoire
        généré à chaque reset() — jamais None après un reset().
        None uniquement avant le premier reset().
        """
        return self._effective_seed

    def _pick_seed(self) -> int | None:
        """Choisit le seed pour le prochain épisode, stocke dans _effective_seed."""
        if self._seed_pool is not None:
            self._current_seed_idx = self._rng.randrange(len(self._seed_pool))
            self._effective_seed = self._seed_pool[self._current_seed_idx]
        elif self._seed is None:
            # Mode full-random : génère un seed aléatoire et le mémorise pour le logging
            self._effective_seed = self._rng.randrange(10 ** 6)
        else:
            self._current_seed_idx = 0
            self._effective_seed = self._seed
        return self._effective_seed

    def _observe(self) -> dict:
        """Construit le dict d'observation depuis l'état courant."""
        grid = self._state.grid
        grid_enc: list[int] = [
            _TILE_ENC[grid.get_tile(x, y)]
            for y in range(Grid.HEIGHT)
            for x in range(Grid.WIDTH)
        ]
        return {
            "grid":     grid_enc,
            "char_pos": self._state.char_pos,
            "exit_pos": self._state.exit_pos,
        }
