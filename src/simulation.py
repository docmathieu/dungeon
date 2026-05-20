import queue
import threading
import time

from game_state import GameState


REPAINT = "repaint"


class Simulation(threading.Thread):
    def __init__(
        self,
        game_state: GameState,
        instructions: str,
        ui_queue: queue.Queue | None = None,
    ):
        super().__init__(daemon=True)
        self._state = game_state
        self._moves = self._parse(instructions)
        self._queue = ui_queue
        self._stop_event = threading.Event()

    @staticmethod
    def _parse(instructions: str) -> list[str]:
        mapping = {"←": "LEFT", "↑": "UP", "→": "RIGHT", "↓": "DOWN"}
        result = []
        for ch in instructions:
            if ch in mapping:
                result.append(mapping[ch])
        return result

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        for move in self._moves:
            if self._stop_event.is_set() or self._state.won:
                break
            self._state.apply_move(move)
            if self._queue is not None:
                self._queue.put(REPAINT)
                time.sleep(0.3)

        if not self._stop_event.is_set() and not self._state.won:
            self._state.score = 0
            self._state.info = "PERDU"
            if self._queue is not None:
                self._queue.put(REPAINT)
