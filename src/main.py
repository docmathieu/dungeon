import sys
import os

# ensure src/ is on the path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from ui import GameUI


def main() -> None:
    ui = GameUI()
    ui.run()


if __name__ == "__main__":
    main()
