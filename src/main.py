import sys
import os

# ensure src/ is on the path when run directly
# Cette ligne rend le script exécutable depuis n'importe quel endroit.
sys.path.insert(0, os.path.dirname(__file__))

from ui import GameUI


def main() -> None:
    ui = GameUI()
    ui.run()


""""
garde-fou classique en Python.

__name__ est une variable automatique que Python définit différemment selon comment le fichier est utilisé :

Si tu lances python main.py directement → __name__ vaut "__main__" → main() est appelé
Si un autre fichier fait import main → __name__ vaut "main" → main() n'est pas appelé
Sans cette condition, importer main.py dans un test ou un autre module démarrerait involontairement la fenêtre pygame.
"""
if __name__ == "__main__":
    main()
