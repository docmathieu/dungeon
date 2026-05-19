Build a standalone Windows executable for the dungeon game.

## Steps

<!-- Étape 1 : S'assurer que l'environnement virtuel Python existe et que les dépendances sont installées. Sans .venv, PyInstaller n'est pas disponible. -->
1. Check that `.venv` exists. If not, run:
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

<!-- Étape 2 : Vérifier que le code source du jeu est présent. PyInstaller ne peut pas créer l'exe sans ce fichier d'entrée. -->
2. Check that `src/main.py` exists. If not, invoke `/generate-game` first.

<!-- Étape 3 : Lancer PyInstaller pour compiler le jeu en un seul exécutable Windows autonome, sans fenêtre console. -->
3. Run PyInstaller:
   ```
   .venv\Scripts\pyinstaller --onefile --noconsole --name dungeon src/main.py
   ```

<!-- Étape 4 : Confirmer que la compilation a réussi en vérifiant la présence du fichier .exe dans le dossier dist/. -->
4. Verify that `dist/dungeon.exe` was created.

<!-- Étape 5 : Communiquer le résultat à l'utilisateur : chemin complet, taille du fichier et tout avertissement émis par PyInstaller. -->
5. Report:
   - Full path to the executable
   - File size
   - Any warnings or errors from PyInstaller

<!-- Étape 6 : Cas particulier — si pygame n'est pas détecté automatiquement par PyInstaller, forcer son inclusion avec --hidden-import et relancer. -->
6. If PyInstaller raises a missing-module warning for pygame, add:
   ```
   --hidden-import pygame
   ```
   and retry.

## Notes
- The `--noconsole` flag hides the terminal window when the exe runs on Windows.
- The `--onefile` flag bundles everything into a single .exe for easy distribution.
- The `dist/` and `build/` folders and the `dungeon.spec` file are normal PyInstaller output — they can be ignored in version control.
