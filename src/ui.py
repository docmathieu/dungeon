import random
import sys
import threading
import tkinter
import tkinter.filedialog

import pygame

from pathlib import Path

from grid import Grid, TileType
from game_state import GameState


# --- colours ---
BLACK  = (  0,   0,   0)
WHITE  = (255, 255, 255)
GREEN  = ( 34, 139,  34)
GREY   = (128, 128, 128)
BLUE   = ( 30, 144, 255)
YELLOW = (255, 255,   0)
RED    = (220,  50,  50)
DARK   = ( 40,  40,  40)
CYAN   = (  0, 220, 220)
ORANGE = (255, 140,   0)   # couleur unique des tracés IA

OPTIMAL_TRAIL_OFFSET = 2   # px shift so optimal trail doesn't overlap player trail

TILE_PX   = 40   # rendered cell size (pixels)
SEP_PX    = 1    # separator between cells
CELL_STEP = TILE_PX + SEP_PX   # 41 px

GRID_W = Grid.WIDTH  * CELL_STEP - SEP_PX   # 409 px
GRID_H = Grid.HEIGHT * CELL_STEP - SEP_PX   # 409 px

HUD_TOP_H = 80    # height of the top HUD strip
HUD_BOT_H = 110   # height of the bottom HUD strip (2 rows of buttons + stats)

INPUT_ACTIVE_COL = ( 60,  60,  80)   # fond du champ de saisie actif
INPUT_BORDER_COL = (100, 100, 200)   # bordure du champ actif
SEED_MAX_DIGITS  = 6                  # longueur maximale de la saisie

WIN_W = GRID_W
WIN_H = HUD_TOP_H + GRID_H + HUD_BOT_H

ANIM_INTERVAL_MS = 200   # délai entre deux trails en mode multi


def _tile_rect(gx: int, gy: int) -> pygame.Rect:
    x = gx * CELL_STEP
    y = HUD_TOP_H + gy * CELL_STEP
    return pygame.Rect(x, y, TILE_PX, TILE_PX)


class GameUI:
    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Dungeon")
        self._font = pygame.font.SysFont("Arial", 16)
        self._clock = pygame.time.Clock()

        self._grid: Grid | None = None
        self._state: GameState | None = None

        # Champ de saisie de seed
        self._seed_input_text: str = ""
        self._seed_input_active: bool = False
        self._seed_input_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._current_seed: int | None = None

        # Mode IA simple (un seul modèle)
        self._ai_trail: list[tuple[int, int]] | None = None
        self._ai_net = None   # torch.nn.Module chargé

        # Mode IA multi (animation multi-checkpoints)
        self._ai_trails: list[dict] = []      # [{trail, color, alpha, stage_idx}]
        self._ai_nets_cache: list[dict] = []  # [{net, color, alpha, stage_idx}] — persistant entre resets
        self._ai_run_dir: Path | None = None  # dossier *_run/ chargé (conservé après reset)

        # Chemin optimal IA (rouge, affiché en fin d'épisode/animation)
        self._ai_optimal_path: list[tuple[int, int]] | None = None
        # Statistiques des épisodes IA (wins, note moy) — remis à None à chaque reset terrain
        self._ai_stats: dict | None = None
        self._anim_idx: int = -1            # index du trail actuellement visible
        self._anim_last_ms: int = 0
        self._loading_progress: float | None = None   # None=inactif, 0..1=chargement
        self._loading_thread: threading.Thread | None = None
        self._trail_surf: pygame.Surface | None = None   # surface réutilisable pour alpha

        # Rects des boutons (initialisés dans _draw_hud_bot)
        self._restart_rect:    pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._seed_input_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._ai_simple_rect:  pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._ai_multi_rect:   pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._ai_restart_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)

        self._reset()

    # ------------------------------------------------------------------
    def _reset(self) -> None:
        seed = random.randrange(10 ** SEED_MAX_DIGITS)
        self._state = GameState.create_solvable(seed=seed)
        self._grid = self._state.grid
        self._current_seed = seed
        self._ai_trail        = None
        self._ai_trails       = []
        self._anim_idx        = -1
        self._loading_progress = None
        self._ai_optimal_path = None
        self._ai_stats        = None

    def _reset_with_seed(self, seed: int) -> None:
        state = GameState.create_solvable(seed=seed)
        self._state = state
        self._grid  = state.grid
        self._current_seed = seed
        self._ai_trail        = None
        self._ai_trails       = []
        self._anim_idx        = -1
        self._loading_progress = None
        self._ai_optimal_path  = None
        self._ai_stats         = None

    # ------------------------------------------------------------------
    def _draw(self) -> None:
        self._screen.fill(BLACK)
        self._draw_grid()
        self._draw_trail()
        self._draw_optimal_trail()
        self._draw_ai_trail()         # mode simple
        self._draw_ai_trails()        # mode multi (animation)
        self._draw_ai_optimal_path()  # chemin optimal rouge (fin d'épisode IA)
        self._draw_character()
        self._draw_exit()
        self._draw_hud_top()
        self._draw_hud_bot()
        pygame.display.flip()

    def _draw_grid(self) -> None:
        for gy in range(Grid.HEIGHT):
            for gx in range(Grid.WIDTH):
                tile = self._grid.get_tile(gx, gy)
                colour = {
                    TileType.GRASS: GREEN,
                    TileType.ROCK:  GREY,
                    TileType.WATER: BLUE,
                }[tile]
                pygame.draw.rect(self._screen, colour, _tile_rect(gx, gy))

    def _draw_optimal_trail(self) -> None:
        if self._state.info == "" or self._state.optimal_path is None:
            return
        o = OPTIMAL_TRAIL_OFFSET
        path = self._state.optimal_path
        for i in range(len(path) - 1):
            ax, ay = path[i]
            bx, by = path[i + 1]
            ra = _tile_rect(ax, ay)
            rb = _tile_rect(bx, by)
            pygame.draw.line(
                self._screen, RED,
                (ra.centerx + o, ra.centery + o),
                (rb.centerx + o, rb.centery + o),
                2,
            )

    def _draw_ai_optimal_path(self) -> None:
        """Dessine le chemin optimal (rouge) en fin d'épisode IA.

        - Mode simple  : affiché dès que le trail est chargé.
        - Mode multi   : affiché uniquement quand l'animation est terminée.
        """
        if not self._ai_optimal_path or len(self._ai_optimal_path) < 2:
            return
        if self._ai_trail is not None:
            show = True   # simple : toujours
        elif self._ai_trails and self._anim_idx >= len(self._ai_trails) - 1:
            show = True   # multi : animation complète
        else:
            show = False
        if not show:
            return
        o = OPTIMAL_TRAIL_OFFSET
        for i in range(len(self._ai_optimal_path) - 1):
            ax, ay = self._ai_optimal_path[i]
            bx, by = self._ai_optimal_path[i + 1]
            ra = _tile_rect(ax, ay)
            rb = _tile_rect(bx, by)
            pygame.draw.line(
                self._screen, RED,
                (ra.centerx + o, ra.centery + o),
                (rb.centerx + o, rb.centery + o),
                2,
            )

    def _draw_ai_trail(self) -> None:
        """Dessine le tracé du mode IA simple (cyan uni)."""
        if not self._ai_trail or len(self._ai_trail) < 2:
            return
        for i in range(len(self._ai_trail) - 1):
            ax, ay = self._ai_trail[i]
            bx, by = self._ai_trail[i + 1]
            ra = _tile_rect(ax, ay)
            rb = _tile_rect(bx, by)
            pygame.draw.line(
                self._screen, ORANGE,
                (ra.centerx, ra.centery),
                (rb.centerx, rb.centery),
                3,
            )

    def _draw_ai_trails(self) -> None:
        """Dessine les trails animés du mode multi-checkpoints (avec alpha)."""
        if not self._ai_trails or self._anim_idx < 0:
            return
        if self._trail_surf is None:
            self._trail_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for td in self._ai_trails[:self._anim_idx + 1]:
            trail = td["trail"]
            if len(trail) < 2:
                continue
            r, g, b = td["color"]
            alpha   = td["alpha"]
            self._trail_surf.fill((0, 0, 0, 0))
            for i in range(len(trail) - 1):
                ax, ay = trail[i]
                bx, by = trail[i + 1]
                ra = _tile_rect(ax, ay)
                rb = _tile_rect(bx, by)
                pygame.draw.line(
                    self._trail_surf, (r, g, b, alpha),
                    (ra.centerx, ra.centery),
                    (rb.centerx, rb.centery),
                    3,
                )
            self._screen.blit(self._trail_surf, (0, 0))

    def _draw_trail(self) -> None:
        trail = self._state.trail
        for i in range(len(trail) - 1):
            ax, ay = trail[i]
            bx, by = trail[i + 1]
            ra = _tile_rect(ax, ay)
            rb = _tile_rect(bx, by)
            pygame.draw.line(
                self._screen, YELLOW,
                (ra.centerx, ra.centery),
                (rb.centerx, rb.centery),
                2,
            )

    def _draw_character(self) -> None:
        gx, gy = self._state.char_pos
        r = _tile_rect(gx, gy)
        cx, cy = r.centerx, r.centery
        pygame.draw.circle(self._screen, YELLOW, (cx, cy - 8), 7)
        pygame.draw.line(self._screen, YELLOW, (cx, cy - 1), (cx, cy + 10), 2)
        pygame.draw.line(self._screen, YELLOW, (cx - 7, cy + 4), (cx + 7, cy + 4), 2)
        pygame.draw.line(self._screen, YELLOW, (cx, cy + 10), (cx - 6, cy + 18), 2)
        pygame.draw.line(self._screen, YELLOW, (cx, cy + 10), (cx + 6, cy + 18), 2)

    def _draw_exit(self) -> None:
        gx, gy = self._state.exit_pos
        r = _tile_rect(gx, gy)
        door = pygame.Rect(r.centerx - 6, r.centery - 9, 12, 18)
        pygame.draw.rect(self._screen, YELLOW, door, 2)
        pygame.draw.circle(self._screen, YELLOW, (r.centerx + 4, r.centery), 2)

    # --- HUD ---
    def _label(self, text: str, x: int, y: int, colour=WHITE) -> None:
        surf = self._font.render(text, True, colour)
        self._screen.blit(surf, (x, y))

    def _draw_hud_top(self) -> None:
        s = self._state
        self._label(f"Déplacements : {s.move_count}", 4, 8)
        self._label(f"Note : {s.score}", 180, 8)
        info_col = (80, 255, 80) if s.info == "GAGNE" else (255, 80, 80) if s.info == "PERDU" else WHITE
        self._label(f"Information : {s.info}", 280, 8, info_col)
        self._label("← ↑ → ↓  pour se déplacer  |  R pour restart", 4, 32, GREY)
        self._label(f"Seed : {self._current_seed}", 4, 56, GREY)

    def _draw_hud_bot(self) -> None:
        bot_y = HUD_TOP_H + GRID_H

        # ── Ligne 1 : restart + seed ──────────────────────────────────
        restart_rect = pygame.Rect(4, bot_y + 12, 70, 28)
        pygame.draw.rect(self._screen, DARK, restart_rect)
        pygame.draw.rect(self._screen, WHITE, restart_rect, 1)
        self._label("restart", restart_rect.x + 6, restart_rect.y + 6)
        self._restart_rect = restart_rect

        self._label("Seed :", 84, bot_y + 18, GREY)
        input_rect = pygame.Rect(124, bot_y + 12, 80, 28)
        bg_col  = INPUT_ACTIVE_COL if self._seed_input_active else DARK
        bdr_col = INPUT_BORDER_COL if self._seed_input_active else GREY
        pygame.draw.rect(self._screen, bg_col, input_rect)
        pygame.draw.rect(self._screen, bdr_col, input_rect, 1)
        display_text = self._seed_input_text if self._seed_input_active else (
            str(self._current_seed) if self._current_seed is not None else ""
        )
        self._label(display_text, input_rect.x + 4, input_rect.y + 6)
        if self._seed_input_active and (pygame.time.get_ticks() // 500) % 2 == 0:
            cursor_x = input_rect.x + 4 + self._font.size(display_text)[0]
            pygame.draw.line(self._screen, WHITE,
                             (cursor_x, input_rect.y + 4),
                             (cursor_x, input_rect.y + 22), 1)
        self._seed_input_rect = input_rect
        self._label("↵ valider", input_rect.right + 6, bot_y + 18, GREY)

        # ── Ligne 2 : boutons IA ──────────────────────────────────────
        row2_y = bot_y + 52

        # [IA simple model]
        ai_simple_rect = pygame.Rect(4, row2_y, 122, 28)
        pygame.draw.rect(self._screen, DARK, ai_simple_rect)
        pygame.draw.rect(self._screen, CYAN, ai_simple_rect, 1)
        self._label("IA simple model", ai_simple_rect.x + 4, ai_simple_rect.y + 6, CYAN)
        self._ai_simple_rect = ai_simple_rect

        # [IA multi model]
        loading = self._loading_progress is not None
        multi_col = GREY if loading else CYAN
        ai_multi_rect = pygame.Rect(132, row2_y, 116, 28)
        pygame.draw.rect(self._screen, DARK, ai_multi_rect)
        pygame.draw.rect(self._screen, multi_col, ai_multi_rect, 1)
        self._label("IA multi model", ai_multi_rect.x + 4, ai_multi_rect.y + 6, multi_col)
        self._ai_multi_rect = ai_multi_rect

        # [IA restart]
        can_restart = (bool(self._ai_trails) or bool(self._ai_nets_cache)
                       or self._ai_net is not None or self._ai_run_dir is not None)
        rst_col = CYAN if can_restart else GREY
        ai_restart_rect = pygame.Rect(254, row2_y, 90, 28)
        pygame.draw.rect(self._screen, DARK, ai_restart_rect)
        pygame.draw.rect(self._screen, rst_col, ai_restart_rect, 1)
        self._label("IA restart", ai_restart_rect.x + 4, ai_restart_rect.y + 6, rst_col)
        self._ai_restart_rect = ai_restart_rect

        # ── Ligne 3 : trail info + statistiques IA ───────────────────────
        row3_y = bot_y + 86

        if loading:
            # Barre de progression pendant le chargement
            bar_w = int(WIN_W * self._loading_progress)
            pygame.draw.rect(self._screen, DARK, (0, row3_y, WIN_W, 8))
            pygame.draw.rect(self._screen, CYAN, (0, row3_y, bar_w, 8))

        # Stats IA (affichées pendant ET après chargement)
        parts: list[str] = []
        if self._ai_trails:
            n     = len(self._ai_trails)
            shown = max(0, self._anim_idx + 1)
            parts.append(f"Trail {shown}/{n}")
        if self._ai_stats:
            wins  = self._ai_stats["wins"]
            total = self._ai_stats["total"]
            parts.append(f"Victoires : {wins}/{total}")
            if wins > 0:
                note = self._ai_stats["note_moy"]
                parts.append(f"Note moy : {note:.0f}")
            else:
                parts.append("Note moy : —")
        if parts:
            text_y = row3_y + (12 if loading else 2)
            self._label("   |   ".join(parts), 4, text_y, GREY)

    # ------------------------------------------------------------------
    def _run_ai_simple(self) -> None:
        """File picker → charge un modèle → joue un épisode complet (mode simple)."""
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path_str = tkinter.filedialog.askopenfilename(
            title="Choisir un checkpoint (.pt)",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("Tous les fichiers", "*.*")],
        )
        root.destroy()
        if not path_str:
            return
        try:
            from exploit import load_net, run_one_episode_info
            self._ai_net = load_net(Path(path_str))
            trail, won, score = run_one_episode_info(self._ai_net, seed=self._current_seed)
            self._ai_trail        = trail
            self._ai_optimal_path = self._state.optimal_path   # même seed → même chemin
            self._ai_stats = {
                "wins":     1 if won else 0,
                "note_moy": score if won else 0,
                "total":    1,
            }
        except Exception as exc:
            print(f"[IA simple] Erreur : {exc}")

    def _run_ai_multi(self) -> None:
        """Directory picker → mémorise le dossier → lance le chargement."""
        if self._loading_progress is not None:
            return   # chargement déjà en cours
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        run_dir_str = tkinter.filedialog.askdirectory(
            title="Choisir un dossier de run (*_run)",
        )
        root.destroy()
        if not run_dir_str:
            return
        self._ai_run_dir      = Path(run_dir_str)
        self._ai_optimal_path = self._state.optimal_path   # capturé avant le thread
        self._load_multi(self._ai_run_dir, self._current_seed)

    def _load_multi(self, run_dir: Path, seed: int | None) -> None:
        """Lance le chargement de tous les checkpoints du run en thread de fond."""
        if self._loading_progress is not None:
            return
        self._ai_trails = []
        self._anim_idx  = -1
        self._loading_progress = 0.0

        def _load() -> None:
            from exploit import scan_run_dir, load_net, run_one_episode_info
            try:
                checkpoints = scan_run_dir(run_dir)
                if not checkpoints:
                    self._loading_progress = None
                    return

                n = len(checkpoints)
                stage_counts: dict[int, int] = {}
                for cp in checkpoints:
                    s = cp["stage_idx"]
                    stage_counts[s] = stage_counts.get(s, 0) + 1
                stage_seen: dict[int, int] = {}

                nets_cache: list[dict] = []
                trails:     list[dict] = []
                wins = 0
                scores_sum = 0
                for i, cp in enumerate(checkpoints):
                    net               = load_net(cp["pt_path"])
                    trail, won, score = run_one_episode_info(net, seed=seed)
                    if won:
                        wins      += 1
                        scores_sum += score
                    s     = cp["stage_idx"]
                    stage_seen[s] = stage_seen.get(s, 0) + 1
                    k     = stage_seen[s]
                    n_s   = stage_counts[s]
                    alpha = int(50 + 170 * k / n_s)
                    nets_cache.append({"net": net, "color": cp["color"],
                                       "alpha": alpha, "stage_idx": s})
                    trails.append({"trail": trail, "color": cp["color"],
                                   "alpha": alpha, "stage_idx": s})
                    self._loading_progress = (i + 1) / n
                    # mise à jour incrémentale pendant le chargement
                    self._ai_stats = {
                        "wins":     wins,
                        "note_moy": scores_sum / wins if wins > 0 else 0,
                        "total":    i + 1,
                    }

                self._ai_nets_cache = nets_cache   # persistant — réutilisé sans disque
                self._ai_trails     = trails
                self._anim_idx      = -1
                self._anim_last_ms  = pygame.time.get_ticks()
            except Exception as exc:
                print(f"[IA multi] Erreur : {exc}")
            finally:
                self._loading_progress = None

        self._loading_thread = threading.Thread(target=_load, daemon=True)
        self._loading_thread.start()

    def _rerun_from_cache(self, seed: int | None) -> None:
        """Rejoue tous les modèles en cache sur un nouveau seed, sans relire le disque."""
        if not self._ai_nets_cache or self._loading_progress is not None:
            return
        self._ai_trails       = []
        self._anim_idx        = -1
        self._ai_optimal_path = self._state.optimal_path   # capturé avant le thread

        def _rerun() -> None:
            from exploit import run_one_episode_info
            try:
                trails: list[dict] = []
                wins = 0
                scores_sum = 0
                for entry in self._ai_nets_cache:
                    trail, won, score = run_one_episode_info(entry["net"], seed=seed)
                    if won:
                        wins      += 1
                        scores_sum += score
                    trails.append({"trail": trail, "color": entry["color"],
                                   "alpha": entry["alpha"], "stage_idx": entry["stage_idx"]})
                self._ai_stats = {
                    "wins":     wins,
                    "note_moy": scores_sum / wins if wins > 0 else 0,
                    "total":    len(self._ai_nets_cache),
                }
                self._ai_trails    = trails
                self._anim_idx     = -1
                self._anim_last_ms = pygame.time.get_ticks()
            except Exception as exc:
                print(f"[IA restart multi] Erreur : {exc}")

        threading.Thread(target=_rerun, daemon=True).start()

    def _restart_ai_anim(self) -> None:
        """Relance l'animation ou recalcule les trails pour le terrain courant.

        - Trails présents (terrain inchangé) → redémarre l'animation depuis le début.
        - Nouveau terrain + nets en cache → rejoue les épisodes sans relire le disque.
        - Nouveau terrain + run_dir connu, cache vide → rechargement complet depuis disque.
        - Nouveau terrain + modèle simple → rejoue le modèle simple.
        """
        if self._ai_trails:
            self._anim_idx     = -1
            self._anim_last_ms = pygame.time.get_ticks()
        elif self._ai_nets_cache:
            self._rerun_from_cache(self._current_seed)
        elif self._ai_run_dir is not None:
            self._load_multi(self._ai_run_dir, self._current_seed)
        elif self._ai_net is not None:
            try:
                from exploit import run_one_episode_info
                trail, won, score = run_one_episode_info(self._ai_net, seed=self._current_seed)
                self._ai_trail = trail
                self._ai_stats = {
                    "wins":     1 if won else 0,
                    "note_moy": score if won else 0,
                    "total":    1,
                }
            except Exception as exc:
                print(f"[IA restart] Erreur : {exc}")

    # ------------------------------------------------------------------
    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._restart_rect.collidepoint(event.pos):
                self._reset()
                return
            if self._ai_simple_rect.collidepoint(event.pos):
                self._run_ai_simple()
                return
            if self._ai_multi_rect.collidepoint(event.pos):
                self._run_ai_multi()
                return
            if self._ai_restart_rect.collidepoint(event.pos):
                self._restart_ai_anim()
                return
            if self._seed_input_rect.collidepoint(event.pos):
                self._seed_input_active = True
                self._seed_input_text   = ""
            else:
                self._seed_input_active = False

        if event.type == pygame.KEYDOWN:
            if self._seed_input_active:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if self._seed_input_text:
                        self._reset_with_seed(int(self._seed_input_text))
                    self._seed_input_active = False
                    self._seed_input_text   = ""
                elif event.key == pygame.K_ESCAPE:
                    self._seed_input_active = False
                    self._seed_input_text   = ""
                elif event.key == pygame.K_BACKSPACE:
                    self._seed_input_text = self._seed_input_text[:-1]
                elif event.unicode.isdigit() and len(self._seed_input_text) < SEED_MAX_DIGITS:
                    self._seed_input_text += event.unicode
                return

            arrow_map = {
                pygame.K_LEFT:  "LEFT",
                pygame.K_RIGHT: "RIGHT",
                pygame.K_UP:    "UP",
                pygame.K_DOWN:  "DOWN",
            }
            if event.key in arrow_map and not self._state.won:
                self._state.apply_move(arrow_map[event.key])
            elif event.key == pygame.K_r:
                self._reset()

    # ------------------------------------------------------------------
    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                self._handle_event(event)

            # Avancement automatique de l'animation multi-trails
            if self._ai_trails and self._anim_idx < len(self._ai_trails) - 1:
                now = pygame.time.get_ticks()
                if self._anim_idx == -1 or now - self._anim_last_ms >= ANIM_INTERVAL_MS:
                    self._anim_idx    += 1
                    self._anim_last_ms = now

            self._draw()
            self._clock.tick(60)
