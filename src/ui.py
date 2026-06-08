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

TRAIL_WIDTH  = 3   # épaisseur de tous les traits (px)
TRAIL_OFFSET = 5   # décalage latéral — rouge = −offset (gauche), jaune = 0 (centre), orange = +offset (droite)
_TRAIL_COLORKEY = (1, 0, 1)  # magenta pur — couleur-clé transparente pour la surface multi-trails

TILE_PX   = 80   # rendered cell size (pixels)
SEP_PX    = 0    # separator between cells (0 = no gap)
CELL_STEP = TILE_PX + SEP_PX   # 80 px

_ASSETS_DIR = Path(__file__).parent.parent / "assets" / "tiles"

GRID_W = Grid.WIDTH  * CELL_STEP - SEP_PX   # 409 px
GRID_H = Grid.HEIGHT * CELL_STEP - SEP_PX   # 409 px

HUD_TOP_H = 96    # height of the top HUD strip (stats + seed + all buttons + loading bar)
HUD_BOT_H = 0     # plus de HUD bas — tous les boutons sont en HUD_TOP

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
        pygame.display.set_caption("Dungeon RL")
        self._font = pygame.font.SysFont("Arial", 16)
        self._clock = pygame.time.Clock()
        self._tile_surfaces  = self._load_tile_surfaces()
        self._player_surface = self._load_sprite("player.png")
        self._castle_surface = self._load_sprite("castle.png")

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

        # État chargement modèles IA (mutuellement exclusifs)
        self._sm_loaded: bool = False   # modèle simple chargé → bouton bleu
        self._mm_loaded: bool = False   # multi modèles chargés → bouton bleu

        # Rects des boutons (initialisés dans _draw_hud_top)
        self._restart_rect:    pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._seed_input_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._ai_simple_rect:  pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._restart_sm_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._ai_multi_rect:   pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._restart_mm_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)

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
        self._draw_castle()
        self._draw_hud_top()
        self._draw_hud_bot()
        pygame.display.flip()

    @staticmethod
    def _load_tile_surfaces() -> dict:
        """Charge et redimensionne les PNG de tuiles (40×40 px).

        Retourne un dict {TileType → Surface}.
        Si un fichier est absent, la valeur est None → fallback couleur dans _draw_grid.
        """
        mapping = {
            TileType.GRASS: "grass.png",
            TileType.ROCK:  "rock.png",
            TileType.WATER: "water.png",
        }
        surfaces = {}
        for tile_type, filename in mapping.items():
            path = _ASSETS_DIR / filename
            if path.exists():
                img = pygame.image.load(str(path)).convert_alpha()
                surfaces[tile_type] = pygame.transform.scale(img, (TILE_PX, TILE_PX))
            else:
                surfaces[tile_type] = None   # fallback couleur
        return surfaces

    @staticmethod
    def _load_sprite(filename: str) -> "pygame.Surface | None":
        """Charge un sprite depuis assets/tiles/ et le redimensionne à TILE_PX×TILE_PX.

        Retourne None si le fichier est absent.
        """
        path = _ASSETS_DIR / filename
        if path.exists():
            img = pygame.image.load(str(path)).convert_alpha()
            return pygame.transform.scale(img, (TILE_PX, TILE_PX))
        return None

    def _draw_grid(self) -> None:
        _fallback = {
            TileType.GRASS: GREEN,
            TileType.ROCK:  GREY,
            TileType.WATER: BLUE,
        }
        for gy in range(Grid.HEIGHT):
            for gx in range(Grid.WIDTH):
                tile = self._grid.get_tile(gx, gy)
                surf = self._tile_surfaces.get(tile)
                rect = _tile_rect(gx, gy)
                if surf is not None:
                    self._screen.blit(surf, rect)
                else:
                    pygame.draw.rect(self._screen, _fallback[tile], rect)

    def _draw_optimal_trail(self) -> None:
        if self._state.info == "" or self._state.optimal_path is None:
            return
        o = -TRAIL_OFFSET
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
                TRAIL_WIDTH,
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
        o = -TRAIL_OFFSET
        for i in range(len(self._ai_optimal_path) - 1):
            ax, ay = self._ai_optimal_path[i]
            bx, by = self._ai_optimal_path[i + 1]
            ra = _tile_rect(ax, ay)
            rb = _tile_rect(bx, by)
            pygame.draw.line(
                self._screen, RED,
                (ra.centerx + o, ra.centery + o),
                (rb.centerx + o, rb.centery + o),
                TRAIL_WIDTH,
            )

    def _draw_ai_trail(self) -> None:
        """Dessine le tracé du mode IA simple (orange)."""
        if not self._ai_trail or len(self._ai_trail) < 2:
            return
        o = TRAIL_OFFSET
        for i in range(len(self._ai_trail) - 1):
            ax, ay = self._ai_trail[i]
            bx, by = self._ai_trail[i + 1]
            ra = _tile_rect(ax, ay)
            rb = _tile_rect(bx, by)
            pygame.draw.line(
                self._screen, ORANGE,
                (ra.centerx + o, ra.centery + o),
                (rb.centerx + o, rb.centery + o),
                TRAIL_WIDTH,
            )

    def _draw_ai_trails(self) -> None:
        """Dessine les trails animés du mode multi-checkpoints (avec alpha).

        Utilise colorkey + set_alpha (surface normale) plutôt que SRCALPHA pour
        garantir que TRAIL_WIDTH est respecté quelle que soit la version SDL2.
        """
        if not self._ai_trails or self._anim_idx < 0:
            return
        if self._trail_surf is None:
            self._trail_surf = pygame.Surface((WIN_W, WIN_H))
            self._trail_surf.set_colorkey(_TRAIL_COLORKEY)
        o = TRAIL_OFFSET
        for td in self._ai_trails[:self._anim_idx + 1]:
            trail = td["trail"]
            if len(trail) < 2:
                continue
            r, g, b = td["color"]
            self._trail_surf.fill(_TRAIL_COLORKEY)        # efface → transparent via colorkey
            self._trail_surf.set_alpha(td["alpha"])        # alpha global du trail
            for i in range(len(trail) - 1):
                ax, ay = trail[i]
                bx, by = trail[i + 1]
                ra = _tile_rect(ax, ay)
                rb = _tile_rect(bx, by)
                pygame.draw.line(
                    self._trail_surf, (r, g, b),           # couleur RGB (sans alpha)
                    (ra.centerx + o, ra.centery + o),
                    (rb.centerx + o, rb.centery + o),
                    TRAIL_WIDTH,
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
                TRAIL_WIDTH,
            )

    def _draw_character(self) -> None:
        gx, gy = self._state.char_pos
        r = _tile_rect(gx, gy)
        if self._player_surface is not None:
            self._screen.blit(self._player_surface, r)
        else:
            cx, cy = r.centerx, r.centery
            pygame.draw.circle(self._screen, YELLOW, (cx, cy - 8), 7)
            pygame.draw.line(self._screen, YELLOW, (cx, cy - 1), (cx, cy + 10), 2)
            pygame.draw.line(self._screen, YELLOW, (cx - 7, cy + 4), (cx + 7, cy + 4), 2)
            pygame.draw.line(self._screen, YELLOW, (cx, cy + 10), (cx - 6, cy + 18), 2)
            pygame.draw.line(self._screen, YELLOW, (cx, cy + 10), (cx + 6, cy + 18), 2)

    def _draw_castle(self) -> None:
        gx, gy = self._state.exit_pos
        r = _tile_rect(gx, gy)
        if self._castle_surface is not None:
            self._screen.blit(self._castle_surface, r)
        else:
            door = pygame.Rect(r.centerx - 6, r.centery - 9, 12, 18)
            pygame.draw.rect(self._screen, YELLOW, door, 2)
            pygame.draw.circle(self._screen, YELLOW, (r.centerx + 4, r.centery), 2)

    # --- HUD ---
    def _label(self, text: str, x: int, y: int, colour=WHITE) -> None:
        surf = self._font.render(text, True, colour)
        self._screen.blit(surf, (x, y))

    def _draw_hud_top(self) -> None:
        s       = self._state
        loading = self._loading_progress is not None

        # ── Row 1 (y=4) : [Génération terrain]  Seed:[input]  stats jeu  touches ──
        restart_rect = pygame.Rect(4, 4, 160, 22)
        pygame.draw.rect(self._screen, DARK, restart_rect)
        pygame.draw.rect(self._screen, WHITE, restart_rect, 1)
        self._label("Génération terrain", restart_rect.x + 6, restart_rect.y + 3)
        self._restart_rect = restart_rect

        self._label("Seed :", restart_rect.right + 6, 8)
        input_rect = pygame.Rect(restart_rect.right + 50, 4, 78, 22)
        bg_col  = INPUT_ACTIVE_COL if self._seed_input_active else DARK
        bdr_col = INPUT_BORDER_COL if self._seed_input_active else WHITE
        pygame.draw.rect(self._screen, bg_col,  input_rect)
        pygame.draw.rect(self._screen, bdr_col, input_rect, 1)
        display_text = self._seed_input_text if self._seed_input_active else (
            str(self._current_seed) if self._current_seed is not None else ""
        )
        self._label(display_text, input_rect.x + 4, input_rect.y + 3)
        if self._seed_input_active and (pygame.time.get_ticks() // 500) % 2 == 0:
            cursor_x = input_rect.x + 4 + self._font.size(display_text)[0]
            pygame.draw.line(self._screen, WHITE,
                             (cursor_x, input_rect.y + 2),
                             (cursor_x, input_rect.y + 19), 1)
        self._seed_input_rect = input_rect

        stats_x = input_rect.right + 12
        info_col = (80, 255, 80) if s.info == "GAGNE" else (255, 80, 80) if s.info == "PERDU" else WHITE
        self._label(f"Déplacements : {s.move_count}", stats_x, 8)
        self._label(f"Note : {s.score}", stats_x + 155, 8)
        self._label(f"Info : {s.info}", stats_x + 245, 8, info_col)
        keys_text = "← ↑ → ↓   |   R restart"
        self._label(keys_text, WIN_W - self._font.size(keys_text)[0] - 4, 8, GREY)

        # ── Row 2 (y=34) : [IA simple model]  [restart SM]  stats simple ──
        row2_y = 34
        sm_bdr = GREY if loading else (BLUE if self._sm_loaded else WHITE)
        ai_simple_rect = pygame.Rect(4, row2_y, 130, 22)
        pygame.draw.rect(self._screen, DARK, ai_simple_rect)
        pygame.draw.rect(self._screen, sm_bdr, ai_simple_rect, 1)
        self._label("IA simple model", ai_simple_rect.x + 4, ai_simple_rect.y + 3, sm_bdr)
        self._ai_simple_rect = ai_simple_rect

        can_rst_sm = self._ai_net is not None and not loading
        rst_sm_col = CYAN if can_rst_sm else GREY
        rst_sm_rect = pygame.Rect(142, row2_y, 88, 22)
        pygame.draw.rect(self._screen, DARK, rst_sm_rect)
        pygame.draw.rect(self._screen, rst_sm_col, rst_sm_rect, 1)
        if loading and self._sm_loaded:
            dots = "." * ((pygame.time.get_ticks() // 400) % 4)
            rst_sm_label = f"Calcul{dots}"
        else:
            rst_sm_label = "restart SM"
        self._label(rst_sm_label, rst_sm_rect.x + 4, rst_sm_rect.y + 3, rst_sm_col)
        self._restart_sm_rect = rst_sm_rect

        if self._sm_loaded and self._ai_stats:
            wins  = self._ai_stats["wins"]
            total = self._ai_stats["total"]
            note  = self._ai_stats["note_moy"]
            sm_parts = [f"Victoires : {wins}/{total}", f"Note moy : {note:.0f}"]
            self._label("   |   ".join(sm_parts), rst_sm_rect.right + 10, row2_y + 3)

        # ── Row 3 (y=60) : [IA multi model]  [restart MM]  stats multi ──
        row3_y = 60
        mm_bdr = GREY if loading else (BLUE if self._mm_loaded else WHITE)
        ai_multi_rect = pygame.Rect(4, row3_y, 130, 22)
        pygame.draw.rect(self._screen, DARK, ai_multi_rect)
        pygame.draw.rect(self._screen, mm_bdr, ai_multi_rect, 1)
        self._label("IA multi model", ai_multi_rect.x + 4, ai_multi_rect.y + 3, mm_bdr)
        self._ai_multi_rect = ai_multi_rect

        can_rst_mm = self._mm_loaded and not loading
        rst_mm_col = CYAN if can_rst_mm else GREY
        rst_mm_rect = pygame.Rect(142, row3_y, 88, 22)
        pygame.draw.rect(self._screen, DARK, rst_mm_rect)
        pygame.draw.rect(self._screen, rst_mm_col, rst_mm_rect, 1)
        if loading and self._mm_loaded:
            dots = "." * ((pygame.time.get_ticks() // 400) % 4)
            rst_mm_label = f"Calcul{dots}"
        else:
            rst_mm_label = "restart MM"
        self._label(rst_mm_label, rst_mm_rect.x + 4, rst_mm_rect.y + 3, rst_mm_col)
        self._restart_mm_rect = rst_mm_rect

        shown = 0
        mm_parts: list[str] = []
        if self._ai_trails:
            n     = len(self._ai_trails)
            shown = max(0, self._anim_idx + 1)
            mm_parts.append(f"Trail {shown}/{n}")
        if self._ai_trails and not loading and shown > 0:
            trails_shown = self._ai_trails[:shown]
            wins = sum(1 for t in trails_shown if t.get("won", False))
            note = sum(t.get("score", 0) for t in trails_shown) / shown
            mm_parts.append(f"Victoires : {wins}/{shown}")
            mm_parts.append(f"Note moy : {note:.0f}")
        elif self._mm_loaded and self._ai_stats and not self._ai_trails:
            wins  = self._ai_stats["wins"]
            total = self._ai_stats["total"]
            note  = self._ai_stats["note_moy"]
            mm_parts.append(f"Victoires : {wins}/{total}")
            mm_parts.append(f"Note moy : {note:.0f}")
        if mm_parts:
            self._label("   |   ".join(mm_parts), rst_mm_rect.right + 10, row3_y + 3)

        # ── Row 4 (y=88) : barre de chargement (espace toujours réservé) ──
        if loading:
            bar_w = int(WIN_W * self._loading_progress)
            pygame.draw.rect(self._screen, DARK, (0, 88, WIN_W, 8))
            pygame.draw.rect(self._screen, CYAN, (0, 88, bar_w, 8))

    def _draw_hud_bot(self) -> None:
        pass   # HUD bas supprimé — tous les boutons sont en HUD_TOP

    # ------------------------------------------------------------------
    def _load_ai_simple(self) -> None:
        """File picker → charge le modèle simple sans jouer d'épisode.

        Le bouton [IA simple model] passe en bleu. Cliquer [restart SM] lance l'épisode.
        """
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path_str = tkinter.filedialog.askopenfilename(
            title="Choisir un checkpoint (.pt DQN ou .zip PPO)",
            filetypes=[
                ("Checkpoint IA", "*.pt *.zip"),
                ("PyTorch DQN", "*.pt"),
                ("SB3 PPO", "*.zip"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        root.destroy()
        if not path_str:
            return
        try:
            from exploit import load_model
            net = load_model(Path(path_str))
            # Passer en mode simple : effacer état multi
            self._ai_nets_cache   = []
            self._ai_run_dir      = None
            self._ai_trails       = []
            self._anim_idx        = -1
            self._ai_trail        = None
            self._ai_optimal_path = None
            self._ai_stats        = None
            self._ai_net  = net
            self._sm_loaded = True
            self._mm_loaded = False
        except Exception as exc:
            print(f"[IA simple load] Erreur : {exc}")

    def _restart_sm(self) -> None:
        """Lance ou relance un épisode avec le modèle simple chargé."""
        if self._ai_net is None or self._loading_progress is not None:
            return
        self._ai_trail        = None
        self._ai_optimal_path = None
        self._ai_stats        = None
        net     = self._ai_net
        seed    = self._current_seed
        optimal = self._state.optimal_path
        self._loading_progress = 0.5   # loader indéterminé pendant l'épisode unique

        def _run() -> None:
            try:
                from exploit import run_one_episode_info
                trail, won, score = run_one_episode_info(net, seed=seed)
                self._ai_trail        = trail
                self._ai_optimal_path = optimal
                self._ai_stats = {
                    "wins":     1 if won else 0,
                    "note_moy": float(score),
                    "total":    1,
                }
            except Exception as exc:
                print(f"[restart SM] Erreur : {exc}")
            finally:
                self._loading_progress = None

        threading.Thread(target=_run, daemon=True).start()

    def _load_ai_multi(self) -> None:
        """Directory picker → mémorise le dossier de run sans jouer d'épisode.

        Le bouton [IA multi model] passe en bleu. Cliquer [restart MM] lance les épisodes.
        """
        if self._loading_progress is not None:
            return
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        run_dir_str = tkinter.filedialog.askdirectory(
            title="Choisir un dossier de run (*_run)",
        )
        root.destroy()
        if not run_dir_str:
            return
        # Passer en mode multi : effacer état simple
        self._ai_net          = None
        self._ai_trail        = None
        self._ai_trails       = []
        self._ai_nets_cache   = []
        self._anim_idx        = -1
        self._ai_optimal_path = None
        self._ai_stats        = None
        self._ai_run_dir = Path(run_dir_str)
        self._mm_loaded  = True
        self._sm_loaded  = False

    def _restart_mm(self) -> None:
        """Lance ou relance les tracés de tous les modèles multi chargés."""
        if not self._mm_loaded or self._loading_progress is not None:
            return
        self._ai_trails       = []
        self._anim_idx        = -1
        self._ai_optimal_path = self._state.optimal_path
        self._ai_stats        = None
        if self._ai_nets_cache:
            self._rerun_from_cache(self._current_seed)
        elif self._ai_run_dir is not None:
            self._load_multi(self._ai_run_dir, self._current_seed)

    def _load_multi(self, run_dir: Path, seed: int | None) -> None:
        """Lance le chargement de tous les checkpoints du run en thread de fond."""
        if self._loading_progress is not None:
            return
        self._ai_trails = []
        self._anim_idx  = -1
        self._ai_stats  = None   # réinitialisé avant le calcul
        self._loading_progress = 0.0

        def _load() -> None:
            from exploit import scan_run_dir, load_model, run_one_episode_info
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
                    net               = load_model(cp["pt_path"])
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
                                   "alpha": alpha, "stage_idx": s,
                                   "won": won, "score": score})
                    # Mise à jour incrémentale — les trails sont visibles au fur et à mesure
                    self._ai_trails = list(trails)
                    self._loading_progress = (i + 1) / n
                    self._ai_stats = {
                        "wins":     wins,
                        "note_moy": scores_sum / (i + 1),   # échecs comptent comme 0
                        "total":    i + 1,
                    }

                # Persistant — réutilisé sans relire le disque (IA restart)
                self._ai_nets_cache = nets_cache
                self._ai_trails     = list(trails)
                self._anim_idx      = -1   # relance l'animation depuis le début
                # Pas de pygame.time.get_ticks() ici — non thread-safe sur Windows ;
                # la boucle principale gère anim_last_ms dès le premier tick.
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
        self._ai_trails        = []
        self._anim_idx         = -1
        self._ai_stats         = None   # réinitialisé avant le calcul
        self._ai_optimal_path  = self._state.optimal_path   # capturé avant le thread
        self._loading_progress = 0.0   # active le loader dès le clic

        def _rerun() -> None:
            from exploit import run_one_episode_info
            try:
                n = len(self._ai_nets_cache)
                trails: list[dict] = []
                wins = 0
                scores_sum = 0
                for i, entry in enumerate(self._ai_nets_cache):
                    trail, won, score = run_one_episode_info(entry["net"], seed=seed)
                    if won:
                        wins      += 1
                        scores_sum += score
                    trails.append({"trail": trail, "color": entry["color"],
                                   "alpha": entry["alpha"], "stage_idx": entry["stage_idx"],
                                   "won": won, "score": score})
                    # Mise à jour incrémentale — trails et stats visibles au fur et à mesure
                    self._ai_trails        = list(trails)
                    self._loading_progress = (i + 1) / n
                    self._ai_stats = {
                        "wins":     wins,
                        "note_moy": scores_sum / (i + 1),
                        "total":    i + 1,
                    }
                self._anim_idx = -1   # relance l'animation depuis le début
            except Exception as exc:
                print(f"[IA restart multi] Erreur : {exc}")
            finally:
                self._loading_progress = None   # retire le loader

        threading.Thread(target=_rerun, daemon=True).start()

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
                self._load_ai_simple()
                return
            if self._restart_sm_rect.collidepoint(event.pos):
                self._restart_sm()
                return
            if self._ai_multi_rect.collidepoint(event.pos):
                self._load_ai_multi()
                return
            if self._restart_mm_rect.collidepoint(event.pos):
                self._restart_mm()
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
            if (self._ai_trails and self._anim_idx < len(self._ai_trails) - 1
                    and self._loading_progress is None):
                now = pygame.time.get_ticks()
                if self._anim_idx == -1 or now - self._anim_last_ms >= ANIM_INTERVAL_MS:
                    self._anim_idx    += 1
                    self._anim_last_ms = now

            self._draw()
            self._clock.tick(60)
