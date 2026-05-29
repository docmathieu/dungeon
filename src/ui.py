import random
import sys

import pygame

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

OPTIMAL_TRAIL_OFFSET = 2   # px shift so optimal trail doesn't overlap player trail

TILE_PX   = 40   # rendered cell size (pixels)
SEP_PX    = 1    # separator between cells
CELL_STEP = TILE_PX + SEP_PX   # 41 px

GRID_W = Grid.WIDTH  * CELL_STEP - SEP_PX   # 409 px
GRID_H = Grid.HEIGHT * CELL_STEP - SEP_PX   # 409 px

HUD_TOP_H = 80    # height of the top HUD strip
HUD_BOT_H = 60    # height of the bottom HUD strip

INPUT_ACTIVE_COL = ( 60,  60,  80)   # fond du champ de saisie actif
INPUT_BORDER_COL = (100, 100, 200)   # bordure du champ actif
SEED_MAX_DIGITS  = 6                  # longueur maximale de la saisie

WIN_W = GRID_W
WIN_H = HUD_TOP_H + GRID_H + HUD_BOT_H


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
        self._seed_input_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)  # calculé dans _draw_hud_bot
        self._current_seed: int | None = None   # seed du terrain affiché

        self._reset()

    # ------------------------------------------------------------------
    def _reset(self) -> None:
        seed = random.randrange(10 ** SEED_MAX_DIGITS)
        self._state = GameState.create_solvable(seed=seed)
        self._grid = self._state.grid
        self._current_seed = seed

    def _reset_with_seed(self, seed: int) -> None:
        """Réinitialise le jeu avec un seed précis.  Si le terrain n'est pas
        solvable, affiche un message et ne change pas l'état."""
        state = GameState.create_solvable(seed=seed)
        self._state = state
        self._grid = state.grid
        self._current_seed = seed

    # ------------------------------------------------------------------
    def _draw(self) -> None:
        self._screen.fill(BLACK)
        self._draw_grid()
        self._draw_trail()
        self._draw_optimal_trail()
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
        # head
        pygame.draw.circle(self._screen, YELLOW, (cx, cy - 8), 7)
        # body
        pygame.draw.line(self._screen, YELLOW, (cx, cy - 1), (cx, cy + 10), 2)
        # arms
        pygame.draw.line(self._screen, YELLOW, (cx - 7, cy + 4), (cx + 7, cy + 4), 2)
        # legs
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

        # second row: controls legend
        self._label("← ↑ → ↓  pour se déplacer  |  R pour restart", 4, 32, GREY)

        # third row: current seed
        self._label(f"Seed : {self._current_seed}", 4, 56, GREY)

    def _draw_hud_bot(self) -> None:
        bot_y = HUD_TOP_H + GRID_H

        # restart button
        restart_rect = pygame.Rect(4, bot_y + 16, 70, 28)
        pygame.draw.rect(self._screen, DARK, restart_rect)
        pygame.draw.rect(self._screen, WHITE, restart_rect, 1)
        self._label("restart", restart_rect.x + 6, restart_rect.y + 6)
        self._restart_rect = restart_rect

        # champ de saisie de seed
        self._label("Seed :", 84, bot_y + 22, GREY)
        input_rect = pygame.Rect(124, bot_y + 16, 80, 28)
        bg_col   = INPUT_ACTIVE_COL if self._seed_input_active else DARK
        bdr_col  = INPUT_BORDER_COL if self._seed_input_active else GREY
        pygame.draw.rect(self._screen, bg_col, input_rect)
        pygame.draw.rect(self._screen, bdr_col, input_rect, 1)
        display_text = self._seed_input_text if self._seed_input_active else (
            str(self._current_seed) if self._current_seed is not None else ""
        )
        self._label(display_text, input_rect.x + 4, input_rect.y + 6)
        # curseur clignotant
        if self._seed_input_active and (pygame.time.get_ticks() // 500) % 2 == 0:
            cursor_x = input_rect.x + 4 + self._font.size(display_text)[0]
            pygame.draw.line(
                self._screen, WHITE,
                (cursor_x, input_rect.y + 4),
                (cursor_x, input_rect.y + 22), 1,
            )
        self._seed_input_rect = input_rect
        self._label("↵ valider", input_rect.right + 6, bot_y + 22, GREY)

    # ------------------------------------------------------------------
    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Clic souris
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._restart_rect.collidepoint(event.pos):
                self._reset()
                return
            # Activation / désactivation du champ de saisie
            if self._seed_input_rect.collidepoint(event.pos):
                self._seed_input_active = True
                self._seed_input_text = ""
            else:
                self._seed_input_active = False

        if event.type == pygame.KEYDOWN:
            # --- champ de saisie actif ---
            if self._seed_input_active:
                if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    if self._seed_input_text:
                        self._reset_with_seed(int(self._seed_input_text))
                    self._seed_input_active = False
                    self._seed_input_text = ""
                elif event.key == pygame.K_ESCAPE:
                    self._seed_input_active = False
                    self._seed_input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    self._seed_input_text = self._seed_input_text[:-1]
                elif event.unicode.isdigit() and len(self._seed_input_text) < SEED_MAX_DIGITS:
                    self._seed_input_text += event.unicode
                return   # bloquer les flèches / R quand le champ est actif

            # --- jeu normal ---
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

            self._draw()
            self._clock.tick(60)
