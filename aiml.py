import csv
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import pygame

# ------------------ CONSTANTS ------------------

ROWS = 6
COLS = 7

EMPTY = 0
PLAYER = 1
AI = 2
PLAYER_TWO = 2

SQUARESIZE = 100
WIDTH = COLS * SQUARESIZE
HEIGHT = (ROWS + 1) * SQUARESIZE

BG_COLOR = (18, 18, 18)
BOARD_COLOR = (40, 44, 52)
HOLE_COLOR = (25, 25, 25)
PLAYER_COLOR = (255, 99, 132)
AI_COLOR = (255, 206, 86)
PLAYER_TWO_COLOR = (86, 178, 255)
TEXT_COLOR = (230, 230, 230)
MUTED_TEXT = (165, 170, 180)
WIN_COLOR = (96, 255, 170)
BUTTON_HOVER = (255, 206, 86)

RADIUS = int(SQUARESIZE / 2 - 6)
PLAYER_TIME_LIMIT = 20
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "connect4_match_history.csv")

GAME_MODES = {
    "pva": "Player vs AI",
    "pvp": "Player vs Player",
}

DIFFICULTIES = {
    "Easy": 3,
    "Medium": 4,
    "Hard": 5,
}

player_score = 0
ai_score = 0
player_two_score = 0


# ------------------ BOARD ------------------

def create_board():
    return np.zeros((ROWS, COLS), dtype=int)


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return 0 <= col < COLS and board[0][col] == EMPTY


def get_next_open_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == EMPTY:
            return r
    return None


def get_valid_locations(board):
    return [c for c in range(COLS) if is_valid_location(board, c)]


def get_piece_color(piece, mode="pva"):
    if piece == PLAYER:
        return PLAYER_COLOR
    if mode == "pvp":
        return PLAYER_TWO_COLOR
    return AI_COLOR


# ------------------ WIN CHECK ------------------

def get_winning_cells(board, piece):
    lines = []

    for r in range(ROWS):
        for c in range(COLS - 3):
            cells = [(r, c + i) for i in range(4)]
            lines.append(cells)

    for c in range(COLS):
        for r in range(ROWS - 3):
            cells = [(r + i, c) for i in range(4)]
            lines.append(cells)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            cells = [(r + i, c + i) for i in range(4)]
            lines.append(cells)

    for r in range(3, ROWS):
        for c in range(COLS - 3):
            cells = [(r - i, c + i) for i in range(4)]
            lines.append(cells)

    for cells in lines:
        if all(board[r][c] == piece for r, c in cells):
            return cells

    return []


def winning_move(board, piece):
    return len(get_winning_cells(board, piece)) > 0


def is_draw(board):
    return len(get_valid_locations(board)) == 0


# ------------------ AI ------------------

def is_terminal_node(board):
    return winning_move(board, PLAYER) or winning_move(board, AI) or is_draw(board)


def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER if piece == AI else AI

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 12
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 4

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 18
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score -= 3

    return score


def score_position(board, piece):
    score = 0

    center_array = [int(i) for i in list(board[:, COLS // 2])]
    score += center_array.count(piece) * 6

    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLS - 3):
            score += evaluate_window(row_array[c:c + 4], piece)

    for c in range(COLS):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROWS - 3):
            score += evaluate_window(col_array[r:r + 4], piece)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = [board[r - i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score


def minimax(board, depth, alpha, beta, maximizing, ai_piece=AI):
    valid_locations = get_valid_locations(board)
    human_piece = PLAYER if ai_piece == AI else AI

    if is_terminal_node(board):
        if winning_move(board, ai_piece):
            return None, 1_000_000
        if winning_move(board, human_piece):
            return None, -1_000_000
        return None, 0

    if depth == 0:
        return None, score_position(board, ai_piece)

    ordered_cols = sorted(valid_locations, key=lambda col: abs(COLS // 2 - col))

    if maximizing:
        value = -math.inf
        col_choice = random.choice(valid_locations)

        for col in ordered_cols:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, ai_piece)
            new_score = minimax(temp_board, depth - 1, alpha, beta, False, ai_piece)[1]

            if new_score > value:
                value = new_score
                col_choice = col

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return col_choice, value

    value = math.inf
    col_choice = random.choice(valid_locations)

    for col in ordered_cols:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, human_piece)
        new_score = minimax(temp_board, depth - 1, alpha, beta, True, ai_piece)[1]

        if new_score < value:
            value = new_score
            col_choice = col

        beta = min(beta, value)
        if alpha >= beta:
            break

    return col_choice, value


def choose_ai_move(board, depth, ai_piece=AI):
    valid = get_valid_locations(board)
    if not valid:
        return None
    col, _ = minimax(board, depth, -math.inf, math.inf, True, ai_piece)
    return col if col is not None else random.choice(valid)


# ------------------ DRAWING ------------------

def draw_text_center(screen, font, text, y, color=TEXT_COLOR):
    label = font.render(text, True, color)
    screen.blit(label, label.get_rect(center=(WIDTH // 2, y)))


def draw_button(screen, font, text, rect, mouse_pos):
    hovered = rect.collidepoint(mouse_pos)
    color = BUTTON_HOVER if hovered else BOARD_COLOR
    pygame.draw.rect(screen, color, rect, border_radius=8)
    pygame.draw.rect(screen, TEXT_COLOR, rect, 2, border_radius=8)
    label_color = (20, 20, 20) if hovered else TEXT_COLOR
    label = font.render(text, True, label_color)
    screen.blit(label, label.get_rect(center=rect.center))
    return hovered


def draw_board(board, screen, mode="pva", winning_cells=None):
    screen.fill(BG_COLOR)

    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(
                screen,
                BOARD_COLOR,
                (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE),
            )
            pygame.draw.circle(
                screen,
                HOLE_COLOR,
                (
                    c * SQUARESIZE + SQUARESIZE // 2,
                    r * SQUARESIZE + SQUARESIZE + SQUARESIZE // 2,
                ),
                RADIUS,
            )

    for c in range(COLS):
        for r in range(ROWS):
            piece = board[r][c]
            if piece != EMPTY:
                pygame.draw.circle(
                    screen,
                    get_piece_color(piece, mode),
                    (
                        c * SQUARESIZE + SQUARESIZE // 2,
                        (r + 1) * SQUARESIZE + SQUARESIZE // 2,
                    ),
                    RADIUS,
                )

    if winning_cells:
        for r, c in winning_cells:
            pygame.draw.circle(
                screen,
                WIN_COLOR,
                (
                    c * SQUARESIZE + SQUARESIZE // 2,
                    (r + 1) * SQUARESIZE + SQUARESIZE // 2,
                ),
                RADIUS // 2,
                5,
            )

    pygame.display.update()


def draw_score(screen, font, mode, turn, start_ticks=None):
    pygame.draw.rect(screen, BG_COLOR, (0, 0, WIDTH, SQUARESIZE))

    if mode == "pvp":
        score = f"P1: {player_score}   P2: {player_two_score}"
    else:
        score = f"You: {player_score}   AI: {ai_score}"

    label = font.render(score, True, TEXT_COLOR)
    screen.blit(label, (18, 12))

    if start_ticks is not None and turn in (PLAYER, PLAYER_TWO):
        elapsed = (pygame.time.get_ticks() - start_ticks) // 1000
        remaining = max(0, PLAYER_TIME_LIMIT - elapsed)
        timer = font.render(f"Time: {remaining}s", True, TEXT_COLOR)
        screen.blit(timer, (WIDTH - timer.get_width() - 18, 12))

    hint = font.render("U: Undo   R: Restart   M: Menu", True, MUTED_TEXT)
    screen.blit(hint, hint.get_rect(center=(WIDTH // 2, 68)))
    pygame.display.update()


def draw_hover_piece(screen, board, mode, turn, mouse_x):
    draw_board(board, screen, mode)
    color = get_piece_color(turn, mode)
    pygame.draw.circle(screen, color, (mouse_x, SQUARESIZE // 2), RADIUS)


# ------------------ SOUND ------------------

def create_tone(frequency, duration_ms, volume=0.25):
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, False)
    wave = np.sin(frequency * t * 2 * math.pi) * volume
    audio = np.array(wave * 32767, dtype=np.int16)
    stereo = np.column_stack((audio, audio))
    return pygame.sndarray.make_sound(stereo.copy())


def load_sounds():
    try:
        pygame.mixer.init()
        return {
            "drop": create_tone(420, 90),
            "win": create_tone(720, 180),
            "click": create_tone(260, 70),
        }
    except pygame.error:
        return {}


def play_sound(sounds, name):
    sound = sounds.get(name)
    if sound:
        sound.play()


# ------------------ ANIMATION ------------------

def animate_drop(screen, board, col, row, piece, mode):
    color = get_piece_color(piece, mode)
    x = col * SQUARESIZE + SQUARESIZE // 2
    target_y = (row + 1) * SQUARESIZE + SQUARESIZE // 2

    for y in range(SQUARESIZE // 2, target_y + 1, 24):
        draw_board(board, screen, mode)
        pygame.draw.circle(screen, color, (x, y), RADIUS)
        pygame.display.update()
        pygame.time.delay(8)


# ------------------ MENU SCREENS ------------------

def show_menu(screen, title_font, font, sounds):
    selected_mode = "pva"
    selected_depth = DIFFICULTIES["Medium"]

    while True:
        screen.fill(BG_COLOR)
        mouse = pygame.mouse.get_pos()

        draw_text_center(screen, title_font, "CONNECT 4 AI", 82)
        draw_text_center(screen, font, "Choose mode and difficulty", 134, MUTED_TEXT)

        mode_buttons = [
            ("Player vs AI", "pva", pygame.Rect(140, 190, 190, 56)),
            ("Player vs Player", "pvp", pygame.Rect(370, 190, 190, 56)),
        ]

        difficulty_buttons = [
            ("Easy", 3, pygame.Rect(105, 315, 145, 56)),
            ("Medium", 4, pygame.Rect(278, 315, 145, 56)),
            ("Hard", 5, pygame.Rect(451, 315, 145, 56)),
        ]

        draw_text_center(screen, font, "Mode", 170, MUTED_TEXT)
        for text, mode, rect in mode_buttons:
            active = selected_mode == mode
            pygame.draw.rect(screen, WIN_COLOR if active else BOARD_COLOR, rect, border_radius=8)
            label_color = (20, 20, 20) if active else TEXT_COLOR
            if rect.collidepoint(mouse):
                pygame.draw.rect(screen, BUTTON_HOVER, rect, 3, border_radius=8)
            label = font.render(text, True, label_color)
            screen.blit(label, label.get_rect(center=rect.center))

        draw_text_center(screen, font, "Difficulty", 295, MUTED_TEXT)
        for text, depth, rect in difficulty_buttons:
            active = selected_depth == depth
            pygame.draw.rect(screen, WIN_COLOR if active else BOARD_COLOR, rect, border_radius=8)
            label_color = (20, 20, 20) if active else TEXT_COLOR
            if rect.collidepoint(mouse):
                pygame.draw.rect(screen, BUTTON_HOVER, rect, 3, border_radius=8)
            label = font.render(text, True, label_color)
            screen.blit(label, label.get_rect(center=rect.center))

        start_rect = pygame.Rect(WIDTH // 2 - 120, 485, 240, 64)
        draw_button(screen, font, "Start Game", start_rect, mouse)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                play_sound(sounds, "click")
                for _, mode, rect in mode_buttons:
                    if rect.collidepoint(event.pos):
                        selected_mode = mode
                for _, depth, rect in difficulty_buttons:
                    if rect.collidepoint(event.pos):
                        selected_depth = depth
                if start_rect.collidepoint(event.pos):
                    return selected_mode, selected_depth


def show_winner(screen, title_font, font, message, color, mode, winning_cells=None, board=None):
    if board is not None:
        draw_board(board, screen, mode, winning_cells)

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 175))
    screen.blit(overlay, (0, 0))

    draw_text_center(screen, title_font, message, HEIGHT // 3, color)
    draw_text_center(screen, font, "Click to continue", HEIGHT // 2, TEXT_COLOR)
    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                return


def show_end_options(screen, font, sounds):
    while True:
        mouse = pygame.mouse.get_pos()
        buttons = [
            ("Play Again", "again", pygame.Rect(WIDTH // 2 - 125, 385, 250, 56)),
            ("Main Menu", "menu", pygame.Rect(WIDTH // 2 - 125, 455, 250, 56)),
            ("Quit", "quit", pygame.Rect(WIDTH // 2 - 125, 525, 250, 56)),
        ]

        for text, _, rect in buttons:
            draw_button(screen, font, text, rect, mouse)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                play_sound(sounds, "click")
                for _, action, rect in buttons:
                    if rect.collidepoint(event.pos):
                        return action


# ------------------ MATCH HISTORY ------------------

def save_match_history(mode, difficulty, winner, moves):
    file_exists = os.path.exists(HISTORY_FILE)

    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "mode", "difficulty", "winner", "moves"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            GAME_MODES[mode],
            difficulty,
            winner,
            moves,
        ])


# ------------------ GAME FLOW ------------------

def make_move(screen, board, col, piece, mode, sounds, move_history):
    if not is_valid_location(board, col):
        return False

    row = get_next_open_row(board, col)
    animate_drop(screen, board, col, row, piece, mode)
    drop_piece(board, row, col, piece)
    move_history.append((row, col, piece))
    play_sound(sounds, "drop")
    draw_board(board, screen, mode)
    return True


def undo_last_turn(board, mode, move_history):
    if not move_history:
        return PLAYER

    undo_count = 2 if mode == "pva" and len(move_history) >= 2 else 1
    last_piece = PLAYER

    for _ in range(undo_count):
        if not move_history:
            break
        row, col, piece = move_history.pop()
        board[row][col] = EMPTY
        last_piece = piece

    return last_piece


def handle_game_end(screen, title_font, font, board, mode, difficulty, winner, piece, moves, sounds):
    global player_score, ai_score, player_two_score

    winning_cells = get_winning_cells(board, piece) if piece else []

    if winner == "Player 1":
        player_score += 1
        color = PLAYER_COLOR
    elif winner == "Player 2":
        player_two_score += 1
        color = PLAYER_TWO_COLOR
    elif winner == "AI":
        ai_score += 1
        color = AI_COLOR
    else:
        color = TEXT_COLOR

    play_sound(sounds, "win")
    save_match_history(mode, difficulty, winner, moves)
    show_winner(screen, title_font, font, f"{winner} wins!" if winner != "Draw" else "Game draw", color, mode, winning_cells, board)
    return show_end_options(screen, font, sounds)


def play_round(screen, title_font, font, mode, depth, sounds):
    board = create_board()
    move_history = []
    turn = PLAYER
    start_ticks = pygame.time.get_ticks()
    difficulty_name = next(name for name, value in DIFFICULTIES.items() if value == depth)

    draw_board(board, screen, mode)

    while True:
        draw_score(screen, font, mode, turn, start_ticks)

        if is_draw(board):
            return handle_game_end(screen, title_font, font, board, mode, difficulty_name, "Draw", None, len(move_history), sounds)

        if mode == "pva" and turn == AI:
            draw_text_center(screen, font, "AI thinking...", 42, TEXT_COLOR)
            pygame.display.update()
            pygame.time.delay(120)
            col = choose_ai_move(board, depth, AI)
            if col is not None and make_move(screen, board, col, AI, mode, sounds, move_history):
                if winning_move(board, AI):
                    return handle_game_end(screen, title_font, font, board, mode, difficulty_name, "AI", AI, len(move_history), sounds)
                turn = PLAYER
                start_ticks = pygame.time.get_ticks()

        if turn in (PLAYER, PLAYER_TWO):
            elapsed = (pygame.time.get_ticks() - start_ticks) // 1000
            if elapsed >= PLAYER_TIME_LIMIT:
                valid = get_valid_locations(board)
                if valid:
                    make_move(screen, board, random.choice(valid), turn, mode, sounds, move_history)
                    if winning_move(board, turn):
                        winner = "Player 1" if turn == PLAYER else "Player 2"
                        return handle_game_end(screen, title_font, font, board, mode, difficulty_name, winner, turn, len(move_history), sounds)
                    turn = AI if mode == "pva" else (PLAYER_TWO if turn == PLAYER else PLAYER)
                    start_ticks = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    turn = undo_last_turn(board, mode, move_history)
                    if mode == "pva":
                        turn = PLAYER
                    draw_board(board, screen, mode)
                    start_ticks = pygame.time.get_ticks()
                elif event.key == pygame.K_r:
                    return "again"
                elif event.key == pygame.K_m:
                    return "menu"

            if event.type == pygame.MOUSEMOTION:
                if mode == "pva" and turn == AI:
                    continue
                draw_hover_piece(screen, board, mode, turn, event.pos[0])
                draw_score(screen, font, mode, turn, start_ticks)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if mode == "pva" and turn == AI:
                    continue

                col = event.pos[0] // SQUARESIZE
                if not make_move(screen, board, col, turn, mode, sounds, move_history):
                    continue

                if winning_move(board, turn):
                    if turn == PLAYER:
                        winner = "Player 1"
                    elif mode == "pvp":
                        winner = "Player 2"
                    else:
                        winner = "AI"
                    return handle_game_end(screen, title_font, font, board, mode, difficulty_name, winner, turn, len(move_history), sounds)

                if mode == "pvp":
                    turn = PLAYER_TWO if turn == PLAYER else PLAYER
                else:
                    turn = AI

                start_ticks = pygame.time.get_ticks()


def play_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4 AI")

    title_font = pygame.font.SysFont("arial", 42, bold=True)
    font = pygame.font.SysFont("arial", 24, bold=True)
    sounds = load_sounds()

    mode, depth = show_menu(screen, title_font, font, sounds)

    while True:
        action = play_round(screen, title_font, font, mode, depth, sounds)

        if action == "again":
            continue
        if action == "menu":
            mode, depth = show_menu(screen, title_font, font, sounds)
            continue
        if action == "quit":
            pygame.quit()
            sys.exit()


# ------------------ RUN ------------------

if __name__ == "__main__":
    play_game()
