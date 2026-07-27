import pygame
import chess
import sys
import torch
import numpy as np
import json
import os
import time
import random
from model import ChessBrain

# --- Window & Styling Configuration ---
WIDTH, HEIGHT = 512, 630  
BOARD_SIZE = 512
SQ_SIZE = BOARD_SIZE // 8
BANNER_HEIGHT = 118

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT_COLOR = (186, 202, 68)
DOT_COLOR = (100, 100, 100)
CAPTURE_RING_COLOR = (220, 50, 50)

BG_COLOR = (30, 34, 42)
BANNER_BG = (40, 44, 52)
CARD_BG = (50, 54, 64)
ACTIVE_CARD_BORDER = (255, 215, 0) 

BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER = (100, 149, 237)
TEXT_COLOR = (255, 255, 255)
GOLD_COLOR = (255, 215, 0)
RED_COLOR = (235, 75, 75)

# Initialize Pygame & Fonts
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Engine - Scazy AI Edition")

title_font = pygame.font.SysFont("Arial", 28, bold=True)
font = pygame.font.SysFont("Arial", 18, bold=True)
small_font = pygame.font.SysFont("Arial", 14)
coord_font = pygame.font.SysFont("Arial", 12, bold=True)
piece_font = pygame.font.SysFont("Arial", 36, bold=True)

# --- Leaderboard Management ---
LEADERBOARD_FILE = "leaderboard.json"

def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"Scazy": 0}
    return {"Scazy": 0}

def record_win(winner_name):
    data = load_leaderboard()
    data[winner_name] = data.get(winner_name, 0) + 1
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f, indent=4)

# --- AI & Brain Logic ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
brain = ChessBrain().to(device)

try:
    brain.load_state_dict(torch.load("chess_brain.pth", weights_only=True))
    brain.eval()
    print("AI Brain successfully loaded!")
except FileNotFoundError:
    print("Warning: 'chess_brain.pth' not found. Scazy running with base weights.")

def get_board_tensor_array(board):
    """Converts the board into a 14x8x8 multi-channel matrix for the AI's vision."""
    state = np.zeros((14, 8, 8), dtype=np.float32)
    piece_idx = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5,
                 'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}

    # 1. Map all pieces to their specific channels
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_idx[piece.symbol()]
            row, col = 7 - (square // 8), square % 8
            state[channel][row][col] = 1.0

    # 2. Turn Channel (Channel 12)
    if board.turn == chess.WHITE:
        state[12].fill(1.0)

    # 3. Castling Rights (Channel 13)
    if board.has_kingside_castling_rights(chess.WHITE): state[13][7][6] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): state[13][7][2] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): state[13][0][6] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): state[13][0][2] = 1.0

    return state

def minimax_rl(board, depth, alpha, beta, maximizing_player):
    """Alpha-beta search utilizing the RL Value Network."""
    if depth == 0 or board.is_game_over():
        state_tensor = torch.tensor(get_board_tensor_array(board), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            return brain(state_tensor).item(), None

    best_move = None
    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_val, _ = minimax_rl(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_val > max_eval:
                max_eval = eval_val; best_move = move
            alpha = max(alpha, eval_val)
            if beta <= alpha: break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_val, _ = minimax_rl(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_val < min_eval:
                min_eval = eval_val; best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha: break
        return min_eval, best_move

def get_scazy_move(board, difficulty):
    legal_moves = list(board.legal_moves)
    if not legal_moves: return None
    
    is_white = board.turn == chess.WHITE

    # --- EASY MODE ---
    # High randomness, 1-ply search, occasionally picks random top 3 moves
    if difficulty == "Easy":
        if random.random() < 0.3:  # 30% chance to just pick a random move
            return random.choice(legal_moves)
            
        move_scores = []
        for move in legal_moves:
            board.push(move)
            score, _ = minimax_rl(board, 1, -float('inf'), float('inf'), not is_white)
            board.pop()
            move_scores.append((score, move))
            
        # Sort and pick randomly from the top 3 moves
        move_scores.sort(key=lambda x: x[0], reverse=is_white)
        top_moves = [m[1] for m in move_scores[:3]]
        return random.choice(top_moves)

    # --- MEDIUM MODE ---
    # Balanced 2-ply search, very low randomness
    elif difficulty == "Medium":
        if random.random() < 0.05: # 5% mistake rate
            return random.choice(legal_moves)
        _, move = minimax_rl(board, 2, -float('inf'), float('inf'), is_white)
        return move if move else random.choice(legal_moves)

    # --- HARD MODE ---
    # Strict deterministic evaluation, 4-ply deep search
    elif difficulty == "Hard":
        _, move = minimax_rl(board, 4, -float('inf'), float('inf'), is_white)
        return move if move else random.choice(legal_moves)

# --- UI Helper Components ---
def draw_button(text, rect, is_hovered, color=BUTTON_COLOR):
    pygame.draw.rect(screen, BUTTON_HOVER if is_hovered else color, rect, border_radius=8)
    pygame.draw.rect(screen, TEXT_COLOR, rect, width=2, border_radius=8)
    text_surf = font.render(text, True, TEXT_COLOR)
    screen.blit(text_surf, text_surf.get_rect(center=rect.center))

def draw_input_box(rect, text, is_active, label):
    color = ACTIVE_CARD_BORDER if is_active else CARD_BG
    lbl_surf = small_font.render(label, True, TEXT_COLOR)
    screen.blit(lbl_surf, (rect.x, rect.y - 20))
    pygame.draw.rect(screen, CARD_BG, rect, border_radius=8)
    pygame.draw.rect(screen, color, rect, width=2, border_radius=8)
    txt_surf = font.render(text + ("|" if is_active and time.time() % 1 > 0.5 else ""), True, TEXT_COLOR)
    screen.blit(txt_surf, (rect.x + 10, rect.y + 12))

def format_time(seconds):
    if seconds is None or seconds < 0: return "--:--"
    return f"{int(seconds)//60:02d}:{int(seconds)%60:02d}"

# --- Drawing Board & Game Elements ---
def draw_top_banner(board, game_mode, timer_enabled, white_time, black_time, p1_name, p2_name):
    pygame.draw.rect(screen, BANNER_BG, (0, 0, WIDTH, BANNER_HEIGHT))
    
    w_card = pygame.Rect(10, 10, 190, 60)
    w_border = ACTIVE_CARD_BORDER if board.turn == chess.WHITE and not board.is_game_over() else CARD_BG
    pygame.draw.rect(screen, CARD_BG, w_card, border_radius=8)
    pygame.draw.rect(screen, w_border, w_card, width=2, border_radius=8)
    screen.blit(small_font.render(f"⚪ {p1_name}", True, TEXT_COLOR), (20, 16))
    screen.blit(font.render(format_time(white_time) if timer_enabled else "∞ Unlimited", True, RED_COLOR if (white_time and white_time < 30) else GOLD_COLOR), (20, 38))

    b_card = pygame.Rect(210, 10, 190, 60)
    b_border = ACTIVE_CARD_BORDER if board.turn == chess.BLACK and not board.is_game_over() else CARD_BG
    pygame.draw.rect(screen, CARD_BG, b_card, border_radius=8)
    pygame.draw.rect(screen, b_border, b_card, width=2, border_radius=8)
    screen.blit(small_font.render(f"⚫ {p2_name}", True, TEXT_COLOR), (220, 16))
    screen.blit(font.render(format_time(black_time) if timer_enabled else "∞ Unlimited", True, RED_COLOR if (black_time and black_time < 30) else GOLD_COLOR), (220, 38))

    menu_btn_rect = pygame.Rect(410, 10, 92, 60)
    draw_button("Menu", menu_btn_rect, menu_btn_rect.collidepoint(pygame.mouse.get_pos()))

    if not board.is_game_over():
        turn_str = f"{p1_name}'s Turn" if board.turn == chess.WHITE else (f"{p2_name} Thinking..." if game_mode == "HVC" else f"{p2_name}'s Turn")
        screen.blit(small_font.render(f"Status: {turn_str}", True, GOLD_COLOR), (15, 82))
    return menu_btn_rect

def draw_board(selected_square, show_coords):
    for row in range(8):
        for col in range(8):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            rect = pygame.Rect(col * SQ_SIZE, row * SQ_SIZE + BANNER_HEIGHT, SQ_SIZE, SQ_SIZE)
            pygame.draw.rect(screen, color, rect)
            if selected_square is not None and row == 7 - (selected_square // 8) and col == selected_square % 8:
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect)
            if show_coords:
                if col == 0: screen.blit(coord_font.render(str(8-row), True, DARK_SQUARE if (row+col)%2==0 else LIGHT_SQUARE), (col*SQ_SIZE+4, row*SQ_SIZE+BANNER_HEIGHT+2))
                if row == 7: screen.blit(coord_font.render(chr(ord('a')+col), True, DARK_SQUARE if (row+col)%2==0 else LIGHT_SQUARE), ((col+1)*SQ_SIZE-12, (row+1)*SQ_SIZE+BANNER_HEIGHT-16))

def draw_legal_moves(board, selected_square):
    if selected_square is None: return
    for dest_sq in [m.to_square for m in board.legal_moves if m.from_square == selected_square]:
        cx, cy = (dest_sq % 8) * SQ_SIZE + SQ_SIZE // 2, (7 - (dest_sq // 8)) * SQ_SIZE + BANNER_HEIGHT + SQ_SIZE // 2
        if board.piece_at(dest_sq): pygame.draw.circle(screen, CAPTURE_RING_COLOR, (cx, cy), SQ_SIZE // 2 - 4, width=5)
        else: pygame.draw.circle(screen, DOT_COLOR, (cx, cy), SQ_SIZE // 6)

def draw_pieces(board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            col, row = square % 8, 7 - (square // 8)
            surf = piece_font.render(piece.symbol(), True, (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0))
            screen.blit(surf, surf.get_rect(center=(col * SQ_SIZE + SQ_SIZE//2, row * SQ_SIZE + BANNER_HEIGHT + SQ_SIZE//2)))

# --- Main Application Logic ---
def main():
    state = "MENU"
    game_mode = "HVC"
    difficulty = "Medium"
    timer_preset = 300
    show_coords = True
    
    board = chess.Board()
    selected_square = None
    winner_recorded = False
    winner_display_name = None  
    
    white_time = black_time = timer_preset
    last_tick_time = time.time()
    move_history = []
    analysis_index = 0

    # Player Name Variables
    p1_name = "Player 1"
    p2_name = "Scazy"
    p1_input_temp = ""
    p2_input_temp = ""
    active_input = 1 

    # Button Rectangles
    btn_hvh = pygame.Rect(100, 130, 312, 45)
    btn_hvc = pygame.Rect(100, 190, 312, 45)
    btn_diff = pygame.Rect(100, 250, 312, 40)
    btn_timer = pygame.Rect(100, 300, 312, 40)
    btn_coords = pygame.Rect(100, 350, 312, 40)
    btn_leaderboard = pygame.Rect(100, 420, 312, 45)
    btn_help = pygame.Rect(100, 480, 312, 45)
    btn_exit = pygame.Rect(100, 540, 312, 45)
    btn_back = pygame.Rect(180, 540, 152, 45)

    running = True
    clock = pygame.time.Clock()

    while running:
        dt = clock.tick(30) / 1000.0
        mouse_pos = pygame.mouse.get_pos()

        # 1. MAIN MENU
        if state == "MENU":
            screen.fill(BG_COLOR)
            screen.blit(title_font.render("CHESS AI: SCAZY", True, GOLD_COLOR), title_font.render("CHESS AI: SCAZY", True, GOLD_COLOR).get_rect(center=(WIDTH//2, 50)))
            
            draw_button("Play: Human vs Human", btn_hvh, btn_hvh.collidepoint(mouse_pos))
            draw_button("Play: Human vs Computer", btn_hvc, btn_hvc.collidepoint(mouse_pos))
            draw_button(f"AI Difficulty: {difficulty}", btn_diff, btn_diff.collidepoint(mouse_pos), (50, 90, 140))
            draw_button(f"Timer Control: {'Off' if timer_preset is None else str(timer_preset//60)+' Min'}", btn_timer, btn_timer.collidepoint(mouse_pos), (50, 90, 140))
            draw_button(f"Coordinates: {'On' if show_coords else 'Off'}", btn_coords, btn_coords.collidepoint(mouse_pos), (50, 90, 140))
            draw_button("Leaderboard", btn_leaderboard, btn_leaderboard.collidepoint(mouse_pos))
            draw_button("How to Play / Help", btn_help, btn_help.collidepoint(mouse_pos))
            draw_button("Exit Game", btn_exit, btn_exit.collidepoint(mouse_pos))

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_hvh.collidepoint(mouse_pos):
                        game_mode = "HVH"; p1_input_temp = ""; p2_input_temp = ""; active_input = 1; state = "NAME_ENTRY"
                    elif btn_hvc.collidepoint(mouse_pos):
                        game_mode = "HVC"; p1_input_temp = ""; p2_input_temp = "Scazy"; active_input = 1; state = "NAME_ENTRY"
                    elif btn_diff.collidepoint(mouse_pos): difficulty = ["Easy", "Medium", "Hard"][ (["Easy", "Medium", "Hard"].index(difficulty) + 1) % 3 ]
                    elif btn_timer.collidepoint(mouse_pos): timer_preset = [None, 180, 300, 600][ ([None, 180, 300, 600].index(timer_preset) + 1) % 4 ]
                    elif btn_coords.collidepoint(mouse_pos): show_coords = not show_coords
                    elif btn_leaderboard.collidepoint(mouse_pos): state = "LEADERBOARD"
                    elif btn_help.collidepoint(mouse_pos): state = "HELP"
                    elif btn_exit.collidepoint(mouse_pos): running = False

        # 2. NAME ENTRY SCREEN
        elif state == "NAME_ENTRY":
            screen.fill(BG_COLOR)
            screen.blit(title_font.render("MATCH SETUP", True, GOLD_COLOR), title_font.render("MATCH SETUP", True, GOLD_COLOR).get_rect(center=(WIDTH//2, 80)))
            
            box1 = pygame.Rect(100, 180, 312, 45)
            box2 = pygame.Rect(100, 280, 312, 45)
            btn_start = pygame.Rect(100, 400, 312, 50)
            
            draw_input_box(box1, p1_input_temp, active_input == 1, "White Player Name:")
            if game_mode == "HVH":
                draw_input_box(box2, p2_input_temp, active_input == 2, "Black Player Name:")
            else:
                screen.blit(small_font.render("Opponent:", True, TEXT_COLOR), (100, 260))
                draw_button("Scazy (AI Locked)", box2, False, CARD_BG)

            draw_button("Start Match", btn_start, btn_start.collidepoint(mouse_pos), (40, 160, 80))
            draw_button("Cancel", btn_back, btn_back.collidepoint(mouse_pos), (180, 60, 60))

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if box1.collidepoint(mouse_pos): active_input = 1
                    elif box2.collidepoint(mouse_pos) and game_mode == "HVH": active_input = 2
                    elif btn_back.collidepoint(mouse_pos): state = "MENU"
                    elif btn_start.collidepoint(mouse_pos):
                        p1_name = p1_input_temp.strip() or "Player 1"
                        p2_name = p2_input_temp.strip() or ("Scazy" if game_mode == "HVC" else "Player 2")
                        board.reset(); selected_square = None; winner_recorded = False; winner_display_name = None; move_history = []
                        white_time = black_time = timer_preset; last_tick_time = time.time(); state = "GAME"
                elif event.type == pygame.KEYDOWN:
                    if active_input == 1:
                        if event.key == pygame.K_BACKSPACE: p1_input_temp = p1_input_temp[:-1]
                        elif len(p1_input_temp) < 14 and event.unicode.isprintable(): p1_input_temp += event.unicode
                    elif active_input == 2 and game_mode == "HVH":
                        if event.key == pygame.K_BACKSPACE: p2_input_temp = p2_input_temp[:-1]
                        elif len(p2_input_temp) < 14 and event.unicode.isprintable(): p2_input_temp += event.unicode
                    if event.key == pygame.K_RETURN: 
                        if active_input == 1 and game_mode == "HVH": active_input = 2

        # 3. ACTIVE GAME SCREEN
        elif state == "GAME":
            curr_time = time.time()
            if timer_preset and not board.is_game_over():
                if board.turn == chess.WHITE and white_time > 0: white_time -= (curr_time - last_tick_time)
                elif board.turn == chess.BLACK and black_time > 0: black_time -= (curr_time - last_tick_time)
            last_tick_time = curr_time
            
            timeout_winner = p2_name if (timer_preset and white_time <= 0) else (p1_name if (timer_preset and black_time <= 0) else None)

            draw_board(selected_square, show_coords)
            draw_legal_moves(board, selected_square)
            draw_pieces(board)
            menu_btn = draw_top_banner(board, game_mode, timer_preset is not None, white_time, black_time, p1_name, p2_name)

            if board.is_game_over() or timeout_winner:
                if not winner_recorded:
                    if timeout_winner: 
                        record_win(timeout_winner)
                        winner_display_name = timeout_winner
                    else:
                        res = board.result()
                        if res == '1-0': 
                            record_win(p1_name)
                            winner_display_name = p1_name
                        elif res == '0-1': 
                            record_win(p2_name)
                            winner_display_name = p2_name
                        else:
                            winner_display_name = "Draw"
                    winner_recorded = True

                # Draw End-Game Overlay
                overlay = pygame.Surface((WIDTH, 120))
                overlay.set_alpha(230)
                overlay.fill(BG_COLOR)
                screen.blit(overlay, (0, BANNER_HEIGHT + 180))
                
                status_txt = "TIME EXPIRED!" if timeout_winner else "GAME OVER!"
                screen.blit(font.render(status_txt, True, RED_COLOR), font.render(status_txt, True, RED_COLOR).get_rect(center=(WIDTH//2, BANNER_HEIGHT + 200)))
                
                # Render the explicit winner text
                if winner_display_name == "Draw":
                    win_text = "Match ended in a Draw!"
                else:
                    win_text = f"Winner: {winner_display_name}!"
                
                screen.blit(title_font.render(win_text, True, GOLD_COLOR), title_font.render(win_text, True, GOLD_COLOR).get_rect(center=(WIDTH//2, BANNER_HEIGHT + 235)))

                btn_rematch = pygame.Rect(100, BANNER_HEIGHT + 265, 140, 30)
                btn_analyze = pygame.Rect(270, BANNER_HEIGHT + 265, 140, 30)
                draw_button("Rematch", btn_rematch, btn_rematch.collidepoint(mouse_pos), (40, 160, 80))
                draw_button("Analyze Game", btn_analyze, btn_analyze.collidepoint(mouse_pos), (140, 80, 180))

            if game_mode == "HVC" and board.turn == chess.BLACK and not board.is_game_over() and not timeout_winner:
                pygame.display.flip(); pygame.time.wait(150)
                ai_move = get_scazy_move(board, difficulty)
                if ai_move: board.push(ai_move); move_history.append(ai_move)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if menu_btn.collidepoint(mouse_pos): state = "MENU"
                    elif board.is_game_over() or timeout_winner:
                        if btn_rematch.collidepoint(mouse_pos):
                            board.reset(); selected_square = None; winner_recorded = False; winner_display_name = None; move_history = []
                            white_time = black_time = timer_preset; last_tick_time = time.time()
                        elif btn_analyze.collidepoint(mouse_pos):
                            analysis_index = len(move_history); state = "ANALYZE"
                    elif not board.is_game_over() and not timeout_winner and mouse_pos[1] >= BANNER_HEIGHT:
                        if (board.turn == chess.WHITE) if game_mode == "HVC" else True:
                            clicked_sq = chess.square(mouse_pos[0] // SQ_SIZE, 7 - ((mouse_pos[1] - BANNER_HEIGHT) // SQ_SIZE))
                            if selected_square is None:
                                p = board.piece_at(clicked_sq)
                                if p and p.color == board.turn: selected_square = clicked_sq
                            else:
                                move = chess.Move(selected_square, clicked_sq)
                                if board.piece_at(selected_square) and board.piece_at(selected_square).piece_type == chess.PAWN and chess.square_rank(clicked_sq) in (0, 7):
                                    move = chess.Move(selected_square, clicked_sq, promotion=chess.QUEEN)
                                if move in board.legal_moves: board.push(move); move_history.append(move)
                                selected_square = None

        # 4. GAME ANALYSIS SCREEN
        elif state == "ANALYZE":
            screen.fill(BG_COLOR)
            ana_board = chess.Board()
            for i in range(min(analysis_index, len(move_history))): ana_board.push(move_history[i])
            for row in range(8):
                for col in range(8):
                    rect = pygame.Rect(col * SQ_SIZE, row * SQ_SIZE + 60, SQ_SIZE, SQ_SIZE)
                    pygame.draw.rect(screen, LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE, rect)
            for square in chess.SQUARES:
                piece = ana_board.piece_at(square)
                if piece:
                    surf = piece_font.render(piece.symbol(), True, (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0))
                    screen.blit(surf, surf.get_rect(center=((square % 8) * SQ_SIZE + SQ_SIZE//2, (7 - (square // 8)) * SQ_SIZE + 60 + SQ_SIZE//2)))
            
            screen.blit(font.render(f"ANALYSIS MODE - Move {analysis_index} / {len(move_history)}", True, GOLD_COLOR), (20, 18))
            btn_prev = pygame.Rect(30, 580, 100, 38); btn_next = pygame.Rect(140, 580, 100, 38)
            btn_first = pygame.Rect(250, 580, 80, 38); btn_exit_ana = pygame.Rect(380, 580, 100, 38)

            draw_button("< Prev", btn_prev, btn_prev.collidepoint(mouse_pos))
            draw_button("Next >", btn_next, btn_next.collidepoint(mouse_pos))
            draw_button("First", btn_first, btn_first.collidepoint(mouse_pos))
            draw_button("Menu", btn_exit_ana, btn_exit_ana.collidepoint(mouse_pos), (180, 60, 60))

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and analysis_index > 0: analysis_index -= 1
                    elif event.key == pygame.K_RIGHT and analysis_index < len(move_history): analysis_index += 1
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_prev.collidepoint(mouse_pos) and analysis_index > 0: analysis_index -= 1
                    elif btn_next.collidepoint(mouse_pos) and analysis_index < len(move_history): analysis_index += 1
                    elif btn_first.collidepoint(mouse_pos): analysis_index = 0
                    elif btn_exit_ana.collidepoint(mouse_pos): state = "MENU"

        # 5. LEADERBOARD SCREEN
        elif state == "LEADERBOARD":
            screen.fill(BG_COLOR)
            screen.blit(title_font.render("VICTORY LEADERBOARD", True, GOLD_COLOR), title_font.render("VICTORY LEADERBOARD", True, GOLD_COLOR).get_rect(center=(WIDTH//2, 60)))

            data = load_leaderboard()
            y_offset = 150
            screen.blit(font.render("Player / Agent", True, TEXT_COLOR), (100, 110))
            screen.blit(font.render("Total Wins", True, TEXT_COLOR), (320, 110))
            pygame.draw.line(screen, TEXT_COLOR, (100, 135), (412, 135), 2)

            for player, wins in sorted(data.items(), key=lambda item: item[1], reverse=True):
                screen.blit(font.render(str(player), True, GOLD_COLOR if player == "Scazy" else TEXT_COLOR), (100, y_offset))
                screen.blit(font.render(str(wins), True, TEXT_COLOR), (350, y_offset))
                y_offset += 40

            draw_button("Back to Menu", btn_back, btn_back.collidepoint(mouse_pos))

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_back.collidepoint(mouse_pos): state = "MENU"

        # 6. HELP SCREEN
        elif state == "HELP":
            screen.fill(BG_COLOR)
            screen.blit(title_font.render("HOW TO PLAY", True, GOLD_COLOR), title_font.render("HOW TO PLAY", True, GOLD_COLOR).get_rect(center=(WIDTH//2, 50)))
            instructions = [
                "1. Game Modes & Names:",
                "   - Enter custom player names before the match starts.",
                "   - AI is permanently locked as 'Scazy'.",
                "",
                "2. Professional Time Controls:",
                "   - Toggleable timer: Off, 3m Blitz, 5m Blitz, 10m Rapid.",
                "",
                "3. Visuals & Move Highlights:",
                "   - Dots mark valid empty destination squares.",
                "   - Red rings mark legal capture moves.",
                "",
                "4. Post-Game & Analysis:",
                "   - Custom names track permanently on the Leaderboard."
            ]
            y = 100
            for line in instructions:
                screen.blit(small_font.render(line, True, TEXT_COLOR), (40, y)); y += 24
            draw_button("Back to Menu", btn_back, btn_back.collidepoint(mouse_pos))

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_back.collidepoint(mouse_pos): state = "MENU"

        pygame.display.flip()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()