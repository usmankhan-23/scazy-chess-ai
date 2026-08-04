import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import time
from model import ChessBrain

# --- Hyperparameters ---
EPISODES = 1000
BATCH_SIZE = 128 
GAMMA = 0.95
LR = 0.001
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.99
MAX_MOVES_PER_GAME = 300 
TRAIN_STEPS_PER_EPISODE = 10 
TARGET_UPDATE_FREQ = 15  # Sync target_net every N episodes

WEIGHTS_FILE = "chess_brain.pth"

# --- Setup Networks ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Policy Network (Actively trained)
policy_net = ChessBrain().to(device)

# ==========================================
# STEP 1: LOAD EXISTING KNOWLEDGE
# ==========================================
if os.path.exists(WEIGHTS_FILE):
    try:
        # Load the weights into the model
        policy_net.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device, weights_only=True))
        print(f"SUCCESS: Loaded existing knowledge from '{WEIGHTS_FILE}'. Scazy is building on past training.")
    except Exception as e:
        print(f"WARNING: Could not load weights ({e}). Starting fresh.")
else:
    print(f"Notice: No existing '{WEIGHTS_FILE}' found. Scazy is starting from scratch.")

# 2. Target Network (Frozen mentor for stable target evaluations)
target_net = ChessBrain().to(device)
# Target net copies the policy_net, which now contains our loaded weights
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Freeze layers like dropout/batchnorm if present

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
criterion = nn.MSELoss()
memory = deque(maxlen=MEMORY_SIZE)

# Piece values normalized for step rewards
PIECE_VALUES = {
    chess.PAWN: 0.1, chess.KNIGHT: 0.3, chess.BISHOP: 0.3,
    chess.ROOK: 0.5, chess.QUEEN: 0.9, chess.KING: 0.0
}

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

def calculate_step_reward(board, move):
    """Calculates material difference and positional breadcrumbs caused by a move."""
    reward = 0.0
    
    # --- 1. Material (Captures) ---
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            val = PIECE_VALUES.get(captured_piece.piece_type, 0)
            reward += val if board.turn == chess.WHITE else -val

    # --- UPGRADE B: ADVANCED REWARD SHAPING ---
    moving_piece = board.piece_at(move.from_square)
    if moving_piece:
        # Since White wants positive scores and Black wants negative scores:
        multiplier = 1.0 if board.turn == chess.WHITE else -1.0
        
        # 2. Center Control Breadcrumbs
        # Squares D4, E4, D5, E5
        if move.to_square in [chess.D4, chess.E4, chess.D5, chess.E5]:
            reward += 0.03 * multiplier
            
        # 3. Piece Development
        # Reward moving Knights and Bishops off the starting ranks
        if moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            start_rank = chess.square_rank(move.from_square)
            if (board.turn == chess.WHITE and start_rank == 0) or (board.turn == chess.BLACK and start_rank == 7):
                reward += 0.02 * multiplier
                
        # 4. King Safety (Early Castling)
        if board.is_castling(move):
            reward += 0.15 * multiplier
            
        # 5. Premature King Exposure Penalty
        # Penalize moving the King in the first 10 full moves unless it's castling
        if moving_piece.piece_type == chess.KING and not board.is_castling(move):
            if board.fullmove_number < 10:
                reward -= 0.05 * multiplier
                
    return reward

def get_best_move_fast(board, current_epsilon):
    """Selects move using an optimized Batched Epsilon-Greedy policy via policy_net."""
    legal_moves = list(board.legal_moves)
    if not legal_moves: return None
    
    # Exploration: Random move
    if random.random() < current_epsilon:
        return random.choice(legal_moves)
    
    # Exploitation: Batch evaluate all legal moves at once
    next_states = []
    for move in legal_moves:
        board.push(move)
        next_states.append(get_board_tensor_array(board))
        board.pop()
        
    batch_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Evaluate moves using policy_net
        vals = policy_net(batch_tensor).view(-1).cpu().numpy()
        
    if board.turn == chess.WHITE:
        best_idx = np.argmax(vals)
    else:
        best_idx = np.argmin(vals)
        
    return legal_moves[best_idx]

def train_batch():
    if len(memory) < BATCH_SIZE: return 0.0
    
    batch = random.sample(memory, BATCH_SIZE)
    states, next_states, rewards, dones = zip(*batch)
    
    states = torch.cat(states).to(device)
    next_states = torch.cat(next_states).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    
    # --- Target Network Evaluation ---
    # Current values evaluated by active policy_net
    current_values = policy_net(states)
    
    # Future values evaluated by frozen target_net for training stability
    with torch.no_grad():
        next_values = target_net(next_states)
        targets = rewards + GAMMA * next_values * (1 - dones)
        
    loss = criterion(current_values, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    epsilon = EPSILON_START
    print(f"Starting Target Network RL Training on {device}...")
    print(f"Target Net Sync Frequency: Every {TARGET_UPDATE_FREQ} episodes.")
    start_time = time.time()
    
    try:
        for episode in range(1, EPISODES + 1):
            board = chess.Board()
            steps = 0
            
            # --- PLAY THE GAME ---
            while not board.is_game_over() and steps < MAX_MOVES_PER_GAME:
                state_matrix = get_board_tensor_array(board)
                state_tensor = torch.tensor(state_matrix).unsqueeze(0)
                
                move = get_best_move_fast(board, epsilon)
                
                step_reward = calculate_step_reward(board, move)
                board.push(move)
                
                next_state_matrix = get_board_tensor_array(board)
                next_state_tensor = torch.tensor(next_state_matrix).unsqueeze(0)
                
                done = board.is_game_over() or steps == MAX_MOVES_PER_GAME - 1
                
                if done:
                    res = board.result()
                    if res == '1-0': step_reward += 1.0
                    elif res == '0-1': step_reward -= 1.0
                    else: step_reward += 0.0 # Draw
                    
                memory.append((state_tensor, next_state_tensor, step_reward, done))
                steps += 1
                
            # --- TRAIN AT THE END OF THE GAME ---
            total_loss = 0
            for _ in range(TRAIN_STEPS_PER_EPISODE):
                total_loss += train_batch()
                
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            avg_loss = total_loss / TRAIN_STEPS_PER_EPISODE if len(memory) >= BATCH_SIZE else 0
            
            # --- SYNC TARGET NETWORK ---
            if episode % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            outcome = board.result() if not steps >= MAX_MOVES_PER_GAME else "Draw (Move Limit)"
            
            # Periodic Checkpoint (policy_net weights)
            if episode % 25 == 0:
                torch.save(policy_net.state_dict(), WEIGHTS_FILE)
                
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Ep {episode}/{EPISODES} | Time: {elapsed_time:.1f}s | Moves: {steps} | Result: {outcome} | Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.2f}")
                start_time = time.time()

        # ==========================================
        # STEP 2: SAVE THE FINAL KNOWLEDGE (If loop finishes completely)
        # ==========================================
        torch.save(policy_net.state_dict(), WEIGHTS_FILE)
        print(f"\nTRAINING COMPLETE! Upgraded brain saved securely to {WEIGHTS_FILE}.")

    except KeyboardInterrupt:
        # ==========================================
        # STEP 3: EMERGENCY SAVE (Triggers if you press Ctrl+C)
        # ==========================================
        print("\n\nTraining stopped by user (Ctrl+C)! Saving current progress...")
        torch.save(policy_net.state_dict(), WEIGHTS_FILE)
        print(f"Emergency save complete. All progress up to this point is saved in {WEIGHTS_FILE}.")

if __name__ == "__main__":
    main()