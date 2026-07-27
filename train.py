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
EPISODES = 100 
BATCH_SIZE = 128 
GAMMA = 0.95
LR = 0.001
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.99
MAX_MOVES_PER_GAME = 200  
TRAIN_STEPS_PER_EPISODE = 10 

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessBrain().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
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
    """Calculates material difference caused by a move."""
    reward = 0.0
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            val = PIECE_VALUES.get(captured_piece.piece_type, 0)
            reward = val if board.turn == chess.WHITE else -val
    return reward

def get_best_move_fast(board, current_epsilon):
    """Selects move using an optimized Batched Epsilon-Greedy policy."""
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
        
    # Stack all next states into a single tensor: shape (Batch, 14, 8, 8)
    batch_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # A SINGLE forward pass for all possible moves
        vals = model(batch_tensor).view(-1).cpu().numpy()
        
    # Select best move depending on whose turn it is
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
    
    # Q-learning target
    with torch.no_grad():
        next_values = model(next_states)
        targets = rewards + GAMMA * next_values * (1 - dones)
        
    current_values = model(states)
    loss = criterion(current_values, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    epsilon = EPSILON_START
    print(f"Starting Highly Optimized Self-Play RL Training on {device}...")
    start_time = time.time()
    
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
        
        outcome = board.result() if not steps >= MAX_MOVES_PER_GAME else "Draw (Move Limit)"
        
        # Save Checkpoint
        if episode % 25 == 0:
            torch.save(model.state_dict(), "chess_brain.pth")
            
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Ep {episode}/{EPISODES} | Time: {elapsed_time:.1f}s | Moves: {steps} | Result: {outcome} | Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.2f}")
            start_time = time.time() # Reset timer for next batch of 10

if __name__ == "__main__":
    main()