import torch
import random
import chess
from rl_env import ChessEnvironment
from model import ChessBrain

# 1. Load the trained brain
brain = ChessBrain()
try:
    brain.load_state_dict(torch.load("chess_brain.pth", weights_only=True))
    brain.eval() # Set model to evaluation mode
    print("Loaded trained AI brain for benchmarking!\n")
except FileNotFoundError:
    print("Error: 'chess_brain.pth' not found. Please train the model first.")
    exit()

def get_ai_move(env, brain, is_white):
    """Generates the optimal move based on the neural network evaluation."""
    legal_moves = list(env.board.legal_moves)
    best_move = None
    best_score = -float('inf') if is_white else float('inf')
    
    for move in legal_moves:
        env.board.push(move)
        state_tensor = torch.tensor(env.get_board_state(), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            score = brain(state_tensor).item()
            
        env.board.pop()
        
        if is_white and score > best_score:
            best_score = score
            best_move = move
        elif not is_white and score < best_score:
            best_score = score
            best_move = move
            
    return best_move if best_move else random.choice(legal_moves)

def evaluate_ai(num_games=50):
    env = ChessEnvironment()
    ai_wins = 0
    random_wins = 0
    draws = 0
    
    print(f"Starting Arena Match: Trained AI vs. Random Agent ({num_games} Games)\n")
    
    for game in range(num_games):
        state = env.reset()
        done = False
        
        # AI plays White on even-numbered games, Black on odd-numbered games
        ai_plays_white = (game % 2 == 0)
        
        while not done:
            is_ai_turn = (env.board.turn == chess.WHITE and ai_plays_white) or \
                         (env.board.turn == chess.BLACK and not ai_plays_white)
            
            if is_ai_turn:
                action = get_ai_move(env, brain, is_white=(env.board.turn == chess.WHITE))
            else:
                action = random.choice(list(env.board.legal_moves)) # Random Opponent
                
            state, reward, done = env.step(action)
            
        result = env.board.result()
        if result == '1-0':
            if ai_plays_white:
                ai_wins += 1
            else:
                random_wins += 1
        elif result == '0-1':
            if not ai_plays_white:
                ai_wins += 1
            else:
                random_wins += 1
        else:
            draws += 1
            
        print(f"Game {game + 1}/{num_games} Complete | Result: {result}")

    # Final Statistics
    print("\n" + "="*30)
    print("      ARENA BENCHMARK SUMMARY      ")
    print("="*30)
    print(f"Trained AI Wins : {ai_wins} ({ai_wins/num_games*100:.1f}%)")
    print(f"Random Opponent : {random_wins} ({random_wins/num_games*100:.1f}%)")
    print(f"Draws           : {draws} ({draws/num_games*100:.1f}%)")

if __name__ == "__main__":
    evaluate_ai(num_games=50)