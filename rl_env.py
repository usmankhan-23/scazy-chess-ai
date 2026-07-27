import chess
import numpy as np

class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        """Resets the board for a new game and returns the starting state as numbers."""
        self.board.reset()
        return self.get_board_state()

    def get_board_state(self):
        """Translates the board into a mathematical matrix for the AI."""
        # For our baseline, we will create a simple 8x8 matrix 
        # where empty = 0, White Pawn = 1, Black Pawn = -1, etc.
        piece_map = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
        }
        
        state = np.zeros((8, 8))
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                state[row][col] = piece_map[piece.symbol()]
                
        return state

    def step(self, move):
        """Takes an action, updates the board, and returns (new_state, reward, done)."""
        # Apply the move to the board
        self.board.push(move)
        
        # Calculate Reward
        reward = 0
        done = self.board.is_game_over()
        
        if done:
            result = self.board.result()
            if result == '1-0':    # White wins
                reward = 1
            elif result == '0-1':  # Black wins
                reward = -1
            else:                  # Draw
                reward = 0
                
        return self.get_board_state(), reward, done

# --- Quick Test ---
if __name__ == "__main__":
    env = ChessEnvironment()
    initial_state = env.reset()
    
    print("AI's Mathematical View of the Starting Board:\n")
    print(initial_state)