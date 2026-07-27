import chess

# Initialize the starting board
board = chess.Board()

# The main game loop: keep playing until someone wins or it's a draw
while not board.is_game_over():
    # Print the current state of the board
    print("\n-------------------")
    print(board)
    print("-------------------")
    
    # Show whose turn it is
    if board.turn == chess.WHITE:
        print("White to move.")
    else:
        print("Black to move.")
        
    # Get user input (using UCI format like 'e2e4' or 'g1f3')
    user_move = input("Enter your move: ")
    
    try:
        # Convert the text into a move object
        move = chess.Move.from_uci(user_move)
        
        # Check if the move is legal
        if move in board.legal_moves:
            board.push(move) # Apply the move to the board
        else:
            print("Illegal move! Try again.")
            
    except ValueError:
        # Handles typos or bad input formatting
        print("Invalid format! Please use standard format (e.g., e2e4).")

# Once the loop breaks (game over), print the final result
print("\nGame Over!")
print("Result:", board.result()) # Returns '1-0' (White wins), '0-1' (Black wins), or '1/2-1/2' (Draw)