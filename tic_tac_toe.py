import math

def print_board(board):
    print("\n")
    for row in board:
        print(" | ".join(row))
        print("-" * 9)
    print("\n")

def check_winner(board):
    # Check rows, columns, and diagonals
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != " ":
            return row[0]
    
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != " ":
            return board[0][col]
            
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != " ":
        return board[0][0]
    
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != " ":
        return board[0][2]
    
    if all(cell != " " for row in board for cell in row):
        return "Tie"
    
    return None

def minimax(board, depth, is_maximizing, alpha, beta):
    winner = check_winner(board)
    
    # SCORING: 
    # Prefer winning sooner (10 - depth) to be aggressive.
    # If losing is inevitable, delay it (depth - 10).
    if winner == "X": # AI wins
        return 10 - depth
    if winner == "O": # Human wins
        return depth - 10
    if winner == "Tie":
        return 0

    if is_maximizing:
        best_score = -math.inf
        for r in range(3):
            for c in range(3):
                if board[r][c] == " ":
                    board[r][c] = "X"
                    score = minimax(board, depth + 1, False, alpha, beta)
                    board[r][c] = " "
                    best_score = max(score, best_score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break 
        return best_score
    else:
        best_score = math.inf
        for r in range(3):
            for c in range(3):
                if board[r][c] == " ":
                    board[r][c] = "O"
                    score = minimax(board, depth + 1, True, alpha, beta)
                    board[r][c] = " "
                    best_score = min(score, best_score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
        return best_score

def best_move(board):
    best_score = -math.inf
    best_moves = [] 
    alpha = -math.inf
    beta = math.inf
    
    # 1. FIND ALL OPTIMAL MOVES (Safety Check)
    for r in range(3):
        for c in range(3):
            if board[r][c] == " ":
                board[r][c] = "X"
                score = minimax(board, 0, False, alpha, beta)
                board[r][c] = " "
                
                # If we found a strictly better score, reset the list
                if score > best_score:
                    best_score = score
                    best_moves = [(r, c)]
                # If this move is essentially equal to the best, add to list
                elif score == best_score:
                    best_moves.append((r, c))
    
    # 2. FILTER FOR STRATEGIC ADVANTAGE (The Fork Trick)
    # If multiple moves offer the same result (e.g., Tie), pick the one 
    # that is strategically superior (Center > Corner > Edge).
    
    center = (1, 1)
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
    selected_move = None
    
    # Priority 1: Take Center if it's a "best move"
    if center in best_moves:
        selected_move = center
    # Priority 2: Take a Corner if it's a "best move"
    else:
        for move in best_moves:
            if move in corners:
                selected_move = move
                break
    
    # Priority 3: Take whatever is left (Edges)
    if selected_move is None and best_moves:
        selected_move = best_moves[0]
        
    return selected_move

def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
    print("Tic-Tac-Toe AI (Unbeatable + Aggressive Fork Strategy)")
    print("AI is X (Goes First), You are O")
    
    while True:
        # --- AI TURN (First) ---
        print("AI is thinking...")
        move = best_move(board)
        if move:
            board[move[0]][move[1]] = "X"
            
        print_board(board)
        winner = check_winner(board)
        if winner:
            if winner == "Tie":
                print("It's a Tie!")
            else:
                print(f"Winner: {winner} (AI)")
            break
            
        # --- HUMAN TURN (Second) ---
        while True:
            try:
                user_input = input("Enter row and col (0-2) for 'O': ").split()
                if len(user_input) != 2:
                    print("Please enter two numbers separated by a space.")
                    continue
                row, col = map(int, user_input)
                if row < 0 or row > 2 or col < 0 or col > 2:
                    print("Out of bounds! 0-2 only.")
                    continue
                if board[row][col] != " ":
                    print("Spot taken, try again.")
                    continue
                board[row][col] = "O"
                break
            except ValueError:
                print("Invalid input. Please enter numbers 0-2 (e.g., '1 1').")

        winner = check_winner(board)
        if winner:
            if winner == "Tie":
                print_board(board)
                print("It's a Tie!")
            else:
                print_board(board)
                print(f"Winner: {winner}")
            break

if __name__ == "__main__":
    play_game()