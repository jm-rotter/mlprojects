import copy
import math

X = "X"
O = "O"
EMPTY = None

def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    flat = [n for ns in board for n in ns]
    count_x = sum(1 for n in flat if n == X)
    count_o = sum(1 for n in flat if n == O)
    return X if count_x == count_o else O

def actions(board):
    return [(i, j) for i, row in enumerate(board)
                   for j, cell in enumerate(row) if cell == EMPTY]

def result(board, action):
    nboard = copy.deepcopy(board)
    i, j = action
    if nboard[i][j] is not EMPTY:
        raise ValueError("Invalid action")
    nboard[i][j] = player(board)
    return nboard

def winner(board):
    f = [n for ns in board for n in ns]
    win_pos = [(0,1,2), (3,4,5), (6,7,8),
               (0,3,6), (1,4,7), (2,5,8),
               (0,4,8), (2,4,6)]
    for i,j,k in win_pos:
        if f[i] == f[j] == f[k] and f[i] is not None:
            return f[i]
    return None

def terminal(board):
    return winner(board) is not None or all(cell is not EMPTY for row in board for cell in row)

def utility(board):
    w = winner(board)
    if w == X:
        return 1
    elif w == O:
        return -1
    return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board using alpha-beta pruning.
    """

    def alphabeta(board, alpha, beta):
        if terminal(board):
            return utility(board), None

        best_move = None

        if player(board) == X:
            value = -math.inf
            for action in actions(board):
                eval, _ = alphabeta(result(board, action), alpha, beta)
                if eval > value:
                    value = eval
                    best_move = action
                alpha = max(alpha, value)
                if beta <= alpha:
                    break 
            return value, best_move

        else:
            value = math.inf
            for action in actions(board):
                eval, _ = alphabeta(result(board, action), alpha, beta)
                if eval < value:
                    value = eval
                    best_move = action
                beta = min(beta, value)
                if beta <= alpha:
                    break  
            return value, best_move

    _, move = alphabeta(board, -math.inf, math.inf)
    return move
