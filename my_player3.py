import random
import time
import numpy as np
from copy import deepcopy
from numpy.lib.stride_tricks import as_strided


def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)


def writePass(path="output.txt"):
    with open(path, 'w') as f:
        f.write("PASS")


def writeNextInput(piece_type, previous_board, board, path="input.txt"):
    res = ""
    res += str(piece_type) + "\n"
    for item in previous_board:
        res += "".join([str(x) for x in item])
        res += "\n"

    for item in board:
        res += "".join([str(x) for x in item])
        res += "\n"

    with open(path, 'w') as f:
        f.write(res[:-1])


def readInput(n, path="input.txt"):

    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')]
                          for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')]
                 for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board


def readOutput(path="output.txt"):
    with open(path, 'r') as f:
        position = f.readline().strip().split(',')

        if position[0] == "PASS":
            return "PASS", -1, -1

        x = int(position[0])
        y = int(position[1])

    return "MOVE", x, y


class GO:
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        self.previous_board = None  # Store the previous board
        self.X_move = True  # X chess plays first
        self.died_pieces = []  # Intialize died pieces to be empty
        self.n_move = 0  # Trace the number of moves
        self.max_move = n * n - 1  # The max movement of a Go game
        self.komi = n/2  # Komi rule
        self.verbose = False  # Verbose only when there is a manual player

    def init_board(self, n):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        board = [[0 for x in range(n)]
                 for y in range(n)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = board
        self.previous_board = deepcopy(board)

    def set_board(self, piece_type, previous_board, board):
        '''
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        '''

        # 'X' pieces marked as 1
        # 'O' pieces marked as 2

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        # self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0:
            neighbors.append((i-1, j))
        if i < len(board) - 1:
            neighbors.append((i+1, j))
        if j > 0:
            neighbors.append((i, j-1))
        if j < len(board) - 1:
            neighbors.append((i, j+1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces:
            return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def place_chess(self, i, j, piece_type):
        '''
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        # Remove the following line for HW2 CS561 S2020
        # self.n_move += 1
        return True

    def valid_place_check(self, i, j, piece_type, test_check=False):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(('Invalid placement. row should be in the range 1 to {}.').format(
                    len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(('Invalid placement. column should be in the range 1 to {}.').format(
                    len(board) - 1))
            return False

        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print('Invalid placement. There is already a chess in this position.')
            return False

        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                if verbose:
                    print(
                        'Invalid placement. A repeat move not permitted by the KO rule.')
                return False
        return True

    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''
        self.board = new_board

    def connected_stones(self, curr_board, piece_type):
        q1 = 0
        q3 = 0
        qd = 0
        arr = np.array(curr_board)
        arr = np.pad(arr, pad_width=1)
        arr = np.where(arr == 3-piece_type, 0, arr)
        two_quad = (2, 2)
        view_shape = tuple(np.subtract(arr.shape, two_quad) + 1) + two_quad
        state = as_strided(arr, view_shape, arr.strides * 2)
        state = state.reshape((-1,) + two_quad)
        for i in range(len(state)):
            if state[i][0][0] == piece_type and state[i][0][1] == state[i][1][0] == state[i][1][1] == 0:
                q1 += 1
            if state[i][0][1] == piece_type and state[i][0][0] == state[i][1][0] == state[i][1][1] == 0:
                q1 += 1
            if state[i][1][0] == piece_type and state[i][0][1] == state[i][0][0] == state[i][1][1] == 0:
                q1 += 1
            if state[i][1][1] == piece_type and state[i][0][1] == state[i][1][0] == state[i][0][0] == 0:
                q1 += 1
            if state[i][0][1] == state[i][1][0] == state[i][1][1] == piece_type and state[i][0][0] == 0:
                q3 += 1
            if state[i][0][0] == state[i][1][0] == state[i][1][1] == piece_type and state[i][0][1] == 0:
                q3 += 1
            if state[i][0][0] == state[i][0][1] == state[i][1][1] == piece_type and state[i][1][0] == 0:
                q3 += 1
            if state[i][0][0] == state[i][1][0] == state[i][0][1] == piece_type and state[i][1][1] == 0:
                q3 += 1
            if state[i][0][0] == state[i][1][1] == piece_type and state[i][0][1] == state[i][1][0] == 0:
                qd += 1
            if state[i][0][1] == state[i][1][0] == piece_type and state[i][0][0] == state[i][1][1] == 0:
                qd += 1
        euler = (q1-q3+2*qd)/4
        return euler

    def score(self, max_capture, min_capture):
        '''
        Get score of a player by counting the number of stones.

        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the game should end.
        '''
        cur_board = deepcopy(self.board)
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        cnt = 0
        op_cnt = 0
        l = 0
        op_l = 0
        euler = self.connected_stones(cur_board, player_color)
        op_euler = self.connected_stones(cur_board, 3 - player_color)
        edge = 0
        op_edge = 0

        for i in range(self.size):
            for j in range(self.size):
                if cur_board[i][j] == player_color:
                    for dir_i, dir_j in directions:
                        nei_i, nei_j = i + dir_i, j + dir_j
                        if nei_i < 0 or nei_i >= 5 or nei_j < 0 or nei_j >= 5:
                            continue
                        if cur_board[nei_i][nei_j] == 0 or cur_board[nei_i][nei_j] == "?":
                            l += 1
                            if cur_board[nei_i][nei_j] == 0:
                                cur_board[nei_i][nei_j] = "!"
                            elif cur_board[nei_i][nei_j] == "?":
                                cur_board[nei_i][nei_j] = "#"
                    if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                        edge += 1
                    cnt += 1
                elif cur_board[i][j] == 3 - player_color:
                    for dir_i, dir_j in directions:
                        nei_i, nei_j = i + dir_i, j + dir_j
                        if nei_i < 0 or nei_i >= 5 or nei_j < 0 or nei_j >= 5:
                            continue
                        if cur_board[nei_i][nei_j] == 0 or cur_board[nei_i][nei_j] == "!":
                            op_l += 1
                            if cur_board[nei_i][nei_j] == 0:
                                cur_board[nei_i][nei_j] = "?"
                            elif cur_board[nei_i][nei_j] == "!":
                                cur_board[nei_i][nei_j] = "#"
                    if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                        op_edge += 1
                    op_cnt += 1
        liberty = l - op_l

        if player_color == 1:
            op_cnt += 2.5
            return 5 * (cnt - op_cnt) + min(max(liberty, -4), 4) + (-4 * (euler - op_euler)) - (edge - op_edge) + (max_capture - min_capture)
        else:
            cnt += 2.5
            return (cnt - op_cnt) + min(max(liberty, -4), 4) + (-4 * (euler - op_euler)) - (edge - op_edge) + (max_capture - min_capture)


class MinMaxPlayer():
    def set_side(self, side):
        self.side = side
        self.pass_move = 0

    def a_b_pruning(self, curr_go, depth, alpha, beta, s, piece_type, max_capture, min_capture):
        if depth == 0 or self.pass_move == 2:
            return s
        possible_move = False
        # Max
        if piece_type == player_color:
            best_s = float('-inf')
            for r in range(5):
                for c in range(5):
                    #print("MAX", depth)
                    #print(alpha, beta)
                    if curr_go.valid_place_check(r, c, piece_type, True):
                        possible_move = True
                        if self.pass_move == 1:
                            self.pass_move = 0
                        next_go = deepcopy(curr_go)
                        next_go.place_chess(r, c, piece_type)
                        died_stone = next_go.remove_died_pieces(3 - piece_type)
                        if died_stone:
                            max_capture += len(died_stone)
                        next_s = next_go.score(max_capture, min_capture)
                        curr_best = self.a_b_pruning(
                            next_go, depth - 1, alpha, beta, next_s, 3 - piece_type, max_capture, min_capture)
                        if curr_best > best_s:
                            best_s = curr_best
                        alpha = max(alpha, best_s)
                        if best_s >= beta:
                            return best_s

            if not possible_move:
                self.pass_move += 1
                next_go = deepcopy(curr_go)
                curr_best = self.a_b_pruning(
                    next_go, depth - 1, alpha, beta, s, 3 - piece_type, max_capture, min_capture)
                best_s = max(best_s, curr_best)
            return best_s
        # Min
        else:
            worst_s = float('inf')
            for r in range(5):
                for c in range(5):
                    #print("MIN", depth)
                    #print(alpha, beta)
                    if curr_go.valid_place_check(r, c, piece_type, True):
                        possible_move = True
                        if self.pass_move == 1:
                            self.pass_move = 0
                        next_go = deepcopy(curr_go)
                        next_go.place_chess(r, c, piece_type)
                        died_stone = next_go.remove_died_pieces(3 - piece_type)
                        if died_stone:
                            min_capture += len(died_stone)
                        next_s = next_go.score(max_capture, min_capture)
                        curr_best = self.a_b_pruning(
                            next_go, depth - 1, alpha, beta, next_s, 3 - piece_type, max_capture, min_capture)
                        if curr_best < worst_s:
                            worst_s = curr_best
                        beta = min(beta, worst_s)
                        if worst_s <= alpha:
                            return worst_s

            if not possible_move:
                self.pass_move += 1
                next_go = deepcopy(curr_go)
                curr_best = self.a_b_pruning(
                    next_go, depth - 1, alpha, beta, s, 3 - piece_type, max_capture, min_capture)
                worst_s = min(worst_s, curr_best)
            return worst_s

    def min_max(self, go, cadidates, piece_type, max_depth):
        alpha = float('-inf')
        beta = float('inf')
        moves = []
        max_value = float('-inf')
        for r, c in cadidates:
            max_capture = 0
            min_capture = 0
            next_go = deepcopy(go)
            next_go.place_chess(r, c, piece_type)
            died_stone = next_go.remove_died_pieces(3 - piece_type)
            if died_stone:
                max_capture += len(died_stone)

            next_s = next_go.score(max_capture, min_capture)
            best_s = self.a_b_pruning(
                next_go, max_depth, alpha, beta, next_s, 3 - piece_type, max_capture, min_capture)

            if best_s > max_value:
                max_value = best_s
                moves = [[r, c]]
            elif best_s == max_value:
                moves.append([r, c])
            alpha = max(best_s, alpha)

        return moves

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''
        self.set_side(piece_type)
        possible_placements = []
        initial_moves = []
        max_depth = 2
        for i in range(5):
            for j in range(5):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    if (i == 2 and j == 2) or (i == 2 and j == 3) or (i == 3 and j == 2) or (i == 2 and j == 1) or (i == 1 and j == 2):
                        initial_moves.append([i, j])
                    possible_placements.append([i, j])
        if not possible_placements:
            self.pass_move += 1
            return "PASS"
        if initial_moves and len(possible_placements) > 21:
            if [2, 2] in initial_moves:
                return [2, 2]
            else:
                return initial_moves[-1]
        if len(possible_placements) <= 10:
            max_depth = 3
        moves = self.min_max(go, possible_placements, piece_type, max_depth)
        return (moves[0])


if __name__ == "__main__":
    start = time.process_time()
    N = 5
    piece_type, previous_board, board = readInput(N)
    player_color = piece_type
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = MinMaxPlayer()
    action = player.get_input(go, piece_type)
    writeOutput(action)
    end = time.process_time()
    print("Move spent: " + str(end-start) + " secs")
