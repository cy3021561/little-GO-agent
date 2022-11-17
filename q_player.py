from re import T
from read import readInput
from write import writeOutput
import numpy as np
import json
from json import JSONEncoder
import time
from copy import deepcopy

WIN_REWARD = 2.0
DRAW_REWARD = 0.5
LOSS_REWARD = 0.0


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class QLearnerGo():
    def __init__(self, initial_value=0.5, side=None, alpha=0.5, gamma=0.7):
        self.q_values_dict = {}
        self.alpha = alpha
        self.gamma = gamma
        self.history_states = []
        self.sub_history_states_1 = []
        self.sub_history_states_2 = []
        self.sub_history_states_3 = []
        self.sub_history_states_4 = []
        self.last_sub_board = None
        self.initial_value = initial_value
        self.read_q_file()
        self.board_history = []

    def set_side(self, side):
        self.side = side

    def revese_state(self, state):
        reversed_state = state.replace("1", "#")
        reversed_state = reversed_state.replace("2", "!")
        reversed_state = reversed_state.replace("!", "1")
        reversed_state = reversed_state.replace("#", "2")
        return reversed_state

    # Counter clock-wise
    def rotate_board(self, board, rotate_time):
        for i in range(rotate_time):
            board = [[board[j][i] for j in range(len(board))]
                     for i in range(len(board[0])-1, -1, -1)]
        return board

    def Q(self, board):
        state = self.encode_state(board)
        # r_state = self.revese_state(state)
        if state in self.q_values_dict:
            return self.q_values_dict[state]
        # elif r_state in self.q_values_dict: return self.q_values_dict[r_state]
        else:
            q_val = np.array(
                [[0.3, 0.3, 0.3], [0.3, 0.8, 0.7], [0.3, 0.7, 0.6]])
            self.q_values_dict[state] = q_val
            return self.q_values_dict[state]

    def encode_state(self, board):
        """ Encode the current state of the board as a string
        """
        return ''.join([str(board[i][j]) for i in range(3) for j in range(3)])

    def save_q_file(self):
        with open('q_file.json', 'w+') as f:
            json.dump(self.q_values_dict, f, cls=NumpyArrayEncoder)
            f.close()

    def read_q_file(self):
        with open('q_file.json') as f:
            self.q_values_dict = json.load(f)
            for key in self.q_values_dict:
                value = self.q_values_dict[key]
                self.q_values_dict[key] = np.array(value, dtype=object)
            f.close()

    def counter_clock_rotate(self, i, j, rotate_time):
        for _ in range(rotate_time):
            i, j = i, 2 - j
            i, j = j, i
        return i, j

    def clock_wise_rotate(self, i, j, rotate_time):
        for _ in range(rotate_time):
            i, j = j, i
            i, j = i, 2 - j
        return i, j

    def devide_board(self, board):
        board_1 = [[board[i][j] for j in range(0, 3)] for i in range(0, 3)]
        board_2 = [[board[i][j] for j in range(2, 5)] for i in range(0, 3)]
        board_3 = [[board[i][j] for j in range(0, 3)] for i in range(2, 5)]
        board_4 = [[board[i][j] for j in range(2, 5)] for i in range(2, 5)]

        return board_1, board_2, board_3, board_4

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''

        self.set_side(piece_type)
        board = go.board
        possible_placements = 0
        curr_max = -np.inf
        true_row, true_col = 0, 0
        sub_row, sub_col = 0, 0
        max_sub_board = None
        sub_state = None
        min_max_cadidates = []
        # board = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]
        for start, sub_board in zip([[1, 0, 0], [2, 0, 2], [3, 2, 0], [4, 2, 2]], self.devide_board(board)):
            sub_idx, start_i, start_j = start[0], start[1], start[2]
            if sub_idx == 1:
                rotate_time = 0
            elif sub_idx == 2:
                rotate_time = 1
            elif sub_idx == 3:
                rotate_time = 3
            else:
                rotate_time = 2
            sub_board = self.rotate_board(sub_board, rotate_time)
            q_values = self.Q(sub_board)
            best_sub_board_row = None
            best_sub_board_col = None
            for i in range(3):
                for j in range(3):
                    true_i, true_j = start_i + i, start_j + j
                    if go.valid_place_check(true_i, true_j, piece_type, test_check=True):
                        r, c = self.counter_clock_rotate(i, j, rotate_time)
                        q_plus = 0
                        if self.surround_chess(true_i, true_j, board) == 1:
                            q_plus = 0.1
                        elif self.surround_chess(true_i, true_j, board) == 2:
                            q_plus = 0.2

                        if q_values[r][c] + q_plus > curr_max:
                            curr_max = q_values[r][c]
                            true_row, true_col = true_i, true_j
                            sub_row, sub_col = r, c
                            sub_state = self.encode_state(sub_board)
                            sub_rotate = rotate_time
                            max_sub_board = sub_idx
                        possible_placements += 1
                    else:
                        r, c = self.counter_clock_rotate(i, j, rotate_time)
                        q_values[r][c] = -1.0
            if best_sub_board_col and best_sub_board_row:
                min_max_cadidates.append(
                    [best_sub_board_row, best_sub_board_col])

        if possible_placements == 0:
            print("PASS")
            return "PASS"
        else:
            if max_sub_board == 1:
                self.sub_history_states_1.append(
                    ([sub_state, sub_row, sub_col]))
            elif max_sub_board == 2:
                self.sub_history_states_2.append(
                    ([sub_state, sub_row, sub_col]))
            elif max_sub_board == 3:
                self.sub_history_states_3.append(
                    ([sub_state, sub_row, sub_col]))
            else:
                self.sub_history_states_4.append(
                    ([sub_state, sub_row, sub_col]))
            self.last_sub_board = max_sub_board
            self.history_states.append([sub_state, sub_row, sub_col])
            # self.board_history.append(deepcopy(board))
            return true_row, true_col

    def decode_state(self, state):
        board = []
        start = 0
        for _ in range(3):
            cur = []
            for i in range(3):
                cur.append(int(state[start+i]))
            board.append(cur)
            start += 3
        return board

    def surround_chess(self, row, col, board):
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        surround = 0
        for dir_r, dir_c in directions:
            new_r, new_c = dir_r + row, dir_c + col
            if new_r < 0 or new_c < 0 or new_r >= len(board) or new_c >= len(board[0]):
                continue
            if board[new_r][new_c] == self.side:
                surround += 1
        return surround

    def learn(self, winner):
        result = False
        if winner == 0:
            reward = DRAW_REWARD
        elif winner == self.side:
            # print("WIN")
            result = True
            reward = WIN_REWARD
        else:
            # print("LOSE")
            reward = LOSS_REWARD
        '''
        self.history_states.reverse()
        max_q_value = -1.0
        idx = 1
        for hist in self.history_states:
            state, row, col = hist
            board = self.decode_state(state)
            row, col = int(row), int(col)
            q = self.Q(board)
            # print(state, row, col, self.q_values_dict[state][row][col])
            if idx == 1 and state == "000000000":
                continue
            elif idx == 1:
                q[row][col] = reward
            else:
                q[row][col] = q[row][col] * \
                    (1 - self.alpha) + self.alpha * \
                    self.gamma * max_q_value
            # print(q[row][col])
            max_q_value = np.max(q)
            idx += 1
        '''
        t = 1
        for history in ([self.sub_history_states_1, self.sub_history_states_2, self.sub_history_states_3, self.sub_history_states_4]):
            # print(t)
            # print(history)
            history.reverse()
            idx = 1
            if t == self.last_sub_board and len(history) >= 3:
                max_q_value = -1.0
                for hist in history:
                    state, row, col = hist
                    board = self.decode_state(state)
                    row, col = int(row), int(col)
                    q = self.Q(board)
                    # print(state, row, col, self.q_values_dict[state][row][col])
                    if idx == 1 and state == "000000000":
                        continue
                    elif idx == 1:
                        q[row][col] = reward
                    else:
                        q[row][col] = q[row][col] * (1 - self.alpha) + \
                            self.alpha * self.gamma * max_q_value
                    # print(q[row][col])
                    max_q_value = np.max(q)
                    idx += 1

            t += 1

            # print(state, row, col, self.q_values_dict[state][row][col])

        self.history_states = []
        self.sub_history_states_1 = []
        self.sub_history_states_2 = []
        self.sub_history_states_3 = []
        self.sub_history_states_4 = []
        self.board_history = []
