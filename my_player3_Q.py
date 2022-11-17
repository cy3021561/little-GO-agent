from read import readInput
from write import writeOutput
import numpy as np
import json
import time

from host import GO


class QLearnerGo():
    def __init__(self, initial_value=0.5, side=None):
        self.q_values_dict = {}
        self.history_states = []
        self.initial_value = initial_value
        self.read_q_file()

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
        r_state = self.revese_state(state)
        if state in self.q_values_dict:
            return self.q_values_dict[state]
        elif r_state in self.q_values_dict:
            return self.q_values_dict[r_state]
        else:
            q_val = np.array(
                [[0.3, 0.3, 0.3], [0.3, 0.8, 0.7], [0.3, 0.7, 0.6]])
            self.q_values_dict[state] = q_val
            return self.q_values_dict[state]

    def encode_state(self, board):
        """ Encode the current state of the board as a string
        """
        return ''.join([str(board[i][j]) for i in range(3) for j in range(3)])

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

    def get_input(self, go, piece_type, board):
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
        #board = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
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
            for i in range(3):
                for j in range(3):
                    true_i, true_j = start_i + i, start_j + j
                    if go.valid_place_check(true_i, true_j, piece_type, test_check=True):
                        r, c = self.counter_clock_rotate(i, j, rotate_time)
                        if q_values[r][c] > curr_max:
                            curr_max = q_values[r][c]
                            true_row, true_col = true_i, true_j
                        possible_placements += 1
                    else:
                        r, c = self.counter_clock_rotate(i, j, rotate_time)
                        q_values[r][c] = -1.0

        if possible_placements == 0:
            return "PASS"
        else:
            return true_row, true_col


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = QLearnerGo()
    action = player.get_input(go, piece_type, board)
    writeOutput(action)
