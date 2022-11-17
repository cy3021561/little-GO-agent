from tracemalloc import start
import numpy as np
import json
import time
from json import JSONEncoder

WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSS_REWARD = 0.0
BOARD_SIZE = 5
GAME_NUM = 100000


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class QLearner():
    def __init__(self, alpha=0.7, gamma=0.9, side=None):
        self.side = side
        self.alpha = alpha
        self.gamma = gamma
        self.q_values_dict = {}
        self.history_states = []
        self.read_q_file()

    def save_q_file(self):
        with open('q_file.json', 'w+') as f:
            json.dump(self.q_values_dict, f, cls=NumpyArrayEncoder)
            f.close()

    def revese_state(self, state):
        reversed_state = state.replace("1", "#")
        reversed_state = reversed_state.replace("2", "!")
        reversed_state = reversed_state.replace("!", "1")
        reversed_state = reversed_state.replace("#", "2")
        return reversed_state

    def rotate_board(self, m):
        return [[m[j][i] for j in range(len(m))] for i in range(len(m[0])-1, -1, -1)]

    def encode_state(self, board):
        """ Encode the current state of the board as a string
        """
        return ''.join([str(board[i][j]) for i in range(5) for j in range(5)])

    def decode_state(self, state):
        board = []
        start = 0
        for _ in range(5):
            cur = []
            for i in range(5):
                cur.append(int(state[start+i]))
            board.append(cur)
            start += 5
        return board

    def Q(self, board):
        board_rotate_1 = self.rotate_board(board)
        board_rotate_2 = self.rotate_board(board_rotate_1)
        board_rotate_3 = self.rotate_board(board_rotate_2)

        state = self.encode_state(board)
        reversed_state = self.revese_state(state)
        state_1 = self.encode_state(board_rotate_1)
        reversed_state_1 = self.revese_state(state_1)
        state_2 = self.encode_state(board_rotate_2)
        reversed_state_2 = self.revese_state(state_2)
        state_3 = self.encode_state(board_rotate_3)
        reversed_state_3 = self.revese_state(state_3)

        if state in self.q_values_dict:
            return self.q_values_dict[state], 0
        elif reversed_state in self.q_values_dict:
            return self.q_values_dict[reversed_state], 0
        elif state_1 in self.q_values_dict:
            return self.q_values_dict[state_1], 1
        elif reversed_state_1 in self.q_values_dict:
            return self.q_values_dict[reversed_state_1], 1
        elif state_2 in self.q_values_dict:
            return self.q_values_dict[state_2], 2
        elif reversed_state_2 in self.q_values_dict:
            return self.q_values_dict[reversed_state_2], 2
        elif state_3 in self.q_values_dict:
            return self.q_values_dict[state_3], 3
        elif reversed_state_3 in self.q_values_dict:
            return self.q_values_dict[reversed_state_3], 3
        else:
            q_val = np.zeros((5, 5))
            q_val.fill(self.initial_value)
            self.q_values_dict[state] = q_val
            return self.q_values_dict[state], 0

    def read_q_file(self):
        with open('q_file.json') as f:
            self.q_values_dict = json.load(f)
            for key in self.q_values_dict:
                value = self.q_values_dict[key]
                self.q_values_dict[key] = np.array(value, dtype=object)
            # print(self.q_values_dict)
            f.close()

    def read_h_file(self):
        with open('history.txt') as f:
            for line in f:
                self.history_states.append(line.strip())
            side = self.history_states[0].split(',')[0]
            self.side = side
            f.close()

    def clean_h_file(self):
        with open("history.txt", 'w') as f:
            pass

    def counter_clock_rotate(self, i, j, rotate_time):
        for _ in range(rotate_time):
            i, j = i, 4 - j
            i, j = j, i
        return i, j

    def learn(self, winner):
        if winner == 0:
            reward = DRAW_REWARD
        elif winner == self.side:
            reward = WIN_REWARD
        else:
            reward = LOSS_REWARD
        self.history_states.reverse()
        max_q_value = -1.0
        for hist in self.history_states:
            side, state, row, col = hist.split(',')
            board = self.decode_state(state)
            row, col = int(row), int(col)
            q, rotate_time = self.Q(board)
            row, col = self.counter_clock_rotate(row, col, rotate_time)
            #print(state, row, col, self.q_values_dict[state][row][col])
            if max_q_value < 0:
                q[row][col] = reward
            else:
                q[row][col] = q[row][col] * \
                    (1 - self.alpha) + self.alpha * self.gamma * max_q_value
            max_q_value = np.max(q)
            #print(state, row, col, self.q_values_dict[state][row][col])
        self.history_states = []
