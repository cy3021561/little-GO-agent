from copy import deepcopy
from traing_host import GO
from random_player_for_train import RandomPlayer
from q_player import QLearnerGo
from my_player3 import MinMaxPlayer
import signal
import resource
import time


def time_exceeded(signo, frame):
    print("Time's up!")
    raise SystemExit(1)


def set_max_runtime(seconds):
    # Install the signal handler and set a resource limit
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, hard))
    signal.signal(signal.SIGXCPU, time_exceeded)


TRAIN_ITERATION = 100000

if __name__ == "__main__":
    # set_max_runtime(20)
    N = 5

    player1 = MinMaxPlayer()
    player2 = QLearnerGo()
    curr_win = 0
    q_win = 0
    r_win = 0
    tie = 0
    curr_time = 0

    for i in range(TRAIN_ITERATION):

        curr_time += 1
        try:
            go = GO(N)
            print("Training progress: " + str(i) +
                  " / " + str(TRAIN_ITERATION))
            if q_win + r_win + tie != 0:
                print("Winning rate:" + str(q_win / (q_win + r_win + tie)))
            '''
            if (q_win / (q_win + r_win + tie)) > curr_win and curr_time > 1000:
                player1.save_q_file()
                curr_win = (q_win / (q_win + r_win + tie))
                print(curr_win)
                player1.read_q_file()
                q_win = 1
                r_win = 1
                tie = 1
                curr_time = 0
                print("Training result saved at: " + str(i+1) + " iteration")
            if (i+1) % 10000 == 0 and (q_win / (q_win + r_win + tie)) < curr_win:
                player1.read_q_file()
                q_win = 1
                r_win = 1
                tie = 1
                curr_time = 0
                print("Reloaing q files...")
                '''
            # Initiate Board
            go.init_board(go.size)
            go.n_move = 0
            go.X_move = True
            result = None
            # Q be black
            while not result:
                # First Player
                piece_type = 1 if go.X_move else 2
                if go.game_end(piece_type):
                    result = go.judge_winner()
                    break

                action = player1.get_input(go, piece_type)
                if action != "PASS":
                    if not go.place_chess(action[0], action[1], piece_type):
                        result = 2 if go.X_move else 1
                        break
                    go.died_pieces = go.remove_died_pieces(3 - piece_type)

                else:
                    go.previous_board = deepcopy(go.board)

                go.n_move += 1
                go.X_move = not go.X_move
                # Second Player
                piece_type = 1 if go.X_move else 2
                if go.game_end(piece_type):
                    result = go.judge_winner()
                    break

                action = player2.get_input(go, piece_type)
                if action != "PASS":
                    if not go.place_chess(action[0], action[1], piece_type):
                        result = 2 if go.X_move else 1
                        break
                    go.died_pieces = go.remove_died_pieces(3 - piece_type)

                else:
                    go.previous_board = deepcopy(go.board)
                go.n_move += 1
                go.X_move = not go.X_move
            if result == player1.side:
                print("WWWIIINNN")
                q_win += 1
            elif result == 0:
                tie += 1
            else:
                print("LLLOOOSSSEEE")
                r_win += 1
            # player1.learn(result)
            # time.sleep(10000)

            # Q be white
            go.X_move = True
            go.n_move = 0
            go.init_board(go.size)
            result = None
            while not result:
                # First Player
                piece_type = 1 if go.X_move else 2
                if go.game_end(piece_type):
                    result = go.judge_winner()
                    break

                action = player2.get_input(go, piece_type)
                if action != "PASS":
                    if not go.place_chess(action[0], action[1], piece_type):
                        result = 2 if go.X_move else 1
                        break
                    go.died_pieces = go.remove_died_pieces(3 - piece_type)

                else:
                    go.previous_board = deepcopy(go.board)

                go.n_move += 1
                go.X_move = not go.X_move
                # Second Player
                piece_type = 1 if go.X_move else 2
                if go.game_end(piece_type):
                    result = go.judge_winner()
                    break

                action = player1.get_input(go, piece_type)
                if action != "PASS":
                    if not go.place_chess(action[0], action[1], piece_type):
                        result = 2 if go.X_move else 1
                        break
                    go.died_pieces = go.remove_died_pieces(3 - piece_type)

                else:
                    go.previous_board = deepcopy(go.board)
                go.n_move += 1
                go.X_move = not go.X_move

            if result == player1.side:
                print("WWWIIINNN")
                q_win += 1
            elif result == 0:
                print("LLLOOOSSSEEE")
                tie += 1
            else:
                r_win += 1
            # player1.learn(result)
            del go

        except Exception as e:
            print(e)
