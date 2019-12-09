import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import datetime
try:
    from .parsearguments import parse
    from .HumanPlayer import HumanPlayer
    from .AIPlayer import AIPlayer
    from .model import NeuralNetwork
    from .Game import Game
except (ModuleNotFoundError, ImportError):
    from parsearguments import parse
    from HumanPlayer import HumanPlayer
    from AIPlayer import AIPlayer
    from model import NeuralNetwork
    from Game import Game

import sys
from collections import deque
import pickle
warnings.resetwarnings()
memory = deque(maxlen=50000)


# if wanted, the weights of a previous model can be loaded.
def load_model(run, version):
    if run is None or version is None:
        print("Using untrained Model")
        return NeuralNetwork()

    neural_network = NeuralNetwork()
    m_tmp = neural_network.read(run, version)
    neural_network.model.set_weights(m_tmp.get_weights())
    print("Using model version "+str(version)+" of run "+str(run))
    return neural_network


# a certain amount of games are played between two players and the results are collected
def simulate_matches(player1, player2, episodes, det_, collect):
    game = Game(player1, player2)
    stats = {1: 0, -1: 0, 0: 0}
    for episode in range(episodes):
        print("Playing game #" + str(episode), end="\r")
        game.game_state = game.reset()
        # the search tree needs to be reset because of the high memory consumption
        game.player1.mcts = None
        game.player2.mcts = None
        winner = game.play(noprints=True, det_=det_, collect=collect)
        stats[winner] += 1
    # collecting action values if necessary
    if collect:
        for m in game.memory:
            memory.append(m)
    return stats, game.memory


# the main function
def connect4():
    # the command line arguments are read
    args = parse(sys.argv[1:])

    # for both players the player type can be selected via command line
    if 'x' in args:
        if args['x'] == "h":
            player1 = HumanPlayer()
        elif args['x'] == "a":
            xr = None
            xv = None
            if 'run' in args:
                xr = args['run']
            if 'ver' in args:
                xv = args['ver']
            player1 = AIPlayer(load_model(xr, xv))
        else:
            print("Error when selecting player type")
            player1 = HumanPlayer()
    else:
        # without any specification an AI player with a blank neural network is created for player 1
        neural_network = NeuralNetwork()
        player1 = AIPlayer(neural_network)

    if 'o' in args:
        if args['o'] == "h":
            player2 = HumanPlayer()
        elif args['o'] == "a":
            or_ = None
            ov = None
            if 'run' in args:
                or_ = args['run']
            if 'ver' in args:
                ov = args['ver']
            player2 = AIPlayer(load_model(or_, ov))
        else:
            print("Error when selecting player type")
            player2 = HumanPlayer()
    else:
        # without any specification, player2 is declared to be a human player
        player2 = HumanPlayer()

    # loading the loss progression if available - this is only used for visualisation purposes
    try:
        try:
            with open("run" + str(args['run']) + "/loss_history.p", "rb") as f:
                loss_array = pickle.load(f)
                player2.loss_array = loss_array
        except (FileNotFoundError, KeyError):
            # whenever the run argument is not specified, the run1 folder is used
            with open("run1/loss_history.p", "rb") as f:
                loss_array = pickle.load(f)
                player2.loss_array = loss_array
    except (FileNotFoundError, KeyError):
        pass

    # by specifying the "--mem" argument action values can be loaded
    global memory
    try:
        try:
            with open("run"+str(args['run'])+"/memory/memory" + str(args['mem']) + ".p", "rb") as f:
                memory = pickle.load(f)
                print("Using memory version " + str(args['mem']) + " of run " + str(args['run']))
        except (KeyError, FileNotFoundError):
            with open("run1/memory/memory" + str(args['mem']) + ".p", "rb") as f:
                memory = pickle.load(f)
                print("Using memory version " + str(args['mem']) + " of run " + str(args['run']))
    except (KeyError, FileNotFoundError):
        memory = deque(maxlen=50000)
        print("memory not loaded")

    # if both players are AIPlayers, games will be simulated until the program is stopped
    if isinstance(player1, AIPlayer) and isinstance(player2, AIPlayer):
        # selecting the current version number or starting from 0
        if 'ver' in args:
            nn_version = int(args['ver'])
        else:
            nn_version = 0

        session = 0
        while True:
            session += 1
            # the first part of the training is to let the 2 AIs play against each other to gather data
            print("Training player1 against itself")
            stats, _ = simulate_matches(player1, player1, 10, 10, True)
            print(str(session) + ": Player 1 won " + str(stats[1]) + " times, Player 2 won " + str(
                stats[-1]) + " times, there was a draw " + str(stats[0]) + " times")
            # if enough data is collected the data is replayed to the model of player2 in order to improve it
            if len(memory) >= 10000:
                print("Retraining player2 from memory")
                player2.replay(memory)
                # the loss is saved to a file
                with open("run1/loss_history.p", "wb") as f:
                    pickle.dump(player2.loss_array, f)

                # the two version are compared to each other by playing a certain amount of games
                print("Comparing retrained version...")
                stats, _ = simulate_matches(player1, player2, 5, 0, False)
                print(str(session) + ": Untrained version won " + str(stats[1]) + " times, Retrained version won " + str(
                    stats[-1]) + " times, there was a draw " + str(stats[0]) + " times")
                # if player2 outperforms player1 by a certain margin the weights of the model are copied to player1
                if stats[-1] > stats[1] * 1.3:
                    # when the weights are copied the current memory and the best weights are saved to memory
                    print("Duplicating retrained model!")
                    weights = player2.model.model.get_weights()
                    player1.model.model.set_weights(weights)
                    nn_version += 1
                    player1.model.write(nn_version)
                    try:
                        with open("Bachelorarbeit/run1/memory/memory" + str(nn_version) + ".p", "wb") as f:
                            pickle.dump(memory, f)
                    except FileNotFoundError:
                        with open("run1/memory/memory" + str(nn_version) + ".p", "wb") as f:
                            pickle.dump(memory, f)
            else:
                print(str(len(memory)) + " action values collected. Retraining model at 10000 action values")
    # if either player is a human player, games are played manually
    else:
        game = Game(player1, player2)
        start_time = time.time()
        game.play(det_=0, collect=False, sims=1000)
        end_time = time.time()
        time_in_sec = end_time - start_time
        formated_time = str(datetime.timedelta(seconds=time_in_sec))
        print("Time taken: " + formated_time)
        control = input("Enter [retry] to play another game or enter [quit] to exit: ")
        if control == "quit" or control == "q":
            exit(0)
        else:
            connect4()


connect4()
