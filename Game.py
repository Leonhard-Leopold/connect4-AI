import numpy as np
import random
from collections import deque
import HumanPlayer
import AIPlayer
try:
    from .util import check_for_winner, display_game_state, calculate_id, valid_actions
except (ModuleNotFoundError, ImportError) as e:
    from util import check_for_winner, display_game_state, calculate_id, valid_actions

class Game:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.game_state = GameState(np.array(np.zeros(42), dtype=np.int), 1)
        self.memory = deque(maxlen=10000)

    # reset the game state to start a new one
    def reset(self):
        self.game_state = GameState(np.array(np.zeros(42), dtype=np.int), 1)
        return self.game_state

    # play the game until it is finished
    def play(self, noprints=False, det_=0, collect=True, sims=50):
        if noprints is False:
            display_game_state(self.game_state.array)
        chosen_starter = random.randrange(2)
        if chosen_starter is 0:
            players = {1: self.player1, -1: self.player2}
        else:
            players = {1: self.player2, -1: self.player1}

        turn_counter = 0
        game_memory = deque(maxlen=10000)
        while True:
            turn_counter += 1
            # det = deterministic (whether the AI always chooses the best move)
            det = turn_counter >= det_

            if isinstance(players[self.game_state.active_player], AIPlayer.AIPlayer) and isinstance(players[-self.game_state.active_player], HumanPlayer.HumanPlayer):
                print("AI is choosing a move ...")

            # get the move from the current player (AI or Human)
            action, action_values = players[self.game_state.active_player].get_move(self.game_state, det, sims)

            # check if move is legal
            allowed_actions = valid_actions(self.game_state.array)
            if action not in allowed_actions:
                while True:
                    print("selected move is not allowed!", allowed_actions, action, action_values)
                    action, action_values = players[self.game_state.active_player].get_move(self.game_state, False, sims)
                    if action in allowed_actions:
                        break

            # when action values need to be collected
            if collect and action_values is not None:
                game_memory.append({'game_state': self.game_state, 'action_values': action_values})
                # the vertically flipped game state is also used
                flipped_action_values = np.array([np.fliplr([row]) for row in np.reshape(np.array(action_values), (-1, 7))]).flatten()
                flipped_game_state_array = np.array([np.fliplr([row]) for row in np.reshape(np.array(self.game_state.array), (-1, 7))]).flatten()
                flipped_game_state = GameState(flipped_game_state_array, self.game_state.active_player)
                game_memory.append({'game_state': flipped_game_state, 'action_values': flipped_action_values})

            # take the action and update the game state
            self.game_state = self.game_state.take_action(action)
            # display the game state if necessary
            if noprints is False:
                display_game_state(self.game_state.array)
            # check if the game is over (win or draw)
            winner = check_for_winner(self.game_state.array)
            if winner is not None:
                if noprints is False:
                    if winner == 1 or winner == -1:
                        print(players[-self.game_state.active_player].title + ' won')
                    else:
                        print("It is a draw")
                # if collecting data, the result of the game is added to later feed to the neural network
                if collect:
                    for m in game_memory:
                        if m['game_state'].active_player == winner:
                            m['value'] = 1
                        elif m['game_state'].active_player == -winner:
                            m['value'] = -1
                        else:
                            m['value'] = 0
                        self.memory.append(m)
                break

        # return the winning player for the statistics - it is separated by player instance and not by who started
        if winner == 0:
            return 0
        elif winner == 1:
            if chosen_starter == 0:
                return 1
            else:
                return -1
        elif winner == -1:
            if chosen_starter == 0:
                return -1
            else:
                return 1


class GameState:
    def __init__(self, game_state, active_player):
        self.array = game_state
        self.id = self._calc_id()
        self.active_player = active_player

    # calculate unique Id for the game state
    def _calc_id(self):
        return calculate_id(self.array)

    # applying a move the a game state and returning the new one with the other player as the active player
    def take_action(self, move):
        new_game_state = np.array(self.array)
        new_game_state[move] = self.active_player
        return GameState(new_game_state, (self.active_player * -1))
