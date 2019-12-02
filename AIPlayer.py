try:
    from .util import calculate_id, check_for_winner, valid_actions
    from .MCTS import Node, Edge, MCTS
except (ModuleNotFoundError, ImportError):
    from util import calculate_id, check_for_winner, valid_actions
    from MCTS import Node, Edge, MCTS
import numpy as np
import random
import math


class AIPlayer:
    def __init__(self, model):
        self.mcts = None
        self.model = model
        self.title = "AI"
        self.loss_array = []

    # letting the AI select a move
    def get_move(self, game_state, det, sims):
        # if only one move is available, this one is chosen
        allowed_actions = valid_actions(game_state.array)
        if len(allowed_actions) == 1:
            return allowed_actions[0], None

        # the given game state is set as root of the tree
        if self.mcts is None or game_state.id not in self.mcts.tree:
            self.mcts = MCTS(Node(game_state))
        else:
            self.mcts.root = self.mcts.tree[game_state.id]

        # simulate a number of games starting from the current game state to fill the Monte Carlo Search Tree
        for i in range(sims):
            leaf, chosen_path, new_game_state = self.mcts.simulate_game()

            # checking if the game finished after the simulation or if the end of the tree was reached
            if new_game_state is None or check_for_winner(new_game_state.array) is None:
                # if the game is not finished the model is used to evaluate the game state
                value, probabilities, allowed_actions = self.get_predictions(leaf.game_state)
                # the model also provides a probability distribution of the best move to take in this situation
                probabilities = probabilities[allowed_actions]

                # new edges and nodes are created at the leaf to expand the tree
                for idx, action in enumerate(allowed_actions):
                    new_game_state = leaf.game_state.take_action(action)
                    if new_game_state.id not in self.mcts.tree:
                        node = Node(new_game_state)
                        self.mcts.add_node(node)
                    else:
                        node = self.mcts.tree[new_game_state.id]
                    new_edge = Edge(leaf, node, probabilities[idx], action)
                    leaf.edges.append((action, new_edge))
            else:
                # if the game is finished, the model is not needed because the value of the game state is the result
                value = -1
                if check_for_winner(new_game_state.array) == 0:
                    value = 0

            # after the value of the game state is calculated, the chosen path of the Search Tree is updated
            self.mcts.back_propagation(leaf, chosen_path, value)

        q = np.zeros(42, dtype=np.float32)
        n = np.zeros(42, dtype=np.integer)
        # choosing the best move after the simulations
        for action, edge in self.mcts.root.edges:
            q[action] = edge.Q
            n[action] = edge.N
        # TODO mit .N austauschen wenn es nicht funktioniert
        n = n / (np.sum(n) * 1.0)
        # the values are normalized into a scale of 0 to 1
        allowed_actions = valid_actions(game_state.array)
        normalized = np.zeros(42, dtype=np.float64)
        for index in allowed_actions:
            normalized[index] = (q[index] - min(q)) / (max(q) - min(q))
        normalized = normalized / np.sum(normalized)
        # the selection can rarely lead to an error because of a prior rounding error
        try:
            # either the best move is chosen or a random one depending on whether the deterministic flag is set.
            if det:
                # one of the moves with the highest value is chosen
                actions = np.argwhere(normalized == max(normalized))
                action = random.choice(actions)[0]
            else:
                # semi-randomly selecting a move - the higher the value the more likely it is chosen
                normalized[allowed_actions[-1]] = normalized[allowed_actions[-1]] + (1 - np.sum(normalized))
                action_idx = np.random.multinomial(1, normalized)
                action = np.where(action_idx == 1)[0][0]
        except (ValueError, IndexError):
            # if the error occurs, simply a random allowed move is chose instead
            action = random.choice(allowed_actions)
        # TODO entfernen wenn alles geht
        #  next_game_state = game_state.take_action(action)
        #  next_game_state_value = -self.get_predictions(next_game_state)[0]
        return action, n

    # uses the neural network to get the value of the game state and the quality of the available moves
    def get_predictions(self, game_state):
        # the game state is the input of the model
        input_to_model = np.array([self.model.convert_to_model_input(game_state)])
        preds = self.model.predict(input_to_model)
        # output 1: estimated value of the board position
        value = preds[0][0]
        # output 2: logits (unnormalized predictions) of the best move to chose
        logits = preds[1][0]

        # filter out the relevant moves
        allowed_actions = valid_actions(game_state.array)
        # set all other values to negativ 100
        mask = np.ones(logits.shape, dtype=bool)
        mask[allowed_actions] = False
        logits[mask] = -100
        # SoftMax (value between 0 and 1 where the sum is 1) is applied to the logits in order to normalize the data
        odds = np.exp(logits)
        probs = odds / np.sum(odds)
        for i in range(len(probs)):
            # because of some rounding errors or exceptionally high/low values from the neural network
            if math.isnan(probs[i]):
                if logits[i] > 100:
                    probs[i] = 20
                elif logits[i] < -100:
                    probs[i] = -20
            else:
                probs[i] = int(probs[i] * 10000000) / 10000000

        return value, probs, allowed_actions

    # taking part of the collected action values to fit to model to their outcome - this is where the model learns
    def replay(self, memory):
        for i in range(10):
            sample = random.sample(memory, min(256, len(memory)))
            states = np.array([self.model.convert_to_model_input(mem['game_state']) for mem in sample])
            targets = {'value_head': np.array([mem['value'] for mem in sample]), 'policy_head': np.array([mem['action_values'] for mem in sample])}
            fit = self.model.fit(states, targets, epochs=1)
            # recording the loss
            self.loss_array.append([round(x, 4) for x in fit.history['loss']])

    # getting predictions from the model
    def predict(self, model_input):
        preds = self.model.predict(model_input)
        return preds
