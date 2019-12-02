import numpy as np
try:
    from .Game import Game
    from .util import calculate_id, check_for_winner
except (ModuleNotFoundError, ImportError):
    from Game import Game
    from util import calculate_id, check_for_winner


class MCTS:
    def __init__(self, root):
        self.tree = {}
        self.root = root
        self.add_node(root)

    # add node to tree
    def add_node(self, node):
        self.tree[node.id] = node

    # simulate a game to evaluate the game state until the end of the tree is reached
    def simulate_game(self):
        current = self.root
        # the chosen path is recorded for the back propagation
        chosen_path = []
        new_game_state = None
        while len(current.edges) != 0:
            # the best move in the situation is chosen - exploration and exploitation are considered
            best_node_value = None
            # exploring only at the root because all of the roots edges need to be evaluated
            if current == self.root:
                # controls exploration likelihood
                epsilon = 0.2
                # Dirichlet introduces noise into the exploration (random value distribution totaling to 1)
                noise = np.random.dirichlet([0.8] * len(current.edges))
            else:
                # if we are not at the root the best move is simply chosen
                epsilon = 0
                noise = np.zeros(len(current.edges), dtype=int)

            # the sum of the visit count of all edges is calculated for the exploration formula
            n_sum = 0
            for action, edge in current.edges:
                n_sum += edge.N

            best_edge = None
            best_action = None
            # all edges are analyzed to find the best move
            for index, (action, edge) in enumerate(current.edges):
                # this formula calculates the exploration chance - improved by a low visit count compared to the sum
                # therefore at the beginning the focus is on exploration, later on exploitation
                exploration_chance = (((1-epsilon) * edge.P) + (epsilon * noise[index])) * np.sqrt(n_sum) / (1 + edge.N)

                # the quality of a node is selected by its average value
                node_quality = edge.Q

                # the best node in regard to quality and necessity of exploration is selected
                overall_node_quality = node_quality + exploration_chance
                if best_node_value is None or overall_node_quality > best_node_value:
                    best_node_value = overall_node_quality
                    best_edge = edge
                    best_action = action

            # the move is applied
            chosen_path.append(best_edge)
            new_game_state = current.game_state.take_action(best_action)
            current = best_edge.out_node
        return current, chosen_path, new_game_state

    # after the value of the leaf node is calculated, all nodes in the selected path are updated
    def back_propagation(self, end_node, chosen_path, result):
        for edge in chosen_path:
            if edge.active_player == end_node.game_state.active_player:
                reward_multiplier = 1
            else:
                reward_multiplier = -1
            # the visit count is increased by one
            edge.N += 1
            # W is increased by the result (either the final game state value or one estimated by the model)
            edge.W = edge.W + (result * reward_multiplier)
            # Q is updated to represent the new values
            edge.Q = edge.W / edge.N


# a representation of a game state in the Monte Carlo Search Tree
class Node:
    def __init__(self, game_state):
        self.game_state = game_state
        self.active_player = game_state.active_player
        self.edges = []
        self.id = game_state.id


# a move leading from one game state to another
class Edge:
    # a new edge is initialized with 0 in N, W and Q and P as the probability estimated by the model
    def __init__(self, in_node, out_node, prob, action):
        self.in_node = in_node
        self.out_node = out_node
        self.prob = prob
        self.action = action
        self.active_player = in_node.game_state.active_player
        self.N = 0  # visit count
        self.W = 0  # sum of P of all prior moves
        self.Q = 0  # average value of each prior move -> Q= W/N -> the more moves taken the more accurate Q is
        self.P = prob  # quality of the game state after taking this move

    # creates an Id for the Search Tree
    def get_id(self):
        return self.in_node.game_state.id + '|' + self.out_node.game_state.id
