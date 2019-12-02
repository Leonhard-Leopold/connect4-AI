try:
    from .util import display_game_state, floor_move
except (ModuleNotFoundError, ImportError):
    from util import display_game_state, floor_move


class HumanPlayer:
    def __init__(self):
        self.mcts = None
        self.title = "Human"

    # get a move from a human player by reading the column from the command line and flooring the move afterwards
    def get_move(self, game_state, det, sims):
        while True:
            action = int(input('Enter your chosen action from 1 (left) to 7 (right): ')) - 1
            action_floor = floor_move(action, game_state.array)
            if action_floor:
                break
            else:
                print("This row is filled")

        return action_floor, None