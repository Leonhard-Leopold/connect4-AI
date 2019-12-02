import numpy as np


# prints the game state in a more readable format
def display_game_state(game_state):
    pieces = {'1': 'X', '0': '-', '-1': 'O'}
    for r in range(6):
        print([pieces[str(x)] for x in game_state[7 * r: (7 * r + 7)]])
    print("\n")


def check_for_winner(game_state):
    # with less than 7 stones on the field there never is a winner
    if np.count_nonzero(game_state) < 7:
        return None

    # the game state is transformed into an 2D array
    game_state_2d = game_state.reshape((6, -1))

    # all possible position where 4 stones could be found in a row are checked
    for x in range(0, 4):
        # horizontal
        for y in range(0, 6):
            if game_state_2d[y][x] != 0 and game_state_2d[y][x] == game_state_2d[y][x + 1] and game_state_2d[y][x] == \
                    game_state_2d[y][x + 2] and game_state_2d[y][x] == game_state_2d[y][x + 3]:
                return game_state_2d[y][x]
        # \
        for y in range(3, 6):
            if game_state_2d[y][x] != 0 and game_state_2d[y][x] == game_state_2d[y - 1][x + 1] and game_state_2d[y][
                x] == \
                    game_state_2d[y - 2][x + 2] and game_state_2d[y][x] == game_state_2d[y - 3][x + 3]:
                return game_state_2d[y][x]
        # /
        for y in range(0, 3):
            if game_state_2d[y][x] != 0 and game_state_2d[y][x] == game_state_2d[y + 1][x + 1] and game_state_2d[y][
                x] == \
                    game_state_2d[y + 2][x + 2] and game_state_2d[y][x] == game_state_2d[y + 3][x + 3]:
                return game_state_2d[y][x]

        # vertical
    for x in range(0, 7):
        for y in range(0, 3):
            if game_state_2d[y][x] != 0 and game_state_2d[y][x] == game_state_2d[y + 1][x] and game_state_2d[y][x] == \
                    game_state_2d[y + 2][x] and game_state_2d[y][x] == game_state_2d[y + 3][x]:
                return game_state_2d[y][x]

    # When the board is full the game ends in a draw
    if np.count_nonzero(game_state) == 42:
        return 0

    return None


# check where all legal moves are
def valid_actions(game_state):
    allowed = []
    for i in range(len(game_state)):
        if i >= len(game_state) - 7:
            if game_state[i] == 0:
                allowed.append(i)
        else:
            if game_state[i] == 0 and game_state[i + 7] != 0:
                allowed.append(i)

    return allowed


# create an ID for a game state
def calculate_id(game_state, array=False):
    player1_position = np.zeros(len(game_state), dtype=np.int)
    player1_position[game_state == 1] = 1
    other_position = np.zeros(len(game_state), dtype=np.int)
    other_position[game_state == -1] = 1
    position = np.append(player1_position, other_position)
    if array:
        return position
    return ''.join(map(str, position))


# a move is lowered to the first available spot in each column - simulating the stone falling down when playing the game
def floor_move(input_column, game_state):
    allowed_actions = valid_actions(game_state)
    for i in range(6):
        if (input_column + (i * 7)) in allowed_actions:
            return input_column + (i * 7)
