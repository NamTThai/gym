import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from six import StringIO
import sys
import six


class Game2048Env(gym.Env):
    '''
    Game2048 environment.
    '''
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):
        self._seed()
        self.board_size = 4
        self.min_tile = 0
        self.max_tile = 11

        shape = [3, self.board_size, self.board_size]
        self.observation_space = spaces.Box(np.zeros(shape), np.ones(shape) * self.max_tile)
        self.action_space = spaces.Discrete(4)

        # Filled in by _reset()
        self.board = None

    def _reset(self):
        # Clear out the board and spawn 2 new tiles
        self.board = [[0 for r in xrange(self.board_size)]  for c in xrange(self.board_size)]
        self._set_new_tile()
        self._set_new_tile()
        return np.array(self.board)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self, mode="human", close=False):
        human_readable_board = np.array(self.board)
        for r, row in enumerate(human_readable_board):
            for c, cell in enumerate(row):
                human_readable_board[r][c] = 0 if cell == 0 else 2 ** cell
        print(human_readable_board)

    def _step(self, action):
        action = self._get_action_encoded(action)
        done = False

        # Rotate to board to so that the action becomes left
        rotated_board = np.rot90(self.board, action)

        # reward is the sum of all newly merged tiles
        reward = 0

        # Push everything to the left
        for r, row in enumerate(rotated_board):
            crashed_tile = -1
            for c, cell in enumerate(row):
                newC = c - 1
                while newC >= 0 and newC > crashed_tile:
                    if rotated_board[r][newC] == 0:
                        rotated_board[r][newC] = cell
                        rotated_board[r][newC+1] = 0
                    elif rotated_board[r][newC] == cell:
                        rotated_board[r][newC] += 1
                        reward += rotated_board[r][newC]
                        rotated_board[r][newC+1] = 0
                        crashed_tile = newC
                    else:
                        break
                    newC -= 1

        # Rotate the board back
        self.board = np.rot90(rotated_board, 4 - action).tolist()

        self._set_new_tile()

        # Check if the game is over
        done = self._is_game_over()

        return np.array(self.board), reward, done, {}

    @property
    def _board(self):
        return self.board

    def _get_action_encoded(self, action):
        if action == 'a':
            return 0
        elif action == 'w':
            return 1
        elif action == 'd':
            return 2
        elif action == 's':
            return 3

    def _has_empty_tile(self):
        """Check if there is any empty tile on the board"""
        for row in self.board:
            for cell in row:
                if cell == 0:
                    return True
        return False

    def _is_game_over(self):
        """The game is over under one of the 2 conditions:
        1. There is no legal action
        2. A 2048 tile has spawned
        """
        for row in self.board:
            for cell in row:
                if cell == 11:
                    return True
        for a in range(4):
            if self._is_legal_action(encoded_action=a):
                return False
        return True

    def _is_legal_action(self, encoded_action=0, rotated_board=None):
        """This is an utility function.
        It checks if moving left is a legal action.
        """
        if rotated_board is None:
            rotated_board = np.rot90(self.board, encoded_action)

        for r, row in enumerate(rotated_board):
            # This variable track the end of row. After this variable is set,
            # if there is any tile after it, then the action left can move
            # that tile and hence legal
            row_should_have_ended = False
            # This variable track the value of the previous tile. If this value
            # match the value of current tile, then they can merge and hence the
            # action is legal
            previous_cel_value = -1
            for c, cell in enumerate(row):
                # Check if the cells can be merged
                if cell == previous_cel_value and not cell == 0:
                    return True
                else:
                    previous_cel_value = cell

                # Check if the cells can be moved
                if cell == 0:
                    row_should_have_ended = True
                else:
                    if row_should_have_ended:
                        return True
        return False


    def _set_new_tile(self):
        """Generate a new tile and randomly placing it on the board. There is
        90% chance of getting tile 2 and 10% chance of getting tile 4
        """
        # Check if a new tile can be spawned
        if not self._has_empty_tile():
            return

        # 90% chance of spawning a 2^1, 10% chance of spawning a 2^2
        new_tile = 1 if self.np_random.rand() < 0.9 else 2
        while True:
            r = self.np_random.randint(0,4)
            c = self.np_random.randint(0,4)
            if self.board[r][c] == 0:
                self.board[r][c] = new_tile
                return
