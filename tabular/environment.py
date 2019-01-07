import copy
import frozendict
import random

from types import SimpleNamespace

import numpy as np

ACTIONS = frozendict.frozendict({
    0: 'top',
    1: 'right',
    2: 'bottom',
    3: 'left'
})

REPR = frozendict.frozendict({
    0: '\u25A1',
    1: '\u25A0',
    2: 'G',
    3: 'A'
})

DONE_REWARD = 9
STEP_REWARD = -1


class GridWorld:
    def __init__(self, world_config):
        '''
        Grid:
        0 = empty 
        1 = wall
        2 = goal
        3 = agent
        world is assumed to be surrounded by walls (that's a very sad sentence)
        '''
        self.grid_config = SimpleNamespace(**world_config)

        self.action_space = list(ACTIONS.keys())
        self.grid = self.make_grid()
        self.agent_pos = self.pick_start_position()

    def make_grid(self):
        '''
        init grid to zero, build external walls, then custom walls and finally the reward
        '''
        grid = np.zeros((self.grid_config.size), dtype=int)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        for wall_pos in self.grid_config.walls:
            grid[wall_pos[0], wall_pos[1]] = 1

        r_pos = self.grid_config.reward
        grid[r_pos[0], r_pos[1]] = 2

        return grid

    def pick_start_position(self):
        '''
        random start position
        '''
        x, y = np.where(self.grid == 0)
        index = np.random.choice(len(x))
        return (x[index], y[index])

    def display(self):
        tiles = list()
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.agent_pos == (i, j):
                    tiles.append(REPR[3])
                else:
                    tiles.append(REPR[self.grid[i, j]])
            tiles.append('\n')

        display_str = ''.join(tiles)
        print(display_str, end='')

    def get_observation(self):
        obs = copy.copy(self.grid)
        obs[self.agent_pos] = 3
        assert obs.shape == self.grid_config.size
        return obs

    def get_actions(self):
        '''
        top, right, bottom, left
        '''
        return self.action_space

    def reset(self):
        self.agent_pos = self.pick_start_position()
        return self.get_observation()

    def is_done(self):
        '''
        done when agent is on goal
        '''
        return self.agent_pos == np.where(self.grid == 2)

    def action(self, action):
        if self.is_done():
            raise Exception("Game is over")
        
        next_pos = self.compute_next_pos(action)

        self.agent_pos = copy.copy(next_pos)

        if self.grid[next_pos] == 2:
            return DONE_REWARD, self.get_observation(), True
        elif self.grid[next_pos] in [0, 3]:
            return STEP_REWARD, self.get_observation(), False
        else:
            print("NEXT POS %s" % str(next_pos))
            print(self.grid[next_pos])
            raise Exception("Invalid next pos")

    def compute_next_pos(self, action):
        next_pos = list(copy.copy(self.agent_pos))
        if action == 0:
            next_pos[0] -= 1
        elif action == 1:
            next_pos[1] += 1
        elif action == 2:
            next_pos[0] += 1
        elif action == 3:
            next_pos[1] -= 1
        else:
            raise Exception("Unknown action %s" % action)
        
        next_pos = tuple(next_pos)

        # if we hit a wall we stay in place
        if self.grid[next_pos] == 1:
            next_pos = copy.copy(self.agent_pos)

        assert next_pos[0] < self.grid_config.size[0]
        assert next_pos[1] < self.grid_config.size[1]
        return next_pos
        