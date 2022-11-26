import numpy as np
import contextlib
from itertools import product

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()

#
#
class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         # lake =  [['&', '.', '.', '.'],
         #          ['.', '#', '.', '#'],
         #          ['.', '.', '.', '#'],
         #          ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """

# creating a grip to be used in the frozen lake environment that takes input M which is a matrix
        def create_reward_grid(M):
            M = np.array(M)
            d1 = len(M)
            d2 = len(M[0])
            dim = [d1, d2]
            grid = np.zeros(dim)
            for i in range(d1):
                for j in range(d2):
                    if M[i, j] == '$':
                        grid[i, j] = 1
                    else:
                        grid[i, j] = 0
            return grid

        self.random_state = np.random.RandomState(seed)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.slip = slip
        n_states = self.lake.size + 1
        self.n_states = n_states
        n_actions = 4
        self.n_actions = n_actions

        self.max_steps = max_steps

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.pi = pi

        self.absorbing_state = n_states - 1

        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.grid = create_reward_grid(self.lake)
        #indices-State coordinates
        self.i_s_pairs = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        # add additional coordinates to represent the absorbing state, an out of grid coordination (-1,-1)
        self.i_s_pairs.append((-1, -1))
        # state-indices coordinates
        self.s_i_pairs = {state: index for (index, state) in enumerate(self.i_s_pairs)}

        self._p = np.zeros((self.n_states, self.n_states, self.n_actions))

        for state_index, state in enumerate(self.i_s_pairs):
            for action_index, action in enumerate(self.actions):
                if state_index == self.absorbing_state:
                    next_state == (-1, -1)
                    next_state_index = self.s_i_pairs.get(next_state, state_index)
                    self._p[next_state_index, state_index, action_index] = 1.0  # stay in absorb state if already in absorb state
                elif state_index != self.absorbing_state and (self.lake_flat[state_index] == '#' or self.lake_flat[state_index] == '$'):
                    next_state = (-1, -1)  # stay in position
                    next_state_index = self.s_i_pairs.get(next_state, state_index)  # to absorb state if at hole or reward
                    self._p[next_state_index, state_index, action_index] = 1.0
                else:  # absorb or reward state if not at hole
                    next_state = (state[0] + action[0], state[1] + action[1])  # update position

                    #store the next states
                    states = []
                    for i in range(n_actions):
                        next_state_i = (state[0] + self.actions[i][0], state[1] + self.actions[i][1])
                        states.append(next_state_i)


                    # If next_state is not valid, default to current state index
                    next_state_index = self.s_i_pairs.get(next_state, state_index)
                    state_indices = []
                    for j in range(n_actions):
                        next_index_j = self.s_i_pairs.get(states[j], state_index)
                        state_indices.append(next_index_j)


                    # define a probability
                    self._p[
                        state_indices[0], state_index, action_index] += self.slip / 4  # probabilty to slip in any of 4 directions
                    self._p[
                        state_indices[1], state_index, action_index] += self.slip / 4  # probabilty to slip in any of 4 directions
                    self._p[
                        state_indices[2], state_index, action_index] += self.slip / 4  # probabilty to slip in any of 4 directions
                    self._p[
                        state_indices[3], state_index, action_index] += self.slip / 4  # probabilty to slip in any of 4 directions

                    self._p[
                        next_state_index, state_index, action_index] += 1.0 - self.slip  # probabilty to go in intended direction + probability of ending up there due to slip

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done


    def p(self, next_state, state, action):
        return self._p[next_state, state, action]

    def r(self, next_state, state, action):
        if state != self.absorbing_state:
            return self.grid[self.i_s_pairs[state]]
        else:
            return 0

    def a(self, state):
        return self.actions

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:

            actions = ['^', '<', '_', '>']
            print('Lake: ')
            print(self.lake)

            print('Policy: ')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

    def play(env):
        actions = ['w', 'a', 's', 'd']

        state = env.reset()
        env.render()

        done = False
        while not done:
            c = input('\nMove: ')
            if c not in actions:
                raise Exception('Invalid Action')

            state, r, done = env.step(actions.index(c))

            env.render()
            print('Reward: {0}.'.format(r))

    def is_final(self, state):
        if state == self.absorbing_state:
            return True
        else:
            return False





