
import numpy as np


def epsilon_greedy(epsilon, random_state, n_actions, q):
    if random_state.uniform(0, 1) < epsilon:
        # exploration, returns random action
        return random_state.randint(0, n_actions)
    else:
        # returns the action which maximises q
        # (or randomly selects one from a set of actions if there are multiple actions with the same q-value)
        return random_state.choice(np.flatnonzero(q == np.max(q)))

class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        n_actions = env.n_actions

        a = epsilon_greedy(epsilon[i], random_state, n_actions, q)
        terminal = False
        while not terminal:
            next_s, r, terminal = env.step(a)
            delta = r - q[a]
            q = next_s.dot(theta)

            a_new = epsilon_greedy(epsilon[i], random_state, n_actions, q)

            delta = delta + (gamma * max(q))
            theta = theta + eta[i] * delta * features[a]
            features = next_s
            a = a_new

    return theta
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        terminal = False

        while not terminal:
            n_actions = env.n_actions
            a = epsilon_greedy(epsilon[i], random_state, n_actions, q)
            next_s, r, terminal = env.step(a)
            delta = r - q[a]
            q = next_s.dot(theta)
            delta = delta + (gamma * max(q))
            theta = theta + (eta[i] * delta * features[a])
            features = next_s
    return theta