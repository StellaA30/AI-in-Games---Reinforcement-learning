################ Tabular model-free algorithms ################

model_based = __import__('tabular_model_based')
import numpy as np

def epsilon_greedy(epsilon, random_state, n_actions, q, state):
    if random_state.uniform(0, 1) < epsilon:
        # exploration, returns random action
        return random_state.randint(0, n_actions)
    else:
        max_value = np.max(q[state, :])  # finds maximum q-value of taking each action in a given state
        # creates a list of indices corresponding to the actions resulting in the max q-value
        max_indices = [index for index in range(n_actions) if q[state, index] == max_value]
        # returns the action which maximises q
        # (or randomly selects one from a set if there are multiple actions with the same q-value)
        return random_state.choice(max_indices)



def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)  # learning rate
    epsilon = np.linspace(epsilon, 0, max_episodes)  # exploration factor

    q = np.zeros((env.n_states, env.n_actions))  # initialise q-values
    counter = 0  # initialise episodes counter

    for i in range(max_episodes):  # looping through the episodes
        state = env.reset()
        # TODO:

        n_actions = env.n_actions   # number of actions
        # initialise action using epsilon-greedy for exploration-exploitation
        action = epsilon_greedy(epsilon[i], random_state, n_actions, q, state)

        terminal_state = False
        while not terminal_state:  # checks if terminal state is reached
            next_state, reward, terminal_state = env.step(action)
            # chooses next action according using epsilon-greedy
            next_action = epsilon_greedy(epsilon[i], random_state, n_actions, q, next_state)
            # update q-values according to sarsa algorithm
            q[state, action] += eta[i]*(reward + gamma*q[next_state, next_action] - q[state, action])
            state = next_state  # updates the state
            action = next_action  # updates the action
        counter += 1  # adds to counter after each episode
        policy = q.argmax(axis=1)
        value = q.max(axis=1)
        # computes value using policy evaluation
        policy_evaluation_value = model_based.policy_evaluation(env, policy, gamma, theta=0.001, max_iterations=100)
        # check if optimal policy has been reached by comparing value of sarsa with policy evaluation
        if all(abs(policy_evaluation_value[count] - value[count]) < 0.1 for count in range(len(policy_evaluation_value))):
            print('episodes:', counter)
            return policy, value

    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    print('episodes:', counter)


    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)  # learning rate
    epsilon = np.linspace(epsilon, 0, max_episodes)  # exploration factor

    q = np.zeros((env.n_states, env.n_actions))  # initialise q-values
    n_actions = env.n_actions  # number of actions
    counter = 0  # initialise episode counter

    for i in range(max_episodes):  # looping through the episodes
        state = env.reset()
        # TODO:

        terminal_state = False
        while not terminal_state:  # checks if terminal state is reached
            # selects action using epsilon-greedy for exploration-exploitation
            action = epsilon_greedy(epsilon[i], random_state, n_actions, q, state)
            next_state, reward, terminal_state = env.step(action)
            # update q-values according to q-learning algorithm
            q[state, action] += eta[i]*(reward + gamma*(np.max(q[next_state])) - q[state, action])
            state = next_state  # update state

        counter += 1  # adds to counter after each episode
        policy = q.argmax(axis=1)
        value = q.max(axis=1)
        # computes value using policy evaluation
        policy_evaluation_value = model_based.policy_evaluation(env, policy, gamma, theta=0.001, max_iterations=100)
        # check if optimal policy has been reached by comparing value of q-learning with policy evaluation
        if all(abs(policy_evaluation_value[count] - value[count]) < 0.1 for count in range(len(policy_evaluation_value))):
            print('episodes:', counter)
            return policy, value

    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    print('episodes:', counter)

    return policy, value
