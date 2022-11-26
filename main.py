# environment = __import__('frozen_lake')
environment = __import__('environment')
tabular_model_free = __import__('tabular_model_free')
tabular_model_based = __import__('tabular_model_based')
non_tabular = __import__('Non-tabular_model-free_algorithms')

def main():
    seed = 0

    # Small frozen lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    # #Big frozen lake
    # lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
    #             ['.', '.', '.', '.', '.', '.', '.', '.'],
    #             ['.', '.', '.', '#', '.', '.', '.', '.'],
    #             ['.', '.', '.', '.', '.', '#', '.', '.'],
    #             ['.', '.', '.', '#', '.', '.', '.', '.'],
    #             ['.', '#', '#', '.', '.', '.', '#', '.'],
    #             ['.', '#', '.', '.', '#', '.', '#', '.'],
    #             ['.', '.', '.', '#', '.', '.', '.', '$']]

    env = environment.FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print('')

    print('## Policy iteration')
    policy, value = tabular_model_based.policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = tabular_model_based.value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    print('')

    print('## Sarsa')
    policy, value = tabular_model_free.sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    print('## Q-learning')
    policy, value = tabular_model_free.q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    linear_env = non_tabular.LinearWrapper(env)

    print('## Linear Sarsa')

    parameters = non_tabular.linear_sarsa(linear_env, max_episodes, eta,
                              gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')

    parameters = non_tabular.linear_q_learning(linear_env, max_episodes, eta,
                                   gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)


x = main()
