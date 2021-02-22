import numpy as np
import gym


def policy_evaluation(env, policy, gamma):
    v = np.zeros(env.env.nS)
    delta = 1e-10
    while True:
        old_v = np.copy(v)
        for s in range(env.env.nS):
            state_action = policy[s]
            v[s] = sum([p*(r+gamma*old_v[s_]) for p, s_, r, _ in env.env.P[s][state_action]])
        if np.sum((np.fabs(old_v - v))) <= delta:
            break

    return v


def policy_improvement(env, gamma, current_value):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_a_list = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_a_list[a] = sum([p*(r+gamma*current_value[s_]) for p, s_, r, _ in env.env.P[s][a]])
        policy[s] = np.argmax(q_a_list)
    return policy


def policy_iteration(env, gamma=1.0):
    policy = np.random.choice(env.env.nA, size=(env.env.nS))  # initialize a random policy
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        current_value = policy_evaluation(env, policy, gamma)
        new_policy = policy_improvement(env, gamma, current_value)
        if (np.all(new_policy == policy)):
            print("Iteration converged at step ", i+1)
            break
        policy = new_policy
    return policy


def run_episode(env, policy, gamma=1.0, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += reward
        step_idx += 1
        if done:
            break
    return total_reward


def test_policy(env, optimal_policy, gamma=1.0, test_num=10):
    scores = [run_episode(env, optimal_policy, gamma, True) for _ in range(test_num)]
    print(np.mean(scores))


if __name__ == "__main__":
    env_name = 'FrozenLake-v0'  # 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    env.reset()
    #env.render()

    optimal_policy = policy_iteration(env, gamma=1.0)
    test_policy(env, optimal_policy, gamma=1.0, test_num=10)