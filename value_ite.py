import numpy as np
import gym

def value_iteration(env, gamma=1.0):
    v = np.zeros(env.env.nS)
    max_ite_num = 100000
    delta = 1e-10
    for i in range(max_ite_num):
        old_v = np.copy(v)
        for s in range(env.env.nS):
            q_a_list = np.zeros(env.env.nA)
            for a in range(env.env.nA):
                q_a_list[a] = sum([p*(r + gamma * old_v[s_]) for p, s_, r, _ in env.env.P[s][a]])
            v[s] = np.max(q_a_list)

        if np.sum(np.fabs(v - old_v)) < delta:
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break

    return v
#
#
def extract_policy(v, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

#
#
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


def my_test_policy(env, optimal_policy, gamma=1.0, test_num=10):
    scores = [run_episode(env, optimal_policy, gamma, True) for _ in range(test_num)]
    print(np.mean(scores))


if __name__ == "__main__":
    env_name = 'FrozenLake-v0'  # 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    env.reset()
    env.render()

    v = value_iteration(env, gamma=1.0)
    optimal_policy = extract_policy(v, gamma=1.0)
    my_test_policy(env, optimal_policy, gamma=1.0, test_num=10)