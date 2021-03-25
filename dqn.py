import os
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, obs_dim:int, max_size:int, batch_size:int=32):
        self.obs_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([max_size], dtype=np.float32)
        self.rews_buf = np.zeros([max_size], dtype=np.float32)
        self.done_buf = np.zeros([max_size], dtype=np.float32)
        self.max_size, self.batch_size = max_size, batch_size
        self.ptr, self.size = 0, 0

    def store(self, obs, act, next_obs, rew, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act

        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = self.ptr % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self):
        idxs = np.random.choice(self.size, self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self):
        return self.size


class NetWork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NetWork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(self, env, memory_size, batch_size, target_update, epsilon_decay,
                 max_epsilon=1, min_epsilon=0.1, gamma=0.99):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device', self.device)
        self.dqn = NetWork(obs_dim, action_dim).to(self.device)
        self.dqn_target = NetWork(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=1e-3)
        self.transition = list()

        self.is_test = False

    def select_action(self, state):
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()

        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax().item()
           # print(type(selected_action))

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        if not self.is_test:
            self.transition += [next_state, reward, done]
            self.memory.store(*self.transition)

            return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch()
        loss = self._compute_dqn_loss(samples)
        self.optimizer.zero_grad()
        loss.backward()

        # for param in self.dqn.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss.item()

    def train(self, num_frames, plotting_interval=10000):
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames+1):
            action = self.select_action(state)
           # print('act: ', action)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                self.epsilon = max(self.min_epsilon,
                                   self.epsilon-(self.max_epsilon-self.min_epsilon)*self.epsilon_decay)

                epsilons.append(self.epsilon)

                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

                    # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)

        self.env.close()

    def test(self):
        self.is_test = True
        state = self.env.reset()
        done = False
        score = 0

        frames = []
        while not done:
            frames.append(self.env.render(mode='rgb_array'))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        return frames

    def _compute_dqn_loss(self, samples):
        device = self.device
        state = torch.FloatTensor(samples['obs']).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1- done
        target = (reward + self.gamma*next_q_value*mask).to(self.device)
        # loss = F.smooth_l2_loss(curr_q_value, target)
        loss_fn = nn.MSELoss()
        loss = loss_fn(curr_q_value, target)
        return loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
            self,
            frame_idx: int,
            scores,
            losses,
            epsilons,
    ):
        """Plot the training progresses."""
       #  clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# environment
env_id = "CartPole-v0"
env = gym.make(env_id)

seed = 0


np.random.seed(seed)
seed_torch(seed)
env.seed(seed)

num_frames = 200000
memory_size = 2000
batch_size = 50
target_update = 20
epsilon_decay = 1 / 20000

agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
agent.train(num_frames)












