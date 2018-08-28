import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import gym
from torch.autograd import Variable
import random
from collections import namedtuple

class DQN(nn.Module):
    def __init__(self, dueling=True):
        super().__init__()
        self.dueling = dueling
        self.fc1 = nn.Linear(4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        if dueling:
            self.v = nn.Linear(512, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.dueling:
            v = self.v(x)
            a = self.fc3(x)
            q = v + a
        else:
            q = self.fc3(x)
        return q
        
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Agent(object):
    def __init__(self, gamma=0.8, batch_size=128):
        self.target_Q = DQN()
        self.Q = DQN()
        self.gamma = gamma
        self.batch_size = 128
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.0001)
    
    def act(self, x, epsilon=0.1):
        p = random.uniform(0, 1)
        if p < epsilon :
            return Variable(torch.Tensor([int(np.round(random.uniform(0, 1)))])).type(torch.LongTensor)
        Q_sa = self.Q(x)
        argmax = Variable(Q_sa.data.max(0)[1]) 
        return argmax
    
    def backward(self, transitions):
        batch = Transition(*zip(*transitions))

        state = Variable(torch.cat(batch.state))
        action = Variable(torch.from_numpy(np.array(batch.action)))
        next_state = Variable(torch.cat(batch.next_state))
        reward = Variable(torch.cat(batch.reward))
        done = Variable(torch.from_numpy(np.array(batch.done)))

        Q_sa = self.Q(next_state).detach()
        target = self.target_Q(next_state).detach() 
        _, argmax = Q_sa.max(dim=1, keepdim=True)
        target = target.gather(1, argmax)

        currentQvalues = self.Q(state).gather(1,action.unsqueeze(1)).squeeze()
        y = (reward.unsqueeze(1) + self.gamma * (target * (1-done.unsqueeze(1)))).squeeze()

        loss = F.smooth_l1_loss(currentQvalues,y)  
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.target_Q, self.Q, 0.995)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

env = gym.make('CartPole-v0')
agent = Agent()
memory = ReplayMemory(100000)
batch_size = 128

epsilon = 1
rewards = []

for i in range(5000):
    obs = env.reset()
    done = False
    total_reward = 0
    epsilon *= 0.99
    while not done:
        if i > 4990:
            env.render()
        epsilon = max(epsilon, 0.1)
        obs_input = Variable(torch.from_numpy(obs).type(torch.FloatTensor))
        action = agent.act(obs_input, epsilon)
        next_obs, reward, done, _ = env.step(action.data.numpy()[0])
        memory.push(obs_input.data.view(1,-1), action.data, 
                    torch.from_numpy(next_obs).type(torch.FloatTensor).view(1,-1), torch.Tensor([reward]),
                   torch.Tensor([done]))
        obs = next_obs
        total_reward += reward
    rewards.append(total_reward)
    print("Episode : ", i)
    if memory.__len__() > 10000:
        batch = memory.sample(batch_size)
        agent.backward(batch)

pd.DataFrame(rewards).rolling(50, center=False).mean().plot()
plt.show()