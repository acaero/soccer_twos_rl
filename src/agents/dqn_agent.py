# https://arxiv.org/pdf/2111.05992 (Goal state)
# https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        # Output a Q-value for each action
        self.fc5 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)  # Output raw Q-values


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.003, gamma=0.99, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=5e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(
            self.qnetwork.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = []
        self.batch_size = 64

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 100000:
            self.memory.pop(0)


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random action for exploration from the set {0, 1, 2}
            return np.random.choice([0, 1, 2], 3).tolist()

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.qnetwork(state)

        q_values = q_values.squeeze().tolist()
        actions = []
        for q in q_values:
            if q > -1:
                actions.append(int(round(q)))
            else:
                actions.append(2)

        # Ensure actions are only from the set {0, 1, 2}
        valid_actions = [0, 1, 2]
        actions = [valid_actions[min(range(len(valid_actions)), key=lambda i: abs(
            valid_actions[i] - action))] for action in actions]

        return actions


    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute the current Q-values for the batch of states
        q_values = self.qnetwork(states)

        # Ensure actions tensor has the same number of dimensions as q_values
        state_action_values = q_values.gather(1, actions)

        # Compute the next Q-values for the batch of next states
        with torch.no_grad():
            next_q_values = self.qnetwork(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.criterion(state_action_values, targets)

        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
