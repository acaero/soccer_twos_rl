import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# Define the QNetwork
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(
            64, action_size * 3
        )  # 3 options for each of the 3 dimensions

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x).view(
            -1, 3, 3
        )  # Reshape to (batch_size, 3 dimensions, 3 options)


# Define the DDQN Agent
class DDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.002,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        batch_size=1024,
        buffer_size=10240,
        tau=1e-3,
    ):
        self.num_agents = 1
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    # update both q networks only after all batches
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([0, 1, 2], 3).tolist()

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        actions = q_values.max(2)[1].squeeze().tolist()
        return actions

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.qnetwork_local(states)
        state_action_values = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)

        next_state_actions = self.qnetwork_local(next_states).max(2)[1].unsqueeze(2)
        next_state_values = (
            self.qnetwork_target(next_states).gather(2, next_state_actions).squeeze(2)
        )

        expected_state_action_values = rewards + (
            self.gamma * next_state_values * (1 - dones)
        )

        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, filename):
        torch.save(
            {
                "qnetwork_local_state_dict": self.qnetwork_local.state_dict(),
                "qnetwork_target_state_dict": self.qnetwork_target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.qnetwork_local.load_state_dict(checkpoint["qnetwork_local_state_dict"])
        self.qnetwork_target.load_state_dict(checkpoint["qnetwork_target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
