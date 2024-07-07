from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(
            64, action_size * 3
        )  # 3 actions, each with 3 possibilities

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.softmax(
            self.fc4(x).view(-1, 3, 3), dim=2
        )  # Apply softmax to each group of 3


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size * 3, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class DDPGAgent:
    def __init__(
        self,
        state_size,
        action_size,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        gamma=0.99,
        tau=1e-3,
        buffer_size=100000,
        batch_size=64,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agents = 1

        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )

        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate
        )

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Initialize target networks weights with the main networks weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_buffer = deque(maxlen=buffer_size)

    def act(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action_probs = self.actor(state).cpu().data.numpy().squeeze()
        self.actor.train()

        if add_noise:
            noise = np.random.normal(0, 0.1, size=action_probs.shape)
            action_probs += noise
            action_probs = np.clip(action_probs, 0, 1)
            action_probs /= action_probs.sum(axis=1, keepdims=True)

        discrete_actions = np.argmax(action_probs, axis=1)
        return discrete_actions.tolist()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Convert discrete action to one-hot encoded form
        action_one_hot = torch.zeros(self.batch_size, 3, 3).to(self.device)
        for i in range(3):
            action_one_hot[:, i, action[:, i]] = 1
        action_one_hot = action_one_hot.view(self.batch_size, -1)

        # Update Critic
        next_action_probs = self.actor_target(next_state)
        next_q = self.critic_target(
            next_state, next_action_probs.view(self.batch_size, -1)
        )
        q_target = reward + (self.gamma * next_q * (1 - done))
        q_pred = self.critic(state, action_one_hot)
        critic_loss = nn.MSELoss()(q_pred, q_target.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(
            state, self.actor(state).view(self.batch_size, -1)
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filename):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
