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
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


class DDPGAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.002,
        gamma=0.99,
        batch_size=1024,
        memory_buffer=10240,
        tau=1e-3,
    ):
        self.num_agents = 1
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        self.actor_local = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=learning_rate
        )

        self.critic_local = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=learning_rate
        )

        self.gamma = gamma
        self.tau = tau

        self.memory = []
        self.memory_size = memory_buffer
        self.batch_size = batch_size

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action_probs = self.actor_local(state).cpu().data.numpy().squeeze()
        self.actor_local.train()

        # Add noise and renormalize
        noise = np.random.normal(0, 0.1, size=action_probs.shape)
        action_probs += noise
        action_probs = np.clip(action_probs, 0, 1)
        action_probs /= action_probs.sum(axis=1, keepdims=True)  # Renormalize

        discrete_actions = np.argmax(action_probs, axis=1)
        return discrete_actions

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Convert discrete actions to one-hot encoded form
        actions_one_hot = torch.zeros(self.batch_size, self.action_size, 3).to(
            self.device
        )
        actions_one_hot.scatter_(2, actions.long().unsqueeze(-1), 1)
        actions_one_hot = actions_one_hot.view(self.batch_size, -1)

        # Update Critic
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(
            next_states, next_actions.view(self.batch_size, -1)
        )
        q_targets = rewards + (self.gamma * next_q_values * (1 - dones))

        q_expected = self.critic_local(states, actions_one_hot)
        critic_loss = nn.MSELoss()(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(
            states, actions_pred.view(self.batch_size, -1)
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        return critic_loss.item()

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
                "actor_local_state_dict": self.actor_local.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_local_state_dict": self.critic_local.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor_local.load_state_dict(checkpoint["actor_local_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_local.load_state_dict(checkpoint["critic_local_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
