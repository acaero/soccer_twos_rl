import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


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
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_logits = self.fc4(x).view(
            -1, 3, 3
        )  # Reshape to (batch_size, 3 actions, 3 possibilities)
        return action_logits


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action.view(action.size(0), -1)], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class SACAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.002,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic1 = Critic(state_size, action_size).to(self.device)
        self.critic2 = Critic(state_size, action_size).to(self.device)
        self.critic1_target = Critic(state_size, action_size).to(self.device)
        self.critic2_target = Critic(state_size, action_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def update(self, states, actions, rewards, next_states, dones):
        if len(states) == 0:
            return

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute the target Q value
        with torch.no_grad():
            next_action_probs = self.actor(next_states)
            next_action_probs = F.softmax(next_action_probs, dim=-1)
            next_action = torch.argmax(next_action_probs, dim=-1)
            next_action_one_hot = F.one_hot(next_action, num_classes=3).float()

            next_q1 = self.critic1_target(next_states, next_action_one_hot)
            next_q2 = self.critic2_target(next_states, next_action_one_hot)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute the current Q value
        current_q1 = self.critic1(states, F.one_hot(actions, num_classes=3).float())
        current_q2 = self.critic2(states, F.one_hot(actions, num_classes=3).float())

        # Compute critic loss
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Compute actor loss
        action_probs = self.actor(states)
        action_probs = F.softmax(action_probs, dim=-1)
        action_log_probs = torch.log(action_probs + 1e-8)

        q1 = self.critic1(states, action_probs)
        q2 = self.critic2(states, action_probs)
        min_q = torch.min(q1, q2)

        actor_loss = (self.alpha * action_log_probs - min_q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_logits = self.actor(state)
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.argmax(action_probs, dim=-1)
        return action.squeeze().cpu().numpy().tolist()

    def save(self, filename):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(
            checkpoint["critic1_optimizer_state_dict"]
        )
        self.critic2_optimizer.load_state_dict(
            checkpoint["critic2_optimizer_state_dict"]
        )


class SACAgents:
    def __init__(
        self,
        num_agents,
        state_size,
        action_size,
        learning_rate=0.002,
        gamma=0.99,
        tau=1e-3,
        alpha=0.2,
    ):
        self.num_agents = num_agents
        self.agents = [
            SACAgent(state_size, action_size, learning_rate, gamma, tau, alpha)
            for _ in range(num_agents)
        ]

    def act(self, states):
        return {i: self.agents[i].act(states[i]) for i in range(self.num_agents)}

    def update(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.agents[i].update(
                states[i], actions[i], rewards[i], next_states[i], dones
            )

    def save(self, filename):
        checkpoint = {}
        for i, agent in enumerate(self.agents):
            checkpoint[f"agent_{i}"] = {
                "actor_state_dict": agent.actor.state_dict(),
                "critic1_state_dict": agent.critic1.state_dict(),
                "critic2_state_dict": agent.critic2.state_dict(),
                "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
                "critic1_optimizer_state_dict": agent.critic1_optimizer.state_dict(),
                "critic2_optimizer_state_dict": agent.critic2_optimizer.state_dict(),
            }
        torch.save(checkpoint, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        for i, agent in enumerate(self.agents):
            agent_checkpoint = checkpoint[f"agent_{i}"]
            agent.actor.load_state_dict(agent_checkpoint["actor_state_dict"])
            agent.critic1.load_state_dict(agent_checkpoint["critic1_state_dict"])
            agent.critic2.load_state_dict(agent_checkpoint["critic2_state_dict"])
            agent.actor_optimizer.load_state_dict(
                agent_checkpoint["actor_optimizer_state_dict"]
            )
            agent.critic1_optimizer.load_state_dict(
                agent_checkpoint["critic1_optimizer_state_dict"]
            )
            agent.critic2_optimizer.load_state_dict(
                agent_checkpoint["critic2_optimizer_state_dict"]
            )
