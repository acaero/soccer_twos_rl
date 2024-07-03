import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_size * 3),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        action_probs = self.actor(state).view(-1, 3, 3)
        action_probs = F.softmax(action_probs, dim=2)
        state_value = self.critic(state)
        return action_probs, state_value


class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.002,
        gamma=0.99,
        epsilon=0.2,
        epochs=10,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.action_size = action_size

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
        action_probs = action_probs.cpu().numpy().squeeze()

        actions = []
        chosen_probs = []
        for probs in action_probs:
            action = np.random.choice(3, p=probs)
            actions.append(action)
            chosen_probs.append(
                probs[action]
            )  # Store the probability of the chosen action

        return np.array(actions), np.array(chosen_probs)

    def update(self, states, actions, old_probs, rewards, dones):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_probs = torch.FloatTensor(np.array(old_probs)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        for _ in range(self.epochs):
            action_probs, state_values = self.actor_critic(states)

            dist = Categorical(action_probs)
            new_probs = dist.log_prob(actions)

            advantages = rewards - state_values.detach()
            ratio = torch.exp(
                new_probs - torch.log(old_probs)
            )  # Convert old_probs to log probabilities
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values, rewards)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def save(self, filename):
        torch.save(
            {
                "actor_critic_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class PPOAgents:
    def __init__(
        self,
        num_agents,
        state_size,
        action_size,
        learning_rate=0.002,
        gamma=0.99,
        epsilon=0.2,
        epochs=10,
    ):
        self.num_agents = num_agents
        self.agents = [
            PPOAgent(state_size, action_size, learning_rate, gamma, epsilon, epochs)
            for _ in range(num_agents)
        ]

    def act(self, state):
        actions = {}
        probs = {}
        for i in range(self.num_agents):
            action, prob = self.agents[i].act(state[i])
            actions[i] = action
            probs[i] = prob
        return actions, probs

    def update(self, states, actions, old_probs, rewards, dones):
        for i in range(self.num_agents):
            self.agents[i].update(
                states[i], actions[i], old_probs[i], rewards[i], dones
            )

    def save(self, filename):
        checkpoint = {}
        for i, agent in enumerate(self.agents):
            checkpoint[f"agent_{i}"] = {
                "actor_critic_state_dict": agent.actor_critic.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
            }
        torch.save(checkpoint, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        for i, agent in enumerate(self.agents):
            agent_checkpoint = checkpoint[f"agent_{i}"]
            agent.actor_critic.load_state_dict(
                agent_checkpoint["actor_critic_state_dict"]
            )
            agent.optimizer.load_state_dict(agent_checkpoint["optimizer_state_dict"])
