import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size * 3),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action_logits = self.actor(state)
        action_probs = F.softmax(action_logits.view(state.size(0), 3, 3), dim=2)
        state_value = self.critic(state)
        return action_probs, state_value


class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.0003,
        gamma=0.99,
        clip_size=0.2,
        epochs=10,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = clip_size
        self.epochs = epochs
        self.num_agents = 1

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
        action_probs = action_probs.squeeze(0).cpu().numpy()
        actions = []
        chosen_probs = []
        for probs in action_probs:
            probs = np.clip(probs, 1e-10, 1.0)  # Clip probabilities to avoid zeros
            probs /= probs.sum()  # Renormalize
            action = np.random.choice(3, p=probs)
            actions.append(action)
            chosen_probs.append(probs[action])
        return np.array(actions).flatten(), np.array(chosen_probs).flatten()

    def update(self, states, actions, old_probs, rewards, dones):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_probs = torch.FloatTensor(np.array(old_probs)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        returns = self.compute_returns(rewards, dones)

        for _ in range(self.epochs):
            action_probs, state_values = self.actor_critic(states)
            action_probs = F.softmax(action_probs.view(-1, 3), dim=-1)

            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                print("NaN or Inf detected in action_probs")
                action_probs = torch.nan_to_num(
                    action_probs, nan=1e-6, posinf=1 - 1e-6, neginf=1e-6
                )

            dist = torch.distributions.Categorical(action_probs)

            actions_reshaped = actions.view(-1)
            new_probs = dist.log_prob(actions_reshaped)
            old_probs_reshaped = old_probs.view(-1)

            ratio = torch.exp(new_probs - old_probs_reshaped)
            ratio = ratio.view(states.size(0), -1)

            advantages = returns - state_values.squeeze(-1).detach()
            advantages_expanded = advantages.unsqueeze(1).expand_as(ratio)

            surr1 = ratio * advantages_expanded
            surr2 = (
                torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                * advantages_expanded
            )

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(-1), returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.tensor(returns).to(self.device)

    def save(self, filename):
        torch.save(self.actor_critic.state_dict(), filename)

    def load(self, filename):
        self.actor_critic.load_state_dict(torch.load(filename))
