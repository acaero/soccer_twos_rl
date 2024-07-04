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
        # Ensure state is 2D: (batch_size, state_size)
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
        self.num_agents = 1

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)

        action_probs = action_probs.squeeze(0).cpu().numpy()
        actions = []
        chosen_probs = []
        for probs in action_probs:
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

        for _ in range(self.epochs):
            action_probs, state_values = self.actor_critic(states)
            dist = torch.distributions.Categorical(action_probs.view(-1, 3))

            # Reshape actions to match the distribution
            actions_reshaped = actions.view(-1)

            new_probs = dist.log_prob(actions_reshaped)
            old_probs_reshaped = old_probs.view(-1)

            advantages = rewards - state_values.squeeze(-1).detach()

            ratio = torch.exp(new_probs - torch.log(old_probs_reshaped))

            # Reshape ratio and advantages to match
            ratio = ratio.view(states.size(0), -1)  # Should be (64, 3)
            advantages = advantages.unsqueeze(1).expand_as(ratio)  # Should be (64, 3)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(-1), rewards)

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
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "epochs": self.epochs,
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.gamma = checkpoint["gamma"]
        self.epochs = checkpoint["epochs"]
