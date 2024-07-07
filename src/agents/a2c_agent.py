import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, action_size * 3)
        self.action_size = action_size

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_logits = self.fc4(x).view(-1, self.action_size, 3)
        return action_logits


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class A2CAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.99,
        entropy_coef=0.01,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
        )
        self.num_agents = 1

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.action_size = action_size

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute advantage
        action_logits = self.actor(states)
        action_probs = F.softmax(action_logits, dim=-1)
        state_values = self.critic(states)
        next_state_values = self.critic(next_states)

        delta = rewards + self.gamma * next_state_values * (1 - dones) - state_values
        advantage = delta.detach()

        # Compute actor loss
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        actions_one_hot = F.one_hot(actions, num_classes=3).float()
        selected_action_log_probs = (action_log_probs * actions_one_hot).sum(dim=(1, 2))
        actor_loss = -(selected_action_log_probs * advantage.squeeze()).mean()

        # Compute critic loss
        critic_loss = F.mse_loss(
            state_values, rewards + self.gamma * next_state_values * (1 - dones)
        )

        # Compute entropy bonus
        entropy = -(action_probs * action_log_probs).sum(dim=(1, 2)).mean()

        # Total loss
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
                "critic_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
