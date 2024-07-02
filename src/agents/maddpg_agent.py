import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class MultiAgentReplayBuffer:
    def __init__(self, buffer_size, batch_size, num_agents, state_size, action_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

    def add(self, states, actions, rewards, next_states, done):
        experience = (states, actions, rewards, next_states, done)
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences])

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


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
    def __init__(self, state_size, action_size, num_agents):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(
            state_size * num_agents + action_size * 3 * num_agents, 1024 * num_agents
        )
        self.fc2 = nn.Linear(1024 * num_agents, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, states, actions):
        x = torch.cat([states, actions * 3], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


class MADDPGAgents:
    def __init__(
        self,
        num_agents,
        state_size,
        action_size,
        lr_actor=0.002,
        lr_critic=0.002,
        gamma=0.99,
        tau=1e-3,
        buffer_size=10240,
        batch_size=1024,
    ):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        # Initialize actors and critics for each agent
        self.actors_local = [
            Actor(state_size, action_size).to(self.device) for _ in range(num_agents)
        ]
        self.actors_target = [
            Actor(state_size, action_size).to(self.device) for _ in range(num_agents)
        ]
        self.actors_optimizer = [
            optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors_local
        ]

        self.critics_local = [
            Critic(state_size, action_size, num_agents).to(self.device)
            for _ in range(num_agents)
        ]
        self.critics_target = [
            Critic(state_size, action_size, num_agents).to(self.device)
            for _ in range(num_agents)
        ]
        self.critics_optimizer = [
            optim.Adam(critic.parameters(), lr=lr_critic)
            for critic in self.critics_local
        ]

        self.memory = MultiAgentReplayBuffer(
            buffer_size, batch_size, num_agents, state_size, action_size
        )
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def act(self, obs, add_noise=True):
        actions = {}
        for i, (agent_id, state) in enumerate(obs.items()):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.actors_local[i].eval()
            with torch.no_grad():
                action_probs = self.actors_local[i](state).cpu().data.numpy().squeeze()
            self.actors_local[i].train()

            if add_noise:
                noise = np.random.normal(0, 0.1, size=action_probs.shape)
                action_probs += noise
                action_probs = np.clip(action_probs, 0, 1)
                action_probs /= action_probs.sum(axis=1, keepdims=True)  # Renormalize

            discrete_actions = np.argmax(action_probs, axis=1)
            actions[agent_id] = discrete_actions
        return actions

    def remember(self, obs, actions, rewards, next_obs, done):
        states = np.array([obs[agent_id] for agent_id in sorted(obs.keys())])
        next_states = np.array(
            [next_obs[agent_id] for agent_id in sorted(next_obs.keys())]
        )
        actions_list = np.array(
            [actions[agent_id] for agent_id in sorted(actions.keys())]
        )
        rewards_list = np.array(
            [rewards[agent_id] for agent_id in sorted(rewards.keys())]
        )
        done_all = done["__all__"]
        self.memory.add(states, actions_list, rewards_list, next_states, done_all)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        for i in range(self.num_agents):
            # Convert discrete actions to one-hot encoded form
            actions_one_hot = torch.zeros(
                self.batch_size, self.num_agents, self.action_size, 3
            ).to(self.device)
            actions_one_hot.scatter_(3, actions.long().unsqueeze(-1), 1)
            actions_one_hot = actions_one_hot.view(self.batch_size, -1)

            # Update critics
            next_actions = torch.cat(
                [
                    self.actors_target[j](next_states[:, j]).view(self.batch_size, -1)
                    for j in range(self.num_agents)
                ],
                dim=1,
            )
            q_targets_next = self.critics_target[i](
                next_states.view(self.batch_size, -1), next_actions
            )
            q_targets = rewards[:, i].unsqueeze(1) + (
                self.gamma * q_targets_next * (1 - dones)
            )

            q_expected = self.critics_local[i](
                states.view(self.batch_size, -1), actions_one_hot
            )
            critic_loss = nn.MSELoss()(q_expected, q_targets)

            self.critics_optimizer[i].zero_grad()
            critic_loss.backward()
            self.critics_optimizer[i].step()

            # Update actors
            actions_pred = [
                (
                    self.actors_local[j](states[:, j]).view(self.batch_size, -1)
                    if j == i
                    else actions_one_hot[
                        :, j * self.action_size * 3 : (j + 1) * self.action_size * 3
                    ].detach()
                )
                for j in range(self.num_agents)
            ]
            actions_pred = torch.cat(actions_pred, dim=1)

            actor_loss = -self.critics_local[i](
                states.view(self.batch_size, -1), actions_pred
            ).mean()

            self.actors_optimizer[i].zero_grad()
            actor_loss.backward()
            self.actors_optimizer[i].step()

            self.soft_update(self.critics_local[i], self.critics_target[i], self.tau)
            self.soft_update(self.actors_local[i], self.actors_target[i], self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * local_param.data
            )

    def save(self, filename):
        checkpoint = {
            "actors_local_state_dict": [
                actor.state_dict() for actor in self.actors_local
            ],
            "actors_target_state_dict": [
                actor.state_dict() for actor in self.actors_target
            ],
            "critics_local_state_dict": [
                critic.state_dict() for critic in self.critics_local
            ],
            "critics_target_state_dict": [
                critic.state_dict() for critic in self.critics_target
            ],
            "actors_optimizer_state_dict": [
                opt.state_dict() for opt in self.actors_optimizer
            ],
            "critics_optimizer_state_dict": [
                opt.state_dict() for opt in self.critics_optimizer
            ],
        }
        torch.save(checkpoint, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        for i in range(self.num_agents):
            self.actors_local[i].load_state_dict(
                checkpoint["actors_local_state_dict"][i]
            )
            self.actors_target[i].load_state_dict(
                checkpoint["actors_target_state_dict"][i]
            )
            self.critics_local[i].load_state_dict(
                checkpoint["critics_local_state_dict"][i]
            )
            self.critics_target[i].load_state_dict(
                checkpoint["critics_target_state_dict"][i]
            )
            self.actors_optimizer[i].load_state_dict(
                checkpoint["actors_optimizer_state_dict"][i]
            )
            self.critics_optimizer[i].load_state_dict(
                checkpoint["critics_optimizer_state_dict"][i]
            )
