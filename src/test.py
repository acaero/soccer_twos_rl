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
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return torch.tanh(self.fc5(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, num_agents):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size * num_agents + action_size * num_agents, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


class MADDPGAgent:
    def __init__(
        self,
        num_agents,
        state_size,
        action_size,
        lr_actor=0.001,
        lr_critic=0.001,
        gamma=0.99,
        tau=1e-3,
        buffer_size=10000,
        batch_size=64,
    ):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                action = self.actors_local[i](state).cpu().data.numpy().flatten()
            self.actors_local[i].train()
            if add_noise:
                action += np.random.normal(0, 0.1, size=self.action_size)
            actions[agent_id] = np.clip(action, -1, 1)
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

        print(f"States shape: {states.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Rewards shape: {rewards.shape}")
        print(f"Next states shape: {next_states.shape}")
        print(f"Dones shape: {dones.shape}")

        for i in range(self.num_agents):
            # Update critics
            next_actions = torch.cat(
                [
                    self.actors_target[j](next_states[:, j])
                    for j in range(self.num_agents)
                ],
                dim=1,
            )
            q_targets_next = self.critics_target[i](
                next_states.reshape(self.batch_size, -1), next_actions
            )
            q_targets = rewards[:, i].unsqueeze(1) + (
                self.gamma * q_targets_next * (1 - dones)
            )

            q_expected = self.critics_local[i](
                states.reshape(self.batch_size, -1),
                actions.reshape(self.batch_size, -1),
            )
            critic_loss = nn.MSELoss()(q_expected, q_targets)

            self.critics_optimizer[i].zero_grad()
            critic_loss.backward()
            self.critics_optimizer[i].step()

            # Update actors
            actions_pred = [
                self.actors_local[j](states[:, j]) if j == i else actions[:, j].detach()
                for j in range(self.num_agents)
            ]
            actions_pred = torch.cat(actions_pred, dim=1)

            actor_loss = -self.critics_local[i](
                states.reshape(self.batch_size, -1), actions_pred
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


def exploratory_test_maddpg(n_episodes=10):
    import soccer_twos
    from src.utils import RewardShaper

    env = soccer_twos.make(render=True)
    reward_shaper = RewardShaper()

    # Initialize MADDPG agents
    num_agents = 4
    state_size = 336
    action_size = 3
    maddpg_agent = MADDPGAgent(num_agents, state_size, action_size)

    # Collect initial experiences
    initial_experiences = 1000
    experiences_count = 0

    while experiences_count < initial_experiences:
        obs = env.reset()
        done = False
        while not done:
            actions = maddpg_agent.act(obs)
            next_obs, rewards, dones, info = env.step(actions)
            done = dones["__all__"]

            maddpg_agent.remember(obs, actions, rewards, next_obs, dones)
            experiences_count += 1

            obs = next_obs
            if experiences_count >= initial_experiences:
                break

    print(f"Collected {experiences_count} initial experiences")

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_rewards = {agent_id: 0 for agent_id in obs.keys()}
        while not done:
            actions = maddpg_agent.act(obs)
            next_obs, rewards, dones, info = env.step(actions)
            done = dones["__all__"]

            # Calculate and log rewards
            shaped_rewards = {}
            for agent_id in obs.keys():
                shaped_reward = reward_shaper.calculate_reward(
                    obs[agent_id], next_obs[agent_id], info, int(agent_id)
                )
                shaped_rewards[agent_id] = shaped_reward
                total_rewards[agent_id] += rewards[agent_id] + shaped_reward

            maddpg_agent.remember(obs, actions, rewards, next_obs, dones)
            maddpg_agent.replay()

            obs = next_obs

        print(f"Episode {episode + 1}/{n_episodes} - Total Rewards: {total_rewards}")

    env.close()


if __name__ == "__main__":
    exploratory_test_maddpg(n_episodes=10)