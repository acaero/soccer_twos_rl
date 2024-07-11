import numpy as np
import pandas as pd
from tqdm import tqdm
from src.agents.baseline_agent import BaselineAgent, RandomAgent
from src.agents.ddpg_agent import DDPGAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.a2c_agent import A2CAgent
from src.agents.ppo_agent import PPOAgent
import soccer_twos
from stable_baselines3 import PPO, A2C


class UnifyAgentWrapper:
    def __init__(self, agent, name=None):
        self.agent = agent
        self.name = name

        if not name:
            self.name = agent.__class__.__name__

    def act(self, obs, info, player_id, env):
        if isinstance(self.agent, BaselineAgent):
            return self.agent.act(info, player_id)
        elif isinstance(self.agent, RandomAgent):
            return self.agent.act(obs[player_id])[0]
        elif isinstance(self.agent, DDPGAgent):
            return self.agent.act(obs[player_id])
        elif isinstance(self.agent, DQNAgent):
            return self.agent.act(obs[player_id])
        elif isinstance(self.agent, A2CAgent):
            return self.agent.act(obs[player_id])
        elif isinstance(self.agent, PPOAgent):
            return self.agent.act(obs[player_id])[0]
        elif isinstance(self.agent, PPO):
            self.agent.set_env(env)
            return self.agent.predict(np.array(obs[0]))[0].tolist()
        elif isinstance(self.agent, A2C):
            self.agent.set_env(env)
            return self.agent.predict(np.array(obs[0]))[0].tolist()
        # ...


def play_matches(
    agent1: UnifyAgentWrapper, agent2: UnifyAgentWrapper, n_games=10, render=False
) -> pd.DataFrame:
    print("Team 1 loaded: ", agent1.name, "| Team 2 loaded: ", agent2.name)
    goals = pd.DataFrame(
        columns=["agent1", "agent2", "agent1_goals", "agent2_goals", "n_games"]
    )

    if render:
        env = soccer_twos.make(render=True, time_scale=1, quality_level=5, worker_id=2)
    else:
        env = soccer_twos.make(render=False, worker_id=2)

    actions = {
        0: [0, 0, 0],
        1: [0, 0, 0],
        2: [0, 0, 0],
        3: [0, 0, 0],
    }
    obs, reward, done, info = env.step(actions)

    team1_goals = 0
    team2_goals = 0

    for _ in tqdm(range(n_games)):
        done = False

        while not done:
            env.reset()

            actions = {
                player_id: (
                    agent1.act(obs, info, player_id, env)
                    if player_id < 2
                    else agent2.act(obs, info, player_id, env)
                )
                for player_id in range(4)
            }

            obs, reward, done, info = env.step(actions)

            if reward[0] > 0:
                team1_goals += 1
            elif reward[2] > 0:
                team2_goals += 1

            done = done["__all__"]

    goals.loc[len(goals)] = [
        agent1.name,
        agent2.name,
        team1_goals,
        team2_goals,
        n_games,
    ]

    env.close()

    return goals


def play_arena(agent_list, n_games=10, render=False):
    goals = pd.DataFrame(
        columns=["agent1", "agent2", "agent1_goals", "agent2_goals", "n_games"]
    )
    for i, agent1 in enumerate(agent_list):
        for j, agent2 in enumerate(agent_list[i + 1 :], start=i + 1):
            out = play_matches(agent1, agent2, n_games, render)
            goals = pd.concat([goals, out], ignore_index=True)
    return goals


# These are the best versions of all self programmed agents
dqn_agent_self = UnifyAgentWrapper(DQNAgent(336, 3))
ddpg_agent_self = UnifyAgentWrapper(DDPGAgent(336, 3))
a2c_agent_self = UnifyAgentWrapper(A2CAgent(336, 3))
ppo_agent_self = UnifyAgentWrapper(PPOAgent(336, 3))
random_agent_self = UnifyAgentWrapper(RandomAgent())
baseline_agent_self = UnifyAgentWrapper(BaselineAgent())

# These are the best versions of all stable baselines agents
ppo_agent_3M = UnifyAgentWrapper(
    PPO.load(r"src\visualization\models\bestmodel_ppo_2,8M.zip")
)
ppo_proximity_single = UnifyAgentWrapper(
    PPO.load(r"src\visualization\models\ppo_proximity_final_single.zip")
)
ppo_proximity_random = UnifyAgentWrapper(
    PPO.load(r"src\visualization\models\ppo_proximity_final_random.zip")
)
ppo_speed_random = UnifyAgentWrapper(
    PPO.load(r"src\visualization\models\ppo_speed_final_random.zip")
)
a2c_proximity_single = UnifyAgentWrapper(
    A2C.load(r"src\visualization\models\a2c_proximity_final_single.zip")
)
a2c_proximity_random = UnifyAgentWrapper(
    A2C.load(r"src\visualization\models\a2c_proximity_final_random.zip")
)
a2c_speed_single = UnifyAgentWrapper(
    A2C.load(r"src\visualization\models\a2c_speed_final_single.zip")
)
a2c_speed_random = UnifyAgentWrapper(
    A2C.load(r"src\visualization\models\a2c_speed_final_random.zip")
)

dqn_agent_self.agent.load(r"src\visualization\models\best_model_dqn_189_dqn_v1.pth")
ddpg_agent_self.agent.load(r"src\visualization\models\best_model_ddpg_106_ddpg_v1.pth")
a2c_agent_self.agent.load(r"src\visualization\models\best_model_a2c_106_a2c_v1.pth")
ppo_agent_self.agent.load(r"src\visualization\models\best_model_ppo_3_ppo_v1.pth")

ppo_agent_3M.agent.set_parameters(
    load_path_or_dict=r"src\visualization\models\bestmodel_ppo_2,8M.zip"
)
ppo_proximity_single.agent.set_parameters(
    load_path_or_dict=r"src\visualization\models\ppo_proximity_final_single.zip"
)
ppo_proximity_random.agent.set_parameters(
    load_path_or_dict=r"src\visualization\models\ppo_proximity_final_random.zip"
)
ppo_speed_random.agent.set_parameters(
    load_path_or_dict=r"src\visualization\models\ppo_speed_final_random.zip"
)
a2c_proximity_random.agent.set_parameters(
    load_path_or_dict=r"src\visualization\models\a2c_proximity_final_random.zip"
)
a2c_proximity_single.agent.set_parameters(
    load_path_or_dict=r"src\visualization\models\a2c_proximity_final_single.zip"
)
a2c_speed_single.agent.set_parameters(
    load_path_or_dict=r"src\visualization\models\a2c_speed_final_single.zip"
)
a2c_speed_random.agent.set_parameters(
    load_path_or_dict=r"src\visualization\models\a2c_speed_final_random.zip"
)

arena_list = [
    dqn_agent_self,
    ddpg_agent_self,
    a2c_agent_self,
    ppo_agent_self,
    random_agent_self,
    baseline_agent_self,
    ppo_agent_3M,
    ppo_proximity_single,
    ppo_proximity_random,
    ppo_speed_random,
    a2c_proximity_single,
    a2c_proximity_random,
    a2c_speed_single,
    a2c_speed_random,
]
if __name__ == "__main__":

    out = play_matches(ppo_agent_3M, baseline_agent_self, n_games=5, render=True)
    print(out)

    # WARNING - This will take a long time to run, especially the more agents and games you have
    # out = play_arena(arena_list, n_games=10, render=False)
