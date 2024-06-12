import soccer_twos
from src.reinforcement_agent import ReinforcementLearningAgent
from src.logger import CustomLogger


# Create a logger
logger = CustomLogger().logger 


# Initialize first action and environment
env = soccer_twos.make(
    render=False,
    time_scale=1,
    quality_level=5,
)

actions = {
        0: [0, 0, 0],
        1: [0, 0, 0],
        2: [0, 0, 0],
        3: [0, 0, 0],
    }

obs, reward, done, info = env.step(actions)

# Assuming state_size and action_size are defined according to your observation and action space
state_size = len(obs[0])  # Adjust based on actual observation space size
action_size = 3  # There are 3 discrete actions: 0, 1, and 2

# Create four RL agents
agents = [ReinforcementLearningAgent(state_size, action_size) for _ in range(4)]

# Reward per team
team_blue_reward = 0
team_orange_reward = 0

# Main loop 
i = 0
while True:
    i += 1

    actions = {}
    for player_id in range(4):
        state = obs[player_id]
        action = agents[player_id].act(state)
        actions[player_id] = action  # Use discrete actions

    next_obs, reward, done, info = env.step(actions)

    # Debugging: Print the structure of `done`
    print(f"Done: {done}")

    # Store experiences and train agents
    for player_id in range(4):
        next_state = next_obs[player_id]
        agents[player_id].remember(
            obs[player_id], 
            actions[player_id], 
            reward[player_id], 
            next_state, 
            done[player_id] if player_id in done else done['__all__']  # Adjust this line based on the printed structure
        )
        agents[player_id].replay()

    obs = next_obs

    # Calculate the total reward for each team and reset if the game is over
    team_blue_reward += reward[0] + reward[1]
    team_orange_reward += reward[2] + reward[3]

    # Log results 
    logger.info(f"ReinforcementLearningAgent", 
                extra={'custom_fields': 
                {
                    'reward': str(reward),
                    'done': str(done),
                    'info': str(info),
                    'team_blue_reward': team_blue_reward,
                    'team_orange_reward': team_orange_reward
                }})

    if done["__all__"]:
        print("Total Reward: ", team_blue_reward, " and ", team_orange_reward)
        team_blue_reward = 0
        team_orange_reward = 0
        obs = env.reset()
