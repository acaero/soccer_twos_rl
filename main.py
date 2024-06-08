import soccer_twos
from src.baseline import SoccerAgent
from src.logger import CustomLogger


# Create a logger
logger = CustomLogger().logger 


# Initialize first action and environment
env = soccer_twos.make(
    render=True,
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

# Reward per team
team_blue_reward = 0
team_orange_reward = 0

# Example instance of the SoccerAgent (baseline) class
agent = SoccerAgent()

# Main loop 
i = 0
while True:
    i += 1

    # Example usage of the baseline to determine actions
    actions = {}
    for player_id in range(4):
       player_info = info[player_id]["player_info"]
       target_pos = agent.defend_and_attack(info[0]["ball_info"]["position"], player_id, player_info["position"])
       actions[player_id] = agent.move_to_point(player_info, target_pos)

    obs, reward, done, info = env.step(actions)

    # Calculate the total reward for each team and reset if the game is over
    team_blue_reward += reward[0] + reward[1]
    team_orange_reward += reward[2] + reward[3]

    # Log results 
    logger.info(f"{agent.__class__.__name__}", 
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
        env.reset()