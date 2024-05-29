import soccer_twos
import numpy as np

env = soccer_twos.make(
    render=True,
    time_scale=1,
    quality_level=5,
)

team0_reward = 0
team1_reward = 0


def movement_to_ball(single_info):
    player_pos = np.array(single_info["player_info"]['position'])
    current_player_rot = single_info["player_info"]['rotation_y']
    ball_pos = np.array(single_info["ball_info"]['position'])

    # Calculate the vector from player to ball
    vector_to_ball = ball_pos - player_pos

    # Calculate the angle of this vector
    theta_ball = np.arctan2(vector_to_ball[1], vector_to_ball[0])
    
    # Convert current player rotation from degrees to radians
    theta_player = np.deg2rad(current_player_rot)
    
    # Calculate the angle difference
    delta_theta = theta_ball - theta_player
    
    # Normalize the angle to the range [-π, π]
    delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
    
    # Determine the rotation direction
    if delta_theta > 0:
        rot = 1
    elif delta_theta < 0:
        rot = -1
    else:
        rot = 0

    return [0, 0, rot]


action = {
    0: env.action_space.sample(),
    1: env.action_space.sample(),
    2: env.action_space.sample(),
    3: env.action_space.sample(),
}

i = 0
while True:
    i += 1

    obs, reward, done, info = env.step(action)

    # print("Observation von Spieler 1 nach step 1: ", obs[0])

    action = {
        0: movement_to_ball(info[0]),
        1: movement_to_ball(info[1]),
        2: movement_to_ball(info[2]),
        3: movement_to_ball(info[3]),
    }

    print("Action von Spieler 1 bei init: ",
          action[0], " \n #0 [-1,1] Vorwärts/Rückwärts \n #1 [-1,1] Links/Rechts \n #2 [-1,1] Drehung")

    if i % 100 == 0:
        print(
            f"obs: {obs}",
            f"reward: {reward}",
            f"done: {done}",
            f"info: {info}",
            sep="\n\n")

    team0_reward += reward[0] + reward[1]
    team1_reward += reward[2] + reward[3]
    if done["__all__"]:
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_reward = 0
        team1_reward = 0
        env.reset()
