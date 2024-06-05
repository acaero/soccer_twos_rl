import soccer_twos
import numpy as np
import time
import math

class BaselineActions:

    def __init__(self, forward_angle = 30, is_defending = True) -> None:
        self.forward_angle = forward_angle
        self.is_defending = is_defending

    def vector_from_angle_custom(self, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        x = math.sin(angle_radians)
        y = math.cos(angle_radians)
        return (x, y)

    def calculate_angle_between_vectors(self, v1, v2):
        x1, y1 = v1
        x2, y2 = v2
        dot_product = x1 * x2 + y1 * y2
        magnitude_v1 = math.sqrt(x1**2 + y1**2)
        magnitude_v2 = math.sqrt(x2**2 + y2**2)
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
        angle = math.degrees(math.acos(cos_theta))
        return angle

    def determine_rotation_direction(self, v1, v2):
        x1, y1 = v1
        x2, y2 = v2
        cross_product = x1 * y2 - y1 * x2
        return "left" if cross_product > 0 else "right"
    
    def get_action(self, info):
        
        actions = {}
        
        for player_id in range(4):
            player_info = info[player_id]["player_info"]
            ball_position = info[player_id]["ball_info"]["position"]

            player_pos = np.array(player_info["position"][:2])
            ball_pos = np.array(ball_position[:2])

            direction_to_ball = ball_pos - player_pos
            direction_to_ball /= np.linalg.norm(direction_to_ball)

            player_rotation_y = player_info['rotation_y']
            player_forward_vector = self.vector_from_angle_custom(player_rotation_y)

            angle_to_ball = self.calculate_angle_between_vectors(player_forward_vector, direction_to_ball)
            rotation_direction = self.determine_rotation_direction(player_forward_vector, direction_to_ball)

            if not self.is_defending:

                forward = 1 if angle_to_ball < self.forward_angle else 0
                turn = 1 if rotation_direction == "left" else 2

            elif self.is_defending:

                blue_goal = np.array([-15.34, 1.82])
                orange_goal = np.array([15.34, 1.82])

                blue_attacking = True if ball_pos[0] > 1 else False
                orange_attacking = True if ball_pos[0] < 1 else False

                if blue_attacking and player_id in [2, 3]:
                    direction_to_goal = orange_goal - ball_pos
                    
                elif orange_attacking and player_id in [0, 1]:
                    direction_to_goal = blue_goal - ball_pos

                    

            actions[player_id] = [forward, 0, turn]

        return actions

env = soccer_twos.make(
    render=True,
    time_scale=1,
    quality_level=5,
)

team0_reward = 0
team1_reward = 0

actions = {
    0: [0, 0, 0],
    1: [0, 0, 0],
    2: [0, 0, 0],
    3: [0, 0, 0],
}

baseline_actions = BaselineActions(forward_angle=30, is_defending=False)
i = 0
while True:
    i += 1

    obs, reward, done, info = env.step(actions)

    actions = baseline_actions.get_action(info)

    print("Rotation von Spieler 1: ", info[0]["player_info"]['rotation_y'],
          "Position von Spieler 1: ", info[0]["player_info"]["position"],
          "Position von Ball: ", info[0]["ball_info"]["position"])

    team0_reward += reward[0] + reward[1]
    team1_reward += reward[2] + reward[3]
    if done["__all__"]:
        print("Total Reward: ", team0_reward, " x ", team1_reward)
#        time.sleep(5)
        team0_reward = 0
        team1_reward = 0
        env.reset()
