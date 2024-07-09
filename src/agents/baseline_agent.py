import numpy as np
from src.utils import calculate_angle_between_vectors, vector_from_angle_custom


class BaselineAgent:
    def __init__(self, num_agents=1):
        self.num_agents = num_agents

    def act(self, info, player_id):
        player_info = info[player_id]["player_info"]
        ball_info = info[0]["ball_info"]
        target_pos = self.defend_and_attack(
            ball_info["position"], player_id, player_info["position"]
        )
        return self.move_to_point(player_info, target_pos)

    def determine_rotation_direction(self, v1, v2):
        x1, y1 = v1
        x2, y2 = v2
        cross_product = x1 * y2 - y1 * x2
        return "left" if cross_product > 0 else "right"

    def calculate_multiplier(self, ball_x, x_goal):
        start_x = 1
        if ball_x == start_x:
            return 1  # To avoid division by zero if the ball is exactly at start_x
        distance_to_start = abs(ball_x - start_x)
        distance_to_goal = abs(x_goal - start_x)
        multiplier = 1 - (distance_to_start / distance_to_goal)
        multiplier = max(0, min(1, multiplier))
        return multiplier

    def move_to_point(self, player_info, target_pos):
        player_pos = np.array(player_info["position"][:2])
        direction_to_target = target_pos - player_pos
        distance_to_target = np.linalg.norm(direction_to_target)
        direction_to_target /= distance_to_target
        player_rotation_y = player_info["rotation_y"]
        player_forward_vector = vector_from_angle_custom(player_rotation_y)
        angle_to_target = calculate_angle_between_vectors(
            player_forward_vector, direction_to_target
        )
        rotation_direction = self.determine_rotation_direction(
            player_forward_vector, direction_to_target
        )
        if distance_to_target < 0.1:
            forward = 0
            turn = 0
        elif angle_to_target < 30:
            forward = 1
            turn = 0
        else:
            forward = 0
            turn = 1 if rotation_direction == "left" else 2
        return [forward, 0, turn]

    def defend_and_attack(self, ball_position, player_id, player_position):
        blue_goal = np.array([-15.34, 1.82])
        orange_goal = np.array([15.34, 1.82])
        target_pos = [0, 0]
        drift = 0.4
        if player_id in [0, 1]:  # blue team
            if ball_position[0] < 1:
                target_pos = blue_goal + (
                    ball_position - blue_goal
                ) * self.calculate_multiplier(ball_position[0], blue_goal[0])
            elif ball_position[0] < player_position[0] and ball_position[0] > 1:
                random_value = np.random.uniform(-1 - drift, 1 + drift)
                target_pos = np.array(ball_position) * np.array([1, random_value])
            else:
                target_pos = np.array(ball_position)
        elif player_id in [2, 3]:  # orange team
            if ball_position[0] > 1:
                target_pos = orange_goal + (
                    ball_position - orange_goal
                ) * self.calculate_multiplier(ball_position[0], orange_goal[0])
            elif ball_position[0] > player_position[0] and ball_position[0] < 1:
                random_value = np.random.uniform(-1 - drift, 1 + drift)
                target_pos = np.array(ball_position) * np.array([1, random_value])
            else:
                target_pos = np.array(ball_position)
        return target_pos

    def save(self, path):
        pass


class RandomAgent:
    def __init__(self, num_agents=1):
        self.num_agents = num_agents

    def act(self, observation):
        # Return a random action for each agent
        return [np.random.uniform(-1, 1, 3) for _ in range(self.num_agents)]

    def save(self, path):
        pass
