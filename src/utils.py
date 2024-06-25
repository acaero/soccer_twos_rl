import numpy as np
import json
from src.config import REWARD_SHAPING


class RewardShaper:
    def __init__(self, field_width=16, field_height=10):
        self.field_width = field_width
        self.field_height = field_height

    def calculate_reward(self, old_obs, new_obs, info, player_id):
        if not REWARD_SHAPING:
            return 0
        reward = 0
        # Ball possession (2)
        if self._has_ball_possession(info, player_id):
            reward += 0.1
        # Field position (3)
        reward += self._field_position_reward(old_obs, new_obs, player_id)
        # Energy efficiency (6)
        reward -= 0.002  # Penalty for taking an action
        # Time pressure (8)
        reward -= 0.0002
        return reward

    def _has_ball_possession(self, info, player_id):
        player_pos = np.array(info[player_id]["player_info"]["position"])
        ball_pos = np.array(info[player_id]["ball_info"]["position"])
        distance = np.linalg.norm(player_pos - ball_pos)
        return distance < 1.0  # Assume possession if within 1 unit of the ball

    def _field_position_reward(self, old_obs, new_obs, player_id):
        old_distance = self._distance_to_opponent_goal(old_obs, player_id)
        new_distance = self._distance_to_opponent_goal(new_obs, player_id)
        return 0.05 * (old_distance - new_distance)

    def _distance_to_opponent_goal(self, obs, player_id):
        player_x = obs[0]  # Assuming player's x position is the first element
        goal_x = self.field_width / 2 if player_id in [0, 1] else -self.field_width / 2
        return abs(player_x - goal_x)


def convert_arrays_to_lists(d):
    if isinstance(d, dict):
        return {k: convert_arrays_to_lists(v) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return d.tolist()
    else:
        return d


def read_json_log_file(file_path):
    log_entries = []
    try:
        with file_path.open("r") as file:
            for line in file:
                try:
                    # Parse each line as a JSON object and append to the list
                    log_entry = json.loads(line.strip())
                    log_entries.append(log_entry)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {e}")
    except Exception as e:
        print(f"Error reading JSON log file: {e}")
    return log_entries


def scaled_distance(vector1, vector2, scale=1.0, min_value=0.0, max_value=1.0):
    """
    Calculate the scaled distance between two vectors and normalize it to a specified range.

    Parameters:
    - vector1: array-like, first vector
    - vector2: array-like, second vector
    - scale: float, scaling factor for the distance
    - min_value: float, minimum value of the output range
    - max_value: float, maximum value of the output range

    Returns:
    - float, scaled and normalized distance between the vectors within the specified range
    """
    # Calculate the Euclidean distance between the two vectors
    distance = np.linalg.norm(np.array(vector1) - np.array(vector2))

    # Apply the scaling factor
    scaled_distance = distance * scale

    # Calculate the normalized distance
    # Here, we assume a max distance for normalization. You can customize this based on your context.
    max_possible_distance = (
        np.linalg.norm(np.array(vector1) - np.zeros_like(vector1)) * 2
    )
    normalized_distance = (scaled_distance - min_value) / (
        max_possible_distance - min_value
    ) * (max_value - min_value) + min_value

    # Clip the normalized distance to be within the range [min_value, max_value]
    normalized_distance = np.clip(normalized_distance, min_value, max_value)

    return normalized_distance
