import numpy as np
import json
from src.config import REWARD_SHAPING


def shape_rewards(info, player_id):
    if not REWARD_SHAPING:
        return 0

    player_pos = np.array(info[player_id]["player_info"]["position"])
    ball_pos = np.array(info[player_id]["ball_info"]["position"])
    distance = np.linalg.norm(player_pos - ball_pos)

    print(distance)

    return -distance


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
