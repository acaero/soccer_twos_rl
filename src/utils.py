import numpy as np
import json
from src.config import REWARD_SHAPING
import math
import multiprocessing as mp
import soccer_twos


def shape_rewards(info, player_id):
    if not REWARD_SHAPING:
        return 0

    extra_reward = 0

    player_pos = np.array(info[player_id]["player_info"]["position"])
    ball_pos = np.array(info[player_id]["ball_info"]["position"])
    distance = np.linalg.norm(player_pos - ball_pos)
    # Normalize the distance
    normalized_distance = 1 - distance / 16.5
    # print("distance: ", distance)

    # Calculate the proximity reward using an exponential function
    proximity_reward = normalized_distance
    # print("reward: ", normalized_distance)
    # Add this to your existing reward
    extra_reward += proximity_reward

    # Calculate the angle reward based on the angle between the player's forward vector and the direction to the ball
    # direction_to_target = ball_pos - player_pos
    # distance_to_target = np.linalg.norm(direction_to_target)
    # direction_to_target /= distance_to_target
    # player_rotation_y = info[player_id]["player_info"]["rotation_y"]
    # player_forward_vector = vector_from_angle_custom(player_rotation_y)
    # angle_to_target = calculate_angle_between_vectors(
    #     player_forward_vector, direction_to_target
    # )

    # angle_reward = np.exp(-3 * angle_to_target / 180) / 6

    # Add this to your existing reward
    # extra_reward += angle_reward

    return extra_reward


def vector_from_angle_custom(angle_degrees):
    angle_radians = math.radians(angle_degrees)
    x = math.sin(angle_radians)
    y = math.cos(angle_radians)
    return (x, y)


def calculate_angle_between_vectors(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    dot_product = x1 * x2 + y1 * y2
    magnitude_v1 = math.sqrt(x1**2 + y1**2)
    magnitude_v2 = math.sqrt(x2**2 + y2**2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle = math.degrees(math.acos(cos_theta))
    return angle


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


def worker_process(pipe, id):
    env = soccer_twos.make(render=True, worker_id=id)
    while True:
        cmd, data = pipe.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            pipe.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            pipe.send(obs)
        elif cmd == "close":
            env.close()
            pipe.close()
            break


class ParallelSoccerEnv:
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.processes = []
        self.pipes = []
        for i in range(n_envs):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=worker_process, args=(child_conn, i))
            p.start()
            self.processes.append(p)
            self.pipes.append(parent_conn)

    def reset(self):
        for pipe in self.pipes:
            pipe.send(("reset", None))
        return np.array([pipe.recv() for pipe in self.pipes])

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(("step", action))
        results = [pipe.recv() for pipe in self.pipes]
        obs, rewards, dones, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), infos

    def close(self):
        for pipe in self.pipes:
            pipe.send(("close", None))
        for p in self.processes:
            p.join()
