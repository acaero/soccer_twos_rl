import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import gym


def plotLearning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    # ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    # ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    # ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    # ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


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
    max_possible_distance = np.linalg.norm(
        np.array(vector1) - np.zeros_like(vector1)) * 2
    normalized_distance = (scaled_distance - min_value) / (
        max_possible_distance - min_value) * (max_value - min_value) + min_value

    # Clip the normalized distance to be within the range [min_value, max_value]
    normalized_distance = np.clip(normalized_distance, min_value, max_value)

    return normalized_distance


def adjust_rewards(reward, info):
    """
    Adjust rewards based on players' closeness to the ball and proximity to their teammates.

    Parameters:
    - reward: dict, current rewards for each player
    - info: dict, information containing player positions and ball position

    Returns:
    - dict, adjusted rewards for each player
    """
    for player_id in range(4):
        player_position = info[player_id]["player_info"]["position"]
        ball_position = info[0]["ball_info"]["position"]

        # Reward for being close to the ball
        # Inverting distance to reward closeness
        closeness_to_ball = 1 - scaled_distance(player_position, ball_position)
        reward[player_id] += closeness_to_ball

        # Slight negative reward for players of the same team being close to each other
        if player_id in [0, 1]:  # Team 1
            other_team_players = [0, 1]
        else:  # Team 2
            other_team_players = [2, 3]

        for teammate_id in other_team_players:
            if teammate_id != player_id:
                teammate_position = info[teammate_id]["player_info"]["position"]
                closeness_to_teammate = scaled_distance(
                    player_position, teammate_position)
                reward[player_id] -= closeness_to_teammate * \
                    0.1  # Adjust the scaling factor as needed

    return reward
