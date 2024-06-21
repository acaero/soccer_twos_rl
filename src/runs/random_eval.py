import numpy as np
import soccer_twos
from src.utils import convert_arrays_to_lists
from src.logger import CustomLogger

if __name__ == "__main__":

    env = soccer_twos.make()
    logger = CustomLogger().logger 
    n_games = 10000
    
    scores = []    

    for i in range(n_games):
        agent_scores = [0, 0, 0, 0]
        done = False
        obs = env.reset()
        while not done:
            actions = {player_id: env.action_space.sample() for player_id in range(4)}  # Use random actions
            next_obs, reward, done, info = env.step(actions)

            for player_id in range(4):
                agent_scores[player_id] += reward[player_id]

            scores.append(agent_scores)
            obs = next_obs

        avg_score = np.mean([score[0] for score in scores[-100:]], axis=0)
        print("episode:", i, "scores:", agent_scores, "average score %.3f" % avg_score)

        # Log results 
        logger.info(f"RandomAgent", 
            extra={'custom_fields': 
            {
                'episode': i,
                'scores': agent_scores,
                'average_score': avg_score,
                'reward': str(convert_arrays_to_lists(reward)),
                'done': str(convert_arrays_to_lists(done)),
                'info': str(convert_arrays_to_lists(info)),
            }})

