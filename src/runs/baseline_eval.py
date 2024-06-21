import soccer_twos
import numpy as np
from src.agents.baseline_agent import SoccerAgent
from src.logger import CustomLogger
from src.utils import convert_arrays_to_lists


env = soccer_twos.make()
logger = CustomLogger().logger 
n_games = 10000

agent = SoccerAgent()

scores = []    

actions = {
        0: [0, 0, 0],
        1: [0, 0, 0],
        2: [0, 0, 0],
        3: [0, 0, 0],
    }

obs, reward, done, info = env.step(actions)

for i in range(n_games):
        agent_scores = [0, 0, 0, 0]
        done = False
        obs = env.reset()
        while not done:
            actions = {}
            for player_id in range(4):
                player_info = info[player_id]["player_info"]
                target_pos = agent.defend_and_attack(info[0]["ball_info"]["position"], player_id, player_info["position"])
                actions[player_id] = agent.move_to_point(player_info, target_pos)
            
            next_obs, reward, done, info = env.step(actions)

            for player_id in range(4):
                agent_scores[player_id] += reward[player_id]

            scores.append(agent_scores)
            obs = next_obs

        avg_score = np.mean([score[0] for score in scores[-100:]], axis=0)
        print("episode:", i, "scores:", agent_scores, "average score %.3f" % avg_score)

        # Log results 
        logger.info(f"BaselineAgent", 
            extra={'custom_fields': 
            {
                'episode': i,
                'scores': agent_scores,
                'average_score': avg_score,
                'reward': str(convert_arrays_to_lists(reward)),
                'done': str(convert_arrays_to_lists(done)),
                'info': str(convert_arrays_to_lists(info)),
            }})
        
