import numpy as np
from src.utils import plotLearning, adjust_rewards, convert_arrays_to_lists
import soccer_twos
from src.agents.dqn_agent import DQNAgent
from src.logger import CustomLogger

if __name__ == "__main__":

    env = soccer_twos.make()
    logger = CustomLogger().logger 
    n_games = 10000
    
    scores, eps_history = [], []    
    agents = [DQNAgent(336, 3) for _ in range(4)]

    for i in range(n_games):
        agent_scores = [0, 0, 0, 0]
        done = False
        obs = env.reset()
        while not done:
            actions = {}
            for player_id in range(4):
                state = obs[player_id]
                action = agents[player_id].act(state)
                actions[player_id] = action  # Use discrete actions
            next_obs, reward, done, info = env.step(actions)

            # Store experiences and train agents
            losses = []
            for player_id in range(4):
                next_state = next_obs[player_id]
                agents[player_id].remember(
                    obs[player_id], 
                    actions[player_id], 
                    reward[player_id], 
                    next_state, 
                    done[player_id] if player_id in done else done['__all__']
                )
                loss = agents[player_id].replay()
                if loss is not None:
                    losses.append(loss)

                agent_scores[player_id] += reward[player_id]

            scores.append(agent_scores)
            eps_history.append([agent.epsilon for agent in agents])
            obs = next_obs

            avg_score = np.mean([score[0] for score in scores[-100:]], axis=0)
            print("episode:", i, "scores:", agent_scores, "average score %.3f" % avg_score, "epsilon %.3f" % eps_history[-1][0])

            # Log results 
            logger.info(f"{DQNAgent}", 
                extra={'custom_fields': 
                {
                    'episode': i,
                    'scores': agent_scores,
                    'average_score': avg_score,
                    'epsilon': eps_history[-1][0],
                    'reward': str(convert_arrays_to_lists(reward)),
                    'done': str(convert_arrays_to_lists(done)),
                    'info': str(convert_arrays_to_lists(info)),
                }})
    x = [i + 1 for i in range(n_games)]
    filename = 'soccer_twos_dqn.png'
    plotLearning(x, [score[0] for score in scores], [eps[0] for eps in eps_history], filename)


    