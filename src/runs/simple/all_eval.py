from src.runs.simple.ddpg_eval import train_ddpg
from src.runs.simple.ddqn_eval import train_ddqn
from src.runs.simple.maddpg_eval import train_maddpg
from src.runs.simple.baseline_eval import train_random, train_baseline
from src.config import N_GAMES

if __name__ == "__main__":
    train_random(n_games=N_GAMES, n_agents=1)
    train_baseline(n_games=N_GAMES, n_agents=1)
    train_ddpg(n_games=N_GAMES, n_agents=1)
    train_ddqn(n_games=N_GAMES, n_agents=1)
    train_maddpg(n_games=N_GAMES, n_agents=1)
