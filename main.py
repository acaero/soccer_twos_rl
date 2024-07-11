from src.runs.ddpg_train import train_ddpg
from src.runs.dqn_train import train_dqn
from src.runs.baseline_train import train_random, train_baseline
from src.config import N_GAMES

if __name__ == "__main__":
    train_random(n_games=N_GAMES, n_agents=1)
    train_baseline(n_games=N_GAMES, n_agents=1)
    train_ddpg(n_games=N_GAMES, n_agents=1)
    train_dqn(n_games=N_GAMES, n_agents=1)
