from src.runs.simple.ddpg_train import train_ddpg
from src.runs.simple.ddqn_train import train_ddqn
from src.runs.simple.maddpg_train import train_maddpg
from src.runs.simple.baseline_train import train_random, train_baseline
from src.runs.simple.ppo_train import train_ppo
from src.runs.simple.sac_train import train_sac
from src.config import N_GAMES

if __name__ == "__main__":
    train_random(n_games=N_GAMES, n_agents=1)
    train_baseline(n_games=N_GAMES, n_agents=1)
    train_ddpg(n_games=N_GAMES, n_agents=1)
    train_ddqn(n_games=N_GAMES, n_agents=1)
    train_maddpg(n_games=N_GAMES, n_agents=1)
    train_ppo(n_games=N_GAMES, n_agents=1)
    train_sac(n_games=N_GAMES, n_agents=1)
