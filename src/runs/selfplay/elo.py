save_steps = 10000
team_change = 5 * save_steps
swap_steps = 10000
play_against_latest_model_ratio = 0.5
window = 5

initial_elo = 1200


class Elo:
    def __init__(self, initial_elo=1200) -> None:
        self.initial_elo = initial_elo
        pass
