from trainee import MyAgent


class MyAgent(MyAgent):
    def __init__(self) -> None:
        super().__init__()

    def play(self, percepts, player, step, time_left):
        super().play(percepts, player, step, time_left, _train = False)