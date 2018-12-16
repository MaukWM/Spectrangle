from abc import ABC, abstractmethod

import move
import state


class Agent(ABC):
    def __init__(self, index, n_players):
        self.index = index
        self.n_players = n_players

    def on_game_start(self, s: state.State):
        pass

    @abstractmethod
    def get_move(self, s: state.State) -> move.Move:
        pass

    def on_game_end(self, s: state.State):
        pass
