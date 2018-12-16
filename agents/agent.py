from abc import ABC, abstractmethod

import move
import state


class Agent(ABC):
    def __init__(self, index):
        self.index = index

    def on_game_start(self, s: state.State):
        pass

    @abstractmethod
    def get_move(self, s: state.State) -> move.Move:
        pass

    def on_game_end(self, s: state.State):
        pass
