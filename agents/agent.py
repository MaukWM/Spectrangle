from abc import ABC

import move
import state


class Agent(ABC):
    def on_game_start(self, s: state.State):
        pass

    def get_move(self, s: state.State) -> move.Move:
        pass

    def on_game_end(self, s: state.State):
        pass
