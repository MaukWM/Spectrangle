import random

import move
import state
from agents import agent


class RandomAgent(agent.Agent):
    def get_move(self, s: state.State) -> move.Move:
        possible_moves = s.get_all_possible_moves(self.index)
        possible_moves = list(possible_moves)
        mv = random.choice(possible_moves)
        return mv
