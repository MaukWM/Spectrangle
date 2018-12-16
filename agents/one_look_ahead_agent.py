import random

import move
import state
from agents import agent


class OneLookAheadAgent(agent.Agent):
    def get_move(self, s: state.State) -> move.Move:
        possible_moves = s.get_all_possible_moves(self.index)
        possible_moves = list(possible_moves)
        highest_score_gain = 0
        best_move = None
        for possible_move in possible_moves:
            r, new_state, terminal = s.generate_step(possible_move, self.index)
            if r > highest_score_gain:
                highest_score_gain=r
                best_move = possible_move
        return best_move
