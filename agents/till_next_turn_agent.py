import move
import state
from agents import agent
import heapq


class TillNextTurnAgent(agent.Agent):
    def __init__(self, index, n_players, top_n, deeper_top_n=None):
        super().__init__(index, n_players)
        self.top_n = top_n
        self.deeper_top_n = deeper_top_n if deeper_top_n is not None else top_n

    def get_move(self, s: state.State) -> move.Move:
        action, _ = self.get_best_value_action(s, self.index, self.index, True, top_n=self.top_n)
        return action

    def generate_ar_tuple(self, s: state.State, a, p):
        r = s.calculate_action_score(a, p)
        return a, r

    def get_best_value_action(self, s: state.State, current_player, desired_player, initial=False, top_n=None):
        if current_player == desired_player and not initial:
            return None, []

        if top_n is not None:
            ar_set = heapq.nlargest(top_n, [self.generate_ar_tuple(s, a, current_player) for a in s.get_all_possible_moves(current_player)], key=lambda ar: ar[1])
        else:
            ar_set = [self.generate_ar_tuple(s, a, current_player) for a in s.get_all_possible_moves(current_player)]

        best_action = None
        best_value = None
        best_values = None
        for a, r in ar_set:
            sp, _, _ = s.generate_step(a, current_player)
            _, values =  self.get_best_value_action(sp, (current_player+1)%self.n_players, desired_player, False, self.deeper_top_n)
            values = [r] + values
            # All values except our own are bad for us
            value = sum(v if i ==0 else -v for i,v in enumerate(values))
            if best_value is None or value > best_value:
                best_value = value
                best_action = a
                best_values = values
        return best_action, best_values

    def __repr__(self):
        return "TillNextTurnAgent "+str(self.index)
