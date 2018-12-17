import random

import move
import state
#
# s = state.State()
# print(s)
# player = 0
# for i in range(50):
#     possible_moves = list(s.get_all_possible_moves(player))
#     move = random.choice(possible_moves)
#     s.step(move, player)
#     print(s)
#     player = (player + 1)%4


s = state.State()
print(s.initial)
for mv, sp in [(mv, s.generate_step(mv, 0)) for mv in s.get_all_possible_moves(0)]:
    print(mv)
    if isinstance(mv, move.PlaceMove):
        print(mv.row, mv.column, ((mv.row, mv.column) in state.bonus_set))
    print(sp)
