import random

import state

s = state.State()
print(s)
player = 0
for i in range(50):
    possible_moves = list(s.get_all_possible_moves(player))
    move = random.choice(possible_moves)
    s.step(move, player)
    print(s)
    player = (player + 1)%4