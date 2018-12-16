import random

import state

s = state.State()
print(s)
player = 0
for i in range(10):
    possible_moves = list(s.get_all_possible_moves(player))
    print(possible_moves)
    move = random.choice(possible_moves)
    print(move)
    s.step(move, player)
    print(s)
    player = (player + 1)%4