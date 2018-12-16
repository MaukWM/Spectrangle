import random

import agents.agent
import state

def play_game(agents):
    s = state.State(4)
    agent = 0
    for i in range(len(agents)):
        agents[i].on_game_start()
    while True:
        s.step(agents[agent].get_move(), agent)
        agent += 1 % len(agents)


if __name__=="__main__":
    pass
