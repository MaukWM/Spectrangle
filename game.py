import random

import agents.agent
import state
from agents.random_agent import RandomAgent


def play_game(agents):
    s = state.State(4)
    agent = 0
    for i in range(len(agents)):
        agents[i].on_game_start(s)
    while True:
        s.step(agents[agent].get_move(s), agent)
        agent =  (agent + 1) % len(agents)


if __name__=="__main__":
    play_game([
        RandomAgent(0),
        RandomAgent(1)
    ])
