import random

import agents.agent
import state
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent

def play_game(agents, visualise: bool):
    s = state.State(len(agents))
    agent = 0
    if visualise:
        print(s)
    for i in range(len(agents)):
        agents[i].on_game_start(s)
    while True:
        s.step(agents[agent].get_move(s), agent)
        agent = (agent + 1) % len(agents)
        if visualise:
            print(s)


if __name__=="__main__":
    play_game(
    agents=[
        HumanAgent(0),
        RandomAgent(1)
    ],
    visualise=True
    )
