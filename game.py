import random
import math

import agents.agent
import state
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from agents.one_look_ahead_agent import OneLookAheadAgent


def play_game(agents, visualise: bool, shuffle_agents=False, starting_agent_index=0):
    agents = agents[:]

    if shuffle_agents:
        random.shuffle(agents)

    for i, agent in enumerate(agents):
        agent.index = i
        agent.n_players = len(agents)

    stop_point = -1
    s = state.State(len(agents))
    agent = starting_agent_index
    if visualise:
        print(s)
    for i in range(len(agents)):
        agents[i].on_game_start(s)
    while True:
        if stop_point == agent:
            break
        reward, terminate = s.step(agents[agent].get_move(s), agent)
        if visualise:
            print(s)
        if terminate:
            if stop_point == -1:
                stop_point = agent
        agent = (agent + 1) % len(agents)
    if visualise:
        print("Game Over!\n")

    best_score = -math.inf
    best_agent = None

    for agent in agents:
        if visualise:
            print(str(agent) + " ended with " + str(s.scores[agent.index]))
        if s.scores[agent.index] > best_score:
            best_score = s.scores[agent.index]
            best_agent = agent

    if visualise:
        print(str(best_agent) + " wins!")
    return {agent: score for agent, score in zip(agents, s.scores)}


if __name__=="__main__":
    play_game(
    agents=[
        OneLookAheadAgent(0, 2),
        RandomAgent(1, 2)
    ],
    visualise=True
    )
