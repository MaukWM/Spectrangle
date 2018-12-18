from collections import defaultdict

import game
from agents.one_look_ahead_agent import OneLookAheadAgent
import numpy as np
import keras as ks

from agents.q_agent import QAgent
from agents.till_next_turn_agent import TillNextTurnAgent

agents = [
    QAgent(0, 4, ks.models.load_model("win_loss_model.h5"), gamma=0.99, use_win_rewards=True),
    QAgent(1, 4, ks.models.load_model("big_net.h5"), gamma=0.99, use_win_rewards=False),
    # TillNextTurnAgent(2, 4, 15, deeper_top_n=2),
    # OneLookAheadAgent(3, 4)
]

game.play_game(agents, True, shuffle_agents=True)

scores = []
wins = defaultdict(lambda: 0)
# switched = False
for i in range(100):
    score = game.play_game(agents, False, shuffle_agents=True)

    score_list = []
    for ag in agents:
        score_list.append(score[ag])

    scores.append(score_list)
    winner = np.argmax(score_list)
    wins[winner] += 1
    if i % 10 == 0:
        print(np.mean(np.array(scores), axis=0), "wins: ", wins)