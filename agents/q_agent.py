import random
import time

import numpy as np
import keras as ks

import move
import state
from agents import agent


def to_oh(i, length):
    out = [0] * length
    out[i] = 1
    return out


def c_to_oh(c):
    out = [0] * 8
    if c is None:
        out[0] = 1
    elif c is "EMPTY":  # TODO: Actually implement this :)
        out[1] = 1
    else:
        out[2 + c.value] = 1
    return out


def transform_state(s: state.State, turns_till_turn, player_index, zero_new_item=False):
    # Triangles:

    triangle_states = []

    for row in range(len(s.board)):
        for column in range(len(s.board[row])):
            colours = s.get_tile_colours(row, column)
            if colours is None:
                colours = [None, None, None]
            for c, nb_c in zip(colours, s.get_neighbouring_tile_colours(row, column)):
                oh_array = c_to_oh(c) + c_to_oh(nb_c)
                triangle_states.append(oh_array)

    hand_state = []
    for i in range(4):
        if len(s.hands[player_index]) <= i:
            hand_state += [c_to_oh(None) + [0]*8]*3
        elif i == 3 and zero_new_item:
            hand_state += [[0]*16]*3
        else:
            for colour in s.hands[player_index][i].colours:
                hand_state.append(c_to_oh(colour) + [0]*8)

    info = [to_oh(turns_till_turn, 4) + [0]*(8+4)]
    return np.array(info + hand_state + triangle_states)


class QAgent(agent.Agent):
    def __init__(self, index, n_players, model: ks.Model, gamma=1.0, reward_scale=1/100):
        super().__init__(index, n_players)
        self.model = model
        self.gamma = gamma
        self.replay = []
        self.replay_size = 10000
        self.reward_scale = reward_scale

        # Make fixed model
        self.fixed_model = ks.models.model_from_json(self.model.to_json())
        self.fixed_model.set_weights(self.model.get_weights())

    def get_move(self, s: state.State) -> move.Move:
        possible_moves, values = self.get_moves_with_values(s)
        best_move_index = int(np.argmax(values))
        best_move = possible_moves[best_move_index]
        return best_move

    def get_moves_with_values(self, s: state.State, player=None):
        if player is None:
            player = self.index
        possible_moves = list(s.get_all_possible_moves(player))
        new_steps = [s.generate_step(mv, player) for mv in possible_moves]
        states, rewards, dones = zip(*new_steps)
        all_new_states = np.stack([
            transform_state(ss, self.n_players - 1, player, zero_new_item=True) for ss in states]
            , axis=0
        )
        values = self.model.predict(all_new_states)[:, 0]
        values = np.array(rewards)*self.reward_scale + self.gamma * values
        return possible_moves, values

    def sample_batch(self):
        batch = []
        for i in range(32):
            s, r, sp, term = random.choice(self.replay[-self.replay_size:])
            batch.append((s, r, sp, term))

        states = np.stack([exp[0] for exp in batch], axis=0)
        targets = np.zeros((32, 1), dtype=np.float32)

        next_states = np.stack([exp[2] for exp in batch], axis=0)
        vps = self.fixed_model.predict(next_states)

        # Get the max for each next state to get V(sp)
        vps = np.max(vps, axis=1)

        terms = np.stack([exp[3] for exp in batch], axis=0)
        vps[terms] = 0

        # Add rewards
        vps = np.stack([exp[1] for exp in batch], axis=0) + self.gamma * vps

        for i, exp in enumerate(batch):
            # Set the target to the new Q(s, a) value for the taken action index (exp[1])
            targets[i, 0] = vps[i]
        return states, targets

    def train(self, epochs, epsilon, epsilon_decay_per_episode, epsilon_min, test_against=[], test_every_n_epsiodes=10, number_of_test_games=10):
        win_rates = []
        epsiodes = []
        try:
            for episode in range(epochs):

                if episode%test_every_n_epsiodes == 1:
                    print("Running tests: ")
                    agents = [self] + test_against
                    wins = 0
                    for i in range(number_of_test_games):
                        for j in range(len(agents)):
                            score = game.play_game(agents, False, False, starting_agent_index=j)
                            winner = max(score.keys(), key=lambda key: score[key])
                            if winner == self:
                                wins += 1
                    win_rate = wins/(number_of_test_games * len(agents))
                    epsiodes.append(episode)
                    win_rates.append(win_rate)
                    print("Episode: ", episode, " winrate: ", win_rate)

                done_counter = 4
                player = 0
                s = state.State()
                o0 = transform_state(s, 0, self.index)
                o1 = transform_state(s, 1, self.index)
                o2 = transform_state(s, 2, self.index)
                o3 = transform_state(s, 3, self.index)

                print("Training episode ", episode, "epsilon: ", max(epsilon*(epsilon_decay_per_episode**episode), epsilon_min))

                while done_counter > 0:
                    if random.random() > max(epsilon*(epsilon_decay_per_episode**episode), epsilon_min):
                        moves, values = self.get_moves_with_values(s, player)
                        i = int(np.argmax(values))
                        mv = moves[i]
                    else:
                        mv = random.choice(list(s.get_all_possible_moves(player)))

                    s_prime, r, done = s.generate_step(mv, player)
                    if done:
                        done_counter -= 1

                    o_p0 = transform_state(s_prime, 0, self.index)
                    o_p1 = transform_state(s_prime, 1, self.index)
                    o_p2 = transform_state(s_prime, 2, self.index)
                    o_p3 = transform_state(s_prime, 3, self.index)

                    self.replay += [
                        (o0, r*self.reward_scale, o_p3, done),
                        (o1, -r*self.reward_scale, o_p0, False),
                        (o2, -r*self.reward_scale, o_p1, False),
                        (o3, -r*self.reward_scale, o_p2, False)
                    ]

                    x, y = self.sample_batch()
                    self.model.fit(x, y, verbose=False)
                    player = (player + 1) % 4
                    s = s_prime
                    o0 = o_p0
                    o1 = o_p1
                    o2 = o_p2
                    o3 = o_p3

                # Update fixed weights
                self.fixed_model.set_weights(self.model.get_weights())
        except KeyboardInterrupt:
            pass
        return epsiodes, win_rates

    def __repr__(self):
        return "QAgent " + str(self.index)


if __name__ == "__main__":
    import game
    from collections import defaultdict
    from agents.human_agent import HumanAgent
    from agents.random_agent import RandomAgent
    from agents.one_look_ahead_agent import OneLookAheadAgent
    import matplotlib.pyplot as plt

    LOAD_MODEL = False
    if LOAD_MODEL:
        model = ks.models.load_model("temp.h5")
    else:
        inp = ks.Input((1 + 3*4 + 3*36, 16))
        info = ks.layers.Lambda(lambda x: x[:, 0])(inp)
        hand = ks.layers.Lambda(lambda x: x[:, 1:3*4+1])(inp)
        board = ks.layers.Lambda(lambda x: x[:, 3*4+1:])(inp)

        triangle_conv1 = ks.layers.Conv1D(64, 1, activation='selu')
        triangle_conv2 = ks.layers.Conv1D(64, 3, strides=3, activation='selu')
        triangle_conv3 = ks.layers.Conv1D(32, 1, strides=1, activation='selu')

        board = triangle_conv1(board)
        triangles = triangle_conv2(board)
        triangles = triangle_conv3(triangles)
        board = ks.layers.Flatten()(triangles)
        board = ks.layers.Dense(64, activation='selu')(board)

        hand = triangle_conv1(hand)
        hand = triangle_conv2(hand)
        hand = triangle_conv3(hand)
        hand = ks.layers.Flatten()(hand)
        hand = ks.layers.Dense(24, activation='selu')(hand)

        x = ks.layers.Concatenate(axis=1)([info, hand, board])
        x = ks.layers.Dense(64, activation='selu')(x)
        x = ks.layers.Dense(32, activation='selu')(x)
        out = ks.layers.Dense(1, activation='linear')(x)

        model = ks.Model(inputs=inp, outputs=out)
        model.compile(optimizer=ks.optimizers.Adam(0.0001), loss='mse')
    model.summary()

    q_ag = QAgent(0, 4, model, gamma=0.99)

    agents = [
        q_ag,
        OneLookAheadAgent(1, 4),
        OneLookAheadAgent(2, 4),
        OneLookAheadAgent(3, 4),
    ]

    episodes, winrates = \
        q_ag.train(200, 1.0, 0.97, epsilon_min=0.01, test_every_n_epsiodes=50, number_of_test_games=10, test_against=agents[1:])

    plt.plot(episodes, winrates)
    plt.xlabel("Episodes")
    plt.ylabel("Win rate")
    plt.show()

    game.play_game(agents, True)

    model.save("temp.h5")

    scores = []
    wins = defaultdict(lambda: 0)
    # switched = False
    for i in range(100):
        score = game.play_game(agents, False, shuffle_agents=True)

        score_list = []
        for ag in agents:
            score_list.append(score[ag])
        # if switched:
        #     score[0], score[1] = score[1], score[0]

        # agents = agents[::-1]
        # switched = not switched
        # for j, agent in enumerate(agents):
        #     agent.index = j

        scores.append(score_list)
        winner = np.argmax(score_list)
        wins[winner] += 1
        if i % 10 == 0:
            print(np.mean(np.array(scores), axis=0), "wins: ", wins)


