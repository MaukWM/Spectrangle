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


def softmax(v, temperature):
    exps = np.exp(v/temperature)
    return exps/np.sum(exps)

def transform_hand(hand, zero_new_item):
    hand_state = []
    for i in range(4):
        if len(hand) <= i:
            hand_state += [c_to_oh(None) + [0]*8]*3
        elif i == 3 and zero_new_item:
            hand_state += [[0]*16]*3
        else:
            for colour in hand[i].colours:
                hand_state.append(c_to_oh(colour) + [0]*8)
    return hand_state


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

    hand_state = transform_hand(s.hands[player_index], zero_new_item)
    hand_state_1 = transform_hand(s.hands[(player_index+1)%4], False)
    hand_state_2 = transform_hand(s.hands[(player_index+2)%4], False)
    hand_state_3 = transform_hand(s.hands[(player_index+3)%4], False)

    info = [to_oh(turns_till_turn, 4) + [s.scores[player_index]/100, max(s.scores)/100, int(max(s.scores) == s.scores[player_index])] + [0]*(8+1)]
    return np.array(info + hand_state + hand_state_1 + hand_state_2 + hand_state_3 + triangle_states)


class QAgent(agent.Agent):
    def __init__(self, index, n_players, model: ks.Model, gamma=1.0, reward_scale=1/100, use_win_rewards=False):
        super().__init__(index, n_players)
        self.model = model
        self.gamma = gamma
        self.replay = []
        self.replay_size = 100000
        self.reward_scale = reward_scale

        self.use_win_rewards = use_win_rewards

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
        if not self.use_win_rewards:
            values = np.array(rewards)*self.reward_scale + self.gamma * values
        else:
            values = self.gamma * values
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

    def train(self, epochs, epsilon, epsilon_decay_per_episode, epsilon_min, test_against=[], test_every_n_epsiodes=10, number_of_test_games=10, use_softmax_with_temperature=False):
        """
        Trains the Agent
        :param epochs: Number of epochs/episodes
        :param epsilon: Starting exploration rate (when using softmax this is used as temperature)
        :param epsilon_decay_per_episode: After every episode epsilon is multiplied with this factor
        :param epsilon_min: The minimum value for epsilon
        :param test_against: Test agents used to determine winrates
        :param test_every_n_epsiodes: Number of episodes between evaluation
        :param number_of_test_games: Number of test games per evaluation (each game is played 4 times to vary start positions!)
        :param use_softmax_with_temperature: If true: use softmax with the epsilon as temperature. This allows for more directed exploration
        :return: tuple of 2 lists: episode numbers and winrate for that moment in time
        """
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
                o0 = transform_state(s, 0, 0)
                o1 = transform_state(s, 1, 1)
                o2 = transform_state(s, 2, 2)
                o3 = transform_state(s, 3, 3)

                print("Training episode ", episode, "epsilon: ", max(epsilon*(epsilon_decay_per_episode**episode), epsilon_min))
                sum_loss = 0
                steps = 0
                while done_counter > 0:
                    steps += 1
                    if not use_softmax_with_temperature:
                        if random.random() > max(epsilon*(epsilon_decay_per_episode**episode), epsilon_min):
                            moves, values = self.get_moves_with_values(s, player)
                            i = int(np.argmax(values))
                            mv = moves[i]
                        else:
                            mv = random.choice(list(s.get_all_possible_moves(player)))
                    else:
                        moves, values = self.get_moves_with_values(s, player)
                        X = random.random()
                        probs = softmax(values, max(epsilon*(epsilon_decay_per_episode**episode), epsilon_min))

                        cum_prob = 0
                        mv = moves[-1]
                        for index, p in enumerate(probs):
                            cum_prob += p
                            if cum_prob > X:
                                mv = moves[index]

                    s_prime, r, done = s.generate_step(mv, player)
                    if done:
                        done_counter -= 1

                    o_p0 = transform_state(s_prime, 0, (player + 1) % 4)
                    o_p1 = transform_state(s_prime, 1, (player + 2) % 4)
                    o_p2 = transform_state(s_prime, 2, (player + 3) % 4)
                    o_p3 = transform_state(s_prime, 3, (player + 0) % 4)

                    if self.use_win_rewards:
                        r = 0

                    self.replay += [
                        (o0, r*self.reward_scale, o_p3, False),
                        (o1, -r*self.reward_scale, o_p0, False),
                        (o2, -r*self.reward_scale, o_p1, False),
                        (o3, -r*self.reward_scale, o_p2, False)
                    ]

                    x, y = self.sample_batch()
                    history = self.model.fit(x, y, verbose=False, batch_size=32)
                    loss = history.history['loss']
                    loss = np.mean(loss)
                    sum_loss += loss
                    player = (player + 1) % 4
                    s = s_prime
                    o0 = o_p0
                    o1 = o_p1
                    o2 = o_p2
                    o3 = o_p3

                if self.use_win_rewards:
                    winner = np.argmax(s.scores)
                    winner_i = (winner - player)%4
                    self.replay += [
                        (o0, 1 if winner_i == 0 else -1, o0, True),
                        (o1, 1 if winner_i == 1 else -1, o1, True),
                        (o2, 1 if winner_i == 2 else -1, o2, True),
                        (o3, 1 if winner_i == 3 else -1, o3, True)
                    ]
                else:
                    self.replay += [
                        (o0, 0, o0, True),
                        (o1, 0, o1, True),
                        (o2, 0, o2, True),
                        (o3, 0, o3, True)
                    ]

                mean_loss = sum_loss/steps
                print("Mean training loss: ", mean_loss)

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
        reg = ks.regularizers.l2(0.00005)

        inp = ks.Input((1 + 3*4*4 + 3*36, 16))
        info = ks.layers.Lambda(lambda x: x[:, 0])(inp)
        hand = ks.layers.Lambda(lambda x: x[:, 1:3*4*4+1])(inp)
        board = ks.layers.Lambda(lambda x: x[:, 3*4*4+1:])(inp)

        triangle_conv1 = ks.layers.Conv1D(24, 1, activation='selu', kernel_regularizer=reg)
        triangle_conv2 = ks.layers.Conv1D(64, 3, strides=3, activation='selu', kernel_regularizer=reg)
        triangle_conv3 = ks.layers.Conv1D(32, 1, strides=1, activation='selu', kernel_regularizer=reg)

        board = triangle_conv1(board)
        triangles = triangle_conv2(board)
        triangles = triangle_conv3(triangles)
        board = ks.layers.Flatten()(triangles)
        board = ks.layers.Dense(128, activation='selu', kernel_regularizer=reg)(board)

        hand = triangle_conv1(hand)
        hand = triangle_conv2(hand)
        hand = triangle_conv3(hand)
        hand = ks.layers.Flatten()(hand)
        hand = ks.layers.Dense(24, activation='selu', kernel_regularizer=reg)(hand)

        x = ks.layers.Concatenate(axis=1)([info, hand, board])
        x = ks.layers.Dense(64, activation='selu', kernel_regularizer=reg)(x)
        x = ks.layers.Dense(32, activation='selu', kernel_regularizer=reg)(x)
        out = ks.layers.Dense(1, activation='linear')(x)

        model = ks.Model(inputs=inp, outputs=out)
        model.compile(optimizer=ks.optimizers.Adam(0.0001), loss='mse')
    model.summary()

    q_ag = QAgent(0, 4, model, gamma=0.93, use_win_rewards=False)

    agents = [
        q_ag,
        OneLookAheadAgent(1, 4),
        OneLookAheadAgent(2, 4),
        OneLookAheadAgent(3, 4),
    ]

    episodes, winrates = \
        q_ag.train(2000, 0.3, 0.97, epsilon_min=0.005, test_every_n_epsiodes=50, number_of_test_games=10, test_against=agents[1:], use_softmax_with_temperature=True)

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


