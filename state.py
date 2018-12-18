import copy
import random
import move
import triangle

bonus_tiles = [(1, 1, 3),
               (3, 1, 2),
               (3, 2, 4),
               (3, 4, 4),
               (3, 5, 2),
               (4, 4, 4),
               (5, 1, 3),
               (5, 5, 2),
               (5, 9, 3)]

bonus_set = {(r, c) for r, c, b in bonus_tiles}

class Tile(object):
    def __init__(self, points_up: bool, bonus: int = 1):
        self.bonus = bonus
        self.contents = None
        self.owner = None
        self.points_up = points_up

    def __repr__(self):
        if self.contents is None:
            if self.bonus != 1:
                if self.points_up:
                    return "/" + str(self.bonus) + "\\"
                else:
                    return "\\" + str(self.bonus) + "/"
            else:
                if self.points_up:
                    return "/_\\"
                else:
                    return "\\¯/"
        if self.contents.colours[0] == triangle.Colour.WHITE:
            if self.points_up:
                return "\033[1m/_\\\033[0m"
            else:
                return "\033[1m\\¯/\033[0m"
        c0, c1, c2 = self.contents.colours

        if self.points_up:
            background = "\033[38;5;%sm_" % triangle.colour_strings[c1]
            char_0 = "\033[38;5;%sm/" % triangle.colour_strings[c0]
            char_1 = "\033[38;5;%sm\\" % triangle.colour_strings[c2]
            return char_0 + background + char_1 + "\033[39m"
        else:
            background = "\033[38;5;%sm¯" % triangle.colour_strings[c0]
            char_0 = "\033[38;5;%sm\\" % triangle.colour_strings[c1]
            char_1 = "\033[38;5;%sm/" % triangle.colour_strings[c2]
            return char_0 + background + char_1 + "\033[39m"


class State(object):
    def __init__(self, players=4):
        self.players = players
        # Build new board
        self.board = self.generate_board()
        # Init players and hands
        self.hands = [[], [], [], []]
        self.bag = triangle.all_triangles[:]
        self.scores = [0, 0, 0, 0]
        self.initial = True
        for player in range(players):
            self.fill_hand(player)

    def get_tile(self, row, column):
        if row > 5 or row < 0:
            return None
        elif column < 0 or column >= len(self.board[row]):
            return None
        tile = self.board[row][column]

        return tile

    def get_tile_colours(self, row, column):
        """
        Gets the colours of your tile. For pointing_up: [left, down, right], for down: [top, left, right]
        :param row:
        :param column:
        :return:
        """
        tile = self.get_tile(row, column)
        if tile is None:
            return None
        else:
            if tile.contents is None:
                return None
            else:
                return tile.contents.colours

    def get_neighbouring_tiles(self, row, column):
        """
        Gets the neighbouring tiles. For pointing_up: [left, down, right], for down: [top, left, right]
        :param row: ja
        :param column: oke
        :return: iets
        """
        tile = self.get_tile(row, column)
        if tile.points_up:
            return self.get_tile(row, column - 1), self.get_tile(row + 1, column + 1), self.get_tile(row, column + 1)
        else:
            return self.get_tile(row - 1, column - 1), self.get_tile(row, column - 1), self.get_tile(row, column + 1)

    def get_neighbouring_tile_colours(self, row, column):
        """
        Gets the neighbouring tile colours. For pointing_up: [left, down, right], for down: [top, left, right]
        :param row: ja
        :param column: oke
        :return: iets
        """
        neighbours = self.get_neighbouring_tiles(row, column)

        def get_color(nb, i):
            if nb is None or nb.contents is None:
                return None
            else:
                return nb.contents.colours[i]

        own_tile = self.get_tile(row, column)
        if own_tile.points_up:
            return get_color(neighbours[0], 2), get_color(neighbours[1], 0), get_color(neighbours[2], 1)
        else:
            return get_color(neighbours[0], 1), get_color(neighbours[1], 2), get_color(neighbours[2], 0)

    def generate_board(self):
        board = []
        for i in range(6):
            row = []
            row_length = 1 + i * 2
            for j in range(row_length):
                # Get potential bonus
                (x, y) = (i, j)
                bonus = 1
                for k in range(len(bonus_tiles)):
                    if bonus_tiles[k][0] == x and bonus_tiles[k][1] == y:
                        bonus = bonus_tiles[k][2]
                if j % 2 == 0:
                    row.append(Tile(points_up=True, bonus=bonus))
                else:
                    row.append(Tile(points_up=False, bonus=bonus))
            board.append(row)
        return board

    def calculate_score(self, row, column, tri=None):
        if tri is None:
            colours = self.get_tile_colours(row, column)
        else:
            colours = tri.colours
        nb_colours = self.get_neighbouring_tile_colours(row, column)

        matching = 0
        for c, nb_c in zip(colours, nb_colours):
            if c == nb_c or c == triangle.Colour.WHITE and nb_c is not None or nb_c == triangle.Colour.WHITE:
                matching += 1
        tile = self.get_tile(row, column)
        if self.initial:
            matching = 1
        if tri is None:
            score = tile.contents.score
        else:
            score = tri.score
        score *= matching
        score *= tile.bonus
        return score

    def calculate_action_score(self, action, player):
        if not isinstance(action, move.PlaceMove):
            return 0
        else:
            tri = self.hands[player][action.hand_index].rotate(action.rotation)
            return self.calculate_score(action.row, action.column, tri)

    def matches_colour(self, tri, nb_colours):
        matching = 0
        for c, nb_c in zip(tri.colours, nb_colours):
            if c == nb_c or c == triangle.Colour.WHITE and nb_c is not None or nb_c == triangle.Colour.WHITE:
                matching += 1
        return matching > 0

    def get_all_empty_tiles(self):
        result = []
        for row in range(6):
            for column in range(len(self.board[row])):
                if self.board[row][column].contents is None:
                    result.append((row, column))
        return result

    def get_all_possible_moves(self, player: int):
        possible_place_actions = set()
        empty_tiles = self.get_all_empty_tiles()
        for hand_index, tri in enumerate(self.hands[player]):
            for row, column in empty_tiles:
                if not self.initial or (self.initial and (row, column) not in bonus_set):
                    nb_colours = self.get_neighbouring_tile_colours(row, column)
                    for rotation in range(3):
                        if self.initial or self.matches_colour(tri.rotate(rotation), nb_colours):
                            possible_place_actions.add(move.PlaceMove(row, column, hand_index, rotation))
        if possible_place_actions:
            return possible_place_actions

        possible_moves = {move.SkipMove()}
        if len(self.bag) > 0:
            for hand_index in range(len(self.hands[player])):
                possible_moves.add(move.ExchangeMove(hand_index))
        return possible_moves

    # @requires shit doesn't break
    def step(self, mv: move.Move, player):
        if isinstance(mv, move.SkipMove):
            term_counter = 0
            for pl in range(self.players):
                pl_moves = self.get_all_possible_moves(pl)
                if len(pl_moves) == 1 and isinstance(pl_moves.pop(), move.SkipMove):
                    term_counter += 1
            if term_counter == self.players:
                deduction_points = sum([x.score for x in self.hands[player]])
                self.scores[player] -= deduction_points
                return -deduction_points, True
            else:
                return 0, False
        elif isinstance(mv, move.ExchangeMove):
            tri = self.hands[player].pop(mv.hand_index)
            new_tri = random.choice(self.bag)
            self.hands[player].append(new_tri)
            self.bag.append(tri)
            self.bag.remove(tri)
            return 0, False
        elif isinstance(mv, move.PlaceMove):
            # In place!
            if self.get_tile(mv.row, mv.column).contents is not None:
                raise Exception("Tried to place at a non-empty tile: (r: %d, c: %s)" % (mv.row, mv.column))
            tri = self.hands[player].pop(mv.hand_index)
            tri = tri.rotate(mv.rotation)
            self.board[mv.row][mv.column].contents = tri
            self.board[mv.row][mv.column].owner = player

            reward = self.calculate_score(mv.row, mv.column)
            self.scores[player] += reward
            self.fill_hand(player)

            self.initial = False
            return reward, False
        else:
            raise ValueError("Not a valid move! " + str(mv))

    def generate_step(self, mv, player):
        # Not in place
        new_state = copy.deepcopy(self)
        r, done = new_state.step(mv, player)
        return new_state, r, done

    def __repr__(self):
        result = ""
        # Board
        rows = ""
        for i in range(len(self.board)):
            spaces_amount = len(self.board) - i
            spaces = "".join([" " for j in range(spaces_amount * 3)])
            row = spaces
            for j in range(len(self.board[i])):
                row += (str(self.board[i][j]))
            rows += row + "\n"
        result += rows + "\n"

        # Hands
        hands = ""
        for i in range(len(self.hands)):
            if i < self.players:
                hand = "Hand player " + str(i) + ": "
                for j in range(len(self.hands[i])):
                    hand += str(self.hands[i][j]) + " "
                hands += hand + "\n"
        result += hands + "\n"

        # Score
        scores = ""
        for i in range(len(self.scores)):
            if i < self.players:
                score = "Score player " + str(i) + ": " + str(self.scores[i]) + "\n"
                scores += score
        result += scores

        return result

    def take_triangle(self):
        if len(self.bag) != 0:
            tri = random.choice(self.bag)
            self.bag.remove(tri)
            return tri
        else:
            return None

    def fill_hand(self, player):
        while len(self.hands[player]) < 4:
            new_triangle = self.take_triangle()
            if new_triangle is not None:
                self.hands[player].append(new_triangle)
            else:
                break


if __name__ == "__main__":
    state = State()
    state.board[0][0].contents = triangle.all_triangles[0]
    state.board[1][0].contents = triangle.all_triangles[0]
    state.board[1][1].contents = triangle.all_triangles[0]
    state.board[1][2].contents = triangle.all_triangles[3]
    state.board[2][0].contents = triangle.all_triangles[4]
    state.board[2][1].contents = triangle.all_triangles[-1]

    state.fill_hand(0)
    state.fill_hand(1)

    state.board[2][1].contents = triangle.all_triangles[2]

    print(state.get_neighbouring_tile_colours(1, 1))
    print(state.get_neighbouring_tile_colours(2, 4))
    print(state.get_neighbouring_tile_colours(0, 0))

    print(state.calculate_score(1, 1))
    print(state)

