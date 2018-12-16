import copy
import random
import move
import triangle


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
    def __init__(self):
        # Build new board
        self.board = self.generate_board()
        # Init players and hands
        self.hands = [[], [], [], []]
        self.bag = triangle.all_triangles[:]
        self.scores = [0, 0, 0, 0]

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
            return self.get_tile(row, column - 1), self.get_tile(row - 1, column - 1), self.get_tile(row, column + 1)

    def get_neighbouring_tile_colours(self, row, column):
        """
        Gets the neighbouring tile colours. For pointing_up: [left, down, right], for down: [top, left, right]
        :param row: ja
        :param column: oke
        :return: iets
        """
        neighbours = self.get_neighbouring_tiles(row, column)
        print(neighbours)

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
        bonus_tiles = [(1, 1, 3),
                       (3, 1, 2),
                       (3, 2, 4),
                       (3, 4, 4),
                       (3, 5, 2),
                       (4, 4, 4),
                       (5, 1, 3),
                       (5, 5, 2),
                       (5, 9, 3)]
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

    def calculate_score(self, row, column):
        colours = self.get_tile_colours(row, column)
        nb_colours = self.get_neighbouring_tile_colours(row, column)

        matching = 0
        for c, nb_c in zip(colours, nb_colours):
            if c == nb_c or c == triangle.Colour.WHITE or nb_c == triangle.Colour.WHITE:
                matching += 1
        tile = self.get_tile(row, column)
        score = tile.contents.score
        score *= matching
        score *= tile.bonus
        return score

    def get_all_empty_tiles(self):
        result = []
        for row in range(6):
            for column in range(len(self.board[row])):
                if self.board[row][column].contents is None:
                    result.append((row, column))
        return result



    def get_all_possible_moves(self, player: int):
        pass

    # @requires shit doesn't break
    def step(self, mv: move.Move, player):

        if isinstance(mv, move.SkipMove):
            return 0, False  # TODO: Check terminal and final score
        elif isinstance(mv, move.ExchangeMove):
            tri = self.hands[player].pop(mv.hand_index)
            new_tri = random.choice(self.bag)
            self.hands.append(new_tri)
            self.bag += tri
            return 0, False
        elif isinstance(mv, move.PlaceMove):
            # In place!
            if self.get_tile(mv.row, mv.column).contents is not None:
                raise Exception("Tried to place at a non-empty tile: (r: %d, c: %s)" % (mv.row, mv.column))
            tri = self.hands[player].pop(mv.hand_index)
            tri = tri.rotate(mv.rotation)
            self.board[mv.row][mv.column] = tri

            reward = self.calculate_score(mv.row, mv.column)
            self.scores[player] += reward
            if len(self.bag) != 0
                self.hands[player].append(random.choice(self.bag))
            else:
                #TODO: Paniek
            return reward, False
        else:
            raise ValueError("Not a valid move!")

    def generate_step(self, mv, player):
        # Not in place
        new_state = copy.deepcopy(self)
        new_state.step(mv, player)
        return new_state

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
            if len(self.hands[i]) != 0:
                hand = "Hand player " + str(i) + ": "
                for j in range(len(self.hands[i])):
                    hand += str(self.hands[i][j]) + " "
                hands += hand + "\n"
        result += hands + "\n"

        # Score
        scores = ""
        for i in range(len(self.scores)):
            if len(self.hands[i]) != 0:
                score = "Score player " + str(i) + ": " + str(self.scores[i]) + "\n"
                scores += score
        result += scores + "\n"

        return result

    def take_triangle(self):
        tri = random.choice(self.bag)
        self.bag.remove(tri)
        return tri

    def fill_hand(self, player):
        while len(self.hands[player]) < 4:
            self.hands[player].append(self.take_triangle())


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
