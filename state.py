import copy
import random

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
        try:
            tile = self.board[row][column]
        except IndexError:
            tile = None
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
        pass

    def get_neighbouring_tiles(self, row, column):
        """
        Gets the neighbouring tiles. For pointing_up: [left, down, right], for down: [top, left, right]
        :param row: ja
        :param column: oke
        :return: iets
        """
        tile = self.get_tile(row, column)
        if tile.points_up:
            return self.get_tile(row, column-1), self.get_tile(row+1, column+1), self.get_tile(row, column+1)
        else:
            return self.get_tile(row, column-1), self.get_tile(row-1, column-1), self.get_tile(row, column+1)

    def get_neighbouring_tile_colours(self, row, column):
        """
        Gets the neighbouring tile colours. For pointing_up: [left, down, right], for down: [top, left, right]
        :param row: ja
        :param column: oke
        :return: iets
        """
        pass

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
            row_length = 1 + i*2
            for j in range(row_length):
                # Get potential bonus
                (x, y) = (i, j)
                bonus = 1
                for k in range(len(bonus_tiles)):
                    if bonus_tiles[k][0] == x and bonus_tiles[k][1] == y:
                        bonus = bonus_tiles[k][2]
                if j%2 == 0:
                    row.append(Tile(points_up=True, bonus=bonus))
                else:
                    row.append(Tile(points_up=False, bonus=bonus))
            board.append(row)
        return board

    def calculate_score(self):
        # For all players
        pass

    def get_all_possible_moves(self, player: int):
        pass

    def step(self, row, column, hand_index, rotation, player):
        # In place!
        if self.get_tile(row, column).contents is not None:
            raise Exception("Tried to place at a non-empty tile: (r: %d, c: %s)" % (row, column))
        # tri = self.hands[player].pop(hand_index)
        # tri = tri.rotate(rotation)




    def generate_step(self, row, column, hand_index, rotation, player):
        # Not in place
        new_state = copy.deepcopy(self)
        new_state.step(row, column, hand_index, rotation, player)
        return new_state

    def __repr__(self):
        result = ""
        # Board
        rows = ""
        for i in range(len(self.board)):
            spaces_amount = len(self.board) - i
            spaces = "".join([" " for j in range(spaces_amount*3)])
            row = spaces
            for j in range(len(self.board[i])):
                row += (str(self.board[i][j]))
            rows += row + "\n"
        result += rows

        # Hands
        hands = ""
        for i in range(len(self.hands)):
            hand = "Hand player " + i + ": "
            for j in range(len(self.hands[i])):
                hand += self.hands[i][j]
            hands += hand
        result += hands

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
    state.board[1][0].contents = triangle.all_triangles[1]
    state.board[1][1].contents = triangle.all_triangles[2]
    state.board[1][2].contents = triangle.all_triangles[3]
    state.board[2][0].contents = triangle.all_triangles[4]
    state.board[2][1].contents = triangle.all_triangles[-1]



    print(state)

