import copy
import triangle

class Tile(object):
    def __init__(self, points_up: bool, bonus: int = 1):
        self.bonus = bonus
        self.contents = None
        self.owner = None
        self.points_up = points_up

    def __repr__(self):
        if self.contents is None:
            return "EEE"
        c0, c1, c2 = self.contents.colours

        if self.points_up:
            background = "\033[3%sm_" % triangle.colour_strings[c1]
            char_0 = "\033[3%sm/" % triangle.colour_strings[c0]
            char_1 = "\033[3%sm\\" % triangle.colour_strings[c2]
            return char_0 + background + char_1 + "\033[39m"
        else:
            background = "\033[3%sm¯" % triangle.colour_strings[c0]
            char_0 = "\033[3%sm\\" % triangle.colour_strings[c1]
            char_1 = "\033[3%sm/" % triangle.colour_strings[c2]
            return char_0 + background + char_1 + "\033[39m"


class State(object):
    def __init__(self, state=None):
        if state is None:
            # Build new board
            self.board = self.generate_board()
            # Init players and hands
            self.hands = tuple()
            self.in_bag = list()
            pass
        else:
            # Deep copy old state
            pass

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

    def step(self, row, column, triangle):
        # In place!
        pass

    def generate_step(self, row, column, hand_index):
        # Not in place
        new_state = copy.deepcopy(self)
        new_state.step(row, column, hand_index)
        return new_state


if __name__ == "__main__":
    state = State()
    state.board[1][1].contents = triangle.all_triangles[2]
    print(state.board)

