
class Tile(object):
    def __init__(self, bonus: int, points_up: bool):
        self.bonus = bonus
        self.contents = None
        self.owner = None
        self.points_up = points_up


class State(object):
    def __init__(self, state=None):
        if state is None:
            # Build new board
            self.board = [[], []]
            # Init players and hands
            self.hands = tuple()
            self.in_bag = list()
            pass
        else:
            # Deep copy old state
            pass

    def get_tile(self, row, column):
        pass

    def get_tile_colours(self, row, column):
        pass

    def get_neighbouring_tiles(self, row, column):
        """
        Gets the neighbouring tiles. For pointing_up: [left, down, right], for down: [top, left, right]
        :param row: ja
        :param column: oke
        :return: iets
        """
        pass

    def get_neighbouring_tile_colours(self, row, column):
        """
        Gets the neighbouring tile colours. For pointing_up: [left, down, right], for down: [top, left, right]
        :param row: ja
        :param column: oke
        :return: iets
        """
        pass

    def generate_board(self):
        pass

    def calculate_score(self):
        # For all players
        pass

    def get_all_possible_moves(self, player:int):
        pass

    def step(self, row, column, triangle):
        # In place!
        pass

    def deepcopy(self):
        pass
