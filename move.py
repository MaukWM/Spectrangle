class Move(object):
    pass


class SkipMove(Move):
    pass


class ExchangeMove(Move):
    def __init__(self, hand_index):
        self.hand_index = hand_index


class PlaceMove(Move):
    def __init__(self, row, column, hand_index, rotation):
        self.row = row
        self.column = column
        self.hand_index = hand_index
        self.rotation = rotation
