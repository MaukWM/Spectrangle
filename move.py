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

    @staticmethod
    def eq(self, other):
        return self.row == other.row and self.column == other.column and self.hand_index == other.hand_index \
               and self.rotation == other.rotation

