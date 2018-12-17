class Move(object):
    pass


class SkipMove(Move):
    pass


class ExchangeMove(Move):
    def __init__(self, hand_index):
        self.hand_index = hand_index

    def __repr__(self):
        return "ExchangeMove(index=%d)" % (self.hand_index,)


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

    def __repr__(self):
        return "PlaceMove(row=%d, col=%d, h_i=%d, rot=%d)" % (self.row, self.column, self.hand_index, self.rotation)

