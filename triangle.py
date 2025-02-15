from enum import Enum


class Colour(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2
    YELLOW = 3
    PURPLE = 4
    WHITE = 5


colour_strings = {
    Colour.RED: "196",
    Colour.BLUE: "32",
    Colour.GREEN: "82",
    Colour.YELLOW: "11",
    Colour.PURPLE: "129",
    Colour.WHITE: "256"
}

class Triangle(object):
    def __init__(self, colours: tuple, score: int):
        self.colours = colours
        self.score = score

    def rotate(self, rotation):
        # Returns rotated triangle

        c0, c1, c2 = self.colours

        if rotation == 0:
            return Triangle((c0, c1, c2), self.score)
        elif rotation == 1:
            return Triangle((c2, c0, c1), self.score)
        elif rotation == 2:
            return Triangle((c1, c2, c0), self.score)
        else:
            # This should not happen, but will work in case it does
            return self.rotate(rotation % 3)

    def __repr__(self):
        if self.colours[0] == Colour.WHITE:
            return "\033[1m/_\\\033[0m"
        else:
            c0, c1, c2 = self.colours
            background = "\033[38;5;%sm" % colour_strings[c1] + str(self.score)
            char_0 = "\033[38;5;%sm/" % colour_strings[c0]
            char_1 = "\033[38;5;%sm\\" % colour_strings[c2]
            return char_0 + background + char_1 + "\033[39m"



# Consists of all triangles possible in the game
all_triangles = []

normal_colours = [Colour.RED, Colour.BLUE, Colour.GREEN, Colour.YELLOW, Colour.PURPLE]
all_triangles += [Triangle((c, c, c), 6) for c in normal_colours]

five_point_colours = [
    (Colour.RED, Colour.YELLOW),
    (Colour.RED, Colour.PURPLE),
    (Colour.BLUE, Colour.RED),
    (Colour.BLUE, Colour.PURPLE),
    (Colour.GREEN, Colour.RED),
    (Colour.GREEN, Colour.BLUE),
    (Colour.YELLOW, Colour.GREEN),
    (Colour.YELLOW, Colour.BLUE),
    (Colour.PURPLE, Colour.YELLOW),
    (Colour.PURPLE, Colour.GREEN),
]
all_triangles += [Triangle((c0, c0, c1), 5) for c0, c1 in five_point_colours]

four_point_colours = [
    (Colour.RED, Colour.BLUE),
    (Colour.RED, Colour.GREEN),
    (Colour.BLUE, Colour.GREEN),
    (Colour.BLUE, Colour.YELLOW),
    (Colour.GREEN, Colour.YELLOW),
    (Colour.GREEN, Colour.PURPLE),
    (Colour.YELLOW, Colour.RED),
    (Colour.YELLOW, Colour.PURPLE),
    (Colour.PURPLE, Colour.RED),
    (Colour.PURPLE, Colour.BLUE),
]
all_triangles += [Triangle((c0, c0, c1), 4) for c0, c1 in four_point_colours]

three_point_colours = [
    (Colour.YELLOW, Colour.BLUE, Colour.PURPLE),
    (Colour.RED, Colour.GREEN, Colour.YELLOW),
    (Colour.BLUE, Colour.GREEN, Colour.PURPLE),
    (Colour.GREEN, Colour.RED, Colour.BLUE),
]

all_triangles += [Triangle((c0, c1, c2), 3) for c0, c1, c2 in three_point_colours]

two_point_colours = [
    (Colour.BLUE, Colour.RED, Colour.PURPLE),
    (Colour.YELLOW, Colour.PURPLE, Colour.RED),
    (Colour.YELLOW, Colour.PURPLE, Colour.GREEN),
]

all_triangles += [Triangle((c0, c1, c2), 2) for c0, c1, c2 in two_point_colours]

one_point_colours = [
    (Colour.GREEN, Colour.RED, Colour.PURPLE),
    (Colour.BLUE, Colour.YELLOW, Colour.GREEN),
    (Colour.RED, Colour.YELLOW, Colour.BLUE),
    (Colour.WHITE, Colour.WHITE, Colour.WHITE),
]

all_triangles += [Triangle((c0, c1, c2), 1) for c0, c1, c2 in one_point_colours]

if __name__ == "__main__":
    print(len(all_triangles))

    tri = all_triangles[30]
    print(list([tri.rotate(i) for i in range(3)]))