from enum import Enum


class Colour(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2
    YELLOW = 3
    PURPLE = 4
    WHITE = 5


class Triangle(object):
    def __init__(self, colours: tuple, score: int):
        self.colours = colours
        self.score = score

    def rotate(self, rotation):
        # Returns rotated triangle
        pass


# Consists of all triangles possible in the game
all_triangles = None
