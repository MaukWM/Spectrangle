import random

import move
import state
from agents import agent


class HumanAgent(agent.Agent):

    def get_move(self, s: state.State) -> move.Move:
        possible_moves = s.get_all_possible_moves(self.index)
        possible_moves = list(possible_moves)
        while True:
            print("Hand index, Row and Col are 1 indexed!")
            hand_index = input("Position in hand: ")
            row = input("Row: ")
            column = input("Column: ")
            rotation = input("<rot 0>: " + str(s.hands[self.index][int(hand_index) - 1].rotate(0)) + " " +
                             "<rot 1>: " + str(s.hands[self.index][int(hand_index) - 1].rotate(1)) + " " +
                             "<rot 2>: " + str(s.hands[self.index][int(hand_index) - 1].rotate(2)) + ": ")

            valid_move = False
            mv = move.PlaceMove(int(row)-1, int(column)-1, int(hand_index)-1, int(rotation))
            for possible_move in possible_moves:
                if move.PlaceMove.eq(mv, possible_move):
                    valid_move = True

            if valid_move:
                return mv
            else:
                print("Invalid move! Try again...\n")
                continue
