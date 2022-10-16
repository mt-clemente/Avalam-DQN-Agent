#!/usr/bin/env python3
"""
Avalam agent.
Copyright (C) 2022, <<<<<<<<<<< YOUR NAMES HERE >>>>>>>>>>>
Polytechnique Montréal

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

"""

from avalam import *


class BestMove():
    def __init__(self) -> None:
        self.move = None
    
    def update(self,act) -> None:
        self.move = act
    

class MyAgent(Agent):

    """My Avalam agent."""

    def play(self, percepts, player, step, time_left):
        """
        This function is used to play a move according
        to the percepts, player and time left provided as input.
        It must return an action representing the move the player
        will perform.
        :param percepts: dictionary representing the current board
            in a form that can be fed to `dict_to_board()` in avalam.py.
        :param player: the player to control in this step (-1 or 1)
        :param step: the current step number, starting from 1
        :param time_left: a float giving the number of seconds left from the time
            credit. If the game is not time-limited, time_left is None.
        :return: an action
            eg; (1, 4, 1 , 3) to move tower on cell (1,4) to cell (1,3)
        """


        MAX_DEPTH = 2 + step // 25
        board = dict_to_board(percepts)

        best_move = BestMove()


        abs(alphabeta(board,0,max_depth=MAX_DEPTH,player = player,alpha = -999,beta = 999,best_move=best_move))
        return best_move.move


def alphabeta(state: Board,depth: int,max_depth: int,player:int, alpha, beta, best_move: BestMove, action = None):
    
    if depth == max_depth or state.is_finished():
        return state.get_score() * 5

    actions = state.get_actions()
    val = - player * 999
    while 1:
        
        try:
            act = next(actions)
        except StopIteration:
            break

        # highly inefficient TODO: implement unplay action and remove temp
        temp = state.clone()
        temp = temp.play_action(act)

        ab = alphabeta(temp, depth + 1, max_depth,-player, alpha, beta, best_move, act)

        if player == 1 :

            if depth == 0 and ab > val :
                print("1 ",val,ab)
                print("nbm")
                best_move.update(act)
                

            val = max(val, ab)
            alpha = max(alpha,val)
            if val > beta:
                break

        else:

            if depth == 0 and ab < val:
                print("2 ",val,ab)
                best_move.update(act)

            val = min(val, ab)
            beta = min(beta,val)
            if val < alpha:
                break

    return val        


def heuristic(state):
    ...

if __name__ == "__main__":
    agent_main(MyAgent())

