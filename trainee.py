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

from datetime import datetime
import numpy as np
from DQN_heuristics import DQN, BestMove
from avalam import *
import torch

INF = 2**32-1

TIME_PROP = [0.09,0.11,0.11,0.09,0.08,0.08,0.06,0.06]

class MyAgent(Agent):

    """My Avalam agent."""

    def __init__(self) -> None:


        # ---------- AlphaBeta Param ----------

        self.TARGET_DEPTH = [11] * 13
        self.BEAM_WIDTH = 6
        self.PLAY_TIME = None
        self.start_time = None
        self.TIME_BREAK_FLAG = False

        # ---------- DQN init ----------

        self.device = torch.device("cpu")

        board_height = 9
        board_width = 9

        self.policy_net = DQN(board_height, board_width,
                                1, self.device).to(self.device)
        self.policy_net.load_state_dict(torch.load('currDQN.pt',))



    def play(self, percepts, player, step, time_left):

        
        if step == 1 or step == 2 or step == 2 * len(TIME_PROP) -1  or step == 2 * len(TIME_PROP):
            self.PLAY_TIME = time_left


        self.TIME_BREAK_FLAG = False
        self.start_time = datetime.now()

        if player == 1:
            board: Board = dict_to_board(percepts)
        else:
            board: Board = Board(percepts['m'],invert= True)
        best_move = BestMove()


        # play normally  using the policy net as heuristic
        self.policy_net.eval()
        with torch.no_grad():
                t = datetime.now()
                self.alphabeta(
                        board,
                        0,
                        player=1,
                        alpha=-INF,
                        beta=INF,
                        best_move=best_move,
                        step=step
                        )

                self.update_depth(step)
                    

                return best_move.move



    # Alpha Beta pruning algorithm, best_move is modified in place.

    def alphabeta(self, state: Board, depth: int, player: int, alpha: float, beta: float, best_move: BestMove, step: int):


        if state.is_finished():
            return np.sign(state.get_score()) * 100


        # test is okay because of pyhotn lazy bool evaluation         
        if  (step - 1) // 2 < len(self.TARGET_DEPTH) and  depth == self.TARGET_DEPTH[(step - 1)//2]:
            return self.policy_net(torch.tensor(state.m).float()[None,None])


        actions = list(state.get_actions())


        outcomes_np = get_all_possible_outcomes(state)
        outcomes = torch.from_numpy(outcomes_np).unsqueeze(1).float().to(self.device)
        heur_vals = self.policy_net(outcomes).cpu().numpy()
        heur_vals = heur_vals.reshape((heur_vals.shape[0],))


        # beam search of width self.BEAM_WIDTH
        if step < 1 + 2 * len(TIME_PROP):
            part_idx = np.argsort(- player * heur_vals)[:self.BEAM_WIDTH]
        else:  
            part_idx = np.argsort(- player * heur_vals)

        val = - player * INF

        for idx in part_idx:

            if depth != 0 and self.timebreak(step):
                if val != - player * INF:
                    return val
                else:
                    return self.policy_net(torch.tensor(state.m).float()[None,None])


            act = actions[idx]
            temp = state.clone()
            temp = temp.play_action(act)

            ab = self.alphabeta(temp, depth + 1, -player, alpha,
                        beta, best_move,step)

            if player == 1:

                if depth == 0 and ab > val:
                    best_move.update(act)

                val = max(val, ab)
                alpha = max(alpha, val)
                if val > beta:
                    break

            else:
                if depth == 0 and ab < val:
                    best_move.update(act)

                val = min(val, ab)
                beta = min(beta, val)
                if val < alpha:
                    break


        return val
        

    def timebreak(self,step):

        # return False

        # choose allocated time based on time needed per move according to TIME_PROP
        if step <= 2 * len(TIME_PROP):
            alloc_mv_time = self.PLAY_TIME * TIME_PROP[(step-1) // 2]

        # allocated evenly remaining time for the rest of the moves
        else:
            alloc_mv_time = self.PLAY_TIME / (19 - len(TIME_PROP))    
        
        if (datetime.now() - self.start_time).total_seconds() > alloc_mv_time:

            self.TIME_BREAK_FLAG = True
            return True

        return False

    def update_depth(self,step):

        if (step - 1) // 2 < len(self.TARGET_DEPTH):
            if self.TIME_BREAK_FLAG:
                self.TARGET_DEPTH[(step-1)//2] -= 1
            else:
                self.TARGET_DEPTH[(step-1)//2] += 1




# ---------- UTIL METHODS ----------



def get_all_possible_outcomes(state : Board):

    actions = list(state.get_actions())
    m = np.array(state.m)
    outcomes = np.apply_along_axis(lambda action :_play_action(m,action),axis=1,arr = actions)
    return outcomes


def _play_action(state, action):

    temp = state.copy()
    i1, j1, i2, j2 = action
    h1 = abs(temp[i1][j1])
    h2 = abs(temp[i2][j2])
    if temp[i1][j1] < 0:
        temp[i2][j2] = -(h1 + h2)
    else:
        temp[i2][j2] = h1 + h2
    temp[i1][j1] = 0
    return temp



# ----------- MAIN CALL -----------



if __name__ == "__main__":
    agent_main(MyAgent())
