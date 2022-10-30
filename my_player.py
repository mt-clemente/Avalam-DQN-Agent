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

from collections import namedtuple
from random import random
import numpy as np
from DQN_heuristics import DQN,ReplayMemory,optim, optimize_model
from avalam import *
import torch

class BestMove():
    def __init__(self) -> None:
        self.move = None
    
    def update(self,act) -> None:
        self.move = act
    

class MyAgent(Agent):

    """My Avalam agent."""
    def __init__(self) -> None:

        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_episode = 0
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        self.steps_done = 0


        #FIXME:
        board_height = 20
        board_width = 20

        try :
            self.policy_net = torch.load()
        except FileNotFoundError:
            self.policy_net = DQN(board_height, board_width, 1, self.device).to(self.device) 
        
        self.target_net = DQN(board_height, board_width, 1).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000,self.Transition)




    def play(self, percepts, player, step, time_left, _train = False):
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
        
        board = dict_to_board(percepts)

        MAX_DEPTH = 2 + step // 25

        best_move = BestMove()



        # play normally
        if not _train:
            abs(alphabeta(board,0,max_depth=MAX_DEPTH,player = player,alpha = -999,beta = 999,best_move=best_move,heuristic=self.policy_net))
            return best_move.move


        # ----------- TRAIN THE MODEL -----------

            
        # Select and perform an action

        # epsilon policy
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:  
            abs(alphabeta(board,0,max_depth=MAX_DEPTH,player = player,alpha = -999,beta = 999,best_move=best_move,heuristic=self.policy_net))
        else:
            abs(alphabeta(board,0,max_depth=MAX_DEPTH,player = player,alpha = -999,beta = 999,best_move=best_move))
            
        reward = calc_reward(board,best_move)
        reward = torch.tensor([reward], device=self.device)

        # Observe new state
        next_board = board.clone()
        next_board.play_action(best_move.move)

        #TODO: handle end of game issues

        # Store the transition in memory
        self.memory.push(board, best_move.move, next_board, reward)

        # Move to the next board
        board = next_board

        # Perform one step of the optimization (on the policy network)
        optimize_model(self)



        #TODO: handle end of training
        #TODO: handle number of episodes -- see how the end of the match is handled
        # Update the target network, copying all weights and biases in DQN
        if self.num_episode % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())



        return best_move.move


def calc_reward(state: Board, action):
    state.play_action(action).get_score()


def alphabeta(state: Board,depth: int,max_depth: int,player:int, alpha, beta, best_move: BestMove, heuristic = None):
    
    if depth == max_depth or state.is_finished():
        if not heuristic:
            return state.get_score() * 5
        else:
            return heuristic(state)

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



if __name__ == "__main__":
    agent_main(MyAgent())

