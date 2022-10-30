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
from datetime import datetime
import random
import sys
import traceback
import numpy as np
from DQN_heuristics import DQN,ReplayMemory, optimize_model
from avalam import *
import torch
import torch.optim as optim


class BestMove():
    def __init__(self) -> None:
        self.move = None
    
    def update(self,act) -> None:
        self.move = act
    

class MyAgent(Agent):

    """My Avalam agent."""
    def __init__(self) -> None:

        self.date = datetime.now()
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
        self.eog_flag = 0


        board_height = 9
        board_width = 9

        #try to load the model obtained from previous training
        try :
            self.policy_net = DQN(board_height, board_width, 1, self.device).to(self.device) 
            self.policy_net.load_state_dict(torch.load('models/curr_DQN.pt'))
        except FileNotFoundError:
            self.policy_net = DQN(board_height, board_width, 1, self.device).to(self.device) 
        
        self.target_net = DQN(board_height, board_width, 1,self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000,self.Transition)





    def play(self, percepts, player, step, time_left, _train = True):
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

        if step < self.eog_flag:
            self.num_episode += 1

        self.eog_flag = step
        
        board = dict_to_board(percepts)

        MAX_DEPTH = 2 #+ step // 25

        best_move = BestMove()



        # play normally  using the policy net as heuristic
        if not _train:
            abs(alphabeta(board,0,max_depth=MAX_DEPTH,player = player,alpha = -999,beta = 999,best_move=best_move,heuristic=self.policy_net))
            return best_move.move


        # ----------- TRAIN THE MODEL -----------

        # epsilon policy
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        # using policy net
        if random.random() > eps_threshold: 
            abs(alphabeta(board,0,max_depth=MAX_DEPTH,player = player,alpha = -999,beta = 999,best_move=best_move,heuristic = self.policy_net)) 
        
        # random exploration
        else:
            abs(alphabeta(board,0,max_depth=MAX_DEPTH,player = player,alpha = -999,beta = 999,best_move=best_move, heuristic=self.policy_net))

        try:
            # calculate the reward based on the selected move
            reward = calc_reward(board,best_move.move)

        except:
            raise BaseException("1")
        
        print(reward,file = sys.stderr)
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
        # save the policy net
        if self.num_episode % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

            torch.save(self.policy_net.state_dict(),f'models/mod_{self.date}/pt')
            torch.save(self.policy_net.state_dict(),f'models/currDQN/pt')


        return best_move.move


def calc_reward(state: Board, action):
    return state.play_action(action).get_score()


def alphabeta(state: Board,depth: int,max_depth: int,player:int, alpha, beta, best_move: BestMove, action = None, heuristic: DQN = None):
    

    if depth == max_depth or state.is_finished():
        if not heuristic:
            return state.get_score() * 5
        else:
            temp = torch.tensor(state.m).float()
            temp = temp[None,None]
            try:
                return heuristic(temp)
            except:
                raise ValueError
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

        ab = alphabeta(temp, depth + 1, max_depth,-player, alpha, beta, best_move, action = act, heuristic= heuristic)

        if player == 1 :

            if depth == 0 and ab > val :
                best_move.update(act)
                

            val = max(val, ab)
            alpha = max(alpha,val)
            if val > beta:
                break

        else:

            if depth == 0 and ab < val:
                best_move.update(act)

            val = min(val, ab)
            beta = min(beta,val)
            if val < alpha:
                break

    return val        



if __name__ == "__main__":
    agent_main(MyAgent())

