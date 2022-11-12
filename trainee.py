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
import torch.nn as nn
import numpy as np
from DQN_heuristics import DQN, PrioritizedReplayMemory,  BestMove, MoveBuffer
from avalam import *
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

INF = 2**32-1


class MyAgent(Agent):

    """My Avalam agent."""

    def __init__(self, _train = True) -> None:

        self._train = _train

        # ---------- AlphaBeta init ----------

        self.MAX_DEPTH = 1

        # ---------- DQN init ----------

        self.date = datetime.now()
        self.Transition = namedtuple('Transition',
                                     ('state', 'next_state', 'reward','weights','indices'))

        print(f"CUDA available : {torch.cuda.is_available()}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.num_episode = 0
        self.BATCH_SIZE = 64
        self.GAMMA = 0.95
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 125
        self.steps_done = 0
        self.eog_flag = 0
        self.BACTH_ROUNDS = 5
        board_height = 9
        board_width = 9
        self.lr = 1 * 10 ** -4

        self.alpha = 1.8
        self.beta = 0.4
        self.prio_epsilon = 1e-6

        # as the opponent is not learning, the oppnent's moves must be
        # considered an environment reaction. Which means that to store a
        # transition we have to save the next state as the board resulting
        # from the opponent's move, hence the need for a buffer

        self.buffer = MoveBuffer()

        # logging
        self.writer = SummaryWriter()
        self.ep_score = 0


        # try to load the model obtained from previous training
        try:
            self.policy_net = DQN(board_height, board_width,
                                  1, self.device).to(self.device)
            self.policy_net.load_state_dict(torch.load('models/currDQN.pt',map_location=torch.device(self.device)))
        except FileNotFoundError:
            print("NEW MODEL",sys.stderr)
            self.policy_net = DQN(board_height, board_width,
                                  1, self.device).to(self.device)

        self.target_net = DQN(board_height, board_width,
                              1, self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        

        self.optimizer = optim.Adam(self.policy_net.parameters(),amsgrad=True,lr =self.lr)
        self.memory = PrioritizedReplayMemory(
            size = 50000,
            Transition = self.Transition,
            alpha = self.alpha,
            batch_size=self.BATCH_SIZE
            )

    def play(self, percepts, player, step, time_left):


        if player == 1:
            board: Board = dict_to_board(percepts)
        else:
            board: Board = Board(percepts['m'],invert= True)
        best_move = BestMove()


        # play normally  using the policy net as heuristic
        if not self._train:
            abs(alphabeta(
                    board,
                    0,
                    max_depth=self.MAX_DEPTH,
                    player=player,
                    alpha=-INF,
                    beta=INF,
                    best_move=best_move,
                    heuristic=self.policy_net
                    ))
            return best_move.move



        # ----------- TRAIN THE MODEL -----------

        self.update_best_move(board = board,player = player,best_move = best_move)

        reward = calc_reward(board.clone(), best_move.move)

        # Observe new state
        next_board = board.clone()
        next_board.play_action(best_move.move)


        if self.buffer.statem != None:

            self.memory.push(
                self.buffer.statem,
                board,
                self.buffer.reward)

        self.buffer.update(
            statem=torch.FloatTensor(next_board.m)[None, None],
            reward=torch.tensor([reward], device=self.device)
        )


        # Perform one step of the optimization (on the policy network)
        self.optimize_model(self.BACTH_ROUNDS)
        
        # detect end of episode
        if step < self.eog_flag:
            self.num_episode += 1
            self.buffer.reset()
            self.writer.add_scalar("Performance/Episode score",next_board.get_score(), self.num_episode)
        else:
            self.ep_score = next_board.get_score()
        
        self.eog_flag = step
        
        
        # Update the target network and checkpoint the policy net
        if self.num_episode % self.TARGET_UPDATE == 0:

            self.target_net.load_state_dict(self.policy_net.state_dict())
            torch.save(self.policy_net.state_dict(), f'models/currDQN.pt')



        return best_move.move


    def optimize_model(self,batch_number):


        for i in range(batch_number):

            if len(self.memory) < self.BATCH_SIZE:
                return


            transitions = self.memory.sample(self.beta)

            batch = transitions#self.Transition(*zip(*transitions))
            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            """ non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.bool)

            non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None]) """


            state_batch =  torch.from_numpy(batch.state).float().unsqueeze(1).to(self.device)
            reward_batch =  torch.from_numpy(batch.reward).to(self.device)
            weights = torch.tensor(batch.weights, device=self.device).float()

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch)

            self.writer.add_scalar("Q values/State Values",torch.mean(state_action_values).item(),self.steps_done * self.BACTH_ROUNDS + i)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            mask = [i for i in range(self.BATCH_SIZE) if batch.next_state[i]]


            

            with torch.no_grad():

                outcomes = [get_all_possible_outcomes(batch.next_state[i]) for i in mask]
                sizes = [len(o) for o in outcomes]
                
                temp = np.concatenate(outcomes,axis=0)

                outcomes = torch.from_numpy(temp).unsqueeze(1).float().to(self.device)

                target_vals = self.target_net(outcomes)

                indices = torch.zeros((len(sizes),max(sizes)),dtype=torch.int64,device=self.device)
                
                k = 0
                for i in range(len(sizes)):
                    indices[i,0:sizes[i]] = torch.arange(k,k+sizes[i],dtype=torch.int64,device=self.device)
                    k+= sizes[i]
                
                # padded_target_values = torch.cat((torch.tensor([0],device=self.device).resize(1,1),target_vals))

                rep = torch.transpose(target_vals.repeat((1,len(sizes))),0,1)
                grouped_target_vals = torch.gather(rep,1,indices)
                
                # there are a lot of recurring values due to our process, max works faster with sparse tensors.
                next_state_values[mask]  =torch.max(grouped_target_vals - grouped_target_vals[0,0],dim = 1)[0] + - grouped_target_vals[0,0]




            expected_state_action_values = (
                next_state_values * self.GAMMA) + reward_batch

            self.writer.add_scalar("Q values/Expected Q values",torch.mean(expected_state_action_values).item(),self.steps_done * self.BACTH_ROUNDS + i)
            self.writer.add_scalar("Performance/Rewards",torch.mean(reward_batch),self.steps_done * self.BACTH_ROUNDS + i)


            # Compute Huber loss
            eltwise_criterion = nn.HuberLoss(reduction="none")
            eltwise_loss = eltwise_criterion(state_action_values,
                            expected_state_action_values.unsqueeze(1))
            
            loss = torch.mean(eltwise_loss * weights)
            self.writer.add_scalar("Q values/Loss",torch.mean(expected_state_action_values).item(),self.steps_done * self.BACTH_ROUNDS + i)

            self.writer.add_scalar("Expected Q value",torch.mean(loss).item(),self.steps_done * self.BACTH_ROUNDS + i)
            # Optimize the model

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            #update priorities for PER
            prio_loss = eltwise_loss.detach().cpu().numpy()
            new_priorities = prio_loss + self.prio_epsilon
            self.memory.update_priorities(batch.indices, new_priorities)


    def update_best_move(self, board , player, best_move) -> None:
        
        # greedy epsilon policy
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * self.steps_done / self.EPS_DECAY)

        self.writer.add_scalar("Epsilon",eps_threshold,self.steps_done)
        self.steps_done += 1

        # using policy net
        if random.random() > eps_threshold:
            abs(alphabeta(board, 0, max_depth=self.MAX_DEPTH, player=player, alpha=-
                INF, beta=INF, best_move=best_move, heuristic=self.policy_net))
            if best_move.move == None:
                raise BaseException("No best move found")
        # random exploration
        else:
            actions = list(board.get_actions())
            act = actions[np.random.randint(len(actions))]
            best_move.update(act)


    def extract_groups(self,values,sizes):

        t = torch.zeros((self.BATCH_SIZE,np.max(sizes)),device= self.device)
        s = 0 

        for i in range(self.BATCH_SIZE):   
            t[i][s:s+sizes[i]] = values[s:s+sizes[i]]
            s+=sizes[i]

        return t

        

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



def calc_reward(state: Board, action):
    new_state = state.clone().play_action(action)

    return new_state.get_score() - state.get_score()



# Alpha Beta pruning algorithm, best_move is modified in place
def alphabeta(state: Board, depth: int, max_depth: int, player: int, alpha, beta, best_move: BestMove, action=None, heuristic: DQN = None):

    if depth == max_depth or state.is_finished():
        temp = torch.tensor(state.m).float()
        temp = temp[None, None]
        return heuristic(temp)

    actions = state.get_actions()
    val = - player * INF
    while 1:

        try:
            act = next(actions)
        except StopIteration:
            break

        # highly inefficient TODO: implement unplay action and remove temp
        temp = state.clone()
        temp = temp.play_action(act)

        ab = alphabeta(temp, depth + 1, max_depth, -player, alpha,
                       beta, best_move, action=act, heuristic=heuristic)

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


if __name__ == "__main__":
    agent_main(MyAgent(_train = True))
