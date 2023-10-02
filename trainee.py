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
from itertools import islice
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

TIME_PROP = [0.09,0.11,0.11,0.09,0.08,0.08,0.06,0.06]


class MyAgent(Agent):

    """My Avalam agent."""

    def __init__(self, _train = True,_test = False) -> None:

        self._train = _train
        self._test = _test 

        # ---------- AlphaBeta Param ----------

        # As we dont know exactly how fast the computer the agent is run works,
        # the max depth is mutable, if some time is left after finishing alphabeta
        # the target depth for this move will increase. On the other hand,
        # if time runs out before the end, target depth will decrease.
        self.TARGET_DEPTH = [9] * 13
        self.BEAM_WIDTH = 6
        self.PLAY_TIME = None
        self.start_time = None
        self.TIME_BREAK_FLAG = False

        # ---------- DQN init ----------

        self.date = datetime.now()
        self.Transition = namedtuple('Transition',
                                     ('state', 'next_state', 'reward','weights','indices'))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and _train else "cpu")

        self.num_episode = 0
        self.BATCH_SIZE = 64
        self.GAMMA = 0.95
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1500
        self.TARGET_UPDATE = 25 # number of episodes between target network updates
        self.steps_done = 0
        self.eog_flag = 0

        # optimize our model every TRAIN_FREQ episodes for BATCH_ROUNDS gives
        # us better stability and more importantly is more optimized for our GPU
        self.TRAIN_FREQ = 20
        self.BACTH_ROUNDS = 20
        board_height = 9
        board_width = 9
        self.lr = 3 * 10 ** -4

        self.alpha = 0.7
        self.beta = 0.5
        self.prio_epsilon = 1e-6


        # as the opponent is not learning, the oppnent's moves must be
        # considered an environment reaction. Which means that to store a
        # transition we have to save the next state as the board resulting
        # from the opponent's move, hence the need for a buffer

        self.buffer = MoveBuffer()

        # tensorboard logging
        if _train:
            self.writer = SummaryWriter()
        self.ep_score = 0


        # try to load the model obtained from previous training or create a new one
        try:
            self.policy_net = DQN(board_height, board_width,
                                  1, self.device).to(self.device)
            self.policy_net.load_state_dict(torch.load('models/currDQN.pt'))
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
            size = 400000,
            Transition = self.Transition,
            alpha = self.alpha,
            batch_size=self.BATCH_SIZE
            )



    def play(self, percepts, player, step, time_left):

        # used to divide time between steps        
        if step == 1 or step == 2 or step == 2 * len(TIME_PROP) -1  or step == 2 * len(TIME_PROP):
            self.PLAY_TIME = time_left


        self.TIME_BREAK_FLAG = False
        self.start_time = datetime.now()


        #to help the network train, we always play the positive values
        if player == 1:
            board: Board = dict_to_board(percepts)
        else:
            board: Board = Board(percepts['m'],invert= True)
        best_move = BestMove()


        # play normally  using the policy net as heuristic
        if not self._train:

            #deactivate batch norm layers
            self.policy_net.eval()
            with torch.no_grad():

                #if the model is being tested, we play using the full algorithm
                if self._test:
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
                        
                    print(time_left,(datetime.now() - t).total_seconds()," SCORE ",board.get_score()," TD = ",self.TARGET_DEPTH, " TIMEBRK :",self.TIME_BREAK_FLAG)

                    return best_move.move
                    
                elif random.random() < 0.95:
                
                    best_move.update(self.greed(board))
                    return best_move.move
                else:
                    return random.choice(list(board.get_actions()))        

        #activate batch norm layers
        self.policy_net.train()


        # ----------- TRAIN THE MODEL -----------


        # the move is selected and updated in place
        self.update_best_move(board = board,best_move = best_move)

        reward = calc_reward(board.clone(),step, best_move.move)


        # Observe new state
        next_board = board.clone()
        next_board.play_action(best_move.move)

        # if a move was played before, the current state is the 'environement's' reaction
        # we can push it into our memory, and store the first half of the next experience
        if self.buffer.statem != None:

            self.memory.push(
                self.buffer.statem,
                board,
                self.buffer.reward)

        self.buffer.update(
            statem=torch.tensor(next_board.m)[None, None].float(),
            reward=torch.tensor([reward], device=self.device)
        )


        # Perform one step of the optimization (on the policy network)
        if self.num_episode % self.TRAIN_FREQ == 0: 
            self.optimize_model(self.BACTH_ROUNDS)
        
        # detect end of episode
        if step < self.eog_flag:
            self.num_episode += 1
            self.buffer.reset()
            self.writer.add_scalar("Performance/Episode score",self.ep_score, self.num_episode)
        else:
            self.ep_score = next_board.get_score()
        
        self.eog_flag = step
        
        
        # Update the target network and checkpoint the policy net
        if self.num_episode % (self.TARGET_UPDATE * self.TRAIN_FREQ) == 0:

            self.target_net.load_state_dict(self.policy_net.state_dict())
            torch.save(self.policy_net.state_dict(), f'models/currDQN.pt')
            torch.save(self.policy_net.state_dict(), f'models/checkpoint/model_{self.num_episode}.pt')



        return best_move.move




    def optimize_model(self,batch_number):


        for i in range(batch_number):

            if len(self.memory) < self.BATCH_SIZE:
                return

            # PER proportional sample
            batch = self.memory.sample(self.beta)

            state_batch =  torch.from_numpy(batch.state).float().unsqueeze(1).to(self.device)
            reward_batch =  torch.from_numpy(batch.reward).to(self.device)
            weights = torch.tensor(batch.weights, device=self.device).float()

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch)

            self.writer.add_scalar("Q values/State Values",torch.mean(state_action_values).item(),self.steps_done * self.BACTH_ROUNDS + i)


            # Compute Q(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            # The eval part is not very readable or intuitive, but it is the 
            # faster alternative of all solutions found with our GPU. 

            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            mask = [i for i in range(self.BATCH_SIZE) if batch.next_state[i]]
            
            with torch.no_grad():

                outcomes = [get_all_possible_outcomes(batch.next_state[i]) for i in mask]
                sizes = [len(o) for o in outcomes]
                
                # concat all the outcomes along the axis considered as batch_size by pytorch conv2D
                temp = np.concatenate(outcomes,axis=0)

                outcomes = torch.from_numpy(temp).unsqueeze(1).float().to(self.device)

                # Evaluate everything in one go, gives us better computation performance and
                # better training with batch normalization than evaluating each experience separatelt
                target_vals = self.target_net(outcomes)

                # each state has a different number of outputs so just reshaping will not do
                # the trick. We need to use sizes the reconstruct lists of outcome Q values 
                # corresponding to each initial state
                indices = torch.zeros((len(sizes),max(sizes)),dtype=torch.int64,device=self.device)
                k = 0
                for i in range(len(sizes)):
                    indices[i,0:sizes[i]] = torch.arange(k,k+sizes[i],dtype=torch.int64,device=self.device)
                    k+= sizes[i]
                
                # repeat the values to gather them. Not memory efficient but the
                # best we can do time-wise
                rep = torch.transpose(target_vals.repeat((1,len(sizes))),0,1)
                grouped_target_vals = torch.gather(rep,1,indices)
                
                next_state_values[mask]  =torch.max(grouped_target_vals,dim = 1)[0]



            expected_state_action_values = (
                next_state_values * self.GAMMA) + reward_batch

            self.writer.add_scalar("Q values/Expected Q values",torch.mean(expected_state_action_values).item(),self.steps_done * self.BACTH_ROUNDS + i)
            self.writer.add_scalar("Performance/Batch Rewards",torch.mean(reward_batch),self.steps_done * self.BACTH_ROUNDS + i)


            # Compute Huber loss, gave slightly better performance than basic MSE
            eltwise_criterion = nn.HuberLoss(reduction="none")
            eltwise_loss = eltwise_criterion(state_action_values,
                            expected_state_action_values.unsqueeze(1))
            
            loss = torch.mean(eltwise_loss * weights)

            self.writer.add_scalar("Loss",torch.mean(loss).item(),self.steps_done * self.BACTH_ROUNDS + i)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            #gradient clipping
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            #update priorities for PER
            prio_loss = eltwise_loss.detach().cpu().numpy()
            new_priorities = prio_loss + self.prio_epsilon
            self.memory.update_priorities(batch.indices, new_priorities)


    # update the best_move object using the epsilon greedy policy
    # the policy is updated every episode but the epsilon decay is tuned accordingly

    def update_best_move(self, board, best_move) -> None:
        
        # greedy epsilon policy
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * self.num_episode / self.EPS_DECAY)
        
        if self._train:
                    self.writer.add_scalar("Epsilon",eps_threshold,self.steps_done)
        self.steps_done += 1

        # using policy net
        if random.random() > eps_threshold:
            
            best_move.update(self.greed(board))

            if best_move.move == None:
                raise BaseException("No best move found")
        # random exploration
        else:
            actions = list(board.get_actions())
            act = actions[np.random.randint(len(actions))]
            best_move.update(act)


    # efficient greedy choice of a move using the onlin policy net
    def greed(self, board: Board):

        outcomes_np = get_all_possible_outcomes(board)
        outcomes = torch.from_numpy(outcomes_np).unsqueeze(1).float().to(self.device)
        target_vals = self.policy_net(outcomes)
        idx = torch.argmax(target_vals)

        return next(islice(board.get_actions(),idx,None),None)



    # Alpha Beta pruning algorithm, best_move is modified in place.
    def alphabeta(self, state: Board, depth: int, player: int, alpha: float, beta: float, best_move: BestMove, step: int):

        # exacerbate win or lose moves
        if state.is_finished():
            return np.sign(state.get_score()) * 100


        # test is okay because of python lazy bool evaluation         
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

            # stop algorithm if running out of time
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


    # update max depth if alphabeta is computed faster than expected
    def update_depth(self,step):

        if (step - 1) // 2 < len(self.TARGET_DEPTH):
            if self.TIME_BREAK_FLAG:
                self.TARGET_DEPTH[(step-1)//2] -= 1
            else:
                self.TARGET_DEPTH[(step-1)//2] += 1




# ---------- UTIL METHODS ----------


# most efficient way found to generate all outcomes
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


# reward system
def calc_reward(state: Board, step,action):

    # the game cannot be finished quicker than 16 moves / player
    # this saves us quite a lot of computing
    if step == 16:
        return 0

    new_state = state.clone().play_action(action)

    if new_state.is_finished():
        return new_state.get_score()
    
    actions = list(new_state.get_actions())
    
    s = []
    for i in range(len(actions)):
        opp_state = new_state.clone().play_action(actions[i])
        if opp_state.is_finished():
            s.append(opp_state.get_score())
    
    if s:
        return min(s)
    
    return 0
    



# ----------- MAIN CALL -----------

if __name__ == "__main__":
    agent_main(MyAgent(_train = True))
