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
from DQN_heuristics import DQN, ReplayMemory
from avalam import *
import torch
import torch.optim as optim

INF = 2**32-1


class BestMove():
    def __init__(self) -> None:
        self.move = None

    def update(self, act) -> None:
        self.move = act


class MoveBuffer():
    def __init__(self) -> None:
        self.state = None
        self.reward = None

    def update(self, state: Board, reward):
        self.state = state
        self.reward = reward
    
    def reset(self):
        self.state = None
        self.reward = None



class MyAgent(Agent):

    """My Avalam agent."""

    def __init__(self) -> None:

        self.date = datetime.now()
        self.Transition = namedtuple('Transition',
                                     ('state', 'next_state', 'reward'))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.num_episode = 0
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 2000
        self.TARGET_UPDATE = 20
        self.steps_done = 0
        self.eog_flag = 0

        board_height = 9
        board_width = 9

        # as the opponent is not learning, the oppnent's moves must be
        # considered an environment reaction. Which means that to store a
        # transition we have to save the next state as the board resulting
        # from the opponent's move, hence the need for a buffer

        self.buffer = MoveBuffer()

        # try to load the model obtained from previous training
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

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000, self.Transition)

    def play(self, percepts, player, step, time_left, _train=True):
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

        # detect end of episode
        if step < self.eog_flag:
            self.num_episode += 1
            self.buffer.reset()

        self.eog_flag = step

        board: Board = dict_to_board(percepts)

        MAX_DEPTH = 1  # + step // 25

        best_move = BestMove()

        # play normally  using the policy net as heuristic
        if not _train:
            abs(alphabeta(
                    board,
                    0,
                    max_depth=MAX_DEPTH,
                    player=player,
                    alpha=-INF,
                    beta=INF,
                    best_move=best_move,
                    heuristic=self.policy_ne
                    ))
            return best_move.move



        # ----------- TRAIN THE MODEL -----------

        # greedy epsilon policy
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        # using policy net
        if random.random() > eps_threshold:
            abs(alphabeta(board, 0, max_depth=MAX_DEPTH, player=player, alpha=-
                INF, beta=INF, best_move=best_move, heuristic=self.policy_net))
            if best_move.move == None:
                raise BaseException("No best move found")
        # random exploration
        else:
            actions = list(board.get_actions())
            act = actions[np.random.randint(len(actions))]
            best_move.update(act)

        reward = calc_reward(board.clone(), best_move.move)

        # Observe new state
        next_board = board.clone()
        next_board.play_action(best_move.move)

        if self.buffer.state != None:

            self.memory.push(
                self.buffer.state,
                board,
                self.buffer.reward)

        self.buffer.update(
            state=torch.FloatTensor(next_board.m)[None, None],
            reward=torch.tensor([reward], device=self.device)
        )

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Update the target network, copying all weights and biases in DQN
        # save the policy net
        if self.num_episode % self.TARGET_UPDATE == 0:

            self.target_net.load_state_dict(self.policy_net.state_dict())

            torch.save(self.policy_net.state_dict(),
                       f'models/mod_{self.date}.pt')
            torch.save(self.policy_net.state_dict(), f'models/currDQN.pt')

        return best_move.move


    def optimize_model(self):

        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = self.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        """ non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]) """

        state_batch = torch.cat(batch.state)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

        with torch.no_grad():
            for i in range(self.BATCH_SIZE):
                # Compute the expected Q values
                state = batch.next_state[i]
                actions = list(state.get_actions())
                if not actions:
                    continue
                try:
                    next_state_values[i] = torch.max(torch.tensor([self.target_net(
                        torch.tensor(
                            state.clone().play_action(act).m)[None, None].float())
                        for act in actions]))
                except:
                    raise BaseException(actions)

        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.HuberLoss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        print(loss, sys.stderr)

        # Optimize the model

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


def calc_reward(state: Board, action):
    new_state = state.clone().play_action(action)

    if new_state.is_finished():
        if new_state.get_score() > 0:
            return +1
        else:
            return -5

    # check if the oponent can win the game
    for act in new_state.get_actions():
        temp = new_state.clone()
        temp.play_action(act)
        if temp.is_finished() and temp.get_score() < 0:
            return -5

    return 0


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
    agent_main(MyAgent())
