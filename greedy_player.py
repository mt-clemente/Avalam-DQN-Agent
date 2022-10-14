#!/usr/bin/env python3
"""
Greedy Avalam agent.
Copyright (C) 2022, Teaching team of the course INF8215 
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
import random
from avalam import *


class GreedyAgent(Agent):

    def __init__(self):

        self.probability = 0.8
    
    """A dumb random agent."""
    def play(self, percepts, player, step, time_left):
        
        
        board = dict_to_board(percepts)
        actions = list(board.get_actions())
        print('step', step, 'player', player, 'actions', len(actions))
                
        def predict_score(board, action):
            board = board.clone()
            board.play_action(action)
            return board.m[action[2]][action[3]]

        def decision(probability):
            return random.random() < probability

        #print(actions)
        order = [player*5,player*4,player*3,player*2,-player*2,-player*3,-player*4,-player*5]
        srt = {b: i for i, b in enumerate(order)}
        sorted_actions = sorted(actions, key=lambda a: srt[predict_score(board, a)])
        
        print(sorted_actions[0])
        if decision(self.probability):
            return sorted_actions[0]      
        else :
            return random.choice(actions)        


if __name__ == "__main__":
    agent_main(GreedyAgent())
