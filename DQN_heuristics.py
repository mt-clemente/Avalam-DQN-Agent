from collections import deque, namedtuple
from matplotlib import pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from my_player import MyAgent


""" Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") """


class ReplayMemory(object):

    def __init__(self, capacity,Transition):
        self.memory = deque([], maxlen=capacity)
        self.Transition = Transition

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    #TODO: fix h and w
    def __init__(self, h, w, outputs,device):
        super(DQN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 1
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))



""" def plot_scores(episode_scores):
    plt.figure(2)
    plt.clf()
    score_t = torch.tensor(episode_scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(score_t.numpy())

    # Take 100 episode averages and plot them too
    if len(score_t) >= 100:
        means = score_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated """


""" 
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

#FIXME:
board_width = len(percepts)
board_height = len(percepts[0])



# Get number of actions from gym action space
n_outputs = 1

policy_net = DQN(board_height, board_width, n_outputs).to(device)
target_net = DQN(board_height, board_width, n_outputs).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0 """


""" # epsilon greedy policy : sometimes choose using the model, sometimes randomly
# with random probability decreasing
def select_action(state: Board):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:  
        MAX_DEPTH = 2
        best_move = BestMove()
        abs(alphabeta(state,0,max_depth=MAX_DEPTH,player = player,alpha = -999,beta = 999,best_move=best_move,heuristic=policy_net))
        return best_move.move

    else:
        return torch.tensor([[random.randrange(n_outputs)]], device=device, dtype=torch.long) """


def optimize_model(agent: MyAgent):
    if len(agent.memory) < agent.BATCH_SIZE:
        return
    transitions = agent.memory.sample(agent.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = agent.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=agent.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = agent.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(agent.BATCH_SIZE, device=agent.device)
    next_state_values[non_final_mask] = agent.target_net(
        non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * agent.GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()





""" #TODO: state management is horrible due to game interfacen try to work around it
def train_model(num_episodes):

    episode_scores = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action

            #TODO: choose reward system, adapt training to avalam
            action = select_action(state)
            _, reward, done, _, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            next_state = state.clone()
            next_state.play_action(action)

            #TODO: handle end of game issues

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            if done:
                episode_scores.append(t + 1)
                plot_scores()
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
 """



