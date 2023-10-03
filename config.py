# ---------- Network config ----------
K1 = 5
K2 = 3
K3 = 2
LINFACT = 32


# ---------- Training config ----------
PORT = 8120
NB_GEN = 1
EP_PER_GEN = 35000
INIT_TRAINER_MODEL = "saved_models/22nov/currDQN.pt"


# ---------- Test config ----------
NB_TEST_EP = 100
A1_PATH = "saved_models/22nov/currDQN.pt"
A2_PATH = "saved_models/17nov-+14400/95vgreedy.pt"
MOVE_TIME = 0.1
# prints score, target depth for alphabeta, was the agent cut short in the
# algorithm or did it reach the target depth
VERBOSE = False 
# You can set a percentage of training game against greedy to prevent cases
# of catastrophic forgetting
GREEDY_PROB=0.2



