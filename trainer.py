from avalam import agent_main
from trainee import MyAgent

# This file is used to use the model as a trainer:
# it doesn't optimize the model and chooses actions with
# a greedy policy using a separate policy net from the one which is
# getting updated
if __name__ == "__main__":
    agent_main(MyAgent(_train = False))