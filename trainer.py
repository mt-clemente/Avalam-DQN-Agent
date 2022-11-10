from avalam import agent_main
from trainee import MyAgent


if __name__ == "__main__":
    agent_main(MyAgent(_train = False))