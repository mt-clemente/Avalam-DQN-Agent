from datetime import datetime
import subprocess
import time
import os
import psutil
import parse

from config import A1_PATH, A2_PATH, MOVE_TIME, NB_TEST_EP

# This file is a chain test launcher
# Its only purpose is to check our model's
# performance against other agents.

try:

    #choose ports
    port1 = 8661
    port2 = 8662

    as_player = 1





    #initialize agents
    os.system(f"cp {A1_PATH} models/currDQN.pt")
    sp1 = subprocess.Popen(f"python3 tester.py -b localhost --port {port1}",shell=True,stderr = subprocess.DEVNULL)

    time.sleep(5)

    os.system(f"cp {A2_PATH} models/currDQN.pt")
    # os.system("cp saved_models/18nov-23000/model_21900.pt models/currDQN.pt")
    sp2 = subprocess.Popen(f"python3 greedy_player.py -b localhost --port {port2}",shell=True,stdout=subprocess.DEVNULL,stderr = subprocess.DEVNULL)
    time.sleep(5)
    
    #test parameters
    episodes = NB_TEST_EP
    start = time.time()

    if as_player == 2:
        port2,port1 = port1,port2

    wins = 0
    played = 0
    for i in range(episodes):
        t = datetime.now()
        p = subprocess.Popen(f"python3 game.py http://localhost:{port1} http://localhost:{port2} --no-gui --time {MOVE_TIME}",shell=True,stdout=subprocess.PIPE)
        out,err = p.communicate()
        score = out.decode().rpartition('Score')[2]
        score = int(parse.parse("{} .{}", score)[0])
        if score * (3 - 2 * as_player)> 0:
            wins += 1
            print(f"episode {i} WON - s = {score}")
        else:
            print(f"episode {i} LOST  - s = {score}")
        played += 1



    end = time.time()



    # noticed some issues when killing processes right away. TODO: fix that

except :
    pass

try:
    print(f'winrate : {wins/played * 100}% on {played} games')
except:
    pass


time.sleep(2)

# we need to carefully kill all the child process in order to free the ports
# if we do not, there will be dead processes occupying them which prevents us from using
# them, uses mem etc.

# player 1
parent_pid = os.getpgid(sp1.pid) 
parent = psutil.Process(parent_pid)
for child in parent.children(recursive=True):
    child.kill()
parent.kill()


# player 2
parent_pid = os.getpgid(sp2.pid) 
parent = psutil.Process(parent_pid)
for child in parent.children(recursive=True):
    child.kill()
parent.kill()
