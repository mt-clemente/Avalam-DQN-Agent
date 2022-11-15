from datetime import datetime
import subprocess
import time
import os
import psutil
import parse

#TODO: Try to multithread (go back to Popen + wait for all process to finish at then end. Maybe not doable for model optimization)



sdout_file = f'logs/stdout/stdout_{datetime.now()}' 

#chose ports
port1 = 8668
port2 = 8669

#initialize agents
sp1 = subprocess.Popen(f"python3 trainee.py -b localhost --port {port2}",shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
sp2 = subprocess.Popen(f"python3 greedy_player.py -b localhost --port {port1}",shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
time.sleep(5)

#test parameters
episodes = 50
start = time.time()

files=[]
wins = 0
for i in range(episodes):
    t = datetime.now()
    p = subprocess.Popen(f"python3 game.py http://localhost:{port1} http://localhost:{port2} --no-gui",shell=True,stdout=subprocess.PIPE)
    out,err = p.communicate()
    score = out.decode().rpartition('Score')[2]
    score = int(parse.parse("{} .{}", score)[0])
    if score > 0:
        wins += 1
        print(f"episode {i} WON - s = {score}")
    else:
        print(f"episode {i} LOST  - s = {score}")



end = time.time()

print(f'winrate : {wins/episodes * 100}% on {episodes} games')


# noticed some issues when killing processes right away. TODO: fix that
time.sleep(2)


# we need to carefully kill all the child process in order to free the ports
# if we do not, there will be dead processes occupying them which leads to errors

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
