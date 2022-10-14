import subprocess
import time
import os
import psutil

#chose ports
port1 = 8001
port2 = 8002

#initialize agents
sp1 = subprocess.Popen(f"python3 greedy_player.py -b localhost --port {port1}",shell=True,stdout=subprocess.DEVNULL,stderr=open("logs/players/log1.txt","w"))
sp2 = subprocess.Popen(f"python3 greedy_player.py -b localhost --port {port2}",shell=True,stdout=subprocess.DEVNULL,stderr=open("logs/players/log2.txt","w"))
time.sleep(2.5)

#training parameters
episodes = 100
start = time.time()

for i in range(episodes):
    s = subprocess.Popen(f"python3 game.py http://localhost:{port1} http://localhost:{port2} --no-gui",shell=True,stdout=open(f"logs/game/log.txt","a"))
    print(i,'th episode done')
end = time.time()


print("Total time = ", end - start)
print("Epoch time = ", (end - start)/episodes)


# noticed some issues when killing processes right away. TODO: fix that
time.sleep(1)


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


# game
parent_pid = os.getpgid(s.pid) 
parent = psutil.Process(parent_pid)
for child in parent.children(recursive=True):
    child.kill()
parent.kill() 