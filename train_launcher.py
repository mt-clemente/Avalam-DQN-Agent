from datetime import datetime
import subprocess
import time
import os
import psutil


#TODO: Try to multithread (go back to Popen + wait for all process to finish at then end. Maybe not doable for model optimization)



batch_dir = f'logs/games/batch_{datetime.now()}' 
sdout_file = f'logs/stdout/stdout_{datetime.now()}' 
os.mkdir(batch_dir)

#chose ports
port1 = 8328
port2 = 8329

#initialize agents
sp1 = subprocess.Popen(f"python3 trainer.py -b localhost --port {port1}",shell=True,stdout=open(f"{sdout_file}.txt","w"),stderr=open("logs/players/log1.txt","w"))
sp2 = subprocess.Popen(f"python3 random_player.py -b localhost --port {port2}",shell=True,stdout=subprocess.DEVNULL,stderr=open("logs/players/log2.txt","w"))
time.sleep(5)

#training parameters
episodes = 100
start = time.time()

s=[]
files=[]
for i in range(episodes):
    f = open(f"{batch_dir}/log_{datetime.now()}.txt","w")
    t = datetime.now()
    p = subprocess.call(f"python3 game.py http://localhost:{port1} http://localhost:{port2} --no-gui",shell=True,stdout=f)
    print(f"episode {i} done in {datetime.now() - t}")


end = time.time()

print("Total time = ", end - start)
print("Epoch time = ", (end - start)/episodes)


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


# game
parent_pid = os.getpgid(s.pid) 
parent = psutil.Process(parent_pid)
for child in parent.children(recursive=True):
    child.kill()
parent.kill() 