from train_launcher import train


from datetime import datetime
import subprocess
import time
import os
import psutil


#TODO: Try to multithread (go back to Popen + wait for all process to finish at then end. Maybe not doable for model optimization)




def train(gen: int, nb_ep: int):


    batch_dir = f'logs/games/batch_{datetime.now()}' 
    sdtout_dir = f'logs/stoudt/batch_{datetime.now()}' 
    os.mkdir(batch_dir)

    #chose ports change them just to be safe in case a port doesnt get freed
    port1 = 8001 + 2 * gen
    port2 = 8002 + 2 * gen

    #initialize agents
    # they both start with the same model but only one is training, the other is idle.

    subprocess.call(f"cp saved_models/model{gen}.pt models/currDQN.pt")
    sp1 = subprocess.Popen(f"python3 trainee.py -b localhost --port {port1}",shell=True,stdout=open(f"{sdtout_dir}.txt","w"),stderr=open("logs/players/log1.txt","w"))
    
    sp2 = subprocess.Popen(f"python3 trainer.py -b localhost --port {port2}",shell=True,stdout=subprocess.DEVNULL,stderr=open("logs/players/log2.txt","w"))
    
    #wait for agents to be running
    time.sleep(5)


    start = time.time()



    # -------- TRAINING LOOP --------

    for i in range(nb_ep):
        f = open(f"{batch_dir}/log_{datetime.now()}.txt","w")
        t = datetime.now()
        subprocess.call(f"python3 game.py http://localhost:{port1} http://localhost:{port2} --no-gui",shell=True,stdout=f)
        print(f"GEN {gen} -- episode {i} done in {datetime.now() - t}")


    end = time.time()

    print("Total time = ", end - start)
    print("Episode avg time = ", (end - start)/nb_ep)


    # noticed some issues when killing processes right away. TODO: fix that
    time.sleep(5)

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
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill() 

    # save the current model to use as trainer for the next generation
    subprocess.call(f"cp models/currDQN.pt saved_models/model{gen + 1}.pt ")
    




# ----------- GENERATION TRAINING

NB_GEN = 10
EP_PER_GEN = 900
INITIAL_MODEL = "greedy"


for gen in range(NB_GEN):


    train(gen,nb_ep=EP_PER_GEN)


