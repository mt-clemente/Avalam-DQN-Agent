from datetime import datetime
import random
import subprocess
import sys
import time
import os
import psutil


def train(gen: int, nb_ep: int, init_model = None):




    #chose ports change them just to be safe in case a port doesnt get freed
    port1 = 8481 + 2 * gen
    port2 = 8482 + 2 * gen

    #initialize agents
    # they both start with the same model but only one is training, the other is idle.



    player = 1
    sp1 = subprocess.Popen(f"python3 trainee.py -b localhost --port {port1}",shell=True)
    
    if init_model:
        sp2 = subprocess.Popen(f"python3 {init_model}.py -b localhost --port {port2}",shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    
    else:
        os.system(f"cp session_models/model_{gen}.pt models/currDQN.pt")
        sp2 = subprocess.Popen(f"python3 trainer.py -b localhost --port {port2}",shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    
    
    #wait for agents to be running
    time.sleep(5)


    start = time.time()

    # -------- TRAINING LOOP --------

    for i in range(nb_ep):
        
        #random playing first or second
        if random.random() > 1:
            port1, port2 = port2, port1
            player = player%2 + 1

        t = datetime.now()
        subprocess.call(f"python3 game.py http://localhost:{port1} http://localhost:{port2} --no-gui",shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

        print(f"GEN {gen} -- episode {i} done in {datetime.now() - t}")


    end = time.time()

    print("Total time = ", end - start)
    print("Episode avg time = ", (end - start)/nb_ep)



    # we need to carefully kill all the child process in order to free the ports
    # if we do not, there will be dead processes occupying them which leads to errors

    # player 1
    parent = psutil.Process(sp1.pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()


    # player 2
    parent = psutil.Process(sp2.pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()


    # save the current model to use as trainer for the next generation
    os.system(f"cp models/currDQN.pt session_models/model_{gen + 1}.pt ")
    




# ---------------  GENERATION TRAINING --------------- 



NB_GEN = 0
EP_PER_GEN = 20
INIT_MODEL = "greedy_player"


date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# For now initialize with greedy as first trainer
if INIT_MODEL:
    train(-1,nb_ep=EP_PER_GEN,init_model=INIT_MODEL)

for gen in range(NB_GEN):

    print(f"--------------- GENERATION {gen} START --------------- ")
    train(gen,nb_ep=EP_PER_GEN)
    print(f"--------------- GENERATION {gen} DONE --------------- \n\n")

