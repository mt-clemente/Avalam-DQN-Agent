from datetime import datetime
import random
import subprocess
import time
import os
import psutil


# This file is used to train our modelit can be used to launch a round of
# training or multiple generations. For later generations, we can train
# against different oponents, the trained player plays randomly as 
# player 1 or player 2.
# If no gen



def train(gen: int, nb_ep: int,port = 8010, trainer: str = None):



    try :
        #chose ports change them just to be safe in case a port doesnt get freed
        port1 = port + 2 * gen
        port_trainer = port + 1 + 2 * gen
        prot_greedy = port - 10
        #initialize agents
        # they both start with the same model but only one is training, the other is idle.



        player = 1
        sp1 = subprocess.Popen(f"python3 trainee.py -b localhost --port {port1}",shell=True)

        time.sleep(5)
        
        sp2 = subprocess.Popen(f"python3 greedy_player.py -b localhost --port {prot_greedy}",shell=True)
    
        time.sleep(5)
        if trainer: 
            os.system(f"cp trainer  models/currDQN.pt")
            sp3 = subprocess.Popen(f"python3 trainer.py -b localhost --port {port_trainer}",shell=True)
        
        else:
            os.system(f"cp models/currDQN.pt session_models/model_{gen + 1}.pt ")
            sp3 = subprocess.Popen(f"python3 trainer.py -b localhost --port {port_trainer}",shell=True)
        
        #wait for agents to be running
        time.sleep(5)


        start = time.time()

        # -------- TRAINING LOOP --------

        for i in range(nb_ep):
            
            #random playing first or second
            if random.random() > 0.2:
                port2 = port_trainer
            else:
                port2 = prot_greedy

            t = datetime.now()
            if random.random() > 0.5:
                player = 1
                subprocess.call(f"python3 game.py http://localhost:{port1} http://localhost:{port2} --no-gui",shell=True,stdout=subprocess.DEVNULL)
            else:
                player = 2
                subprocess.call(f"python3 game.py http://localhost:{port2} http://localhost:{port1} --no-gui",shell=True,stdout=subprocess.DEVNULL)

            

            print(f"GEN {gen} -- episode {i} done in {datetime.now() - t} as player {player}")


        end = time.time()

        print("Total time = ", end - start)
        print("Episode avg time = ", (end - start)/nb_ep)

    except KeyboardInterrupt:
        pass


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
    
    if trainer:
        # player 2
        parent = psutil.Process(sp3.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()

    print("\nPROCESSES TERMINATED")
    
    # save the current model to use as trainer for the next generation
    os.system(f"cp models/currDQN.pt session_models/model_{gen + 1}.pt ")
    





# ---------------  GENERATION TRAINING --------------- 


PORT = 8120
NB_GEN = 1
EP_PER_GEN = 35000
INIT_MODEL = None


date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# For now initialize with greedy as first trainer
if INIT_MODEL:
    train(-1,nb_ep=EP_PER_GEN,init_model=INIT_MODEL,port = PORT)



for gen in range(NB_GEN):

    print(f"--------------- GENERATION {gen} START --------------- ")
    train(gen,nb_ep=EP_PER_GEN, port = PORT)
    print(f"--------------- GENERATION {gen} DONE --------------- \n\n")

