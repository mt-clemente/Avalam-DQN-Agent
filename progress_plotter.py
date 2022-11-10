from datetime import datetime
from matplotlib import pyplot as plt
import os
import parse
from pathlib import Path


# Plot win difference
dir_path = './logs/games'
dirs = filter(os.path.isdir, sorted(Path(dir_path).iterdir(), key=os.path.getmtime))
for dir in dirs:
    
    Y = []
    W = []
    w = 0
    wins = 0
    c = 0
    paths = filter(os.path.isfile, sorted(Path(dir).iterdir(), key=os.path.getmtime))
    if os.path.exists(f'figs/scores/scores_{str(dir).rpartition(os.sep)[2]}.png'):
        print(f"{dir} not updated")
        #continue
    for file in paths:
        try:
            f = open(f"{file}")
            score = f.read().strip().rpartition('Score')[2]
        except:
            continue
        try:
            Y.append(int(parse.parse("{} .{}", score)[0]))
            if Y[-1] > 0:
                w+=1
                wins+=1
            W.append(w)

        # invalid moves are treated as -100
        except:
            Y.append(-100)
            W.append(w)
        c += 1

    X = [i for i in range(c)]
    #plt.plot(X,Y)
    plt.plot(X,[W[i] / (i+1) * 100 for i in range(len(W))])
    plt.xlabel('Epsiode')
    plt.ylabel('Win delta')
    print(dir)
    print(f"winrate : {wins/len(Y) * 100}% out of {len(Y)} games")
    plt.savefig(f'figs/scores/scores_{str(dir).rpartition(os.sep)[2]}.png')
    plt.clf()