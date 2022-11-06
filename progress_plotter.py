from datetime import datetime
from matplotlib import pyplot as plt
import os
import parse
from pathlib import Path


dir_path = './logs/games'

# Iterate directory
dirs = filter(os.path.isdir, sorted(Path(dir_path).iterdir(), key=os.path.getmtime))
for dir in dirs:
    
    Y = []
    W = []
    w = 0
    c = 0
    paths = filter(os.path.isfile, sorted(Path(dir).iterdir(), key=os.path.getmtime))
    if os.path.exists(f'scores/scores_{str(dir).rpartition(os.sep)[2]}.png'):
        print(f"{dir} not updated")
        continue
    for file in paths:
        f = open(f"{file}")
        score = f.read().strip().rpartition('Score')[2]
        try:
            Y.append(int(parse.parse("{} .{}", score)[0]))
            if Y[-1] > 0:
                w+=1
            else:
                w-=1    
            W.append(w)

        # invalid moves are treated as -100
        except ValueError:
            Y.append(-100)
        c += 1

    X = [i for i in range(c)]
    plt.plot(X,Y)
    plt.plot(X,W)
    plt.xlabel('Epsiode')
    plt.ylabel('Score')
    print(dir)
    plt.savefig(f'scores/scores_{str(dir).rpartition(os.sep)[2]}.png')
    plt.clf()