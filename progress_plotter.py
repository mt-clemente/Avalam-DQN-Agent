from datetime import datetime
from matplotlib import pyplot as plt
import os
import parse
from pathlib import Path


dir_path = './logs/games'
c = 0
Y = []

# Iterate directory
dirs = filter(os.path.isdir, sorted(Path(dir_path).iterdir(), key=os.path.getmtime))
for dir in dirs:
    paths = filter(os.path.isfile, sorted(Path(dir).iterdir(), key=os.path.getmtime))
    if os.path.exists(f'scores/scores_{str(dir).rpartition(os.sep)[2]}.png'):
        print("not updated")
        continue
    for file in paths:
        f = open(f"{file}")
        score = f.read().strip().rpartition('Score')[2]
        try:
            Y.append(int(parse.parse("{} .{}", score)[0]))
        
        # invalid moves are treated as -10
        except ValueError:
            Y.append(-10)
        c += 1

    X = [i for i in range(c)]

    plt.plot(X,Y)
    plt.xlabel('Epsiode')
    plt.ylabel('Score')
    print(dir)
    plt.savefig(f'scores/scores_{str(dir).rpartition(os.sep)[2]}.png')