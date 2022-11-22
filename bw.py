APPROX_MOVES =  [290, 276, 263, 250, 237, 224, 212, 201, 189, 178, 167, 156, 146, 136, 127, 117, 108, 100, 91, 83, 75, 68, 61, 54, 47, 41, 35, 29, 24, 19, 14, 10, 6, 2, 0]

def get_max_beamwidth(depth: int, step: int) -> int:
    bw = 1
    while 1:
        s = 0
        p = 1
        for i in range(step, step + depth):
            s+= APPROX_MOVES[i] * bw ** i
            p *= APPROX_MOVES[i]

        
        if  p < s or bw > 292:
            return bw
        else:
            bw += 1


for i in range(25):
    for j in range(5,35 - i):
        print(f"step = {i} | depth = {j} , bw = {get_max_beamwidth(j,i)}")