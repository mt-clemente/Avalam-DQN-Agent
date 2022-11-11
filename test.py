import timeit
import torch

SETUP = """
import torch
a = torch.rand(size = (9,9)).to('cuda')
"""

STMT = """
a [1,2] = a[2,4] + a[3,1]
"""
print(timeit.timeit(setup=SETUP,stmt = STMT))

SETUP = """
import torch
a = torch.rand(size = (9,9)).to('cpu')
"""
print(timeit.timeit(setup=SETUP,stmt = STMT))