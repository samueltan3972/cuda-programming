import argparse

import numpy as np
from timeit import default_timer as timer
from numba import cuda, jit

def FillArrayCPU(a, size):
    for k in range(size):
        a[k] += 1

@jit(parallel=True, cache=True)
def FillArrayJIT(a, size):
    for k in range(size):
        a[k] += 1

def main(size):
    a = np.ones(size, dtype=np.float32)

    start = timer()
    FillArrayCPU(a, size)
    print("CPU execution time:", timer()-start, "seconds")

    start = timer()
    FillArrayJIT(a, size)
    print("JIT execution time:", timer()-start, "seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=1000000)

    args = parser.parse_args()

    main(args.size)