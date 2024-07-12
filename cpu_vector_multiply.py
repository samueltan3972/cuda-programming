import argparse

import numpy as np
from timeit import default_timer as timer

# Demostrate the run time on CPU
def MultiplyVector(a, b, c):
    for i in range(a.size):
        c[i] = a[i] * b[i]

def MultiplyVector_numpy(a, b):
    return a * b

def main(use_numpy=False):
    N = 64000000 # size per declared array

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.ones(N, dtype=np.float32)

    start = timer()
    if not use_numpy:
        MultiplyVector(A, B, C)
    else:
        C = MultiplyVector_numpy(A, B)

    exec_time = timer() - start

    print("C[:6] = ", C[:6])
    print("C[-6:] = ", C[-6:])

    print(f"Execution time on CPU: {exec_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action="store_true", help="To run in numpy way")

    args = parser.parse_args()

    main(args.n)