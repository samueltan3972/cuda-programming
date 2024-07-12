import numpy as np
from timeit import default_timer as timer
from numba import vectorize

@vectorize(["float32(float32,float32)"], target = 'cpu')
def MultiplyVector(a, b):
    return a * b

def main():
    N = 100000000 # size per declared array

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.ones(N, dtype=np.float32)

    start = timer()
    C = MultiplyVector(A, B)
    exec_time = timer() - start

    print("C[:6] = ", C[:6])
    print("C[-6:] = ", C[-6:])

    print(f"Execution time: {exec_time} seconds")

if __name__ == "__main__":
    main()
