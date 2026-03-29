import numpy as np
from model import score
from data import load_data

P = 1479*1.05
P_1 = 33185


def main():
    t, sensor_data = load_data()
    t_n = 10

    ops = np.loadtxt("parameter.txt", dtype=float)
    print(score(ops, sensor_data, t, P, P_1, t_n))
    return 0

if __name__ == "__main__":
    main()