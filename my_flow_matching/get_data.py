from os import link

import numpy as np
import matplotlib.pyplot as plt

def create_data(length: int = 4):
    N: int = 1000
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4

    checkerboard = np.indices((length, length)).sum(axis=0) % 2

    sample_points = []
    while len(sample_points) < N:
        x_sample = np.random.uniform(x_min, x_max)
        y_sample = np.random.uniform(y_min, y_max)

        i = int((x_sample - x_min) / (x_max - x_min) * length)
        j = int((y_sample - y_min) / (y_max - y_min) * length)

        if checkerboard[j, i] == 1:
            sample_points.append((x_sample, y_sample))

    return sample_points


def main():
    sample_points = create_data()
    x = [i[0] for i in sample_points]
    y = [i[1] for i in sample_points]

    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    main()