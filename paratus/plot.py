import matplotlib.pyplot as plt
import numpy as np


def plotEmbedding(X, y=None, render=True):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    if render:
        plt.show()


if __name__ == "__main__":
    plotEmbedding(np.array([
        [0, 1],
        [1, 1]
    ]))

    plotEmbedding(np.array([
        [0, 1],
        [1, 1]
    ]), [2, 1])
