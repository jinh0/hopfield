import numpy as np
import matplotlib.pyplot as plt

# convert array to vector
def get_neurons(image: list[list[int]]):
    return np.reshape(np.array(image).flatten(), (25, 1))


def get_weights(neurons: np.ndarray):
    # The weight matrix is the dot product of the
    # neuron vector with its transpose.
    weights = np.dot(neurons, neurons.T)

    # Set the diagonals to zero, since there are no loops
    # This is a simple graph
    for i in range(25):
        weights[i][i] = 0

    return weights


def draw(image: list[list[int]], title=None):
    plt.matshow(image)

    if title:
        plt.title(title)

    plt.show()


def reconstruct(image: list[list[int]], weights: np.ndarray):
    image = get_neurons(image)

    reconstructed = [0] * 25

    for i in range(25):
        score = sum([weights[i][j] * image[j] for j in range(25)])
        reconstructed[i] = 1 if score > 0 else -1

    return np.reshape(reconstructed, (5, 5))
