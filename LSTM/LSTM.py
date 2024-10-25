### Author: Yuxuan Zhang
### Date: 10/24/2024
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def lstm(sequence):
    # Define the one-hot encoding
    word_dict = {
        "bad": np.array([1, 0, 0, 0]),
        "good": np.array([0, 1, 0, 0]),
        "not": np.array([0, 0, 1, 0]),
        "uh": np.array([0, 0, 0, 1])
    }

    # Initialize h_0 and c_0
    h_t = np.zeros(6)
    c_t = np.zeros(6)

    # For simplicity, set i_t and f_t to constants
    i_t = np.ones(6)
    f_t = np.array([0, 0, 0, 0, 0, 1])

    # Cell Gate matrices
    W_ig = np.zeros((6, 4))
    W_hg = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [-100, 100, 0, 100, -100, 0]
    ])

    b_ig = np.array([100, 100, 100, 100, 100, 0]) 
    b_hg = np.zeros(6) 

    # Output gate matrices
    W_io = np.array([
        [100, 0, 0, 0],
        [0, 100, 0, 0],
        [0, 0, 100, 0],
        [100, 0, 0, 0],
        [0, 100, 0, 0],
        [0, 0, 0, 0]
    ])
    W_ho = np.array([
        [0, 0, -100, 0, 0, 0],
        [0, 0, -100, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 100, 0, 0, 0],
        [0, 0, 100, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    b_io = np.zeros(6) 
    b_ho = np.array([-50, -50, -50, -150, -150, -50] )

    for word in sequence:
        print(f"Processing word: {word}")

        x_t = word_dict[word]
        g_t = tanh(np.dot(W_ig, x_t) + b_ig + np.dot(W_hg, h_t) + b_hg)
        c_t = f_t * c_t + i_t * g_t
        o_t = sigmoid(np.dot(W_io, x_t) + b_io + np.dot(W_ho, h_t) + b_ho)
        h_t = o_t * tanh(c_t)

        print("Updated delayed total sentiment score:", c_t[-1])
        print("-" * 50)

def main():
    sequence = ["uh", "good", "good", "not", "not", "bad", "bad", "bad", "uh"]
    lstm(sequence)

if __name__ == "__main__":
    main()