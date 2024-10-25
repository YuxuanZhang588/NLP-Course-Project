import numpy as np

# Sigmoid and Tanh functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Define the one-hot encoding for the sequence of words ["bad", "good", "not", "uh"]
word_dict = {
    "bad": np.array([1, 0, 0, 0]),
    "good": np.array([0, 1, 0, 0]),
    "not": np.array([0, 0, 1, 0]),
    "uh": np.array([0, 0, 0, 1])
}

# Initialize h_0 and c_0 (initial hidden and cell states) as zeros
h_t = np.zeros(6)  # Hidden state is 6-dimensional
c_t = np.zeros(6)  # Cell state is also 6-dimensional

# Define the weight matrices and biases as given in the problem
W_ig = np.zeros((6, 4))  # Input-to-cell gate weights (6x4 matrix of zeros)
W_hg = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [-10, 10, 0, 10, -10, 0]
])

# Bias vectors
b_ig = np.array([10, 10, 10, 10, 10, 0])  # Bias for input-to-cell gate
b_hg = np.zeros(6)  # Bias for hidden-to-cell gate

# Input gate (i_t) is all ones in this problem
i_t = np.ones(6)
f_t = np.array([0, 0, 0, 0, 0, 1])

# Output gate matrices
W_io = np.array([
    [10, 0, 0, 0],
    [0, 10, 0, 0],
    [0, 0, 10, 0],
    [10, 0, 0, 0],
    [0, 10, 0, 0],
    [0, 0, 0, 0]
])

W_ho = np.array([
    [0, 0, -10, 0, 0, 0],
    [0, 0, -10, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 10, 0, 0, 0],
    [0, 0, 10, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

b_io = np.zeros(6)  # Bias for input-to-output gate
b_ho = np.array([-5, -5, -5, -15, -15, -5] )  # Bias for hidden-to-output gate

# Process each word in the sequence: "bad", "good", "not", "uh"
sequence = ["uh", "not", "uh", "bad", "bad", "good", "uh"]

for word in sequence:
    print(f"Processing word: {word}")
    
    # Get the one-hot encoded vector for the current word
    x_t = word_dict[word]
    
    ### Step 1: Calculate g_t
    g_t = tanh(np.dot(W_ig, x_t) + b_ig + np.dot(W_hg, h_t) + b_hg)
    
    ### Step 2: Calculate i_t * g_t (Element-wise multiplication)
    i_t_times_g_t = i_t * g_t
    
    ### Step 3: Update the cell state c_t
    c_t = f_t * c_t + i_t_times_g_t
    
    ### Step 4: Calculate the output gate o_t
    o_t = sigmoid(np.dot(W_io, x_t) + b_io + np.dot(W_ho, h_t) + b_ho)
    
    ### Step 5: Update the hidden state h_t
    h_t = o_t * tanh(c_t)  # The new hidden state is based on the cell state and output gate
    
    # Print the updated hidden state and cell state
    # print("Updated hidden state h_t:", h_t)
    print("Updated cell state c_t:", c_t)
    print("-" * 50)