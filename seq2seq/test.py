import numpy as np

# Define the weights according to the specified plan
W_e = np.array([
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
])
W_o = np.eye(6)  # Identity matrix for direct mapping in decoding

# Define the tokens and their mapping to indices
token_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, '.': 5}  # '.' is EOS

# Encoder function
def encode(x, h):
    # Concatenate x and h
    combined = np.concatenate((x, h), axis=0)
    # Update hidden state with W_e
    h_next = W_e @ combined
    return h_next

# Decoder function
def decode(h):
    # Apply W_o to get the output counts and EOS indicator
    output = np.maximum(0, W_o @ h)  # ReLU applied here
    return output

# Function to process a sequence with the encoder-decoder
def process_sequence(sequence):
    # Initialize hidden state to zero
    h = np.zeros(6)
    
    # Encoding phase: process each token in the input sequence
    for token in sequence:
        # One-hot encode the token
        x = np.zeros(6)
        x[token_to_index[token]] = 1
        
        # Update hidden state
        h = encode(x, h)
        
        # Stop counting if EOS is encountered
        if h[5] > 0:  # EOS flag is set in the last position of h
            break
    
    # Decoding phase: get the output based on the final hidden state
    output = decode(h)
    return output

# Test the model on some example sequences
sequences = {
    "badcab.": [2, 2, 1, 1, 0, 1],   # Expected output: 221101
    "bababacee.": [3, 3, 1, 0, 1, 1], # Expected output: 331011
    "dadda.": [2, 0, 0, 3, 0, 1]      # Expected output: 200301
}

# Test each sequence
for sequence, expected in sequences.items():
    result = process_sequence(sequence)
    print(f"Input sequence: '{sequence}'")
    print(f"Expected output: {expected}")
    print(f"Model output:   {result.astype(int)}\n")