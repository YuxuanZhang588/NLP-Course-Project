"""Pytorch."""
import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt

FloatArray = NDArray[np.float64]

def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding

def loss_fn(logp: float) -> float:
    """Compute loss to maximize probability."""
    return -logp

class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct uniform initial s
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0.astype(float)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        logp = torch.nn.LogSoftmax(0)(self.s)

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ logp

def calculate_optimal_probabilities(vocabulary, tokens):
    """Calculate optimal probabilities based on token frequencies."""
    token_count = {token: tokens.count(token) for token in vocabulary}
    token_count[None] = len(tokens) - sum(token_count.values())
    total_tokens = sum(token_count.values())
    optimal_probabilities = np.array([token_count[token] / total_tokens for token in vocabulary])
    return optimal_probabilities.reshape(-1, 1)

def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype(float))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations =  1000
    learning_rate =  0.01

    # calculate optimal probabilities and minimum loss
    optimal_probs = calculate_optimal_probabilities(vocabulary, tokens)
    log_optimal_probs = np.log(optimal_probs)
    min_loss = -np.sum(encodings.T @ log_optimal_probs)

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    for _ in range(num_iterations):
        logp_pred = model(x)
        loss = loss_fn(logp_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_history.append(loss.item())

    # Convert None to UNK for plotting
    vocabulary_for_plotting = ["UNK" if token is None else token for token in vocabulary]

    # Plot Final Probabilities with Optimal Probabilities
    final_probs = torch.nn.Softmax(0)(model.s).detach().numpy()
    plt.figure(figsize=(10, 5))
    plt.plot(vocabulary_for_plotting, final_probs, label="Learned Probabilities", marker='o')
    plt.plot(vocabulary_for_plotting, optimal_probs, label="Optimal Probabilities", marker='x')
    for i, prob in enumerate(final_probs):
        plt.text(vocabulary_for_plotting[i], prob, f'{prob[0]:.5f}', ha='right', va='bottom', color='blue', fontsize=5, rotation=45)
    for i, prob in enumerate(optimal_probs):
        plt.text(vocabulary_for_plotting[i], prob, f'{prob[0]:.5f}', ha='left', va='top', color='orange', fontsize=5, rotation=45)
    plt.title('Final Token Probabilities')
    plt.xlabel('Token')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

    # Plot loss over iterations and minimum loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Loss")
    plt.axhline(y=min_loss, color='r', linestyle='--', label="Minimum Loss")
    plt.title('Loss Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    gradient_descent_example()