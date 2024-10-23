"""Latent Dirichlet Allocation

Patrick Wang, 2021
"""
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np

def lda_gen(vocabulary, alpha, beta, xi):
    """Genertate a list of words in a document"""
    document_length = np.random.poisson(xi)
    topic_dist = np.random.dirichlet(alpha)
    words = []
    k = len(alpha)
    V = len(vocabulary)
    for _ in range(document_length):
        topic_index = np.random.choice(k, p=topic_dist)
        word_index = np.random.choice(V, p=beta[topic_index])
        words.append(vocabulary[word_index])
    return words


def test():
    """Test the LDA generator."""
    vocabulary = [
        "bass", "pike", "deep", "tuba", "horn", "catapult",
    ]
    beta = np.array([
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
        [0.3, 0.0, 0.2, 0.3, 0.2, 0.0]
    ])
    alpha = np.array([0.2, 0.2, 0.2])
    xi = 50
    documents = [
        lda_gen(vocabulary, alpha, beta, xi)
        for _ in range(100)
    ]
    print(documents[0])
    # Create a corpus from a list of texts
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    model = LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=3,
    )
    print(model.alpha)
    print(model.show_topics())

if __name__ == "__main__":
    test()
