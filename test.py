from collections import defaultdict, Counter
def build_ngram_model(corpus, n):
    ngrams = defaultdict(Counter)
    # Build all n-grams from unigrams to n-grams
    for order in range(1, n+1):
        for i in range(len(corpus) - order + 1):
            ngram = tuple(corpus[i:i + order - 1])
            next_word = corpus[i + order - 1]
            ngrams[ngram][next_word] += 1
    return ngrams