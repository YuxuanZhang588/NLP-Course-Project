import nltk
import numpy as np
from viterbi import viterbi

nltk.download('brown')
nltk.download('universal_tagset')
tagged_sents = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]

tags = [tag for sent in tagged_sents for _, tag in sent]
words = [word.lower() for sent in tagged_sents for word, _ in sent]
unique_tags = list(set(tags))
unique_words = list(set(words))
unique_words.append('UNK')

tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
word_to_idx = {word: idx for idx, word in enumerate(unique_words)}

n_tags = len(unique_tags)
n_words = len(unique_words)

# Add-one smoothing
transition_matrix = np.ones((n_tags, n_tags)) 
observation_matrix = np.ones((n_tags, n_words))  
initial_probs = np.ones(n_tags)

for sent in tagged_sents:
    first_tag = sent[0][1]
    initial_probs[tag_to_idx[first_tag]] += 1

initial_probs /= np.sum(initial_probs)

for sent in tagged_sents:
    for i in range(len(sent) - 1):
        current_tag = sent[i][1]
        next_tag = sent[i + 1][1]
        current_word = sent[i][0].lower()
        transition_matrix[tag_to_idx[current_tag], tag_to_idx[next_tag]] += 1
        if current_word in word_to_idx:
            observation_matrix[tag_to_idx[current_tag], word_to_idx[current_word]] += 1
        else:
            observation_matrix[tag_to_idx[current_tag], word_to_idx['UNK']] += 1

for sent in tagged_sents:
    last_tag = sent[-1][1]
    last_word = sent[-1][0].lower()
    if last_word in word_to_idx:
        observation_matrix[tag_to_idx[last_tag], word_to_idx[last_word]] += 1
    else:
        observation_matrix[tag_to_idx[last_tag], word_to_idx['UNK']] += 1

transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
observation_matrix /= observation_matrix.sum(axis=1, keepdims=True)

# Infer the sequence of states for sentences 10150 - 10152 of the Brown corpus
test_sents = nltk.corpus.brown.sents()[10150:10153]
true_tags = []
test_sents = []
for tagged_sentence in nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]:
    sentence = []
    tags = []
    for word, tag in tagged_sentence:
        sentence.append(word)
        tags.append(tag)
    test_sents.append(sentence)
    true_tags.append(tags)
for i, sent in enumerate(test_sents):
    obs_seq = [word_to_idx.get(word.lower(), word_to_idx['UNK']) for word in sent]
    best_tags = viterbi(obs_seq, initial_probs, transition_matrix, observation_matrix)
    predicted_tags = [unique_tags[i] for i in best_tags[0]]
    print("Sentence:", sent)
    print("Predicted Tags:", predicted_tags)
    print("True Tags:", true_tags[i])