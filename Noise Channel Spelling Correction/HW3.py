import math
import csv

def load_freq_data():
    freq_dict = {}
    freq_prob = {}
    file_path = 'count_1w.txt'
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            freq_dict[parts[0]] = int(parts[1])
    total = sum(freq_dict.values())
    for word, count in freq_dict.items():
        freq_prob[word] = math.log(count/total)
    return freq_prob

def load_unigram():
    file_path = 'unigrams.csv'
    unigram_data = {}
    with open(file_path, mode='r') as f:
        for row in csv.DictReader(f):
            unigram = row['unigram']
            count = int(row['count'])
            unigram_data[unigram] = count
    return unigram_data

def load_bigram():
    file_path = 'bigrams.csv'
    bigram_data = {}
    with open(file_path, mode='r') as f:
        for row in csv.DictReader(f):
            bigram = row['bigram']
            count = int(row['count'])
            bigram_data[bigram] = count
    return bigram_data
             
def load_addition():
    file_path = 'additions.csv'
    addition_data = {}
    with open(file_path, mode='r') as f:
        for row in csv.DictReader(f):
            prefix = row['prefix']
            added_char = row['added']
            count = int(row['count'])
            addition_data[(prefix, added_char)] = count
    return addition_data

def load_deletion():
    file_path = 'deletions.csv'
    deletion_data = {}
    with open(file_path, mode='r') as f:
        for row in csv.DictReader(f):
            prefix = row['prefix']
            deleted_char = row['deleted']
            count = int(row['count'])
            deletion_data[(prefix, deleted_char)] = count
    return deletion_data

def load_substitution():
    file_path = 'substitutions.csv'
    substitution_data = {}
    with open(file_path, mode='r') as f:
        for row in csv.DictReader(f):
            original = row['original']
            substituted = row['substituted']
            count = int(row['count'])
            substitution_data[(original, substituted)] = count
    return substitution_data

def calculate_insertion_cost(x_i_m1, w_i, w_i_m1, insertion_data, unigram_data):
    insertion_count = insertion_data.get((x_i_m1, w_i), 1)
    unigram_count = unigram_data.get(w_i_m1, 1)
    return insertion_count / unigram_count

def calculate_deletion_cost(x_i_m1, w_i, deletion_data, bigram_data):
    deletion_count = deletion_data.get((x_i_m1, w_i), 1)
    bigram_count = bigram_data.get((x_i_m1, w_i), 1)
    return deletion_count / bigram_count

def calculate_substitution_cost(x_i, w_i, substitution_data, unigram_data):
    substitution_count = substitution_data.get((x_i, w_i), 1)
    unigram_count = unigram_data.get(w_i, 1)
    return substitution_count / unigram_count


def weighted_levenshtein_distance(word1, word2, insertion_data, deletion_data, 
                                  substitution_data, unigram_data, bigram_data):
    n, m = len(word1), len(word2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        prefix = word1[i-2] if i > 1 else '#'
        dp[i][0] = dp[i - 1][0] + calculate_deletion_cost(prefix, word1[i-1], deletion_data, bigram_data)
    for j in range(1, m + 1):
        prefix = word2[i-2] if j > 1 else '#'
        dp[0][j] = dp[0][j - 1] + calculate_insertion_cost(prefix, word2[j-1], word2[:j-1], insertion_data, unigram_data)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prefix_word1 = word1[i-2] if i > 1 else '#'
            prefix_word2 = word2[j-2] if j > 1 else '#'          
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                insert_cost = calculate_insertion_cost(prefix_word2, word2[j-1], word1[i-2] if i>1 else '#', insertion_data, unigram_data)
                delete_cost = calculate_deletion_cost(prefix_word1, word1[i-1], deletion_data, bigram_data)
                substitute_cost = calculate_substitution_cost(word1[i-1], word2[j-1], substitution_data, unigram_data)
                dp[i][j] = min(
                    dp[i - 1][j] + delete_cost, 
                    dp[i][j - 1] + insert_cost,  
                    dp[i - 1][j - 1] + substitute_cost  
                )
    return dp[n][m]

def generate_candidates(word, dictionary):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    inserts = [L + c + R for L, R in splits for c in letters]
    deletes = [L + R[1:] for L, R in splits if R]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    all_candidates = inserts + deletes + replaces
    valid_candidates = []
    for edit in all_candidates:
        if edit not in valid_candidates and edit[0] in dictionary and edit[0] != word:
            valid_candidates.append(edit)
    return valid_candidates

def correct(word):
    word = word.lower()
    addition_data = load_addition()
    deletion_data = load_deletion()
    substitution_data = load_substitution()
    dictionary = load_freq_data()
    unigram_data = load_unigram()
    bigram_data = load_bigram()
    if word in dictionary:
        return word
    candidates = generate_candidates(word, dictionary)
    best_score = -math.inf
    best_candidate = word
    for candidate in candidates:
        edit_prob = prob_func(word, candidate, addition_data, deletion_data, substitution_data, 
                                   unigram_data, bigram_data)
        unigram_prob = dictionary.get(candidate, -math.inf)
        score = math.log(edit_prob) + unigram_prob
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate

def prob_func(word, candidate, addition_data, deletion_data, substitution_data, unigram_data, bigram_data):
    distance = weighted_levenshtein_distance(word, candidate, addition_data, deletion_data, 
                                             substitution_data, unigram_data, bigram_data)
    return 1 / (1 + distance)

if __name__ == "__main__":
    misspelled_word = "corret"
    correction = correct(misspelled_word)
    print(f"Correction for '{misspelled_word}': {correction}")