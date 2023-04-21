#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs


vocab = codecs.open("brown_vocab_100.txt")


# Load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    # Import part 1 code to build dictionary
    word_index_dict[line.rstrip()] = i

word_count = i + 1

f = codecs.open("brown_100.txt")

# Iterate through file and update counts
counts = np.zeros((word_count, word_count))
for line in f:
    words = line.lower().strip().split(" ")
    for i in range(len(words) - 1):
        counts[word_index_dict[words[i]], word_index_dict[words[i + 1]]] += 1

# Normalize counts
probs = normalize(counts, norm='l1', axis=1)

# Writeout bigram probabilities
with open("bigram_probs.txt", "w") as f1:
    print(probs[ word_index_dict["all"],word_index_dict["the"] ], file=f1)
    print(probs[word_index_dict["the"],word_index_dict["jury"]], file=f1)
    print(probs[word_index_dict["the"],word_index_dict["campaign"]], file=f1)
    print(probs[word_index_dict["anonymous"],word_index_dict["calls"]], file=f1)


f.close()