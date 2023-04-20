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
alpha = 0.1

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    word_index_dict[line.rstrip()] = i

word_count = i + 1

f = codecs.open("brown_100.txt")

#TODO: iterate through file and update counts
counts = np.zeros((word_count, word_count))
counts += alpha

for line in f:
    words = line.lower().strip().split(" ")
    for i in range(len(words) - 1):
        counts[word_index_dict[words[i]], word_index_dict[words[i + 1]]] += 1

#TODO: normalize counts
probs = normalize(counts, norm='l1', axis=1)

#TODO: writeout bigram probabilities

with open("smooth_probs.txt", "w") as f1:
    print(probs[ word_index_dict["all"],word_index_dict["the"] ], file=f1)
    print(probs[word_index_dict["the"],word_index_dict["jury"]], file=f1)
    print(probs[word_index_dict["the"],word_index_dict["campaign"]], file=f1)
    print(probs[word_index_dict["anonymous"],word_index_dict["calls"]], file=f1)

#Comparing smoothed probabilities to unsmoothed probabilities

smooth_probs = []
with open("smooth_probs.txt", "r") as f2:
    for line in f2:
        smooth_probs.append(float(line))

unsmooth_probs = []

with open("bigram_probs.txt", "r") as f3:
    for line in f3:
        unsmooth_probs.append(float(line))

print("Smoothed probabilities: ", smooth_probs)
print("Unsmoothed probabilities: ", unsmooth_probs)
print("difference ", np.subtract(unsmooth_probs, smooth_probs))
print("Unsmoothed probabilities are higher than smoothed probabilities")




f.close()