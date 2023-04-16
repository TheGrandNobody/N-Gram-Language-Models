#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import json
import numpy as np
from generate import GENERATE

# load the indices dictionary
word_index_dict = {}
with open("brown_vocab_100.txt", "r") as f:
    for i, line in enumerate(f):
        word_index_dict[line.rstrip()] = i

# load all sentences
with open("brown_100.txt", "r") as f:
    sentences = f.read().splitlines()

# initialize counts to a zero vector
counts = np.zeros(len(word_index_dict.keys())) 

# iterate through file and update counts
for sen in sentences:
    words = sen.lower().strip().split(" ")
    for word in words:
        counts[word_index_dict[word]] += 1

# normalize and writeout counts. 
probs = counts / np.sum(counts)

# write probs as a dict into a text file
word_probs_dict = {}
for word in word_index_dict:
    word_probs_dict[word] = probs[word_index_dict[word]]
with open("unigram_probs.txt", "w") as f:
    json.dump(word_probs_dict, f, indent=4)

# verification
print("%.8f" % word_probs_dict["all"])
print("%.8f" % word_probs_dict["resolution"])