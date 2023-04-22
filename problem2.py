#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import json
import numpy as np
from generate import GENERATE
from random import randint

if __name__ == "__main__":
    # Load the indices dictionary.
    word_index_dict = {}
    with open("brown_vocab_100.txt", "r") as f:
        for i, line in enumerate(f):
            word_index_dict[line.rstrip()] = i

    # Load all sentences.
    with open("brown_100.txt", "r") as f:
        sentences = f.read().splitlines()

    # Initialize counts to a zero vector.
    counts = np.zeros(len(word_index_dict.keys())) 

    # Iterate through file and update counts.
    for sen in sentences:
        words = sen.lower().strip().split(" ")
        for word in words:
            counts[word_index_dict[word]] += 1

    #  Normalize and writeout counts. 
    probs = counts / np.sum(counts)

    # Write probs as a dict into a text file.
    word_probs_dict = {}
    for word in word_index_dict:
        word_probs_dict[word] = probs[word_index_dict[word]]
    with open("unigram_probs.txt", "w") as f:
        json.dump(word_probs_dict, f, indent=4)

    # Verify that the dictionary was created correctly.
    print("%.8f" % word_probs_dict["all"])
    print("%.8f" % word_probs_dict["resolution"])

    # Generate 10 sentences.
    sents = [GENERATE(word_index_dict, word_probs_dict.values(), "unigram", randint(5, 50), "<s>") + "\n" for _ in range(10)]
    with open("unigram_generation.txt", "w") as f:
        f.writelines(sents)

    # Calculating sentence probabilities and perplexities.
    with open("toy_corpus.txt", "r") as f:
        sentences = f.read().splitlines()
    sentprobs, perplexities = [], []
    for sent in sentences:
        sentprob = 1
        words = sent.lower().strip().split(" ")
        for word in words:
            sentprob *= word_probs_dict[word]
        sentprobs.append(sentprob)
        perplexities.append(1/(pow(sentprob, 1.0/len(words))))
    print(sentprobs)
    print(perplexities)
    with open("unigram_eval.txt", "w") as f:
        f.writelines([str(p) + "\n" for p in perplexities])


