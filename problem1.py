#!/usr/bin/env python3
"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import json

if __name__ == "__main__":
    word_index_dict = {}
    # Read the text file and create a dictionary with the words as keys and the indices as values.
    with open("brown_vocab_100.txt", "r") as f:
        for i, line in enumerate(f):
            word_index_dict[line.rstrip()] = i
    # Write the dictionary to a text file.
    with open("word_to_index_100.txt", "w") as f:
        json.dump(word_index_dict, f, indent=4)

    # Verify that the dictionary was created correctly.
    print(word_index_dict['all'])
    print(word_index_dict['resolution'])
    print(len(word_index_dict))