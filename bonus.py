import nltk
nltk.download('brown')
from nltk import corpus, FreqDist
import string, math
from collections import defaultdict

def pmi(word1: str, word2: str, fdist: FreqDist, length: int, count: int) -> float:
    """ 
    Calculates the pmi for two unigrams given a corpus and its frequency distribution.

    Args:
        word1 (str): The first word string
        word2 (str): The second word string
        corpus (list): The corpus as a list of words.
        fdist (FreqDist): The frequency distribution of the corpus.
    """
    # Calculate the pmi
    return math.log(count * length / (fdist[word1] * fdist[word2]))

def check_punct(word: str) -> bool:
    """ Checks if a given word is punctuation or made up of punctuation.

    Args:
        word (str): The specified word

    Returns:
        bool: True if the word is not punctuation and is not made up of punctuation, False otherwise.
    """
    return word not in string.punctuation and len([1 for c in word if c in string.punctuation]) != len(word)

if __name__ == "__main__":
    # Load the brown corpus
    brown = [w.strip().lower() for w in corpus.brown.words()]
    # Remove punctuation and combinations of punctuation from the corpus
    clean_brown = [w.strip().lower() for w in brown if check_punct(w)]
    # Convert the corpus to a frequency distribution
    fdist = FreqDist(clean_brown)
    # Count the occurence of each word pair in the corpus that has a frequency of 10 or more
    pfdist = defaultdict(int)
    for i in range(len(brown) - 1):
        # We use the whole corpus (with punctuation) to maintain the natural order of words
        # A word at the end of a sentence is not considered paired with the word at the beginning of the next sentence
        if check_punct(brown[i]) and check_punct(brown[i + 1]) and fdist[brown[i]] > 9 and fdist[brown[i + 1]] > 9:
            pfdist[f"{brown[i]} {brown[i + 1]}"] += 1
    # Calculate the pmi for each word pair in pfdist
    pmis = sorted({k: pmi(k.split()[0], k.split()[1], fdist, len(clean_brown), v) for k, v in pfdist.items()}.items(), key=lambda x: x[1])
    # Save the results to a file
    with open("pmis.txt", "w") as f:
        # Write the bottom 20 results
        for j in pmis[:20]:
            f.write(f"{j[0]}: {j[1]} \n")
        # Write the top 20 results
        for j in pmis[-20:]:
            f.write(f"{j[0]}: {j[1]} \n")