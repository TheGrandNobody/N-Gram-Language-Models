import numpy as np

def get_unigram_p(unigram: str, word_index_dict: dict, sentences: list, smoothed: bool):
    """ Get the probability of a unigram given a unigram string.

    Args:
        word_index_dict (dict): The dictionary of words and their indices.
        sentences (list): The list of sentences.
        smoothed (bool): Whether the probability should be smoothed or not.
    """
    # Initialize counts to a zero vector
    counts = np.zeros(len(word_index_dict.keys())) 

    # Iterate through file and update counts
    for sen in sentences:
        words = sen.lower().strip().split(" ")
        for word in words:
            counts[word_index_dict[word]] += 1

    smoothing = np.full(shape=len(counts), fill_value=0.1, dtype=np.float64) if smoothed else np.zeros(len(counts))
    # Normalize and writeout counts. 
    probs = (counts + smoothing) / np.sum(counts)

    return probs[word_index_dict[unigram]]

def get_bigram_p(bigram:str, word_index_dict:dict, sentences: list, smoothed: bool):
    """ Get the probability of a bigram given a bigram string.

    Args:
        bigram (str): The bigram string.
        word_index_dict (dict): The dictionary of words and their indices.
        sentences (list): The list of sentences.
        smoothed (bool): Whether the probability should be smoothed or not.
    """

def get_trigram_p(trigram: str, word_index_dict:dict, sentences: list, smoothed: bool):
    """ Get the probability of a trigram given a trigram string.

    Args:
        word_index_dict (dict): The dictionary of words and their indices.
        sentences (list): The list of sentences.
    """
    words = trigram.split(" ")
    p_AB = get_bigram_p(words[0] + " " + words[1], word_index_dict, sentences, smoothed)
    p_BC = get_bigram_p(words[1] + " " + words[2], word_index_dict, sentences, smoothed)
    p_B = get_unigram_p(words[1], word_index_dict, sentences, smoothed)
    return p_AB * p_BC / p_B 


if __name__ == "__main__":
    # Load the indices dictionary
    word_index_dict = {}
    with open("brown_vocab_100.txt", "r") as f:
        for i, line in enumerate(f):
            word_index_dict[line.rstrip()] = i
    # Load all sentences
    with open("brown_100.txt", "r") as f:
        sentences = f.read().splitlines()

    
