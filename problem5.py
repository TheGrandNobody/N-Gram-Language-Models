import numpy as np

def calculate_trigram(trigram_phrase: str, corpus: list, vocab_length: int, alpha=0.0) -> float:
    """ Computes the probability of a given trigram phrase for a given corpus.  

    Args:
        trigram_phrase (str): The specified trigram.
        corpus (list): The corpus to calculate the trigram probability from.
        vocab_length (int): The length of the vocabulary.
        alpha (float, optional): The alpha-smoothing value. Defaults to 0.0.

    Returns:
        float: The probability of the trigram.
    """
    # Split the trigram into words
    word1, word2, word3 = trigram_phrase.split()
    first_two_words_count = alpha
    last_word_count = alpha
    
    # Count the probability for the trigram
    for sentence in corpus:
        sentence = sentence.lower().strip().split(" ")
        for i in range(len(sentence) - 2):
            if sentence[i] == word1 and sentence[i+1] == word2:
                first_two_words_count += 1
                if sentence[i+2] == word3:
                    last_word_count += 1

    return last_word_count / (first_two_words_count + vocab_length * alpha)


if __name__ == "__main__":
    # Load the indices dictionary
    with open("brown_100.txt", "r") as f:
        sentences = f.read().splitlines()

    with open("brown_vocab_100.txt", "r") as f:
        vocab_length = len(f.read().splitlines())

    probablity = calculate_trigram("in the past", sentences, vocab_length)
    print("Probability of 'in the past' without smoothing is: ", probablity)

    probablity = calculate_trigram("in the past", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'in the past' with smoothing of 0.1 is: ", probablity)

    probablity = calculate_trigram("in the time", sentences, vocab_length)
    print("Probability of 'in the time' without smoothing is: ", probablity)

    probablity = calculate_trigram("in the time", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'in the time' with smoothing of 0.1 is: ", probablity)

    probablity = calculate_trigram("the jury said", sentences, vocab_length)
    print("Probability of 'the jury said' without smoothing is: ", probablity)

    probablity = calculate_trigram("the jury said", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'the jury said' with smoothing of 0.1 is: ", probablity)

    probablity = calculate_trigram("the jury recommended", sentences, vocab_length)
    print("Probability of 'the jury recommended' without smoothing is: ", probablity)

    probablity = calculate_trigram("the jury recommended", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'the jury recommended' with smoothing of 0.1 is: ", probablity)

    probablity = calculate_trigram("jury said that", sentences, vocab_length)
    print("Probability of 'jury said that' without smoothing is: ", probablity)

    probablity = calculate_trigram("jury said that", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'jury said that' with smoothing of 0.1 is: ", probablity)

    probablity = calculate_trigram("agriculture teacher ,", sentences, vocab_length)
    print("Probability of 'agriculture teacher ,' without smoothing is: ", probablity)

    probablity = calculate_trigram("agriculture teacher ,", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'agriculture teacher ,' with smoothing of 0.1 is: ", probablity)