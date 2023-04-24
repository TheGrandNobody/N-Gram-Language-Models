import numpy as np

def calculate_trigram(trigram_phrase, corpus, vocab_length, alpha=0.0):


    word1, word2, word3 = trigram_phrase.split()


    first_two_words_count = alpha
    last_word_count = alpha

    for sentence in corpus:
        sentence = sentence.lower().strip().split(" ")
        for i in range(len(sentence) - 2):
            if sentence[i] == word1 and sentence[i+1] == word2:
                first_two_words_count += 1
                if sentence[i+2] == word3:
                    last_word_count += 1

    trigram_probability = last_word_count / (first_two_words_count + vocab_length * alpha)

    return trigram_probability


if __name__ == "__main__":
    # Load the indices dictionary
    with open("brown_100.txt", "r") as f:
        sentences = f.read().splitlines()

    with open("brown_vocab_100.txt", "r") as f:
        vocab_length = len(f.read().splitlines())

    probablity = calculate_trigram("in the past", sentences, vocab_length, alpha=0)
    print("Probability of 'in the past' without smoothing is: ", probablity)

    probablity = calculate_trigram("in the past", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'in the past' with smoothing of 0.1 is: ", probablity)

    probablity = calculate_trigram("in the time", sentences, vocab_length, alpha=0)
    print("Probability of 'in the time' without smoothing is: ", probablity)

    probablity = calculate_trigram("in the time", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'in the time' with smoothing of 0.1 is: ", probablity)

    probablity = calculate_trigram("the jury said", sentences, vocab_length, alpha=0)
    print("Probability of 'the jury said' without smoothing is: ", probablity)

    probablity = calculate_trigram("the jury said", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'the jury said' with smoothing of 0.1 is: ", probablity)

    probablity = calculate_trigram("the jury recommended", sentences, vocab_length, alpha=0)
    print("Probability of 'the jury recommended' without smoothing is: ", probablity)

    probablity = calculate_trigram("the jury recommended", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'the jury recommended' with smoothing of 0.1 is: ", probablity)

    probablity = calculate_trigram("jury said that", sentences, vocab_length, alpha=0)
    print("Probability of 'jury said that' without smoothing is: ", probablity)

    probablity = calculate_trigram("jury said that", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'jury said that' with smoothing of 0.1 is: ", probablity)

    probablity = calculate_trigram("agriculture teacher ,", sentences, vocab_length, alpha=0)
    print("Probability of 'agriculture teacher ,' without smoothing is: ", probablity)

    probablity = calculate_trigram("agriculture teacher ,", sentences, vocab_length, alpha = 0.1)
    print("Probability of 'agriculture teacher ,' with smoothing of 0.1 is: ", probablity)