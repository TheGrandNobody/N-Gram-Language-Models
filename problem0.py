import nltk
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
from nltk import FreqDist, corpus
import matplotlib.pyplot as plt
import string

def check_punct(word: str) -> bool:
    """ Checks if a given word is punctuation or made up of punctuation.

    Args:
        word (str): The specified word

    Returns:
        bool: True if the word is not punctuation and is not made up of punctuation, False otherwise.
    """
    return word not in string.punctuation and len([1 for c in word if c in string.punctuation]) != len(word)

def main():
    genre = ["Corpus", "News", "Romance"]
    # Compute a frequency distribution of the whole corpus.
    fdist = FreqDist(w for w in corpus.brown.words() if check_punct(w))
    # Compute a frequency distribution of the genre "news".
    fdist_news = FreqDist(w for w in corpus.brown.words(categories="news") if check_punct(w))
    # Compute a frequency distribution of the genre "romance".
    fdist_romance = FreqDist(w for w in corpus.brown.words(categories="romance") if check_punct(w))

    tokens, types, words, avg_words, avg_word_length, most_frequent = [], [], [], [], [], []

    for i in range(3):
        print(f"{'Whole corpus' if i == 0 else 'News Genre' if i == 1 else 'Romance Genre'}: ", "Statistics")
        print("----------------------------")
        # Count the number of tokens in corpus.
        tokens.append(len(corpus.brown.words() if i == 0 else corpus.brown.words(categories=genre[i].lower())))
        # Count the number of types in corpus.
        types.append(len(fdist if i == 0 else fdist_news if i == 1 else fdist_romance))
        # Count the total number of words in the corpus.
        words.append(len([w for w in corpus.brown.words() if check_punct(w)] if i == 0\
                     else [w for w in corpus.brown.words(categories=genre[i].lower()) if check_punct(w)]))
        # Count the average number of words per sentence.
        avg_words.append((words[i] / len(corpus.brown.sents())) if i == 0\
                      else (words[i] / len(corpus.brown.sents(categories=genre[i].lower()))))
        # Count the average word length
        avg_word_length.append(sum([len(w) for w in corpus.brown.words() if check_punct(w)]) / types[i])
        # Count the 10 most frequent parts of speech in the dataset.
        pos = nltk.pos_tag(corpus.brown.words() if i == 0 else corpus.brown.words(categories=genre[i].lower()))
        pos_fdist = FreqDist([p[1] for p in pos])
        most_frequent.append(pos_fdist.most_common(10))
        print("Number of tokens: ", tokens[i])
        print("Number of types: ", types[i])
        print("Total number of words: ", words[i])
        print("Average number of words per sentence: ", avg_words[i])
        print("Average word length: ", avg_word_length[i])
        print("10 most frequent parts of speech: ", most_frequent[i])

    # Plot the frequency distribution of the corpus, news genre and romance on both linear and log-log scale.
    fig, axs = plt.subplots(2, 3)
    for i in range(2):
        for j in range(3):
            axs[i,j].set_title(genre[j])
            if i == 0:
                if j == 0:
                    axs[i,j].plot(range(1, types[j] + 1), sorted(fdist.values(), reverse=True))
                elif j == 1:
                    axs[i,j].plot(range(1, types[j] + 1), sorted(fdist_news.values(), reverse=True))
                else:
                    axs[i,j].plot(range(1, types[j] + 1), sorted(fdist_romance.values(), reverse=True))
            else:
                if j == 0:
                    axs[i,j].loglog(range(1, types[j] + 1), sorted(fdist.values(), reverse=True))
                elif j == 1:
                    axs[i,j].loglog(range(1, types[j] + 1), sorted(fdist_news.values(), reverse=True))
                else:
                    axs[i,j].loglog(range(1, types[j] + 1), sorted(fdist_romance.values(), reverse=True))
            axs[i,j].set_xlabel("Position")
            axs[i,j].set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig("freq_dist.png")

if __name__ == "__main__":
    main()