import nltk
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
from nltk import FreqDist, corpus
import matplotlib.pyplot as plt

def main():
    # Compute a frequency distribution of the whole corpus
    fdist = FreqDist(corpus.brown.words())
    # Compute a frequency distribution of the genre "news"
    fdist_news = FreqDist(corpus.brown.words(categories="news"))
    # Compute a frequency distribution of the genre "romance"
    fdist_romance = FreqDist(corpus.brown.words(categories="romance"))
    # Count the number of tokens in corpus
    tokens = sum(fdist.values())
    # Count the number of types in corpus
    types = len(fdist)
    # Count the number of words in corpus
    words = len(nltk.tokenize.TreebankWordDetokenizer().detokenize(corpus.brown.words()))
    # Count the average number of words per sentence
    avg_words = words / len(corpus.brown.sents())
    # Count the average word length
    avg_word_length = sum([len(w) for w in fdist.keys()]) / types

    # Count the 10 most frequent parts of speech in the dataset
    pos = nltk.pos_tag(corpus.brown.words())
    pos_fdist = FreqDist([p[1] for p in pos])
    most_frequent = pos_fdist.most_common(10)

    # Plot the frequency distribution of the corpus, news genre and romance on both linear and log-log scale
    title = ["Linear F. Distribution: ", "Log-Log F. Distribution: "]
    genre = ["Corpus", "News", "Romance"]
    fig, axs = plt.subplots(2, 3)
    for i in range(2):
        for j in range(3):
            axs[i,j].set_title(title[i] + genre[j])
            if i == 0:
                if j == 0:
                    axs[i,j].plot(range(1, types + 1), sorted(fdist.values(), reverse=True))
                elif j == 1:
                    axs[i,j].plot(range(1, len(fdist_news) + 1), sorted(fdist_news.values(), reverse=True))
                else:
                    axs[i,j].plot(range(1, len(fdist_romance) + 1), sorted(fdist_romance.values(), reverse=True))
            else:
                if j == 0:
                    axs[i,j].loglog(range(1, types + 1), sorted(fdist.values(), reverse=True))
                elif j == 1:
                    axs[i,j].loglog(range(1, len(fdist_news) + 1), sorted(fdist_news.values(), reverse=True))
                else:
                    axs[i,j].loglog(range(1, len(fdist_romance) + 1), sorted(fdist_romance.values(), reverse=True))
            axs[i,j].set_xlabel("Position in the frequency distribution")
            axs[i,j].set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig("freq_dist.png")

if __name__ == "__main__":
    main()