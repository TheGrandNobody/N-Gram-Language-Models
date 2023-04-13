import nltk
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
from nltk import FreqDist, corpus

def main():
    # Compute a frequency distribution of the whole corpus
    fdist = FreqDist(corpus.brown.words())
    # Compute a frequency distribution of the genre "news"
    fdist_news = FreqDist(corpus.brown.words(categories="news"))
    # Compute a frequency distribution of the genre "romance"
    fdist_romance = FreqDist(corpus.brown.words(categories="romance"))
    # Count the number of tokens
    tokens = sum(fdist.values())
    # Count the number of types
    types = len(fdist)
    # Count the number of words
    words = len(nltk.tokenize.TreebankWordDetokenizer().detokenize(corpus.brown.words()))
    # Count the average number of words per sentence
    avg_words = words / len(corpus.brown.sents())
    # Count the average word length
    avg_word_length = sum([len(w) for w in fdist.keys()]) / types
    # Count the 10 most frequent parts of speech in the dataset
    pos = nltk.pos_tag(corpus.brown.words())
    pos_fdist = FreqDist([p[1] for p in pos])
    most_frequent = pos_fdist.most_common(10)

    # Plot the frequency distribution of the whole corpus
    fdist.plot(50, cumulative=False)

if __name__ == "__main__":
    main()

