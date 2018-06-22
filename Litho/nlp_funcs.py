import re

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger

# Specify perceptron tagger - otherwise NLTK will instantiate the Perception package every time
# the tagger is called.
tagger = PerceptronTagger()
stemmer = SnowballStemmer("english")
lmtzr = WordNetLemmatizer()


def token_cleanup(tokens):
    """
    Filter out any tokens without letters.

    :param tokens: list[str], tokens to filter

    :returns: list, cleaned list of tokens
    """
    cleaned_tokens = [token for token in tokens if token.isalpha()]

    return cleaned_tokens
# End token_cleanup()


def get_nouns(tokens):
    """
    Run given tokenized text through part-of-speech tagger and retrieve nouns.

    :param tokens: list[str], tokens to tag

    :returns: list[str], noun tokens
    """
    sent = tagger.tag(tokens)
    # sent = nltk.tag.pos_tag(tokens)
    return [s[0] for s in sent if s[1] == 'NN']
# End get_nouns()


def tokenize_and_lemmatize(text, li=None):
    """
    Tokenize a sentence, then individual words to ensure tokenization of punctuation.

    :param text: str, text to tokenize and stem
    :param li: list[str], list of words to filter out

    :returns: list, stemmed and filtered text
    """

    text = re.sub('[^A-Za-z\s]+', ' ', text)

    # tokens = [re.sub('[^A-Za-z0-9\s\.]+', ' ', word) for word in tokens]  # Remove special characters with space
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    cleaned_tokens = token_cleanup(tokens)
    cleaned_tokens = [n for n in cleaned_tokens if len(n) > 3]
    stems = [lmtzr.lemmatize(t) for t in cleaned_tokens]
    if li:
        stems = [x for x in stems if x not in li]

    return stems
# End tokenize_and_stem()



def tokenize_and_stem(text, li=None):
    """
    Tokenize a sentence, then individual words to ensure tokenization of punctuation.

    :param text: str, text to tokenize and stem
    :param li: list[str], list of words to filter out

    :returns: list, stemmed and filtered text
    """

    text = re.sub('[^A-Za-z\s]+', ' ', str(text))

    # tokens = [re.sub('[^A-Za-z0-9\s\.]+', ' ', word) for word in tokens]  # Remove special characters with space
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    cleaned_tokens = token_cleanup(tokens)
    cleaned_tokens = [n for n in cleaned_tokens if len(n) > 3]
    stems = [stemmer.stem(t) for t in cleaned_tokens]
    if li:
        stems = [x for x in stems if x not in li]

    return stems
# End tokenize_and_stem()


def tokenize_only(text, li):
    """
    Similar to `tokenize_and_stem()`, except this function does not stem, but converts each word to lowercase.

    :param text: str, text to tokenize
    :param li: list[str], list of words to filter out
    """
    text = re.sub('[^A-Za-z\s]+', ' ', text)  # Remove special characters

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    cleaned_tokens = token_cleanup(tokens)
    cleaned_tokens = [n for n in cleaned_tokens if len(n) > 3]
    filtered_tokens = [x for x in cleaned_tokens if x not in li]

    return filtered_tokens
# End tokenize_only()


def lemmatize_stems(tokens):
    ret = [stemmer.stem(lmtzr.lemmatize(w)) for w in tokens]
    return ret
# End lemmatize_stems()
