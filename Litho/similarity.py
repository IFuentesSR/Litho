from collections import Counter

from fuzzywuzzy import fuzz, process

from .nlp_funcs import (get_nouns, lemmatize_stems, tokenize_and_stem,
                        tokenize_only)


def check_similarity(row, target, threshold=60):
    """
    Check to see if MajorLithCode similarity score is above threshold.
    Uses `fuzzywuzzy.fuzz.ratio()` to calculate text similarity.

    :param row: DataFrame tuple, of row provided through `.itertuples()`
                'Description' and 'MajorLithCode' columns are expected.
    :param target: str, word to find similarity of
    :param threshold: float, threshold that similarity must be above.

    :returns: None or str, MajorLithCode that is above threshold.
    """
    if fuzz.ratio(row['Description'], target) > threshold:
        return row['MajorLithCode']
    # End if
# End check_similarity()


def match_lithcode(row, target, stopwords, threshold=60):
    """
    Check if given row has a matching instance of the target word and return related LithCode.
    Uses `fuzzywuzzy.fuzz.ratio()` to calculate text similarity.

    :param row: DataFrame tuple, of row provided through `.itertuples()`.
                'Description' and 'MajorLithCode' elements are expected.
    :param target: str, word to find matches for
    :param threshold: float, threshold that similarity must be above.
    """
    desc = row.Description
    if len(target.split()) > 1:
        noun_tokens = get_nouns(tokenize_only(target, stopwords))
        res = (target, fuzz.ratio(' '.join(noun_tokens), ' '.join(tokenize_and_stem(desc, stopwords))))
    else:
        res = (target, 100) if target in desc else False
    # End if

    if not res:
        return

    if res[1] > threshold:
        return row.MajorLithCode

# End match_lithcode()


def jaccard_similarity(query, document):
    """
    Jaccard similarity.
    See: http://billchambers.me/tutorials/2014/12/21/tf-idf-explained-in-python.html

    :param query: list[str], tokenized text
    :param document: list[str], tokenized text to compare against

    :returns: float, a score indicating the similarity between two texts
    """
    query_set = set(query)
    doc_set = set(document)

    intersection = query_set.intersection(doc_set)
    union = query_set.union(doc_set)
    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)
# End jaccard_similarity()


def calc_similarity_score(t1, t2):
    """
    Calculate similarity score.

    :param t1: list[str], tokenized text to compare
    :param t2: list[str], tokenized text to compare against

    :returns: float, similarity score
    """
    score = jaccard_similarity(t1, t2)
    s2 = fuzz.token_set_ratio(" ".join(t1), " ".join(t2)) / 100
    if abs(score - s2) > 0.3:
        # Get the min of average or weighted average
        # score = min((score + s2 / 2), (score + s2 / score**2))
        score = (score + s2) / 2.0
    else:
        score = s2
    # End if

    return score
# End calc_similarity_score()


def print_sim_compare(t1, t2, stopwords):
    """
    Debug/testing function. Prints out similarity scores.

    :param t1: str, text to compare
    :param t2: str, text to compare against

    """
    t1 = get_nouns(tokenize_and_stem(t1.strip(), stopwords))
    t2 = get_nouns(tokenize_and_stem(t2.strip(), stopwords))
    print('Jaccard:', jaccard_similarity(t1, t2))
    print('Ratio:', fuzz.ratio(" ".join(t1), " ".join(t2)) / 100)
    print('Partial Ratio:', fuzz.partial_ratio(" ".join(t1), " ".join(t2)) / 100)
    print('Token Set Ratio:', fuzz.token_set_ratio(" ".join(t1), " ".join(t2)) / 100)
    print('Token Sort Ratio:', fuzz.token_sort_ratio(" ".join(t1), " ".join(t2)) / 100)

    # Calculate similarity score
    print("noun tokens", t1, t2)
    score = calc_similarity_score(t1, t2)

    print("Score would be:", score)
# End print_sim_compare()


def merge_similar_words(lith_df, stopwords, target_cols=None):
    """Merge similar words together.

    :param lith_df: DataFrame
    :param stopwords: list[str], common words to filter out
    :param target_cols: list[str], column names with additional features to use other than 'Description'

    :returns: tuple, `DataFrame` and `set` of unique words
    """
    assert 'Description' in lith_df.columns, "'Description' column not found in DataFrame"

    if not target_cols:
        target_cols = []

    all_words = Counter()
    target_cols = []
    tokenized_desc = []
    for row in lith_df.itertuples():
        tmp = tokenize_and_stem(row.Description, [])

        # Filter words that are less than 3 characters long
        tmp = [w.strip() for w in tmp if w not in stopwords and len(w) > 2]

        for col in target_cols:
            if col == 'Description':
                continue
            tmp.append(getattr(row, col))
        # End for
        tokenized_desc.append(tmp)
        all_words.update({w: 1 for w in tmp})
    # End for

    # We can filter words here based on occurance but leaving that for now...
    all_words = [w for w in all_words]

    word_map = {}
    for word in all_words:
        tmp = [w for w, score in process.extract(word, all_words) if score > 85]
        word_map[word] = min(tmp, key=len)
    # End for

    updated_desc = []
    word_set = set()
    for desc in tokenized_desc:
        tmp = [word_map[w] for w in desc]
        updated_desc.append(tmp)
        word_set.update(tmp)
    # End for

    tokenized_desc = updated_desc
    lith_df['tokens'] = tokenized_desc

    all_words = sorted(list(word_set))

    return lith_df, all_words, word_map
# End merge_similar_words()
