import logging
import timeit

import hnswlib

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from utils import (
    map_async,
    extract_glove_embeddings,
    pre_process,
    generate_random_vectors,
    batch_gen,
    recall, plot_features,
)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger()

GLOVE_EMBEDDINGS = {}  # extract_glove_embeddings()


def generate_sentence_embedding(data: str, ids):
    """
        :param embeddings:
        :param data:
        :param ids:
        :return:
        @MISC {239071,
        TITLE = {Apply word embeddings to entire document, to get a feature vector},
        AUTHOR = {D.W. (https://stats.stackexchange.com/users/2921/d-w)},
        HOWPUBLISHED = {Cross Validated},
        NOTE = {URL:https://stats.stackexchange.com/q/239071 (version: 2016-10-07)},
        EPRINT = {https://stats.stackexchange.com/q/239071},
        URL = {https://stats.stackexchange.com/q/239071}
    }
    """
    running_embedding = list()
    # present_embedding_words = list(set(data.split()).intersection(set(embeddings.keys())))
    # for word in present_embedding_words:
    #     running_embedding.append(embeddings[word])

    for word in data.split():
        try:
            running_embedding.append(GLOVE_EMBEDDINGS[word])
        except KeyError:
            logger.debug(f"NO EMBEDDING FOUND FOR {word}")
    # u_min = np.min(np.array(running_embedding), axis=0)
    # u_max = np.max(np.array(running_embedding), axis=0)
    if len(running_embedding) != 0:
        emd = np.mean(running_embedding, axis=0)
    else:
        emd = np.zeros(300)
    return ids, emd


def tfidf_hashed(x, n_features):
    return HashingVectorizer(n_features=n_features).transform(x)


def tfidf(x):
    return TfidfVectorizer(stop_words="english").fit_transform(x)


def get_tfidf_features(X, attr):
    logger.info(f"TFIDF IN PROCESS")
    return tfidf(X[attr].values.tolist()).toarray()


def get_tfidf_hashed_features(X, attr, n_features) -> np.ndarray:
    logger.info(f"TFIDF IN PROCESS")
    return tfidf_hashed(X[attr].values.tolist(), n_features=n_features).toarray()


def get_glove_embeddings(X, attr):
    embeddings = list()

    for ids, result in map_async(batch_gen(X, attr), generate_sentence_embedding):
        embeddings.append(result)

    return np.array(embeddings)


def ann_search(
    embeddings, data_ids, neighbours=4, ef_construction=200, m=20, set_ef=50
):
    logger.info(f"ANN SEARCH IN PROCESS")
    size, dim = embeddings.shape
    p = hnswlib.Index(space="cosine", dim=dim)
    p.init_index(max_elements=size, ef_construction=ef_construction, M=m)
    p.add_items(embeddings, data_ids)
    p.set_ef(set_ef)
    labels, distances = p.knn_query(embeddings, k=neighbours)
    return labels, distances


def generate_candidates_pairs(candidates: list, similarity: list):
    logger.info(f"CANDIDATE GENERATION IN PROCESS")
    candidate_pairs = list()
    metric = list()
    for ix, can in enumerate(candidates):
        main_can = can[0]
        for yx in range(1, len(can)):
            similarity_score = similarity[ix][yx]
            if similarity_score < 0.20:
                continue

            can2 = can[yx]
            if main_can < can2:
                candidate_pairs.append((main_can, can2))
            else:
                candidate_pairs.append((can2, main_can))
            metric.append(similarity[ix][yx])
        # candidate_pairs.extend(itertools.combinations(can, 2))

    _, candidates_pairs = zip(
        *sorted(zip(metric, candidate_pairs), key=lambda x: x[0], reverse=True)
    )
    return set(candidate_pairs)


def block_with_attr(X, attr):  # replace with your logic.
    """
    This function performs blocking using attr
    :param X: dataframe
    :param attr: attribute used for blocking
    :return: candidate set of tuple pairs
    """
    # X = pd.concat([X] * 500, ignore_index=True)

    logger.info(f"EXECUTION STARTED")
    X = pre_process(X, attr)

    start = timeit.default_timer()
    embeddings = get_tfidf_hashed_features(X, attr, n_features=100)
    # embeddings = get_glove_embeddings(X, attr)
    # embeddings = get_tfidf_features(X, attr)
    # plot_features(embeddings, X, attr)
    labels, distances = ann_search(embeddings, X["id"].to_list(), neighbours=15)
    candidate_pairs = generate_candidates_pairs(
        labels.tolist(), (1 - distances).tolist()
    )
    logging.info(f"TOTAL CANDIDATES: {len(candidate_pairs)}")
    stop = timeit.default_timer()
    logger.info(f"EXECUTION TIME {stop - start}")
    return list(candidate_pairs)


def save_output(
    X1_candidate_pairs, X2_candidate_pairs
):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend(
            [(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs))
        )
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend(
            [(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs))
        )

    all_cand_pairs = (
        X1_candidate_pairs + X2_candidate_pairs
    )  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(
        all_cand_pairs, columns=["left_instance_id", "right_instance_id"]
    )
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


# read the datasets
X1 = pd.read_csv("X1.csv")
X2 = pd.read_csv("X2.csv")

# perform blocking
X1_candidate_pairs = block_with_attr(X1, attr="title")
X2_candidate_pairs = block_with_attr(X2, attr="name")

print(
    f"RECALL FOR X1 - {recall(pd.read_csv('Y1.csv').to_records(index=False).tolist(), X1_candidate_pairs)}"
)
print(
    f"RECALL FOR X2 - {recall(pd.read_csv('Y2.csv').to_records(index=False).tolist(), X2_candidate_pairs)}"
)

# save results
save_output(X1_candidate_pairs, X2_candidate_pairs)
