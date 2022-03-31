import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import timeit

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

SIMILARITY = "similarity"
IDX = "lid"
IDY = "rid"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger()


def train_lsh(tfidf, n_vectors, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dim = tfidf.shape[1]
    random_vectors = generate_random_vectors(dim, n_vectors)

    # partition data points into bins,
    # and encode bin index bits into integers
    bin_indices_bits = tfidf.dot(random_vectors) >= 0
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    bin_indices = bin_indices_bits.dot(powers_of_two)

    # update `table` so that `table[i]` is the list of document ids with bin index equal to i
    table = defaultdict(list)
    for idx, bin_index in enumerate(bin_indices):
        table[bin_index].append(idx)

    # note that we're storing the bin_indices here
    # so we can do some ad-hoc checking with it,
    # this isn't actually required
    model = {'table': table,
             'random_vectors': random_vectors,
             'bin_indices': bin_indices,
             'bin_indices_bits': bin_indices_bits}
    return model


def recall(true, prediction):
    return (len(set(true).intersection(set(prediction)))) / len(true)


def map_async(iterable, func, model, max_workers=os.cpu_count()):
    # https://devdreamz.com/question/825056-how-do-i-use-threads-on-a-generator-while-keeping-the-order
    # Generator that applies func to the input using max_workers concurrent jobs
    def async_iterator():
        iterator = iter(iterable)
        pending_results = []
        has_input = True
        thread_pool = ThreadPoolExecutor(max_workers)
        while True:
            # Submit jobs for remaining input until max_worker jobs are running
            while (
                has_input
                and len([e for e in pending_results if e.running()]) < max_workers
            ):
                try:
                    ids, data = next(iterator)
                    logger.debug(f"Submitting Task")
                    pending_results.append(thread_pool.submit(func, model, data, ids))
                except StopIteration:
                    logger.debug(f"Submitted all task")
                    has_input = False

            # If there are no pending results, the generator is done
            if not pending_results:
                return

            # If the oldest job is done, return its value
            if pending_results[0].done():
                yield pending_results.pop(0).result()
            # Otherwise, yield the CPU, then continue starting new jobs
            else:
                time.sleep(0.01)

    return async_iterator()


def generate_similarity_df(
    similarity_matrix: np.ndarray, match_over: pd.Series
) -> pd.DataFrame:
    similarity_upper_tri_indices = np.nonzero(np.triu(similarity_matrix))
    similarity_upper_tri = similarity_matrix[similarity_upper_tri_indices]

    _rows = similarity_upper_tri_indices[0]
    _cols = similarity_upper_tri_indices[1]

    df = pd.DataFrame(
        {
            IDX: match_over[_rows].values,
            IDY: match_over[_cols].values,
            SIMILARITY: similarity_upper_tri,
        }
    )

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[IDX] != df[IDY]]


def get_matched_pair(
    vector_representation: np.ndarray,
    similarity_over: pd.Series,
):
    similarity_matrix = 1 - squareform(pdist(vector_representation, metric="cosine"))
    matched_pair_id = generate_similarity_df(similarity_matrix, similarity_over)
    matched_pair_id = remove_duplicates(df=matched_pair_id)
    matched_pair_id.sort_values(by=[SIMILARITY], inplace=True, ascending=False)
    matched_pair_id = matched_pair_id[matched_pair_id[SIMILARITY] > 0.1]
    return matched_pair_id


def pre_process(df: pd.DataFrame):
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df = df.applymap(lambda x: re.sub(r"\W+", " ", x) if type(x) == str else x)
    return df


def batch_gen(data: pd.DataFrame, attr: str):
    for i, df_chunk in data.iterrows():
        yield df_chunk["id"], df_chunk[attr]


def tfidf_hashed(x, n_features):
    return HashingVectorizer(n_features=n_features).transform(x)


def generate_random_vectors(dim, n_vectors):
    """
    generate random projection vectors
    the dims comes first in the matrix's shape,
    so we can use it for matrix multiplication.
    """
    return np.random.randn(dim, n_vectors)


def get_similarity_items(x, item_id, topn=5):
    """
    Get the top similar items for a given item id.
    The similarity measure here is based on cosine distance.
    """
    query = x[item_id]
    scores = x.dot(query.reshape(1, -1).T).ravel()
    best = np.argpartition(scores, -topn)[-topn:]
    return sorted(zip(best, scores[best]), key=lambda x: -x[1])


def block_with_attr(X, attr):  # replace with your logic.
    """
    This function performs blocking using attr
    :param X: dataframe
    :param attr: attribute used for blocking
    :return: candidate set of tuple pairs
    """
    logger.info(f"EXECUTION STARTED")

    X = pre_process(X)
    # X = X.append(X * 1000, ignore_index=True)
    # build index from patterns to tuples
    # tfidf = TfidfVectorizer(
    #     analyzer='word',
    #     ngram_range=(1, 3),
    #     min_df=0,
    #     stop_words='english')
    # X_tfidf = tfidf.fit_transform(X[attr].values.tolist())

    Q = generate_random_vectors(864908, 20)

    start = timeit.default_timer()
    embeddings = tfidf_hashed(X[attr].values.tolist(), n_features=20).toarray()
    embeddings_ids = X["id"].values.tolist()
    matched_pair_id = remove_duplicates(
        get_matched_pair(embeddings, similarity_over=pd.Series(embeddings_ids))
    )

    model = train_lsh(embeddings, 100, seed=143)
    similar_items = get_similarity_items(embeddings, 0)

    similar_item_ids = [similar_item for similar_item, _ in similar_items]
    bits1 = model['bin_indices_bits'][similar_item_ids[0]]
    bits2 = model['bin_indices_bits'][similar_item_ids[1]]

    # UNCOMMENT TO DEBUG
    # matched_pair_str = get_matched_pair(
    #     embeddings,
    #     similarity_over=X[attr],
    # )
    # matched_pair_str.to_csv("debug.csv")

    candidate_pairs = list(zip(matched_pair_id[IDX], matched_pair_id[IDY]))
    candidate_pairs_real_ids = list()

    for it in tqdm(candidate_pairs):
        real_id1, real_id2 = it

        # NOTE: This is to make sure in the final output.csv, for a pair id1 and id2 (assume id1<id2),
        # we only include (id1,id2) but not (id2, id1)
        if real_id1 < real_id2:
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))

    stop = timeit.default_timer()
    logger.info(f"EXECUTION TIME {stop - start}")
    return candidate_pairs_real_ids


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
