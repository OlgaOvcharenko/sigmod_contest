import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import timeit

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sparse_dot_topn import awesome_cossim_topn
from tqdm import tqdm


SIMILARITY = "similarity"
IDX = "lid"
IDY = "rid"

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def recall(true, prediction):
    return (len(set(true).intersection(set(prediction)))) / len(true)


def load_vectors(fname):
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = map(float, tokens[1:])
    return data


def get_glove_embeddings(pth):
    logger.info(f"LOADING GLOVE EMBEDDING")
    embeddings_index = dict()
    with open(pth, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype="float32")
    return embeddings_index


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
                    logger.info(f"Submitting Task")
                    pending_results.append(thread_pool.submit(func, model, data, ids))
                except StopIteration:
                    logger.info(f"Submitted all task")
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


def l2_norm(x):
    return np.sqrt(np.sum(x ** 2))


def div_norm(x):
    norm_value = l2_norm(x)
    if norm_value > 0:
        return x * (1.0 / norm_value)
    else:
        return x


def sentence_embedding(embeddings, data: str, ids):
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
    for word in data.split():
        try:
            running_embedding.append(embeddings[word])
        except KeyError:
            logger.error(f"NO EMBEDDING FOUND FOR {word}")
    u_min = np.min(np.array(running_embedding), axis=0)
    u_max = np.max(np.array(running_embedding), axis=0)
    return ids, np.concatenate((u_min, u_max), axis=-1)


def cosine_similarity(
    a: csr_matrix, b: csr_matrix, ntop: int = 10, lower_bound: float = 0.80
) -> csr_matrix:
    # https://github.com/ing-bank/sparse_dot_topn
    return awesome_cossim_topn(a, b, ntop=ntop, lower_bound=lower_bound)


def generate_similarity_df(
    sparse_matrix: csr_matrix, match_over: pd.Series, top=None
) -> pd.DataFrame:
    sparse_matrix_non_zeros = sparse_matrix.nonzero()

    _rows = sparse_matrix_non_zeros[0]
    _cols = sparse_matrix_non_zeros[1]

    df = pd.DataFrame(
        {
            IDX: match_over[_rows].values[:top],
            IDY: match_over[_cols].values[:top],
            SIMILARITY: sparse_matrix.data[:top],
        }
    )

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[IDX] != df[IDY]]


def get_matched_pair(
    vector_representation: csr_matrix,
    similarity_over: pd.Series,
    top_matches: int = 20,
    confidence_score: float = 0.80,
):
    similarity_matrix = cosine_similarity(
        a=vector_representation,
        b=vector_representation.T,
        ntop=top_matches,
        lower_bound=confidence_score,
    )
    matched_pair_id = generate_similarity_df(similarity_matrix, similarity_over)
    matched_pair_id = remove_duplicates(df=matched_pair_id)
    matched_pair_id.sort_values(by=[SIMILARITY], inplace=True, ascending=False)
    return matched_pair_id


def pre_process(df: pd.DataFrame):
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df = df.applymap(lambda x: re.sub(r"\W+", " ", x) if type(x) == str else x)
    return df


def batch_gen(data: pd.DataFrame, attr: str):
    for i, df_chunk in data.iterrows():
        yield df_chunk["id"], df_chunk[attr]


def block_with_attr(X, attr):  # replace with your logic.
    """
    This function performs blocking using attr
    :param X: dataframe
    :param attr: attribute used for blocking
    :return: candidate set of tuple pairs
    """
    X = pre_process(X)
    glove_embeddings = get_glove_embeddings(r"glove.6B.300d.txt")

    # build index from patterns to tuples
    embeddings = list()
    embeddings_ids = list()

    start = timeit.default_timer()

    for ids, result in map_async(
        batch_gen(X, attr), sentence_embedding, glove_embeddings
    ):
        embeddings.append(result)
        embeddings_ids.append(ids)
        logger.info(f"Got Result")
    stop = timeit.default_timer()
    logger.info(f"EXECUTION TIME {stop - start}")
    embeddings = np.array(embeddings)

    vector_representation = csr_matrix(embeddings)
    matched_pair_id = remove_duplicates(
        get_matched_pair(
            vector_representation,
            similarity_over=pd.Series(embeddings_ids),
            top_matches=30,
            confidence_score=0.60,
        )
    )
    # UNCOMMENT TO DEBUG
    # matched_pair_str = get_matched_pair(
    #     vector_representation,
    #     similarity_over=X[attr],
    #     top_matches=30,
    #     confidence_score=0.80,
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
