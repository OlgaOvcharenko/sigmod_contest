import logging
import timeit

import numpy as np
import pandas as pd

from ann_search import HNSWLib
from feature_embeddings import TFIDFHashedEmbeddings, FastTextEmbeddings
from utils import (
    pre_process,
    recall,
)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger()


def block_with_attr(X, attr, threshold):  # replace with your logic.
    """
    This function performs blocking using attr
    :param X: dataframe
    :param attr: attribute used for blocking
    :return: candidate set of tuple pairs
    """
    # X = pd.concat([X] * 800, ignore_index=True)

    logger.info(f"EXECUTION STARTED")
    X = pre_process(X, attr)

    start = timeit.default_timer()
    # fast_text_embeddings.load(X[attr].tolist())
    # emd = fast_text_embeddings.generate(X[attr].tolist())
    emd = feature_embeddings.generate(
        X[attr].tolist(), n_features=50, df=X, attr=attr, batch_size=128, epoch=12
    )

    if type(emd) == pd.DataFrame:
        data_emd = np.array(emd["embeddings"].tolist())
        data_ids = emd["id"].tolist()
    else:
        data_emd = emd
        data_ids = X["id"].to_list()
    data_emd = feature_embeddings.normalize(data_emd)
    nn, distances = ann_search_index.load_and_query(
        data_emd,
        neighbours=64,
        n_bits=30,
        raw_data=X[attr].tolist(),
        hash_size=32,
        hash_table=16,
    )

    candidates_pairs, metric = ann_search_index.generate_candidate_pairs(
        nn, distances, data_ids, distance_threshold=threshold
    )

    candidates_pairs, metric = ann_search_index.sort_based_on_metric(
        candidates_pairs, metric
    )
    candidates_pairs = ann_search_index.remove_duplicate(candidates_pairs)
    logger.info(f"TOTAL CANDIDATES: {len(candidates_pairs)}")

    stop = timeit.default_timer()
    logger.info(f"EXECUTION TIME {stop - start}")
    return list(candidates_pairs)


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


feature_embeddings = TFIDFHashedEmbeddings()
feature_embeddings.load()
# fast_text_embeddings = FastTextEmbeddings()

ann_search_index = HNSWLib()  # LSHRPQuery()  # LSHQuery()  # HNSWLib()


# read the datasets
X1 = pd.read_csv("X1.csv")
X2 = pd.read_csv("X2.csv")

# perform blocking
X1_candidate_pairs = block_with_attr(X1, attr="title", threshold=0.80)
X2_candidate_pairs = block_with_attr(X2, attr="name", threshold=0.60)

# print(
#     f"RECALL FOR X1 - {recall(pd.read_csv('Y1.csv').to_records(index=False).tolist(), X1_candidate_pairs)}"
# )
# print(
#     f"RECALL FOR X2 - {recall(pd.read_csv('Y2.csv').to_records(index=False).tolist(), X2_candidate_pairs)}"
# )

# save results
save_output(X1_candidate_pairs, X2_candidate_pairs)
