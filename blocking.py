import re
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd
from tqdm import tqdm


def block_with_attr(X, id, attr):
    tokens = np.zeros((X.shape[0],), dtype=object)
    doc_frequencies = dict()

    # TODO normalization

    num_words = 0

    # tokenize records patterns and count frequency
    for i in tqdm(range(X.shape[0])):
        attr_i = str(X[attr][i])
        pattern = re.findall("\w+\s\w+\d+", attr_i.lower())  # look for patterns like "thinkpad x1"
        if len(pattern) == 0:
            tokens[i] = []
            continue

        tokens[i] = pattern

        for token in pattern:
            num_words += 1
            frequency = doc_frequencies.get(token)
            if frequency is not None:
                doc_frequencies[token] = [frequency[0] + 1]
            else:
                doc_frequencies[token] = [1]

    # calculate tf-idf values
    tf_idf_avg_list = np.zeros((X.shape[0],), dtype=object)
    for i in tqdm(range(X.shape[0])):
        sum, count = 0, 0
        for token in tokens[i]:
            tf = doc_frequencies.get(token)[0] / num_words
            idf = np.log(X.shape[0] / (doc_frequencies.get(token)[0] + 1))

            tf_idf = tf * idf
            frequencies = doc_frequencies.get(token)
            frequencies.append(tf_idf)
            doc_frequencies[token] = frequencies
            if doc_frequencies[token][0] > 1:
                sum += tf_idf
                count += 1

        tf_idf_avg = sum / count if count > 0 else 0
        tf_idf_avg_list[i] = tf_idf_avg

    # block values by their high value tokens
    blocks = dict()
    for i in tqdm(range(X.shape[0])):
        for token in tokens[i]:
            if doc_frequencies[token][1] > tf_idf_avg_list[i]:
                record_list = [i]
                if token in blocks:
                    record_list.extend(blocks.get(token))

                blocks[token] = record_list
    doc_frequencies.clear()

    # improve block collection as an index, create index of weights for record pairs
    weights_pairs = []
    num_pairs, sum_weights = 0, 0
    for block in tqdm(blocks.items()):
        block_records = block[1]
        all_pairs = [(block_records[i], block_records[j]) for i in range(len(block_records)) for j in range(i+1, len(block_records))]

        for r1_id, r2_id in all_pairs:

            def intersection(lst1, lst2) -> []:
                return list(set(lst1) & set(lst2))

            r1, r2 = tokens[r1_id], tokens[r2_id]
            if r1 != r2:
                r1_original_id, r2_orirginal_id = X[id][r1_id], X[id][r2_id]
                weight = len(intersection(r1, r2))
                weights_pairs.append([weight, r1_original_id, r2_orirginal_id])

                num_pairs += 1
                sum_weights += weight
    blocks.clear()
    tokens = np.empty(1)

    weight_avg = sum_weights / num_pairs if num_pairs > 0 else 0

    # TODO Jaccard similarity or Levenstein distance
    pairs = ([tuple(elem[1:3]) if elem[1] < elem[2] else (elem[2], elem[1]) for elem in tqdm(weights_pairs) if not elem[0] < weight_avg])

    return pairs


def save_output(X1_candidate_pairs,
                X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


def duplicate_with_new_id(X, times_more: int):
    X_new = X
    for i in range(0, times_more):
        X_new = pd.concat([X_new, X])

    return X_new.reset_index()


def naive_blocking(X, num_blocks):
    return np.array_split(X, num_blocks)


def group_and_blocking(X, group_by_cols):
    return X.groupby(group_by_cols)


# read the datasets
X1 = pd.read_csv("X1.csv")
X2 = pd.read_csv("X2.csv")

# X1 = duplicate_with_new_id(X1, 9)
# print("X1 size " + str(len(X1)))
# print("X2 size " + str(len(X2)))

X1_blocks = naive_blocking(X1, 20)
# X2_blocks = naive_blocking(X2, 40)
X2_blocks = group_and_blocking(X2, ["brand"])

# perform blocking
X1_block_pairs = [block_with_attr(X_tmp.reset_index(), id="id", attr="title") for X_tmp in X1_blocks]
X2_block_pairs = [block_with_attr(X_tmp.reset_index(), id="id", attr="name") for _, X_tmp in X2_blocks]

X1_block_pairs = [pairs for pairs in X1_block_pairs if pairs]
X2_block_pairs = [pairs for pairs in X2_block_pairs if pairs]

X1_candidate_pairs = np.vstack(X1_block_pairs).tolist()
X2_candidate_pairs = np.concatenate(X2_block_pairs, axis=0).tolist()

# # perform blocking
# X1_candidate_pairs = block_with_attr(X1, id="id", attr="title")
# X2_candidate_pairs = block_with_attr(X2, id="id", attr="name")

# save results
save_output(X1_candidate_pairs, X2_candidate_pairs)
