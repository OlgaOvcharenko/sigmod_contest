from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import re


def block_with_attr(X, id, attr):
    tokens = dict()
    doc_frequencies = dict()

    # TODO normalization

    num_words = 0

    # tokenize records
    for i in tqdm(range(X.shape[0])):
        attr_i = str(X[attr][i])
        pattern = re.findall("\w+\s\w+\d+", attr_i.lower())  # look for patterns like "thinkpad x1"
        if len(pattern) == 0:
            continue

        tokens[X[id][i]] = [pattern]

        for token in pattern:
            num_words += 1
            frequency = doc_frequencies.get(token)
            if frequency is not None:
                doc_frequencies[token] = frequency + 1
            else:
                doc_frequencies[token] = 1

    # calculate tf-idf values
    tf_idf_vec = np.zeros((len(doc_frequencies),)) # FIXME union with doc_frequencies
    for i in tqdm(range(X.shape[0])):
        sum, count = 0, 0
        for token in tokens.get(X[id][i]):
            tf = doc_frequencies.get(token) / num_words
            idf = np.log(X.shape[0] / (doc_frequencies[token] + 1))

            tf_idf = tf * idf
            tf_idf_vec[token] = tf_idf
            if tf_idf > 1:
                sum += tf_idf
                count += 1

        tf_idf_avg = sum / count
        tokens[X[id][i]] = tokens.get(X[id][i]).append([tf_idf_avg]) # FIXME one dict for all? Maybe lists with i

    # block values by their high value tokens
    blocks = dict()
    for i in tqdm(range(X.shape[0])):
        for token in tokens.get(X[id][i]):
            if tf_idf_vec[token] > tokens[X[id][i]][1][0]:
                record_list = [str(X[attr][i]).lower()]

                if token in blocks:
                    record_list.extend(blocks.get(token))
                blocks[token] = record_list

    # improve block collection as an index, create index of weights for record pairs
    weights = []
    num_pairs, sum_weights = 0, 0
    for block in blocks.keys():
        block_records = blocks.get(block)
        all_pairs = [(block_records[i], block_records[j]) for i in range(len(block_records)) for j in range(i+1, len(block_records))]

        for r1, r2 in zip(all_pairs):
            sim = len(r1.intersection(r2)) / max(len(r1), len(r2))  # Jaccard similarity
            weights.append([r1, r2, sim])  # FIXME ids instead of records

            num_pairs += 1
            sum_weights += sim

    weight_avg = sum_weights / num_pairs

    for i in tqdm(range(weights.shape[0])):
        if weights[i][2] < weight_avg:
            weights.pop(i)

    return weights


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


# read the datasets
X1 = pd.read_csv("X1.csv")
X2 = pd.read_csv("X2.csv")

# perform blocking
X1_candidate_pairs = block_with_attr(X1, id="id", attr="title")
X2_candidate_pairs = block_with_attr(X2, id="id", attr="name")

# save results
save_output(X1_candidate_pairs, X2_candidate_pairs)
