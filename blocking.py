import csv
import os
import re
import multiprocessing
import time
from itertools import combinations

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
# import nltk
# nltk.download('omw-1.4')
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer

candidate_size1, candidate_size2 = 0, 0

def block_with_attr(X, id, attr, is_X1:bool):
    global start
    if time.time() - start > 1500:
        return

    tokens = np.zeros((X.shape[0],), dtype=object)
    doc_frequencies, frequencies_list, token_row_indices = dict(), list(), list()
    token_row_indices_append, frequencies_list_append = token_row_indices.append, frequencies_list.append

    num_words, count_tokens = 0, 0

    # tokenize records patterns and count frequency
    for i in tqdm(range(X.shape[0])):
        attr_i = str(X[attr][i])
        pattern = re.findall("\w+\s\w+\d+", attr_i)  # look for patterns like "thinkpad x1" #FIXME
        if len(pattern) == 0:
            tokens[i] = []
            continue
        pattern = [p.lower() for p in pattern]
        tokens[i] = pattern

        for token in pattern:
            frequency = doc_frequencies.get(token)
            if frequency is not None:
                frequencies_list[frequency] += 1

            else:
                doc_frequencies[token] = count_tokens
                token_row_indices_append(i)
                frequencies_list_append(1)
                count_tokens += 1

        num_words += 1

    frequencies_list, token_row_indices = np.array(frequencies_list), np.array(token_row_indices, dtype=int)
    tf, idf = frequencies_list / num_words, np.log(X.shape[0] / (frequencies_list + 1))
    tf_idfs = tf * idf

    non_unique_indices = (frequencies_list > 1)
    tfidf_row_non_unique_avg = pd.DataFrame(np.vstack([token_row_indices[non_unique_indices], tf_idfs[non_unique_indices]]).transpose(),
                                   columns=["rid", "tfidf"]).groupby("rid").mean()

    # block values by their high value tokens
    blocks = dict()
    for i in tqdm(range(tokens.shape[0])):
        for token in tokens[i]:
            token_index = doc_frequencies[token]

            try:
                tfidf_row_avg = tfidf_row_non_unique_avg.loc[i][0]
            except KeyError:
                tfidf_row_avg = 0

            if tf_idfs[token_index] > tfidf_row_avg:
                record_list = [i]
                if token in blocks:
                    record_list.extend(blocks.get(token))

                blocks[token] = record_list

    doc_frequencies.clear()
    tokens = np.empty(1)

    # improve block collection as an index, create index of weights for record pairs
    pairs = []
    pairs_extend = pairs.extend
    blocks_list = list(blocks.values())
    blocks.clear()
    # for block_records in tqdm(blocks_list):
    #     all_pairs = [(X[id][a], X[id][b]) if X[id][a] < X[id][b] else (X[id][b], X[id][a])
    #                  for idx, a in enumerate(block_records) for b in block_records[idx + 1:] if X[id][b] != X[id][a]]
    #     pairs_extend(all_pairs)

    # improve block collection as an index, create index of weights for record pairs
    weights_pairs = []
    num_pairs, sum_weights = 0, 0
    weights_extend = weights_pairs.extend
    for block_records in tqdm(blocks_list):
        def intersection(lst1, lst2) -> []:
            return list(set(lst1) & set(lst2))
        all_pairs = [[len(intersection(X[attr][a], X[attr][b])), X[id][a], X[id][b]] if X[id][a] < X[id][b]
                     else [len(intersection(X[attr][a], X[attr][b])), X[id][b], X[id][a]]
                     for idx, a in enumerate(block_records) for b in block_records[idx + 1:] if X[id][b] != X[id][a]]
        weights_extend(all_pairs)

    if weights_pairs:
        weights_pairs = np.unique(np.vstack(weights_pairs), axis=0)
    else:
        return

    weight_avg = np.mean(weights_pairs[:, 0])
    pairs = (weights_pairs[weights_pairs[:, 0] >= weight_avg])[:, 1:3]

    write_to_csv(pairs, is_X1)
    return


def write_to_csv(X_candidate_pairs, is_X1: bool, out_file: str = "output.csv"):
    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    global candidate_size1, candidate_size2

    if (is_X1 and candidate_size1 >= expected_cand_size_X1) or (not is_X1 and candidate_size2 >= expected_cand_size_X2):
        pass
    else:
        if is_X1 and (candidate_size1 + len(X_candidate_pairs)) > expected_cand_size_X1:
            X_candidate_pairs = X_candidate_pairs[:expected_cand_size_X1-candidate_size1]
        if not is_X1 and (candidate_size2 + len(X_candidate_pairs)) > expected_cand_size_X2:
            X_candidate_pairs = X_candidate_pairs[:expected_cand_size_X2-candidate_size2]

        candidate_size1 += len(X_candidate_pairs) if is_X1 else 0
        candidate_size2 += len(X_candidate_pairs) if not is_X1 else 0
        with open(out_file, "a") as out:
            csv_out = csv.writer(out)
            csv_out.writerows(X_candidate_pairs)


def fill_output_file_with_0(is_X1: bool, out_file: str = "output.csv"):
    global candidate_size1, candidate_size2
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    X_extended = []
    if is_X1 and candidate_size1 < expected_cand_size_X1:
        X_extended = [(0, 0)] * (expected_cand_size_X1 - candidate_size1)
    if not is_X1 and candidate_size2 < expected_cand_size_X2:
        X_extended = [(0, 0)] * (expected_cand_size_X2 - candidate_size2)

    with open(out_file, "a") as out:
        csv_out = csv.writer(out)
        csv_out.writerows(X_extended)


def fill_output_file():
    file1, file2 = open("out1.csv") if os.path.exists("out1.csv") else [], \
                   open("out2.csv") if os.path.exists("out2.csv") else []
    out1, out2 = list(csv.reader(file1)) if file1 else file1, list(csv.reader(file2)) if file2 else file2
    save_output(out1, out2)


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


if os.path.exists("output.csv"):
    os.remove("output.csv")

with open("output.csv", "a") as header_fp:
    file_header = ["left_instance_id", "right_instance_id"]
    csv_out = csv.writer(header_fp)
    csv_out.writerow(file_header)

start = time.time()

# read the datasets
X1 = pd.read_csv("X1.csv")
X2 = pd.read_csv("X2.csv")

# X1 = duplicate_with_new_id(X1, 10)
# X2 = duplicate_with_new_id(X2, 10)
# print("X1 size " + str(len(X1)))
# print("X2 size " + str(len(X2)))

X1_blocks = naive_blocking(X1, 50)
X2_blocks = naive_blocking(X2, 70)
# X2_blocks = group_and_blocking(X2, ["brand"])

# perform blocking
num_cores = multiprocessing.cpu_count()

# FIXME hardcoded num jobs
_ = Parallel(n_jobs=16, require='sharedmem')(delayed(block_with_attr)(i.reset_index(), id="id", attr="title", is_X1=True) for i in X1_blocks)
fill_output_file_with_0(True)
_ = Parallel(n_jobs=16, require='sharedmem')(delayed(block_with_attr)(i.reset_index(), id="id", attr="name", is_X1=False) for i in X2_blocks)
fill_output_file_with_0(False)

# # check length of the file
# with open("output.csv", "r") as fp:
#     for (count, _) in enumerate(fp, 1):
#        pass
#
#     if count < 3000000:
#         with open("output.csv", "a") as fp1:
#             X_extended = [(0, 0)] * (3000000 - count)
#             csv_out = csv.writer(fp1)
#             csv_out.writerows(X_extended)
#
#     if count > 3000000:
#         csv_in = csv.reader(fp)
#         data = list(csv_in)[:3000000]
#         with open("output.csv", "w") as fp2:
#             csv_out = csv.writer(fp2)
#             csv_out.writerows(data)

# X1_block_pairs = [pairs for pairs in X1_block_pairs if pairs]
# X2_block_pairs = [pairs for pairs in X2_block_pairs if pairs]

# X1_candidate_pairs = np.vstack(X1_block_pairs).tolist() if len(X1_block_pairs) > 0 else []
# X2_candidate_pairs = np.concatenate(X2_block_pairs, axis=0).tolist() if len(X1_block_pairs) > 0 else []
#
# # save results
# save_output(X1_candidate_pairs, X2_candidate_pairs)

# end = time.time()
# print(f"Runtime of the program is {end - start}")
