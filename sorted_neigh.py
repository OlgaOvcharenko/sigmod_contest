import csv
import multiprocessing
import time
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re
import os
import numpy as np
from joblib import Parallel, delayed
import string

x1_matches = []
x2_matches = []

pd.options.mode.chained_assignment = None


def get_recall():
    Y1 = pd.read_csv("Y1.csv").values.tolist()
    Y2 = pd.read_csv("Y2.csv").values.tolist()

    global x1_matches, x2_matches
    count_matches1 = 0
    for i in Y1:
        for j in x1_matches:
            count_matches1 += 1 if i[0] == j[0] and i[1] == j[1] else 0

    print(f"Number pairs X1:  {len(x1_matches)}")
    print(f"Recall X1:  {count_matches1 / len(Y1)}")

    count_matches2 = 0
    for i in Y2:
        for j in x2_matches:
            count_matches2 += 1 if i[0] == j[0] and i[1] == j[1] else 0
    print(f"Number pairs X2:  {len(x2_matches)}")
    print(f"Recall X2:  {count_matches2 / len(Y2)}")

    print(f"Overall recall:  {(count_matches1 + count_matches2) / (len(Y2) + len(Y1))}")
    return


candidate_size1, candidate_size2 = 0, 0


def normalize_string(str_to_normalize: str):
    # lowercase, no punctuation or - | &,
    # remove website names like as amazon.com/ebay/techbuy/alienware/Miniprice.ca,
    # wholesale/new/used/brand,
    # computer/computers/laptop/pc,
    # buy/sale,
    # best/good/quality
    # single letters
    # stopwords as on/in/at/from etc
    # inch, GHz, Hz, cm
    stopwords = {"on", "in", "at", "from", "as", "an", "the", "a", "with", "and", "or", "of", "but", "and",
                 "amazon.com", "ebay", "techbuy", "alienware", "miniprice.ca", "alibaba",
                 "wholesale", "new", "used", "brand",
                 "computer", "computers", "laptops", "laptop", "product", "products", "tablet", "tablets", "pc",
                 "buy", "sale", "best", "good", "quality",
                 "accessories", "kids", ""
                 ",", "|", "/", "@", "!", "?", "-",
                 "1st", "2nd", "3rd",
                 "ghz", "inch", "cm", "mm", "mhz", "gb", "kb", }

    # remove domain names
    pattern_domain_name = "^((?!-)[A-Za-z0-9-]" + "{1,63}(?<!-)\\.)" + "+[A-Za-z]{2,6}"
    no_domain_str = re.sub(pattern_domain_name, '', str_to_normalize.lower())

    # replace 5 cm to 5cm (Hz, inch etc) etc
    pattern_measures_name = "(?:\d+)\s+(inch|cm|mm|m|hz|ghz|gb|mb|g)"
    no_domain_str = re.sub(pattern_measures_name, '', no_domain_str)

    # remove punctuation
    no_punctuation_string = no_domain_str.translate(str.maketrans(string.punctuation, " "*len(string.punctuation)))

    result_words = [word for word in re.split("\W+", no_punctuation_string)
                    if word not in stopwords and len(word) != 1]
    return result_words


def block_with_attr(X, id, attr, is_X1:bool):
    tokens_per_row = np.zeros((X.shape[0],), dtype=object)

    doc_frequencies = dict()
    frequencies_list, token_row_indices = list(), list()  # FIXME
    token_row_indices_append, frequencies_list_append = token_row_indices.append, frequencies_list.append

    num_words, count_tokens = 0, 0

    # tokenize records patterns and count frequency
    for i in tqdm(range(X.shape[0])):
        attr_i = str(X[attr][i])
        result_words = normalize_string(attr_i)
        normalized_str, num_words = " ".join(result_words), len(result_words)

        # For Jaccard distance later
        result_words = sorted(result_words, key=len, reverse=True)
        X[attr][i] = " ".join(result_words)

        # | (\w{2, }\s\w+\s\w+\s\w+\d+) | (\w{2, }\s\w)
        # pattern = re.findall("(\w{2,}\s\w+\d+)", normalized_str)  # FIXME split instead? top 5 longest? or all pairs?
        pattern = result_words
        # pattern = " ".join(result_words)
        if len(pattern) == 0:
            tokens_per_row[i] = result_words[0:3]
        tokens_per_row[i] = pattern

        for token in pattern:
            frequency = doc_frequencies.get(token)
            if frequency is not None:
                frequencies_list[frequency] += 1
            else:
                doc_frequencies[token] = count_tokens
                token_row_indices_append(i)
                frequencies_list_append(1)
                count_tokens += 1

        num_words += num_words

    # calculate tf-idf values
    frequencies_list, token_row_indices = np.array(frequencies_list, dtype=int), np.array(token_row_indices, dtype=int)
    tf, idf = frequencies_list / num_words, np.log(X.shape[0] / (frequencies_list + 1))
    tf_idfs = tf * idf

    # compute average of non-unique token tfidf value
    non_unique_indices = (frequencies_list > 1)
    tfidf_row_non_unique_avg = pd.DataFrame(
        np.vstack([token_row_indices[non_unique_indices], tf_idfs[non_unique_indices]]).transpose(),
        columns=["tid", "tfidf"]).groupby("tid").mean()

    # block values by their high value tokens
    blocks = dict()
    for i in tqdm(range(tokens_per_row.shape[0])):
        # each row tokens
        for token in tokens_per_row[i]:
            token_index = doc_frequencies[token]

            try:
                tfidf_row_avg = tfidf_row_non_unique_avg.loc[i][0]
            except KeyError:
                tfidf_row_avg = None  # unique token - no need

            # block only frequent tokens
            if tfidf_row_avg and tf_idfs[token_index] > tfidf_row_avg:
                record_list = [i]
                token_block = blocks.get(token)
                if token_block is not None:
                    record_list.extend(token_block)
                blocks[token] = record_list

    doc_frequencies.clear()
    tokens = np.empty(1)

    # matching and improve block collection as an index and create pairs, weighted record pairs
    # TODO can be skipped and just all pairs matching then
    weighted_pairs = set()
    weights_pairs_add = weighted_pairs.add
    for _, block_ids in tqdm(blocks.items()):
        def intersection(s1, s2) -> []:
            s1, s2 = set(s1.lower().split()), set(s2.lower().split())
            return len(s1.intersection(s2)) / max(len(s1), len(s2))

        # all pairs with different ids weighted by Levenstein distance
        _ = [weights_pairs_add((intersection(X[attr][a], X[attr][b]), X[id][a], X[id][b])) if X[id][a] < X[id][b]
                     else weights_pairs_add((intersection(X[attr][a], X[attr][b]), X[id][b], X[id][a]))
                     for idx, a in enumerate(block_ids) for b in block_ids[idx + 1:] if X[id][b] != X[id][a]]

    weighted_pairs = np.array(list(weighted_pairs))
    weight_avg = np.mean(weighted_pairs[:, 0])
    pairs = (weighted_pairs[weighted_pairs[:, 0] >= weight_avg])[:, 1:3]

    # pairs = [pairs[:, 0].argsort()[::-1]]

    # write_to_csv(pairs, is_X1)

    global x1_matches, x2_matches
    if is_X1:
        x1_matches = pairs
    else:
        x2_matches = pairs

    return list(pairs)


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

X1_candidate_pairs = block_with_attr(X1, id="id", attr="title", is_X1=True)
X2_candidate_pairs = block_with_attr(X2, id="id", attr="name", is_X1=False)

# save results
save_output(X1_candidate_pairs, X2_candidate_pairs)

end = time.time()
print(f"Runtime of the program is {end - start}")
get_recall()
