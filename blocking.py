import itertools
import re
import time
from collections import defaultdict

import pandas as pd
import random
import pdb

import baseline
from ann_search import LSHRPQuery
from feature_embeddings import TFIDFHashedEmbeddings
from preprocessing import Preprocessor
from lsh import *

start = time.time()


def hash_by_number(name: str, is_X2: bool):
    pattern = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
    all_numbers = set(filter(lambda num1: num1 > 512,
                             map(lambda num: int(num.replace(".", "").replace(",", "")), re.findall(pattern, name))))

    # all_numbers = set(map(lambda num: int(num.replace(".", "").replace(",", "")), re.findall(pattern, name)))

    def cantor_pairing(a, b):
        return (a + b) * ((a + b) / 2) * b

    all_pairs_hashed = []
    if len(all_numbers) == 2:
        all_pairs_hashed = [int(cantor_pairing(pair[0], pair[1])) if pair[0] < pair[1] else
                            int(cantor_pairing(pair[1], pair[0]))
                            for pair in itertools.combinations(all_numbers, 2)]

    elif len(all_numbers) > 6:
        for pair in itertools.combinations(all_numbers, 6):
            pair_sorted = sorted(pair)
            all_pairs_hashed.append(int(
                cantor_pairing(cantor_pairing(
                    cantor_pairing(cantor_pairing(
                        cantor_pairing(pair_sorted[0], pair_sorted[1]), pair_sorted[2]), pair_sorted[3]), pair_sorted[4]), pair_sorted[5])))

    elif len(all_numbers) > 4:
        for pair in itertools.combinations(all_numbers, 4):
            pair_sorted = sorted(pair)
            all_pairs_hashed.append(int(cantor_pairing(cantor_pairing(
                cantor_pairing(pair_sorted[0], pair_sorted[1]), pair_sorted[2]), pair_sorted[3])))

    elif len(all_numbers) > 3:
        for pair in itertools.combinations(all_numbers, 3):
            pair_sorted = sorted(pair)
            all_pairs_hashed.append(int(cantor_pairing(cantor_pairing(
                pair_sorted[0], pair_sorted[1]), pair_sorted[2])))

    return all_pairs_hashed


def get_empty_string_pairs(dataset):
    ds = dataset[dataset['title'] == '']
    if ds.empty:
        return []

    ids = ds['id'].to_list()
    candidates = set()
    for ix1, id1 in enumerate(ids):
        for _, id2 in enumerate(ids[ix1 + 1:]):
            pair = (id1, id2) if id1 < id2 else (id2, id1)
            candidates.add(pair)
    return candidates


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

    # make sure to have the pairs in the first dataset first
    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs
    output_df = pd.DataFrame(
        all_cand_pairs, columns=["left_instance_id", "right_instance_id"]
    )
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


def blocking_step(df_path, k=3, buckets=15, hash_function_count=150, is_X2: bool=False):
    ds = Preprocessor.build(df_path)
    dataset = ds.preprocess()
    all_pairs_hashed = defaultdict(list)

    shingles = []
    for _, row in dataset.iterrows():
        data = row["title"]
        shingles.append((row["id"], k_shingles(data, k)))

        if is_X2:
            [all_pairs_hashed[k].append(row['id']) for k in hash_by_number(data, is_X2)]
        shingles.append((row['id'], k_shingles(data, k)))

    all_shingles = {item for set_ in shingles for item in set_[1]}

    vocab = {}
    for i, shingle in enumerate(list(all_shingles)):
        vocab[shingle] = i

    del all_shingles

    lsh = LSH(buckets)

    arr = gen_minhash(vocab, hash_function_count)
    for id, shingle in shingles:
        if not shingle:
            continue
        ohe = one_hot(shingle, vocab)

        fingerprint = get_fingerprint(arr, ohe)
        lsh.hash(fingerprint, id)

    feature_embeddings = TFIDFHashedEmbeddings()
    feature_embeddings.load()

    emd = feature_embeddings.generate(dataset["title"].tolist(), n_features=50)

    ann_search_index = LSHRPQuery()
    nn, distances = ann_search_index.load_and_query(emd, n_bits=32)
    rp_cp, _ = ann_search_index.generate_candidate_pairs(
        nn, distances, dataset["id"].to_list()
    )
    del nn, distances, ann_search_index
    lsh_cp = lsh.get_candidate_pairs()
    # lsh_cp = lsh_cp.union(rp_cp)

    empty_string_pairs = get_empty_string_pairs(dataset)

    baseline_res = []
    if time.time() - start <= (1700 if is_X2 else 900):
        baseline_res = baseline.block_with_attr(dataset, 'title')

    cand_pairs = []
    if is_X2 and time.time() - start <= 1800:
        for hashed_key in all_pairs_hashed:
            for pair in itertools.combinations(all_pairs_hashed[hashed_key], 2):
                if len(all_pairs_hashed[hashed_key]) > 1:
                    cand_pairs.append((pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0]))

    res = prepare_output(lsh_cp, rp_cp, 1000000 if not is_X2 else 2000000,
                         [empty_string_pairs, cand_pairs, baseline_res])

    print(f"LSH CP \t{len(lsh_cp)}")
    print(f"RP CP \t{len(rp_cp)}")
    print(f"Hashed numbers \t{len(cand_pairs)}")
    print(f"Baseline \t{len(baseline_res)}")
    print(f"By empty string\t{len(empty_string_pairs)}")

    return list(res)


def recall(true, prediction):
    return (len(set(true).intersection(set(prediction)))) / len(true)


def prepare_output(lsh_cp: set, rp_cp: set, expected_output_size: int, other_results):
    # intersection
    both_lsh_pairs = list(lsh_cp.intersection(rp_cp))

    tmp_len = len(both_lsh_pairs)

    # symmetric difference and other
    other_res = list()
    difference_lsh = lsh_cp.symmetric_difference(rp_cp)
    difference_lsh_others = set()
    if tmp_len < expected_output_size:
        for res in other_results:
            other_res += res

        other_res = set(other_res)
        difference_lsh_others = difference_lsh.intersection(other_res)
        tmp_len += len(difference_lsh_others)

    lsh_cp_rest = set()
    if tmp_len < expected_output_size:
        lsh_cp_rest = (lsh_cp.difference(rp_cp)).difference(difference_lsh_others)
        tmp_len += len(lsh_cp_rest)

    rp_cp_rest = set()
    if tmp_len < expected_output_size:
        rp_cp_rest = (rp_cp.difference(lsh_cp)).difference(difference_lsh_others)
        tmp_len += len(rp_cp_rest)

    res = list(both_lsh_pairs) + list(difference_lsh_others) + list(lsh_cp_rest) + list(rp_cp_rest)

    return res


if __name__ == "__main__":
    start = time.time()
    X1_candidate_pairs = blocking_step(df_path="X1.csv", is_X2=False)
    X2_candidate_pairs = blocking_step(df_path="X2.csv", is_X2=True)

    print(f"X1_candidate_pairs: {len(X1_candidate_pairs)}")
    print(f"X2_candidate_pairs: {len(X2_candidate_pairs)}")
    #  pdb.set_trace()
    r1 = recall(
        pd.read_csv("Y1.csv").to_records(index=False).tolist(), X1_candidate_pairs
    )
    r2 = recall(
        pd.read_csv("Y2.csv").to_records(index=False).tolist(), X2_candidate_pairs
    )
    r = (r1 + r2) / 2
    print(f"RECALL FOR X1 \t\t{r1:.3f}")
    print(f"RECALL FOR X2 \t\t{r2:.3f}")
    print(f"RECALL OVERALL  \t{r:.3f}")
    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
