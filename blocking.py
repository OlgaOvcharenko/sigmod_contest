import csv
import gc
import itertools
import os
import re
import string
from collections import defaultdict

import numpy as np
import pandas as pd
# from googletrans import Translator
# from langdetect import detect

import baseline
import preprocessing
from ann_search import LSHRPQuery
from feature_embeddings import TFIDFHashedEmbeddings
from preprocessing import Preprocessor
from lsh import *
from save_to_file import save_output, save_X1, save_X2


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


def minhash_important_numbers(dataset):
    k = 2
    shingles = []
    for _, row in dataset.iterrows():
        n = str(row['important_numbers'])
        if n == '777':
            continue
        shingles.append((row['id'], k_shingles(n, k)))

    all_shingles = {item for set_ in shingles for item in set_[1]}

    vocab = {}
    for i, shingle in enumerate(list(all_shingles)):
        vocab[shingle] = i

    del all_shingles

    buckets = 5
    lsh = LSH(buckets)

    hash_function_count = 50
    arr = gen_minhash(vocab, hash_function_count)
    for id, shingle in shingles:
        if len(shingle) == 0:
            continue
        ohe = one_hot(shingle, vocab)

        fingerprint = get_fingerprint(arr, ohe)
        lsh.hash(fingerprint, id)

    return list(lsh.get_candidate_pairs())


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


def blocking_step(df_path, is_X2: bool):
    all_pairs_hashed = defaultdict(list)

    ds = Preprocessor.build(df_path)
    dataset = ds.preprocess()   # TODO: split into #hyperthreads jobs

    k = 3  # ~5 for small docs (emails), 9 - 10 for large docs(papers)
    shingles = []

    for _, row in dataset.iterrows():
        string = row['title']
        if is_X2:
            [all_pairs_hashed[k].append(row['id']) for k in hash_by_number(string, is_X2)]
        shingles.append((row['id'], k_shingles(string, k)))

    all_pairs = []

    # hash by numbers in string
    cand_pairs = []
    if is_X2:
        for hashed_key in all_pairs_hashed:
            for pair in itertools.combinations(all_pairs_hashed[hashed_key], 2):
                if len(all_pairs_hashed[hashed_key]) > 1:
                    cand_pairs.append((pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0]))

    all_shingles = {item for set_ in shingles for item in set_[1]}

    vocab = {}
    for i, shingle in enumerate(list(all_shingles)):
        vocab[shingle] = i

    del all_shingles

    buckets = 15
    lsh = LSH(buckets)

    hash_function_count = 150
    arr = gen_minhash(vocab, hash_function_count)
    for id, shingle in shingles:
        if len(shingle) == 0:
            continue
        ohe = one_hot(shingle, vocab)

        fingerprint = get_fingerprint(arr, ohe)
        lsh.hash(fingerprint, id)

    lsh_pairs = list(lsh.get_candidate_pairs())

    empty_string_pairs = get_empty_string_pairs(dataset)
    print(f"LSH: \t{len(lsh_pairs)}")
    print(f"Hashed numbers \t{len(cand_pairs)}")
    print(f"By empty string\t{len(empty_string_pairs)}")

    minhash_numbers = []
    if is_X2:
        minhash_numbers = minhash_important_numbers(dataset)
        print(f"By important number\t{len(minhash_numbers)}")

    if is_X2:
        res = set(lsh_pairs + all_pairs + minhash_numbers +
                  empty_string_pairs)
    else:
        res = set(lsh_pairs + cand_pairs + all_pairs + empty_string_pairs)

    del lsh_pairs, all_pairs, cand_pairs, shingles

    print(f"Result set: \t{len(res)}\n")
    return list(res)


def recall_misses(true, prediction, get_false: bool):
    fp = set(true).difference(set(prediction)) if get_false else set()
    return (len(set(true).intersection(set(prediction)))) / len(true), fp


def blocking_step2(df_path):
    ds = Preprocessor.build(df_path)
    dataset = ds.preprocess()  # TODO: split into #hyperthreads jobs

    k = 3  # ~5 for small docs (emails), 9 - 10 for large docs(papers)
    shingles = []
    for _, row in dataset.iterrows():
        data = row["title"]
        shingles.append((row["id"], k_shingles(data, k)))

    all_shingles = {item for set_ in shingles for item in set_[1]}

    vocab = {}
    for i, shingle in enumerate(list(all_shingles)):
        vocab[shingle] = i

    del all_shingles

    buckets = 15
    lsh = LSH(buckets)

    hash_function_count = 150
    arr = gen_minhash(vocab, hash_function_count)
    for id, shingle in shingles:
        if len(shingle) == 0:
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
    lsh_cp = lsh.get_candidate_pairs()

    return list(lsh_cp.union(rp_cp))


def recall(true, prediction):
    return (len(set(true).intersection(set(prediction)))) / len(true)


def main2():
    X1_candidate_pairs = blocking_step("X1.csv", False)
    X2_candidate_pairs = blocking_step("X2.csv", True)

    print(f"X1_candidate_pairs: {len(X1_candidate_pairs)}")
    print(f"X2_candidate_pairs: {len(X2_candidate_pairs)}")
    #  pdb.set_trace()
    r1 = recall(
        pd.read_csv("Y1.csv").to_records(
            index=False).tolist(), X1_candidate_pairs
    )
    r2 = recall(
        pd.read_csv("Y2.csv").to_records(
            index=False).tolist(), X2_candidate_pairs
    )
    r = (r1 + r2) / 2
    print(f"RECALL FOR X1 \t\t{r1:.3f}")
    print(f"RECALL FOR X2 \t\t{r2:.3f}")
    print(f"RECALL OVERALL  \t{r:.3f}")
    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)


def main1():
    X1_candidate_pairs = blocking_step("X1.csv", False)
    print(f'X1_candidate_pairs: {len(X1_candidate_pairs)}')

    r1, fp1 = recall(pd.read_csv('Y1.csv').to_records(
        index=False).tolist(), X1_candidate_pairs, False)
    print(f"RECALL FOR X1 \t\t{r1:.3f}")
    save_X1(X1_candidate_pairs)

    # del X1_candidate_pairs
    # gc.collect()

    X2_candidate_pairs = blocking_step("X2.csv", True)
    print(f'X2_candidate_pairs: {len(X2_candidate_pairs)}')

    r2, fp2 = recall_misses(pd.read_csv('Y2.csv').to_records(
        index=False).tolist(), X2_candidate_pairs, True)
    print(f"RECALL FOR X2 \t\t{r2:.3f}")
    save_X2(X2_candidate_pairs)

    r = (r1 + r2) / 2
    print(f"RECALL OVERALL  \t{r:.3f}")

    print("Dataset 1 misses:")
    y1 = pd.read_csv('X1.csv')
    for f in fp1:
        print(list(y1["title"][y1["id"] == f[0]]))
        print(list(y1["title"][y1["id"] == f[1]]))
        print("\n")

    print("Dataset 2 misses:")
    y1 = pd.read_csv('X2.csv')
    for f in fp2:
        print(list(y1["name"][y1["id"] == f[0]]))
        print(list(y1["name"][y1["id"] == f[1]]))
        print("\n")


if __name__ == '__main__':
    main2()
