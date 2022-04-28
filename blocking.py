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
from save_to_file import save_output, save_X2, save_X1

def hash_by_number(name: str, is_X2: bool):
    pattern = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'

    all_numbers = set(filter(lambda num1: num1 > 99,
                             map(lambda num: int(num.replace(".", "").replace(",", "")), re.findall(pattern, name))))

    # all_numbers = set(map(lambda num: int(num.replace(".", "").replace(",", "")), re.findall(pattern, name)))

    def cantor_pairing(a, b):
        return (a + b) * ((a + b) / 2) * b

    # if len(all_numbers) == 2:
    #     all_pairs_hashed = [cantor_pairing(pair[0], pair[1]) if pair[0] < pair[1] else
    #                         cantor_pairing(pair[1], pair[0])
    #                         for pair in itertools.combinations(all_numbers, 2)]
    #     return all_pairs_hashed

    if len(all_numbers) > 6:
        hashed_triples = []
        for pair in itertools.combinations(all_numbers, 6):
            pair_sorted = sorted(pair)
            hashed_triples.append(
                cantor_pairing(int(cantor_pairing(
                int(cantor_pairing(int(cantor_pairing(
                int(cantor_pairing(pair_sorted[0], pair_sorted[1])), pair_sorted[2])), pair_sorted[3])), pair_sorted[4])), pair_sorted[5]))
        return hashed_triples
    elif is_X2 and 5 > len(all_numbers) > 3:
        hashed_triples = []
        for pair in itertools.combinations(all_numbers, 3):
            pair_sorted = sorted(pair)
            hashed_triples.append(cantor_pairing(int(cantor_pairing(pair_sorted[0], pair_sorted[1])), pair_sorted[2]))
        return hashed_triples
    elif is_X2 and len(all_numbers) > 4:
        hashed_triples = []
        for pair in itertools.combinations(all_numbers, 4):
            pair_sorted = sorted(pair)
            hashed_triples.append(cantor_pairing(int(cantor_pairing(
                int(cantor_pairing(pair_sorted[0], pair_sorted[1])), pair_sorted[2])), pair_sorted[3]))
        return hashed_triples
    else:
        return []


def blocking_step(df_path, is_X2:bool):
    all_pairs_hashed = defaultdict(list)

    ds = Preprocessor.build(df_path)
    dataset = ds.preprocess()   # TODO: split into #hyperthreads jobs

    k = 3  # ~5 for small docs (emails), 9 - 10 for large docs(papers)
    shingles = []

    buckets_brands = {"toshiba": [], "chuwi": [], "mediacom": [], "google": [], "sandisk": [],
                      "vero": [], "msi": [], "xiaomi": [], "microsoft": [], "apple": [], "razer": [],
                      "lg": [], "dell": [], "fujitsu": [], "huawei": [], "lenovo": [], "acer": [],
                      "asus": [], "hp": [], "samsung": [], "kingston": [], "pami": [],
                      "lenovo thinkpad": []}

    i = 0
    for _, row in dataset.iterrows():
        original_str = row['title']
        str = preprocessing.normalize_string(row['title'], is_X2)

        # if is_X2:
        #     [all_pairs_hashed[k].append(row['id']) for k in hash_by_number(original_str, is_X2)]

        # dataset['short_id'][i] = short_id
        shingles.append((row['id'], k_shingles(str, k)))

        # for d in buckets_brands.keys():
        #     if d in str:
        #         buckets_brands[d].append(i)

        i += 1

    all_pairs = []
    # TODO joblib
    # FIXME check ids
    # or no sort and get > 0.5
    # for brand, ids in buckets_brands.items():
    #     def intersection(name1, name2):
    #         s1 = set(name1.lower().split())
    #         s2 = set(name2.lower().split())
    #         return len(s1.intersection(s2)) / max(len(s1), len(s2))
    #
    #     pairs = [(dataset['id'][a], dataset['id'][b]) if dataset['id'][a] < dataset['id'][b]
    #                  else (dataset['id'][b], dataset['id'][a])
    #                  for idx, a in enumerate(ids) for b in ids[idx + 1:] if dataset['id'][b] != dataset['id'][a]]
    #
    #     all_pairs.extend(pairs)

    # hash by numbers in string
    cand_pairs = []
    # for hashed_key in sorted(all_pairs_hashed, key=lambda x: len(all_pairs_hashed[x]), reverse=True):
    #     cand_pairs.extend([(pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
    #      for pair in itertools.combinations(all_pairs_hashed[hashed_key], 2) if len(all_pairs_hashed[hashed_key]) > 1])

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

    # baseline_pairs = baseline.block_with_attr(dataset, 'title')
    # TODO random pairs -> numbers instead

    print(f"LSH: \t{len(lsh_pairs)}")
    print(f"Hashed numbers \t{len(cand_pairs)}")
    print(f"By brand \t{len(all_pairs)}")

    res = set(lsh_pairs + cand_pairs + all_pairs)

    del lsh_pairs, all_pairs, cand_pairs, shingles

    print(f"Result set: \t{len(res)}")
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
        # data = row["title"]
        # original_str = row['title']
        data = preprocessing.normalize_string(row['title'], True)
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


def main1():
    X1_candidate_pairs = blocking_step("X1_large.csv", False)
    print(f'X1_candidate_pairs: {len(X1_candidate_pairs)}')

    r1, fp1 = recall_misses(pd.read_csv('Y1.csv').to_records(
        index=False).tolist(), X1_candidate_pairs, True)
    print(f"RECALL FOR X1 \t\t{r1:.3f}")
    save_X1(X1_candidate_pairs)

    del X1_candidate_pairs
    gc.collect()

    X2_candidate_pairs = blocking_step("X2_large.csv", True)
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
