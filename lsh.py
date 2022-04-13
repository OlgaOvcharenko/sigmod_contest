from itertools import combinations
import pandas as pd
#  from tqdm import tqdm
import numpy as np
import time
import logging
import sys
import pdb


#  X1_df = pd.read_csv("X1_large.csv")
#  X2_df = pd.read_csv("X2_large.csv")
X1_df = pd.read_csv("X1.csv")
X2_df = pd.read_csv("X2.csv")
#  l = logging.Logger("")
#  h = logging.StreamHandler()
#  f = logging.Formatter(fmt="[{filename}:{lineno}] {msg}", style="{")
#  h.setFormatter(f)
#  l.addHandler(h)


# %%
def one_hot(shingles: set, vocab: dict) -> np.ndarray:
    vec = np.zeros(len(vocab))
    for shingle in shingles:
        idx = vocab[shingle]
        vec[idx] = 1
    return vec


# %%
def k_shingles(data: str, k: int) -> set:
    shingles = []

    for i in range(len(data) - k + 1):
        shingles.append(data[i:i+k])

    return set(shingles)


# %%
def gen_minhash(vocab: dict, count: int) -> np.ndarray:
    arr = np.zeros((count, len(vocab.keys())))

    for i in range(count):
        permutation = np.random.permutation(len(vocab)) + 1
        arr[i, :] = permutation.copy()

    return arr.astype(int)


# %%
def get_fingerprint(minhash, ohe):
    idx = np.nonzero(ohe)[0].tolist()
    shingles = minhash[:, idx]
    fingerprint = np.min(shingles, axis=1)
    return fingerprint


# %%
class LSH:

    def __init__(self, b):
        self.bucket_count = b
        self.buckets = []
        [self.buckets.append({}) for _ in range(b)]

    def gen_bands(self, fingerprint):
        fingerprint_size = len(fingerprint)
        assert fingerprint_size % self.bucket_count == 0
        band_size = int(fingerprint_size / self.bucket_count)

        # break fingerprint into bands
        bands = []
        for i in range(0, fingerprint_size, band_size):
            bands.append(fingerprint[i:i+band_size])
        return np.stack(bands)

    def hash(self, fingerprint, id):
        bands = self.gen_bands(fingerprint).astype(str)
        for i, band in enumerate(bands):
            band = ','.join(band)
            if band not in self.buckets[i].keys():
                self.buckets[i][band] = []
            self.buckets[i][band].append(id)

    def get_candidate_pairs(self):
        candidates = []
        for bucket_band in self.buckets:
            keys = bucket_band.keys()
            for bucket in keys:
                hashed_values = bucket_band[bucket]
                if len(hashed_values) > 1:
                    #  pdb.set_trace()
                    candidates.extend(combinations(hashed_values, 2))

        return set(candidates)


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
        X1_candidate_pairs.extend(
            [(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend(
            [(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    # make sure to have the pairs in the first dataset first
    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs
    output_df = pd.DataFrame(all_cand_pairs, columns=[
                             "left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


def blocking_step(X):
    # TODO: split into #hyperthreads jobs

    # %%
    k = 3  # ~5 for small docs (emails), 9 - 10 for large docs(papers)
    shingles = []
    for row in X:
        id, _, data = row.partition(' ')
        shingles.append((id, k_shingles(data, k)))

    # %%
    all_shingles = {item for set_ in shingles for item in set_[1]}

    vocab = {}
    for i, shingle in enumerate(list(all_shingles)):
        vocab[shingle] = i

    del all_shingles

    # %%
    # LSH
    buckets = 15
    lsh = LSH(buckets)

    # %%
    #one_hot = one_hot_encoding(vocab)
    hash_function_count = 150
    arr = gen_minhash(vocab, hash_function_count)
    for id, shingle in shingles:
        ohe = one_hot(shingle, vocab)

        fingerprint = get_fingerprint(arr, ohe)
        lsh.hash(fingerprint, id)

    # %%
    # candidate pairs
    return list(lsh.get_candidate_pairs())


def duplicate_with_new_id(X, times_more: int):
    X_new = X
    for _ in range(0, times_more):
        X_new = pd.concat([X_new, X])

    return X_new.reset_index()


if __name__ == '__main__':

    # %%
    #  start = time.time()
    X1_df['id_title'] = X1_df['id'].astype(str) + ' ' + X1_df['title']
    X1 = X1_df['id_title'].tolist()
    X2_df['id_title'] = X2_df['id'].astype(str) + ' ' + X2_df['name']
    X2 = X2_df['id_title'].tolist()  # TODO: include other attributes!
    #  print("X1 size " + str(len(X1_df)))
    #  print("X2 size " + str(len(X2_df)))
    X1_candidate_pairs = blocking_step(X1)
    #  X2_candidate_pairs = blocking_step(X2)
    X2_candidate_pairs = []
    #
    #  # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    #  end = time.time()
    #  print(f"Runtime of the program is {end - start}.")
