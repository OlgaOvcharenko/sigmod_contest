import itertools
from collections import defaultdict

import pandas as pd
import pdb
import cProfile
import pstats
import io
from pstats import SortKey
import re

from math import sqrt
from queue import PriorityQueue
from preprocessing import Preprocessor
from lsh import *


#  X1_df = pd.read_csv("X1_large.csv")
#  X2_df = pd.read_csv("X2_large.csv")
X1_df = pd.read_csv("X1.csv")
X2_df = pd.read_csv("X2.csv")
isX1 = True
#  l = logging.Logger("")
#  h = logging.StreamHandler()
#  f = logging.Formatter(fmt="[{filename}:{lineno}] {msg}", style="{")
#  h.setFormatter(f)
#  l.addHandler(h)


def hash_by_number(name: str):
    pattern = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'

    all_numbers = set(filter(lambda num1: num1 > 99,
                             map(lambda num: int(num.replace(".", "").replace(",", "")), re.findall(pattern, name))))

    # all_numbers = set(map(lambda num: int(num.replace(".", "").replace(",", "")), re.findall(pattern, name)))

    def cantor_pairing(a, b):
        return (a + b) * ((a + b) / 2) * b

    all_pairs_hashed = []
    # if len(all_numbers) == 2:
    #     all_pairs_hashed = [int(cantor_pairing(pair[0], pair[1])) if pair[0] < pair[1] else
    #                         int(cantor_pairing(pair[1], pair[0]))
    #                         for pair in itertools.combinations(all_numbers, 2)]
    #
    # el
    if len(all_numbers) > 6:
        for pair in itertools.combinations(all_numbers, 6):
            pair_sorted = sorted(pair)
            all_pairs_hashed.append(int(
                cantor_pairing(cantor_pairing(
                    cantor_pairing(cantor_pairing(
                        cantor_pairing(pair_sorted[0], pair_sorted[1]), pair_sorted[2]), pair_sorted[3]),
                    pair_sorted[4]), pair_sorted[5])))

    elif len(all_numbers) > 4:
        for pair in itertools.combinations(all_numbers, 4):
            pair_sorted = sorted(pair)
            all_pairs_hashed.append(int(cantor_pairing(cantor_pairing(
                cantor_pairing(pair_sorted[0], pair_sorted[1]), pair_sorted[2]), pair_sorted[3])))

    elif len(all_numbers) > 3:
        for pair in itertools.combinations(all_numbers, 3):
            pair_sorted = sorted(pair)
            all_pairs_hashed.append(int(cantor_pairing(cantor_pairing(pair_sorted[0], pair_sorted[1]), pair_sorted[2])))

    return all_pairs_hashed


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


def blocking_step(df_path):
    ds = Preprocessor.build(df_path)
    dataset = ds.preprocess()   # TODO: split into #hyperthreads jobs

    k = 3  # ~5 for small docs (emails), 9 - 10 for large docs(papers)
    if isX1:
        K = 10  # kNN-Join K
    else:
        K = 5

    if isX1:
        min_sim = 0.8
    else:
        min_sim = 0.6

    shingles = {}
    source_freq = {}
    index = {}
    common_token_counter = {}
    flags = {}

    id_str = {}

    for _, row in dataset.iterrows():
        data = row['title']
        id = row['id']
        flags[id] = -1
        # TODO:
        #  shngl = k_shingles(data, 9)
        shngl = data.split()
        #  pdb.set_trace()
        #  shngl = [' '.join(shngl[i: i + 2]) for i in range(0, len(shngl), 2)]
        shingles[id] = shngl
        source_freq[id] = len(shngl)
        id_str[id] = data
        for s in shngl:
            if s in index:
                index[s].append(id)
            else:
                index[s] = [id]

    cp = set()
    i = 0
    j = 0
    for id, shingle_set in shingles.items():
        local_cp = []
        i += 1
        # loop over all shingles for a given row
        src_ids = []
        for s in shingle_set:
            src_ids.extend(index[s])
            if not src_ids:
                continue

        # loop over all ids that are in the bucket of one of the shingles

        all_hashed = defaultdict(list)
        [all_hashed[paired_num].append(id) for paired_num in hash_by_number(id_str[id])]

        for src_id in src_ids:
            if src_id == id:
                continue
            if flags[src_id] != id:
                common_token_counter[src_id] = 0
                flags[src_id] = id
            # FIXME is really necessary
            common_token_counter[src_id] = common_token_counter[src_id] + 1
            counter = common_token_counter[src_id]

            if counter > K:
                # pair = (id, src_id) if id < src_id else (src_id, id)
                # if pair not in cp:
                #     similarity = counter / \
                #         sqrt(source_freq[src_id] * len(shingle_set))
                #
                #     if similarity >= min_sim:
                #         pair = (id, src_id) if id < src_id else (src_id, id)
                #         cp.add(pair)
                [all_hashed[paired_num].append(id) for paired_num in hash_by_number(id_str[src_id])]

        for hashed_key in sorted(all_hashed, key=lambda x: len(all_hashed[x]), reverse=True):
            [cp.add((pair[0], pair[1])) if pair[0] < pair[1] else cp.add((pair[1], pair[0]))
             for pair in itertools.combinations(all_hashed[hashed_key], 2) if len(all_hashed[hashed_key]) > 1]

    return list(cp)


def recall(true, prediction):
    return (len(set(true).intersection(set(prediction)))) / len(true)


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    X1_candidate_pairs = blocking_step("X1.csv")
    isX1 = False
    X2_candidate_pairs = blocking_step("X2.csv")
    pr.disable()
    s = io.StringIO()

    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(10)
    print(s.getvalue())

    print(f'X1_candidate_pairs: {len(X1_candidate_pairs)}')
    print(f'X2_candidate_pairs: {len(X2_candidate_pairs)}')
    #  pdb.set_trace()
    r1 = recall(pd.read_csv('Y1.csv').to_records(
        index=False).tolist(), X1_candidate_pairs)
    r2 = recall(pd.read_csv('Y2.csv').to_records(
        index=False).tolist(), X2_candidate_pairs)
    r = (r1 + r2) / 2
    print(f"RECALL FOR X1 \t\t{r1:.3f}")
    print(f"RECALL FOR X2 \t\t{r2:.3f}")
    print(f"RECALL OVERALL  \t{r:.3f}")
    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
