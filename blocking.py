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
        for src_id in src_ids:
            if src_id == id:
                continue
            if flags[src_id] != id:
                common_token_counter[src_id] = 0
                flags[src_id] = id
            common_token_counter[src_id] = common_token_counter[src_id] + 1
            counter = common_token_counter[src_id]

            if counter > 0:
                pair = (id, src_id) if id < src_id else (src_id, id)
                if pair not in cp:
                    similarity = counter / \
                        sqrt(source_freq[src_id] * len(shingle_set))

                    if similarity >= min_sim:
                        pair = (id, src_id) if id < src_id else (src_id, id)
                        cp.add(pair)

        #  if not local_cp:
        #      continue
        #
        #  k_sim = PriorityQueue()
        #  min_sim = 0
        #  if len(local_cp) > max_local_cp:
        #      max_local_cp = len(local_cp)
        #
        #  for candidate in local_cp:
        #      j += 1
        #      common = common_token_counter[candidate]
        #
        #      # cosine sim
        #      similarity = common / \
        #          sqrt(source_freq[candidate] * len(shingle_set))

        #      if min_sim < similarity:
        #          k_sim.put((-similarity, similarity))
        #          if (K < k_sim.qsize()):
        #              min_sim = k_sim.get()[1]
        #
        #  for candidate in local_cp:
        #      common = common_token_counter[candidate]
        #
        #      # cosine sim
        #      similarity = common / \
        #          sqrt(source_freq[candidate] * len(shingle_set))

            #  if similarity >= min_sim:
            #      pair = (id, candidate) if id < candidate else (candidate, id)
            #      cp.add(pair)
    #  print(f"maxlen: {max_local_cp}\ni: {i}\tj: {j}")
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
