import pandas as pd
import pdb

import cProfile
import pstats
import io
from pstats import SortKey

from queue import PriorityQueue
from ann_search import LSHRPQuery
from feature_embeddings import TFIDFHashedEmbeddings
from preprocessing import Preprocessor
from lsh import *
from math import sqrt


#  X1_df = pd.read_csv("X1_large.csv")
#  X2_df = pd.read_csv("X2_large.csv")
X1_df = pd.read_csv("X1.csv")
X2_df = pd.read_csv("X2.csv")
#  l = logging.Logger("")
#  h = logging.StreamHandler()
#  f = logging.Formatter(fmt="[{filename}:{lineno}] {msg}", style="{")
#  h.setFormatter(f)
#  l.addHandler(h)


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


def blocking_step(df_path, k=3, buckets=15, hash_function_count=150,
                  is_X2=False):
    ds = Preprocessor.build(df_path)
    dataset = ds.preprocess()

    shingles = []
    id_to_idx = {}
    for i, row in dataset.iterrows():
        data = row["title"]
        shingles.append((row["id"], k_shingles(data, k)))
        id_to_idx[row["id"]] = i

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

    if not is_X2:
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
        lsh_cp = lsh_cp.union(rp_cp)
        return list(lsh_cp)

    minhash_buckets = lsh.get_buckets()
    candidates = set()
    buckets = 0
    K_limit = 25
    K = 10

    #  pr = cProfile.Profile()
    #  pr.enable()

    for bucket_band in minhash_buckets:
        keys = bucket_band.keys()
        for bucket in keys:
            hashed_values = bucket_band[bucket]
            if len(hashed_values) > 1:  # and len(hashed_values) < 100:
                # prune candidates with knnjoin
                if len(hashed_values) > K_limit:
                    index = {}
                    shingles = {}
                    source_freq = {}
                    common_token_counter = {}
                    flags = {}
                    min_sim = 0.00
                    for id in hashed_values:
                        data = dataset.at[id_to_idx[id], 'title']
                        flags[id] = -1
                        shngl = data.split()
                        shingles[id] = shngl
                        source_freq[id] = len(shngl)
                        for s in shngl:
                            if s in index:
                                index[s].append(id)
                            else:
                                index[s] = [id]
                    for id, shingle_set in shingles.items():
                        shingle_set_len = len(shingle_set)
                        local_cp = []
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
                            #  local_cp.append(src_id)
                            if flags[src_id] != id:
                                common_token_counter[src_id] = 0
                                flags[src_id] = id
                            common_token_counter[src_id] = common_token_counter[src_id] + 1

                        if not local_cp:
                            continue

                        k_sim = PriorityQueue()

                        for candidate in local_cp:
                            common = common_token_counter[candidate]

                            # cosine sim
                            similarity = common / \
                                sqrt(source_freq[candidate] * shingle_set_len)

                            if min_sim < similarity:
                                k_sim.put((-similarity, similarity))
                                if (K < k_sim.qsize()):
                                    min_sim = k_sim.get()[1]

                        for candidate in local_cp:
                            common = common_token_counter[candidate]

                            # cosine sim
                            similarity = common / \
                                sqrt(source_freq[candidate] * shingle_set_len)

                            if similarity >= min_sim:
                                pair = (id, candidate) if id < candidate else (
                                    candidate, id)
                                candidates.add(pair)
                else:
                    for ix1, id1 in enumerate(hashed_values):
                        for _, id2 in enumerate(hashed_values[ix1 + 1:]):
                            pair = (id1, id2) if id1 < id2 else (id2, id1)
                            candidates.add(pair)
                #  pdb.set_trace()
                #  candidates.extend(combinations(hashed_values, 2))
    lsh_cp = list(candidates)
    #  pr.disable()
    #  s = io.StringIO()
    #
    #  sortby = SortKey.CUMULATIVE
    #  ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #  ps.print_stats(25)
    #  print(s.getvalue())

    return lsh_cp


def blocking_groupby(df_path, k=3, buckets=15, hash_function_count=150):
    ds = Preprocessor.build(df_path)
    dataset = ds.preprocess()
    dataset = dataset.groupby('brand')
    all_cp = []

    for _, group in dataset:
        shingles = []
        for _, row in group.iterrows():
            data = row["title"]
            shingles.append((row["id"], k_shingles(data, k)))

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

        emd = feature_embeddings.generate(
            group["title"].tolist(), n_features=50)

        ann_search_index = LSHRPQuery()
        nn, distances = ann_search_index.load_and_query(emd, n_bits=32)
        rp_cp, _ = ann_search_index.generate_candidate_pairs(
            nn, distances, group["id"].to_list()
        )
        del nn, distances, ann_search_index
        lsh_cp = lsh.get_candidate_pairs()
        lsh_cp = lsh_cp.union(rp_cp)

        all_cp.extend(list(lsh_cp))

    return all_cp


def recall(true, prediction):
    return (len(set(true).intersection(set(prediction)))) / len(true)


if __name__ == "__main__":
    X1_candidate_pairs = blocking_step("X1.csv")
    X2_candidate_pairs = blocking_step("X2.csv", k=3, buckets=25,
                                       hash_function_count=200)

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
