import pandas as pd
import pdb
from lsh import *


#  X1_df = pd.read_csv("X1_large.csv")
#  X2_df = pd.read_csv("X2_large.csv")

#  X1_df = pd.read_csv("x1toy.csv")
X1_df = pd.read_csv("X1.csv")
#  X2_df = pd.read_csv("X2.csv")
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
    buckets = 10
    lsh = LSH(buckets)

    # %%
    #one_hot = one_hot_encoding(vocab)
    hash_function_count = 20
    arr = gen_minhash(vocab, hash_function_count)
    for id, shingle in shingles:
        ohe = one_hot(shingle, vocab)

        fingerprint = get_fingerprint(arr, ohe)
        lsh.hash(fingerprint, id)

    # %%
    # candidate pairs
    return list(lsh.get_candidate_pairs())


if __name__ == '__main__':

    # %%
    X1_df['id_title'] = X1_df['id'].astype(str) + ' ' + X1_df['title']
    X1 = X1_df['id_title'].tolist()
    #  X2_df['id_title'] = X2_df['id'].astype(str) + ' ' + X2_df['name']
    #  X2 = X2_df['id_title'].tolist()  # TODO: include other attributes!

    X1_candidate_pairs = blocking_step(X1)
    print(f'X1_candidate_pairs: {len(X1_candidate_pairs)}')
    #  print(f'{X1_candidate_pairs}')
    #  X2_candidate_pairs = blocking_step(X2)
    X2_candidate_pairs = []
    #  pdb.set_trace()

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
