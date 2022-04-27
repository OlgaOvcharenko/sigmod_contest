import pandas as pd
import pdb

from ann_search import LSHRPQuery
from feature_embeddings import TFIDFHashedEmbeddings
from preprocessing import Preprocessor
from lsh import *


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


def blocking_step(df_path):
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


if __name__ == "__main__":

    # %%
    #  X1_df['id_title'] = X1_df['id'].astype(str) + ' ' + X1_df['title']
    #  X1 = X1_df['id_title'].tolist()
    #  X2_df['id_title'] = X2_df['id'].astype(str) + ' ' + X2_df['name']
    #  X2 = X2_df['id_title'].tolist()  # TODO: include other attributes!

    X1_candidate_pairs = blocking_step("X1.csv")
    X2_candidate_pairs = blocking_step("X2.csv")

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
