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

x1_matches = []
x2_matches = []


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


def block_with_attr(X, id, attr, is_X1:bool):
    '''
    This function performs blocking using attr
    :param X: dataframe
    :param attr: attribute used for blocking
    :return: candidate set of tuple pairs
    '''

    global start
    if time.time() - start > 1560:
        return

    # build index from patterns to tuples
    # pattern2id_1 = defaultdict(list)
    pattern2id_2 = defaultdict(list)
    for i in tqdm(range(X.shape[0])):
        attr_i = str(X[attr][i])
        # pattern_1 = attr_i.lower()  # use the whole attribute as the pattern
        # pattern2id_1[pattern_1].append(i)

        pattern_2 = re.findall("\w+\s\w+\d+", attr_i)  # look for patterns like "thinkpad x1"
        if len(pattern_2) == 0:
            continue
        # pattern_2 = list(sorted(pattern_2))
        pattern_2 = [str(it).lower() for it in pattern_2]
        [pattern2id_2[p].append(i) for p in pattern_2]

    # add id pairs that share the same pattern to candidate set
    # candidate_pairs_1 = []
    # candidate_pairs_1_extend = candidate_pairs_1.extend
    # # FIXME
    # pattern2id_1 = dict(filter(lambda elem: len(elem[1]) > 1, pattern2id_1.items()))
    # for pattern in tqdm(pattern2id_1):
    #     ids = list(pattern2id_1[pattern])
    #     if len(ids) > 1:
    #         candidate_pairs_1_extend((a, b) for idx, a in enumerate(ids) for b in ids[idx + 1:])

    # add id pairs that share the same pattern to candidate set
    candidate_pairs_2 = []
    candidate_pairs_2_extend = candidate_pairs_2.extend
    for pattern in tqdm(pattern2id_2):
        ids = list(pattern2id_2[pattern])
        if len(ids) < 1000:  # skip patterns that are too common
            candidate_pairs_2_extend((a, b) for idx, a in enumerate(ids) for b in ids[idx + 1:] if a != b)

    # remove duplicate pairs and take union
    # candidate_pairs = np.vstack([np.unique(candidate_pairs_1, axis=0), np.unique(candidate_pairs_2, axis=0)]) if len(candidate_pairs_1) > 0 else np.unique(candidate_pairs_2, axis=0)
    candidate_pairs = np.unique(candidate_pairs_2, axis=0)

    # sort candidate pairs by jaccard similarity.
    # In case we have more than 1000000 pairs (or 2000000 pairs for the second dataset),
    # sort the candidate pairs to put more similar pairs first,
    # so that when we keep only the first 1000000 pairs we are keeping the most likely pairs
    jaccard_similarities = []
    candidate_pairs_real_ids = []
    for it in tqdm(candidate_pairs):
        id1, id2 = it

        # get real ids
        real_id1 = X['id'][id1]
        real_id2 = X['id'][id2]
        if real_id1 < real_id2: # NOTE: This is to make sure in the final output.csv, for a pair id1 and id2 (assume id1<id2), we only include (id1,id2) but not (id2, id1)
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))

        # compute jaccard similarity
        name1 = str(X[attr][id1])
        name2 = str(X[attr][id2])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))
    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]

    if is_X1:
        x1_matches.extend(candidate_pairs_real_ids)
    else:
        x2_matches.extend(candidate_pairs_real_ids)

    write_to_csv(candidate_pairs_real_ids, is_X1)

    return 


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

X1_blocks = naive_blocking(X1, 2)
X2_blocks = naive_blocking(X2, 2)

# perform blocking
num_cores = multiprocessing.cpu_count()

# FIXME hardcorded num jobs
pairs1 = Parallel(n_jobs=16, require='sharedmem')(delayed(block_with_attr)(i.reset_index(), id="id", attr="title", is_X1=True) for i in X1_blocks)
fill_output_file_with_0(True)
pairs2 = Parallel(n_jobs=16, require='sharedmem')(delayed(block_with_attr)(i.reset_index(), id="id", attr="name", is_X1=False) for i in X2_blocks)
fill_output_file_with_0(False)

# # check length of the file
# with open("output.csv", "r") as fp:
#     for (count, _) in enumerate(fp, 1):
#        pass
#
#     if count < 3000000:
#         with open("output.csv", "a") as fp1:
#             X_extended = [(0, 0)] * (3000000 - count)
#             csv_out = csv.writer(fp1)
#             csv_out.writerows(X_extended)
#         fp1.close()
#
#     if count > 3000000:
#         csv_in = csv.reader(fp)
#         data = list(csv_in)[:3000000]
#         with open("output.csv", "w") as fp2:
#             csv_out = csv.writer(fp2)
#             csv_out.writerows(data)
#         fp2.close()
# fp.close()

# X1_block_pairs = [pairs for pairs in pairs1 if pairs]
# X2_block_pairs = [pairs for pairs in pairs2 if pairs]
#
# X1_candidate_pairs = np.vstack(X1_block_pairs).tolist() if len(X1_block_pairs) > 0 else []
# X2_candidate_pairs = np.concatenate(X2_block_pairs, axis=0).tolist() if len(X1_block_pairs) > 0 else []
#
# # save results
# save_output(X1_candidate_pairs, X2_candidate_pairs)

end = time.time()
print(f"Runtime of the program is {end - start}")
get_recall()
