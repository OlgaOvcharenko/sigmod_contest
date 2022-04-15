import pandas as pd
import pdb
from dataset import Preprocessor
from partitioning import Partitioner


#  X1_df = pd.read_csv("x1toy.csv")
#  X1_df = pd.read_csv("X1_large.csv")
#  X2_df = pd.read_csv("X2_large.csv")
#  X1_df = pd.read_csv("X1.csv")
#  X2_df = pd.read_csv("X2.csv")


def save_output(X1_candidate_pairs,
                X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend(
            [(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend(
            [(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))
    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs
    output_df = pd.DataFrame(all_cand_pairs, columns=[
                             "left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)


def blocking_step(df_path):
    ds = Preprocessor.build(df_path)
    data = ds.preprocess()

    pt = Partitioner.build(df_path, data)
    pt.blocking_step()

    return pt.get_candidate_pairs()


if __name__ == '__main__':
    #  X1_candidate_pairs = blocking_step("X1_large.csv")
    #  X2_candidate_pairs = blocking_step("X2_large.csv")
    X1_candidate_pairs = blocking_step("X1.csv")
    X2_candidate_pairs = blocking_step("X2.csv")
    #  X1_candidate_pairs = []
    #  X2_candidate_pairs = []
    print(f'X1_candidate_pairs: {len(X1_candidate_pairs)}')
    print(f'X2_candidate_pairs: {len(X2_candidate_pairs)}')
    #  pdb.set_trace()

    save_output(X1_candidate_pairs, X2_candidate_pairs)
