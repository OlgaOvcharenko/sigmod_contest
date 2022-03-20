import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sparse_dot_topn import awesome_cossim_topn

SIMILARITY = "similarity"
IDX = "idx"
IDY = "idy"


def get_tf_id_vector(x, ngram_range) -> csr_matrix:
    tf_id_vector = TfidfVectorizer(ngram_range=ngram_range)
    return tf_id_vector.fit_transform(x)


def generate_vector_representation(x, ngram_range: tuple = (1, 4)) -> csr_matrix:
    return get_tf_id_vector(x, ngram_range)


def cosine_similarity(
    a: csr_matrix, b: csr_matrix, ntop: int = 10, lower_bound: float = 0.80
):
    # https://github.com/ing-bank/sparse_dot_topn
    return awesome_cossim_topn(a, b, ntop=ntop, lower_bound=lower_bound)


def get_matches_df(sparse_matrix, name_vector, top=None) -> pd.DataFrame:
    sparse_matrix_non_zeros = sparse_matrix.nonzero()

    _rows = sparse_matrix_non_zeros[0]
    _cols = sparse_matrix_non_zeros[1]

    df = pd.DataFrame(
        {
            IDX: name_vector[_rows].values[:top],
            IDY: name_vector[_cols].values[:top],
            SIMILARITY: sparse_matrix.data[:top],
        }
    )

    return df[df[IDX] != df[IDY]]


def block_with_attr(X, attr):  # replace with your logic.
    """
    This function performs blocking using attr
    :param X: dataframe
    :param attr: attribute used for blocking
    :return: candidate set of tuple pairs
    """

    # build index from patterns to tuples
    vector_representation = generate_vector_representation(X[attr].values.tolist())
    similarity_matrix = cosine_similarity(
        a=vector_representation, b=vector_representation.T, ntop=10, lower_bound=0.8
    )
    matched_pair = get_matches_df(similarity_matrix, X["id"])
    candidate_pairs_real_ids = list(zip(matched_pair[IDX], matched_pair[IDY]))
    return candidate_pairs_real_ids


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

    all_cand_pairs = (
        X1_candidate_pairs + X2_candidate_pairs
    )  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(
        all_cand_pairs, columns=["left_instance_id", "right_instance_id"]
    )
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


# read the datasets
X1 = pd.read_csv("X1.csv")
X2 = pd.read_csv("X2.csv")

# perform blocking
X1_candidate_pairs = block_with_attr(X1, attr="title")
X2_candidate_pairs = block_with_attr(X2, attr="name")

# save results
save_output(X1_candidate_pairs, X2_candidate_pairs)
