import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sparse_dot_topn import awesome_cossim_topn
from tqdm import tqdm

SIMILARITY = "similarity"
IDX = "lid"
IDY = "rid"


def get_tf_id_vector(x, ngram_range) -> csr_matrix:
    tf_id_vector = TfidfVectorizer(ngram_range=ngram_range)
    return tf_id_vector.fit_transform(x)


def generate_vector_representation(x, ngram_range: tuple = (1, 4)) -> csr_matrix:
    return get_tf_id_vector(x, ngram_range)


def cosine_similarity(
    a: csr_matrix, b: csr_matrix, ntop: int = 10, lower_bound: float = 0.80
) -> csr_matrix:
    # https://github.com/ing-bank/sparse_dot_topn
    return awesome_cossim_topn(a, b, ntop=ntop, lower_bound=lower_bound)


def generate_similarity_df(
    sparse_matrix: csr_matrix, match_over: pd.Series, top=None
) -> pd.DataFrame:
    sparse_matrix_non_zeros = sparse_matrix.nonzero()

    _rows = sparse_matrix_non_zeros[0]
    _cols = sparse_matrix_non_zeros[1]

    df = pd.DataFrame(
        {
            IDX: match_over[_rows].values[:top],
            IDY: match_over[_cols].values[:top],
            SIMILARITY: sparse_matrix.data[:top],
        }
    )

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[IDX] != df[IDY]]


def get_matched_pair(
    vector_representation: csr_matrix,
    similarity_over: pd.Series,
    top_matches: int = 20,
    confidence_score: float = 0.80,
):
    similarity_matrix = cosine_similarity(
        a=vector_representation,
        b=vector_representation.T,
        ntop=top_matches,
        lower_bound=confidence_score,
    )
    matched_pair_id = generate_similarity_df(similarity_matrix, similarity_over)
    matched_pair_id = remove_duplicates(df=matched_pair_id)
    matched_pair_id.sort_values(by=[SIMILARITY], inplace=True, ascending=False)
    return matched_pair_id


def block_with_attr(X, attr):  # replace with your logic.
    """
    This function performs blocking using attr
    :param X: dataframe
    :param attr: attribute used for blocking
    :return: candidate set of tuple pairs
    """

    # build index from patterns to tuples
    vector_representation = generate_vector_representation(X[attr].values.tolist())
    matched_pair_id = get_matched_pair(
        vector_representation,
        similarity_over=X["id"],
        top_matches=15,
        confidence_score=0.80,
    )
    # UNCOMMENT TO DEBUG
    # matched_pair_str = get_matched_pair(vector_representation, similarity_over=X[attr], top_matches=15,
    #                                        confidence_score=0.80)

    candidate_pairs = list(zip(matched_pair_id[IDX], matched_pair_id[IDY]))
    candidate_pairs_real_ids = list()

    for it in tqdm(candidate_pairs):
        real_id1, real_id2 = it

        # NOTE: This is to make sure in the final output.csv, for a pair id1 and id2 (assume id1<id2),
        # we only include (id1,id2) but not (id2, id1)
        if real_id1 < real_id2:
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))

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
