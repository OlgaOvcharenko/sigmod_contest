"""
Implementation inspired by:
https://www.pinecone.io/learn/locality-sensitive-hashing/
"""
from itertools import combinations

#  from tqdm import tqdm
import numpy as np
import pdb


class LSH:
    """
    Splits fingerprints into fixed amount of bands.
    A single collision in a band is enough to produce a candidate pair.
    Fingerprint size has to be multiple of band amount.
    :param `b`: Amount of bands/buckets.
    """

    def __init__(self, b):
        self.bucket_count = b
        self.buckets = []
        [self.buckets.append({}) for _ in range(b)]

    def gen_bands(self, fingerprint):
        """
        Splits a single fingerprint into `bucket_count` many bands of same
        size.
        """
        fingerprint_size = len(fingerprint)
        assert fingerprint_size % self.bucket_count == 0
        band_size = int(fingerprint_size / self.bucket_count)

        bands = []
        for i in range(0, fingerprint_size, band_size):
            bands.append(fingerprint[i: i + band_size])
        return np.stack(bands)

    def hash(self, fingerprint, id):
        """
        After generint bands for the fingerprint we 'hash' the
        band, in our case there is no additional hash function, we simply
        convert the band into a string representation to allow for easier
        handling.
        Band number is the same as bucket id.
        :param `fingerprint`: Fingerprint to hash.
        :param `id`: Id of row in input dataset.
        """
        bands = self.gen_bands(fingerprint).astype(str)
        for i, band in enumerate(bands):
            band = ",".join(band)
            if band not in self.buckets[i].keys():
                self.buckets[i][band] = []
            self.buckets[i][band].append(id)

    def get_candidate_pairs(self):
        """
        Returns all possible candidate pairs with the previously hashed
        fingerprints.
        For each bucket, if there's more than 1 id insides we have a collision,
        e.g., possible candidate.
        :returns: Set of all possible candidates.
        """
        candidates = set()
        for bucket_band in self.buckets:
            keys = bucket_band.keys()
            for bucket in keys:
                hashed_values = bucket_band[bucket]
                if len(hashed_values) > 1:  # and len(hashed_values) < 100:
                    for ix1, id1 in enumerate(hashed_values):
                        for _, id2 in enumerate(hashed_values[ix1 + 1:]):
                            pair = (id1, id2) if id1 < id2 else (id2, id1)
                            candidates.add(pair)
                    #  pdb.set_trace()
                    #  candidates.extend(combinations(hashed_values, 2))

        return candidates

    def get_buckets(self):
        return self.buckets


# %%
def one_hot(shingles: set, vocab: dict) -> np.ndarray:
    """
    One-hot encodes a shingle set for a given row of the data set.
    For every shingle in the vocab we check if the shingle is present
    in the shingles for the row, if so -> 1, else 0.
    :param `shingles`: Input shingle set for a single row.
    :param `vocab`: Shingle set containing all unique shingles for all rows.
    :returns: One hot encoded vector.
    """
    vec = np.zeros(len(vocab))
    for shingle in shingles:
        idx = vocab[shingle]
        vec[idx] = 1
    return vec


# %%
def k_shingles(data: str, k: int) -> set:
    """
    Creates a k-shingle set for the input data.
    Basically moves a sliding window of size k over the input data and creates
    a set of all the shingles present.
    :param `data`: Input string (one per row).
    :param `k`: Fixed size of each shingle.
    :returns: Set of shingles.
    """
    shingles = []

    for i in range(len(data) - k + 1):
        shingles.append(data[i: i + k])

    return set(shingles)


# %%
def gen_minhash(vocab: dict, count: int) -> np.ndarray:
    """
    Generates `count` many minhash hashing functions/permutations.
    :param `vocab`: Shingle set containing all unique shingles for all rows.
    :param `count`: How many hash functions we want -> size of fingerprint.
    :returns: Array of `count` minhash functions.
    """
    arr = np.zeros((count, len(vocab.keys())))

    for i in range(count):
        permutation = np.random.permutation(len(vocab)) + 1
        arr[i, :] = permutation.copy()

    return arr.astype(int)


# %%
def get_fingerprint(minhash, ohe):
    """
    Creates the fingerprint for a given one-hot encoded shingle set.
    :param: `minhash`: Set of minhash functions.
    :param: `ohe`: One-hot encoded shingle set.
    :returns: Min-hashed fingerprint.
    """
    nnz_indexes = np.nonzero(ohe)[0].tolist()
    shingles = minhash[:, nnz_indexes]
    fingerprint = np.min(shingles, axis=1)
    return fingerprint
