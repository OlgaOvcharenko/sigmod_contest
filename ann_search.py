import itertools
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger()


class ANN:
    def load_and_query(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def generate_candidate_pairs(
        nn, distances, nn_ids, distance_threshold=0.20, **kwargs
    ):
        logger.info(f"CANDIDATE GENERATION IN PROCESS")
        candidate_pairs = list()
        metric = list()
        for ix, neighbours in enumerate(nn):
            main_can = nn_ids[ix]
            for yx, neighbour in enumerate(neighbours):
                can2 = nn_ids[neighbour]
                candidate_distance = distances[ix][yx]

                if main_can == can2:
                    continue
                if candidate_distance < distance_threshold:
                    continue

                if main_can < can2:
                    candidate_pairs.append((main_can, can2))
                else:
                    candidate_pairs.append((can2, main_can))
                metric.append(candidate_distance)

        return candidate_pairs, metric

    @staticmethod
    def sort_based_on_metric(candidate_pairs, metric):
        logger.info(f"SORTING IN PROGRESS")
        sorted_metric, sorted_candidates_pairs = zip(
            *sorted(zip(metric, candidate_pairs), key=lambda x: x[0], reverse=True)
        )
        return sorted_candidates_pairs, sorted_metric

    @staticmethod
    def remove_duplicate(candidate_pairs):
        logger.info(f"REMOVING DUPLICATES")
        return set(candidate_pairs)


class FaissL2Flat(ANN):
    def load_and_query(self, feature_emd, neighbours, **kwargs):
        size, dim = feature_emd.shape
        _emb = feature_emd.astype(np.float32)

        import faiss

        index = faiss.IndexFlatL2(dim)
        index.add(_emb)

        distances, nn = index.search(_emb, k=neighbours)

        return nn, distances


class HNSWLib(ANN):
    def load_and_query(
        self,
        x,
        neighbours=5,
        ef=500,
        m=64,
        ef_construction=500,
        search_space="cosine",
        **kwargs,
    ):
        size, dim = x.shape
        import hnswlib

        p = hnswlib.Index(space=search_space, dim=dim)
        p.init_index(max_elements=size, ef_construction=ef_construction, M=m)
        p.add_items(x, list(range(0, size)))
        p.set_ef(ef)
        nn, distances = p.knn_query(x, k=neighbours)
        return nn, 1 - distances


class LSHQuery(ANN):
    def __init__(self):
        self._all_hash = None

    def generate_candidate_pairs(
        self, nn, distances, nn_ids, distance_threshold=0.20, **kwargs
    ):

        candidates = set()
        for bucket_hash, bucket in nn.items():
            if len(bucket) > 1:
                for ix1, id1 in enumerate(bucket):
                    rid1 = nn_ids[id1]

                    for _, id2 in enumerate(bucket[ix1 + 1 :]):
                        rid2 = nn_ids[id2]
                        pair = (rid1, rid2) if rid1 < rid2 else (rid2, rid1)
                        candidates.add(pair)
            # else:
            #     bucket_hash_idx = np.where(
            #         np.array(self._all_hash) == bucket_hash
            #     )[0][0]
            #     prev_hash_idx = self._all_hash[bucket_hash_idx - 1]
            #     next_hash_idx = self._all_hash[bucket_hash_idx + 1]
            #     approx_neighbour_bucket = nn[prev_hash_idx] if prev_hash_idx in nn else nn[next_hash_idx]
            #
            #     if len(approx_neighbour_bucket) >= 1:
            #         rid1 = bucket[0]
            #         for _, id2 in enumerate(approx_neighbour_bucket):
            #             rid2 = nn_ids[id2]
            #             pair = (rid1, rid2) if rid1 < rid2 else (rid2, rid1)
            #             candidates.add(pair)

        return candidates, None

    @staticmethod
    def sort_based_on_metric(candidate_pairs, metric):
        logger.info(f"SORTING IN PROGRESS")
        return candidate_pairs, metric

    @staticmethod
    def _get_random_vectors(hash_size, hash_table, dim):
        return np.dstack(
            [np.random.normal(0, 1, (hash_size, dim)) for i in range(hash_table)]
        ).T.swapaxes(-1, -2)

    def load_and_query(self, x, **kwargs):
        fg = (
            self._get_random_vectors(
                kwargs["hash_size"], kwargs["hash_table"], x.shape[-1]
            )
            - 0.5
        )
        _hash_dot = np.tensordot(x, fg.T, axes=([-1], [0]))
        _hash_dot = np.sign(_hash_dot)
        _hash_dot[_hash_dot < 0] = 0
        _hash_dot = _hash_dot.astype(np.uint8)
        for i in range(kwargs["hash_size"]):
            data = _hash_dot[:, i, :]
            data[data > 0] = i + 1
            _hash_dot[:, i, :] = data
        _hash_dot = np.sum(_hash_dot, axis=1)
        _, d = _hash_dot.shape

        self._all_hash = list()
        table = defaultdict(list)
        for i, hash_code in enumerate(_hash_dot):
            hc = "".join([str(ii) for ii in hash_code])
            self._all_hash.append(hc)

            table[hc].append(i)

        self._all_hash.sort(key=float)
        return table, None


class LSHRPQuery(ANN):
    def __init__(self):
        self._binary_hash = None

    @staticmethod
    def hamming(hashed_vec, hashes):
        # get hamming distance between query vec and all buckets in self.hashes
        hamming_dist = np.count_nonzero(hashed_vec != hashes, axis=1).reshape(-1, 1)
        # add hash values to each row
        hamming_dist = np.concatenate((hashes, hamming_dist), axis=1)
        # sort based on distance
        hamming_dist = hamming_dist[hamming_dist[:, -1].argsort()]
        return hamming_dist

    def generate_candidate_pairs(
        self,
        nn,
        distances,
        nn_ids,
        distance_threshold=0.20,
        hanging_id_threshold=3,
        **kwargs,
    ):

        candidates = set()
        for _, bucket in nn.items():
            if len(bucket) > 1:
                for ix1, id1 in enumerate(bucket):
                    rid1 = nn_ids[id1]

                    for _, id2 in enumerate(bucket[ix1 + 1 :]):
                        rid2 = nn_ids[id2]
                        pair = (rid1, rid2) if rid1 < rid2 else (rid2, rid1)
                        candidates.add(pair)
            else:
                approx_neighbour_bucket = list()
                id1 = bucket[0]
                dist = self.hamming(self._binary_hash[id1], self._binary_hash)
                for i in range(1, hanging_id_threshold):
                    closet_pair = nn["".join(dist[i][:-1].astype(str))]
                    approx_neighbour_bucket.extend(closet_pair)

                rid1 = nn_ids[id1]
                for _, id2 in enumerate(approx_neighbour_bucket):
                    rid2 = nn_ids[id2]
                    pair = (rid1, rid2) if rid1 < rid2 else (rid2, rid1)
                    candidates.add(pair)

        return candidates, None

    @staticmethod
    def sort_based_on_metric(candidate_pairs, metric):
        logger.info(f"SORTING IN PROGRESS")
        return candidate_pairs, metric

    def load_and_query(self, x, **kwargs):
        _, dim = x.shape
        plane_norms = np.random.rand(dim, kwargs["n_bits"]) - 0.5

        direction = np.dot(x, plane_norms)
        direction = direction > 0
        self._binary_hash = direction.astype(int)

        table = defaultdict(list)
        for i, hash_code in enumerate(self._binary_hash):
            table["".join([str(ii) for ii in hash_code])].append(i)
        return table, None


class PySparNNCluster(ANN):
    def load_and_query(
        self, x, raw_data, neighbours=5, clusters=3, num_indexes=50, **kwargs
    ):
        import pysparnn.cluster_index as ci

        cp = ci.MultiClusterIndex(x, raw_data, num_indexes=num_indexes)
        nn = cp.search(x, k=neighbours, k_clusters=clusters, return_distance=True)
        return nn


class AnnoyAnn(ANN):
    def load_and_query(self, x, n_tress=10, **kwargs):
        from annoy import AnnoyIndex

        _, dim = x.shape
        a_index = AnnoyIndex(dim, "dot")
        for i, vec in enumerate(x):
            a_index.add_item(i, vec)

        a_index.build(n_tress)

        nn = list()
        distances = list()
        for i, vec in enumerate(x):
            _nn, _d = a_index.get_nns_by_vector(
                vec, n=5, search_k=5, include_distances=True
            )
            nn.append(_nn)
            distances.append(_d)

        return np.array(nn), np.array(distances)
