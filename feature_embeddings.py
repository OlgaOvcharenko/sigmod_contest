import logging

import numpy as np

logger = logging.getLogger()


class Embeddings:
    def load(self, **kwargs):
        raise NotImplementedError

    def generate(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def normalize(x):
        emd = x / np.linalg.norm(x, axis=-1).reshape(-1, 1)
        return emd


class TFIDFHashedEmbeddings(Embeddings):
    def load(self):
        pass

    def generate(self, x, n_features, **kwargs):
        from sklearn.feature_extraction.text import HashingVectorizer

        return (
            HashingVectorizer(n_features=n_features, ngram_range=(1, 1))
            .transform(x)
            .toarray()
        )
