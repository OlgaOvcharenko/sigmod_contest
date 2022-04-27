import logging
import os

import numpy as np
import pandas as pd
from joblib import delayed, Parallel

from utils import map_async, batch_gen

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


class PyMagnitudeSentenceEmbeddings(Embeddings):
    def __init__(self):
        self._vectors = None
        self._pth = "glove.6B.100d.magnitude"

    def load(self):
        from pymagnitude import Magnitude

        self._vectors = Magnitude(self._pth)

    def generate_sentence_embeddings(self, x_id, x):
        running_embedding = list()

        try:
            inx = [_x.split() for _x in x]
            running_embedding = self._vectors.query(inx)
        except KeyError:
            logger.debug(f"NO EMBEDDING FOUND")
        # u_min = np.min(np.array(running_embedding), axis=0)
        # u_max = np.max(np.array(running_embedding), axis=0)

        tdf = pd.DataFrame()
        tdf["id"] = x_id
        tdf["embeddings"] = np.mean(running_embedding, axis=-2).tolist()
        return tdf

    def generate(self, x, **kwargs):

        embeddings = [
            Parallel(n_jobs=os.cpu_count(), backend="threading")(
                delayed(self.generate_sentence_embeddings)(
                    b_df["id"], b_df[kwargs["attr"]].to_list()
                )
                for ix, b_df in kwargs["df"].groupby(
                    np.arange(len(kwargs["df"])) // kwargs["batch_size"]
                )
            )
        ]

        return pd.concat(embeddings[0])


class FastTextEmbeddings(Embeddings):
    def __init__(self):
        self._model = None

    def load(self):
        pass

    def generate_sentence_embeddings(self, x_id, x):
        running_embedding = list()

        for word in x.split():
            try:
                running_embedding.append(self._model.wv[word])
            except KeyError:
                logger.debug(f"NO EMBEDDING FOUND FOR {word}")

        # u_min = np.min(np.array(running_embedding), axis=0)
        # u_max = np.max(np.array(running_embedding), axis=0)

        return x_id, np.mean(running_embedding, axis=0)

    def generate(self, x, epoch=12, **kwargs):
        words_in_sentences = [sentence.split(" ") for sentence in x]
        from gensim.models import FastText

        self._model = FastText(
            words_in_sentences, min_count=1, epochs=epoch, word_ngrams=1, window=2
        )

        embeddings = list()

        for ids, result in map_async(batch_gen(x), self.generate_sentence_embeddings):
            embeddings.append(result)

        # embeddings = [
        #     Parallel(n_jobs=os.cpu_count(), backend="threading")(
        #         delayed(self.generate_sentence_embeddings)(x_chunk["id"], x_chunk[kwargs["attr"]])
        #         for x_chunk_id, x_chunk in kwargs["df"].iterrows()
        #     )
        # ]

        return np.array(embeddings)


class UniModelTF(Embeddings):
    def __init__(self):
        self._model = None

    def load(self, **kwargs):
        import tensorflow_hub as hub

        self._model = hub.load(
            r"tfhub_modules/063d866c06683311b44b4992fd46003be952409c"
        )

    def predict(self, ix, x):
        tdf = pd.DataFrame()
        tdf["id"] = ix
        tdf["embeddings"] = np.array(self._model(x)).tolist()
        return tdf

    def generate(self, x, **kwargs):
        # for _, b_df in kwargs["df"].groupby(np.arange(len(kwargs["df"])) // kwargs["batch_size"]):
        #     embeddings.extend(self._model())
        embeddings = [
            Parallel(n_jobs=os.cpu_count(), backend="threading")(
                delayed(self.predict)(b_df["id"], b_df[kwargs["attr"]].to_list())
                for ix, b_df in kwargs["df"].groupby(
                    np.arange(len(kwargs["df"])) // kwargs["batch_size"]
                )
            )
        ]

        return pd.concat(embeddings[0])


class SentenceTransformersEmd(Embeddings):
    def __init__(self):
        self._model = None

    def load(self, **kwargs):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    def generate(self, x, **kwargs):
        return self._model.encode(x)


class TFIDFEmbeddings(Embeddings):
    def __init__(self):
        self._model = None

    def load(self, **kwargs):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._model = TfidfVectorizer(vocabulary="english", ngram_range=(1, 3))

    def generate(self, x, **kwargs):
        self._model.fit(x)
        features_vec = self._model.transform(x)
        return features_vec
