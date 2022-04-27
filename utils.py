import logging
import os
import re
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from concurrent.futures import ThreadPoolExecutor

GLOVE_PTH = "glove.6B.300d.txt"

logger = logging.getLogger()

STOPWORDS = {
    "on",
    "in",
    "at",
    "from",
    "as",
    "an",
    "the",
    "a",
    "with",
    "and",
    "or",
    "of",
    "but",
    "and",
    "amazon.com",
    "ebay",
    "techbuy",
    "alienware",
    "miniprice.ca",
    "alibaba",
    "google",
    "wholesale",
    "new",
    "used",
    "brand",
    "computer",
    "computers",
    "laptops",
    "laptop",
    "product",
    "products",
    "tablet",
    "tablets",
    "pc",
    "buy",
    "sale",
    "best",
    "good",
    "quality",
    "accessories",
    "kids",
    "" ",",
    "|",
    "/",
    "@",
    "!",
    "?",
    "-",
    "1st",
    "2nd",
    "3rd",
    "ghz",
    "inch",
    "cm",
    "mm",
    "mhz",
    "gb",
    "kb",
}


def generate_random_vectors(dim, n_vectors):
    return np.random.normal(0, 0.5, (n_vectors, dim))


def recall(true, prediction):
    return (len(set(true).intersection(set(prediction)))) / len(true)


def cosine_distance(u, v):
    u = u.reshape(1, -1)
    return 1 - (u @ v.T / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)))


def map_async(iterable, func, **kwargs):
    max_workers = os.cpu_count()

    # https://devdreamz.com/question/825056-how-do-i-use-threads-on-a-generator-while-keeping-the-order
    # Generator that applies func to the input using max_workers concurrent jobs

    def async_iterator():
        iterator = iter(iterable)
        pending_results = []
        has_input = True
        thread_pool = ThreadPoolExecutor(max_workers)
        while True:
            # Submit jobs for remaining input until max_worker jobs are running
            while (
                has_input
                and len([e for e in pending_results if e.running()]) < max_workers
            ):
                try:
                    ids, data = next(iterator)
                    logger.debug(f"Submitting Task")
                    pending_results.append(
                        thread_pool.submit(func, ids, data, **kwargs)
                    )
                except StopIteration:
                    logger.debug(f"Submitted all task")
                    has_input = False

            # If there are no pending results, the generator is done
            if not pending_results:
                return

            # If the oldest job is done, return its value
            if pending_results[0].done():
                yield pending_results.pop(0).result()
            # Otherwise, yield the CPU, then continue starting new jobs
            else:
                time.sleep(0.01)

    return async_iterator()


def plot_features(vector_representation, df, attr, tag="dell"):
    tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")
    embs = tsne.fit_transform(vector_representation)
    df["x"] = embs[:, 0]
    df["y"] = embs[:, 1]
    match = df[df[attr].str.contains(tag)]
    fig, ax = plt.subplots(figsize=(10, 8))
    # Scatter points, set alpha low to make points translucent
    ax.scatter(df.x, df.y, alpha=0.1)
    ax.scatter(match.x, match.y, alpha=0.2, color="green")
    plt.title("Scatter plot using t-SNE")
    plt.show()


def extract_glove_embeddings():
    logger.info(f"LOADING GLOVE EMBEDDING")
    embeddings_index = dict()
    with open(GLOVE_PTH, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype="float32")
    return embeddings_index


def remove_stop_words(x):

    result_words = [
        word for word in re.split("\W+", x) if word not in STOPWORDS and len(word) != 1
    ]

    return " ".join(result_words)


def reg_normalization(x):
    x = str(x).lower()
    x = re.sub(r"\W+", " ", str(x))
    x = re.sub(
        r"^((?!-)[A-Za-z0-9-]" + "{1,63}(?<!-)\\.)" + "+[A-Za-z]{2,6}", "", str(x)
    )
    x = re.sub(r"(?:\d+)\s+(inch|cm|mm|m|hz|ghz|gb|mb|g)", "", x)
    x = re.sub(r"[^\w\s]", "", x)
    x = [
        word for word in re.split("\W+", x) if word not in STOPWORDS and len(word) != 1
    ]
    return " ".join(x)


def pre_process(df: pd.DataFrame, attr):
    df[attr] = df.apply(lambda row: reg_normalization(row[attr]), axis=1)
    return df


def batch_gen(x, **kwargs):
    for x_chunk_id, x_chunk in enumerate(x):
        yield x_chunk_id, x_chunk


def tf_batch_gen(df, batch_size, attr):
    for ix, b_df in df.groupby(np.arange(len(df)) // batch_size):
        yield ix, b_df[attr].to_list()
