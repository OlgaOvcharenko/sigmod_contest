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
                        thread_pool.submit(func, data, ids, **kwargs)
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


def regex_feature_process(data):
    p = re.findall(r"\w+\s\w+\d+", data)

    def _digit(inp_str):
        return np.alltrue(
            np.array([indi_str.isdigit() for indi_str in inp_str.split()])
        )

    if len(p) == 0:
        feat = re.findall(r"(?i)\b[a-z]+\b", data)
    elif np.alltrue(np.array([_digit(s) for s in p])):
        feat = re.findall(r"(?i)\b[a-z]+\b", data)
    else:
        feat = p
    return " ".join(feat)


def pre_process(df: pd.DataFrame, attr):

    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df = df.applymap(lambda x: re.sub(r"\W+", " ", x) if type(x) == str else x)
    df[attr] = df.apply(lambda row: regex_feature_process(row[attr]), axis=1)
    # df = df.applymap(
    #     lambda x: " ".join(
    #         re.findall(r"\w+\s\w+\d+", x)
    #     )
    #     if type(x) == str
    #     else x
    # )

    return df


def batch_gen(data: pd.DataFrame, attr: str):
    for i, df_chunk in data.iterrows():
        yield df_chunk["id"], df_chunk[attr]
