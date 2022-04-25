import re
import string
from collections import defaultdict

import numpy as np
import pandas as pd
# from googletrans import Translator
# from langdetect import detect

import baseline
from preprocessing import Preprocessor
from lsh import *

# languages = set()
# translations_dict = dict()


#  X1_df = pd.read_csv("X1_large.csv")
#  X2_df = pd.read_csv("X2_large.csv")
X1_df = pd.read_csv("X1.csv")
X2_df = pd.read_csv("X2.csv")
#  l = logging.Logger("")
#  h = logging.StreamHandler()
#  f = logging.Formatter(fmt="[{filename}:{lineno}] {msg}", style="{")
#  h.setFormatter(f)
#  l.addHandler(h)


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
        X1_candidate_pairs.extend(
            [(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend(
            [(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    # make sure to have the pairs in the first dataset first
    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs
    output_df = pd.DataFrame(all_cand_pairs, columns=[
                             "left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


def hash_by_number(p: int, q: int):
    all_pairs_hashed = defaultdict(list)

    # extract all numbers
    all_numbers = []  # list of lists

    for num in all_numbers:
        # all pairs
        # hash px + qy and append id
        pass

    # all pairs output


def normalize_string(str_to_normalize: str):
    # lowercase, no punctuation or - | &,
    # remove website names like as amazon.com/ebay/techbuy/alienware/Miniprice.ca,
    # wholesale/new/used/brand,
    # computer/computers/laptop/pc,
    # buy/sale,
    # best/good/quality
    # single letters
    # stopwords as on/in/at/from etc
    # inch, GHz, Hz, cm
    stopwords = {"on", "in", "at", "from", "as", "an", "the", "a", "with", "and", "or", "of", "but", "and", "not",
                 "amazon.com", "ebay", "techbuy", "alienware", "miniprice.ca", "alibaba", "mygofer.com", "uediamarkt", "mediamarkt",
                 "wholesale", "new", "used", "brand", "buy"
                 "computer", "computers", "laptops", "laptop", "product", "products", "tablet", "tablets", "pc",
                 "buy", "sale", "best", "good", "quality", "better"
                 "accessories", "kids", ""
                 ",", "|", "/", "@", "!", "?", "-", "&", "*", "#", "(", ")", "[", "]", "{", "}", "/", "|", '"', "*", "/", '-', '+', "#", "-", '\n',
                 "1st", "2nd", "3rd",
                 "ghz", "inch", "cm", "mm", "mhz", "gb", "kb", }
    replace_dict = {"chrgr": "chargers", "usb-stick": "memory card", "memory da usb": "memory card"}

    # remove domain names
    pattern_domain_name = "^((?!-)[A-Za-z0-9-]" + "{1,63}(?<!-)\\.)" + "+[A-Za-z]{2,6}"
    no_domain_str = re.sub(pattern_domain_name, '', str_to_normalize.lower())

    # replace 5 cm to 5cm (Hz, inch etc) etc
    pattern_measures_name = "(?:\d+)\s+(inch|cm|mm|m|hz|ghz|gb|mb|g)"
    no_domain_str = re.sub(pattern_measures_name, '', no_domain_str)

    # remove punctuation
    no_punctuation_string = no_domain_str.translate(str.maketrans(string.punctuation, " "*len(string.punctuation)))

    result_words = [word if not replace_dict.get(word) else replace_dict.get(word) for word in re.split("\W+", no_punctuation_string)
                    if word not in stopwords and len(word) != 1]

    global translations_dict, languages

    # try:
    #     lang = detect(str(no_punctuation_string))
    #     if lang != 'en':
    #         for word in result_words:
    #             if not any(char.isdigit() for char in word) and len(word) > 4:
    #                 translator = Translator()
    #                 translated_word = translator.translate(word).text
    #                 if translated_word.casefold() != word.casefold():
    #                     translations_dict[word] = translated_word
    #                     languages.add(lang)
    # except:
    #     print("No results")

    res_str = " ".join(sorted(result_words))  # TODO try sort
    return res_str


# def blocking_step(df_path):
#     ds = Preprocessor.build(df_path)


def blocking_step(df_path):
    ds = Preprocessor.build(df_path)
    dataset = ds.preprocess()   # TODO: split into #hyperthreads jobs

    k = 3  # ~5 for small docs (emails), 9 - 10 for large docs(papers)
    shingles = []

    buckets_brands = {"toshiba": [], "chuwi": [], "mediacom": [], "google": [], "sandisk": [],
                      "vero": [], "msi": [], "xiaomi": [], "microsoft": [], "apple": [], "razer": [],
                      "lg": [], "dell": [], "fujitsu": [], "huawei": [], "lenovo": [], "acer": [],
                      "asus": [], "hp": [], "samsung": [], "kingston": [], "pami": []}

    i = 0
    for _, row in dataset.iterrows():
        str = normalize_string(row['title'])
        shingles.append((row['id'], k_shingles(str, k)))

        for d in buckets_brands.keys():
            if d in str:
                buckets_brands[d].append(i)

        i += 1

    all_pairs = []

    # joblib
    # or no sort and get > 0.5
    for brand, ids in buckets_brands.items():
        def intersection(name1, name2):
            s1 = set(name1.lower().split())
            s2 = set(name2.lower().split())
            return len(s1.intersection(s2)) / max(len(s1), len(s2))

        pairs = [(dataset['id'][a], dataset['id'][b]) if dataset['id'][a] < dataset['id'][b]
                     else (dataset['id'][b], dataset['id'][a])
                     for idx, a in enumerate(ids) for b in ids[idx + 1:] if dataset['id'][b] != dataset['id'][a] and
                 intersection(dataset['title'][a], dataset['title'][b]) >= 0.4]

        all_pairs.extend(pairs)
    cand_pairs = all_pairs
    # cand_pairs = np.array(sorted(all_pairs, key=lambda x: x[0], reverse=True))
    # cand_pairs = list(map(tuple, cand_pairs[:, 1:3]))
    all_shingles = {item for set_ in shingles for item in set_[1]}

    vocab = {}
    for i, shingle in enumerate(list(all_shingles)):
        vocab[shingle] = i

    del all_shingles

    buckets = 15
    lsh = LSH(buckets)

    hash_function_count = 150
    arr = gen_minhash(vocab, hash_function_count)
    for id, shingle in shingles:
        if len(shingle) == 0:
            continue
        ohe = one_hot(shingle, vocab)

        fingerprint = get_fingerprint(arr, ohe)
        lsh.hash(fingerprint, id)

    lsh_pairs = list(lsh.get_candidate_pairs())

    # baseline_pairs = baseline.block_with_attr(dataset, 'title')

    # TODO random pairs

    print(len(lsh_pairs))
    res = set(lsh_pairs)
    # print(len(cand_pairs))
    return list(res)


def recall(true, prediction):
    fp = set(true).difference(set(prediction))
    return (len(set(true).intersection(set(prediction)))) / len(true), fp


if __name__ == '__main__':

    # %%
    #  X1_df['id_title'] = X1_df['id'].astype(str) + ' ' + X1_df['title']
    #  X1 = X1_df['id_title'].tolist()
    #  X2_df['id_title'] = X2_df['id'].astype(str) + ' ' + X2_df['name']
    #  X2 = X2_df['id_title'].tolist()  # TODO: include other attributes!

    X1_candidate_pairs = blocking_step("X1.csv")
    X2_candidate_pairs = blocking_step("X2.csv")

    print(f'X1_candidate_pairs: {len(X1_candidate_pairs)}')
    print(f'X2_candidate_pairs: {len(X2_candidate_pairs)}')
    #  pdb.set_trace()
    r1, fp1 = recall(pd.read_csv('Y1.csv').to_records(
        index=False).tolist(), X1_candidate_pairs)
    r2, fp2 = recall(pd.read_csv('Y2.csv').to_records(
        index=False).tolist(), X2_candidate_pairs)
    r = (r1 + r2) / 2
    print(f"RECALL FOR X1 \t\t{r1:.3f}")
    print(f"RECALL FOR X2 \t\t{r2:.3f}")
    print(f"RECALL OVERALL  \t{r:.3f}")
    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)

    print("Dataset 1 misses:")
    y1 = pd.read_csv('X1.csv')
    for f in fp1:
        # a, b = y1["title"][y1["id"] == f[0]], y1["title"][y1["id"] == f[1]]
        # print(a[0])
        print(list(y1["title"][y1["id"] == f[0]]))
        print(list(y1["title"][y1["id"] == f[1]]))
        print("\n")

    print("Dataset 2 misses:")
    y1 = pd.read_csv('X2.csv')
    for f in fp2:
        print(list(y1["name"][y1["id"] == f[0]]))
        print(list(y1["name"][y1["id"] == f[1]]))
        print("\n")

    # print(languages)
    # output_df = pd.DataFrame(list(translations_dict.items()))
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    # output_df.to_csv("translations.csv", index=False)
