import re
import string

import pandas as pd
from collections import Counter
import pdb


SPECIAL_CHAR = ["?", "(", ")", "[", "]", "{", "}",
                  "/", "|", '"', "*", "/", '-', '+', "#", '\n']
translations_dict = pd.read_csv(
    "translations.csv", header=None, index_col=0, squeeze=True).to_dict()


class Preprocessor:

    def __init__(self, df) -> None:
        self.df = pd.read_csv(df)

    def to_lower(self, cols):
        # for c in cols:
        #     self.df[c] = self.df[c].str.lower()
        pass

    def cleanup(self, cols):
        pass
        # for c in cols:
        #     for sc in SPECIAL_CHAR:
        #         self.df[c] = self.df[c].str.replace(sc, ' ')

    def preprocess(self) -> pd.DataFrame:
        self.df = self.df.fillna(value="")
        cols = [c for c in self.df.columns if c != "id"]
        self.to_lower(cols)
        self.cleanup(cols)
        return self._preprocess_X()

    def _preprocess_X(self):
        raise NotImplementedError("Dataset baseclass.")

    def normalize_string(self, str_to_normalize, is_X2: bool):
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
        replace_dict = {"chrgr": "chargers",
            "usb-stick": "memory card", "memory da usb": "memory card"}

        # remove domain names
        pattern_domain_name = "^((?!-)[A-Za-z0-9-]" + \
                                 "{1,63}(?<!-)\\.)" + "+[A-Za-z]{2,6}"
        no_domain_str = re.sub(pattern_domain_name, '',
                               str_to_normalize.lower())

        # replace 5 cm to 5cm (Hz, inch etc) etc
        pattern_measures_name = "(?:\d+)\s+(inch|cm|mm|m|hz|ghz|gb|mb|g)"
        no_domain_str = re.sub(pattern_measures_name, '', no_domain_str)

        # remove punctuation
        no_punctuation_string = no_domain_str.translate(
            str.maketrans(string.punctuation, " "*len(string.punctuation)))

        if is_X2:
            result_words = set(word if not translations_dict.get(word) else translations_dict.get(word) for word in re.split("\W+", no_punctuation_string)
                            if word not in stopwords and len(word) > 1)
        else:
            result_words = set(word for word in re.split(
                "\W+", no_punctuation_string) if word not in stopwords and len(word) > 1)

        res_str = " ".join(
            sorted(result_words, reverse=False))  # TODO try sort
        return res_str

    @staticmethod
    def build(path):
        datasets = {
            "X1_large.csv": X1_Preprocessor,
            "X2_large.csv": X2_Preprocessor,
            "X1.csv": X1_Preprocessor,
            "X2.csv": X2_Preprocessor,
        }

        real_path = path
        if path == "X1_large.csv":
            path = "X1.csv"
        if path == "X2_large.csv":
            path = "X2.csv"
        return datasets[path](real_path)


class X1_Preprocessor(Preprocessor):
    def __init__(self, df):
        super().__init__(df)

    def _preprocess_X(self):
        self.df['title'] = self.df['title'].apply(
            self.normalize_string, is_X2=False)
        return self.df


class X2_Preprocessor(Preprocessor):

    def __init__(self, df):
        super().__init__(df)

    def extract_brand(self):
        brands = [
            "dell",
            "acer",
            "panasonic",
            "asus",
            "lenovo",
            "lexar",
            "pny",
            "sony",
            "toshiba",
            "sandisk",
            "transcend",
            "kingston",
            "samsung",
            "hp",
            "accessoires", "carte", "clé", "disque", "intenso", "karta", "logilink", "pami"
        ]

        stopwords = ["technology", "technologies"]

        for idx, row in self.df.iterrows():
            if row['brand'] is not "":
                b = row['brand'].split(" ") 

                uniquebrand = Counter(b)
                s = " ".join(uniquebrand.keys())

                for sc in SPECIAL_CHAR:
                    s = s.replace(sc, '')

                for w in stopwords:
                    s = s.replace(w, '').strip()

                self.df.at[idx, 'brand'] = s
                continue

            title = row['title']
            for brand in brands:
                if brand in title:
                    self.df.at[idx, 'brand'] = brand
                    break
            self.df.at[idx, 'brand'] = 'zzzzzz'

    def normalize_brand(self, brand):
        return brand.lower()

    def _preprocess_X(self):
        self.df = self.df.rename({'name':'title'}, axis=1)
        self.df['title'] = self.df['title'] + ' ' + self.df['brand'] + self.df['category'] # + ' ' + self.df['description'] + ' ' + self.df['category']  # self.df['price'].astype(str) +
        self.df['title'] = self.df['title'].str.replace('nan', '')
        self.df['title'] = self.df['title'].apply(self.normalize_string, is_X2=True)
        self.df['brand'] = self.df['brand'].str.lower()
        self.extract_brand()
        return self.df
