import numpy as np
import pandas as pd
import pdb
from langdetect import detect



SPECIAL_CHAR = ["(", ")", "[", "]", "{", "}", "/", "|", '"', "*", "/", '-', '+', "#", '\n']

class Preprocessor():

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
        return self.df


class X2_Preprocessor(Preprocessor):
    def __init__(self, df):
        super().__init__(df)

    def _preprocess_X(self):
        self.df = self.df.rename({'name':'title'}, axis=1)
        self.df['title'] = self.df['title'] + ' ' + self.df['price'].astype(str) + ' ' + self.df['brand'] + ' ' + self.df['description'] + ' ' + self.df['category']
        self.df['title'] = self.df['title'].str.replace('nan', '')
        #  pdb.set_trace()
        return self.df
