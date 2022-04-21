import pandas as pd
import pdb


SPECIAL_CHAR = [",", ":", ";", "!", "?",
                "(", ")", "[", "]", "{", "}", "/", "|", '"', "*", "/", '-', '+', '\n']


class Preprocessor():

    def __init__(self, df) -> None:
        self.df = pd.read_csv(df)

    def to_lower(self, cols):
        for c in cols:
            self.df[c] = self.df[c].str.lower()

    def cleanup(self, cols):
        for c in cols:
            for sc in SPECIAL_CHAR:
                self.df[c] = self.df[c].str.replace(sc, ' ')

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
    language_mapping = {"fr": ["carte", "clé", "montres", "cle", "disque", "mémoire"], "es": ["tarjeta", "memoria", ],
                        "hu": ["kártya", "memóriakártya", "pendrive", "pen drive"], "pl": ["karta", "pami"],
                        "it": ["scheda", "memorie", "chiavetta"],
                        "de": ["karte", "speicherkarte", "werk", "für", "bei", "speich"], "nl": ["kaart", "flashstation"],
                        "se": ["minneskort", "usb-minne"], "cz": ["ý"]}

    def __init__(self, df):
        super().__init__(df)

    def extract_language(self):
        for idx, row in self.df.iterrows():
            def predicate(x): return x in row['title']
            for k, v in self.language_mapping.items():
                if any(map(predicate, self.language_mapping[k])):
                    self.df.at[idx, "lang"] = k
                    break

    def _preprocess_X(self):
        self.df = self.df.rename({'name': 'title'}, axis=1)
        self.df["lang"] = "en"
        self.extract_language()
        return self.df
