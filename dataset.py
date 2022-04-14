import pandas as pd


class Dataset():

    def __init__(self, df) -> None:
        self.df = pd.read_csv(df)

    def preprocess(self) -> pd.DataFrame:
        return self.df

    @staticmethod
    def build(path):
        datasets = {
            "X1.csv": X1,
            "X2.csv": X2,
        }
        return datasets[path](path)


class X1(Dataset):
    pass


class X2(Dataset):
    pass
