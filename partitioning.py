import pdb

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
    "hp"
]

cpu_brands = [
    "intel",
    "amd"
]

models = [
    "thinkpad", "latitude", "inspiron", "vostro", "xps", "aspire", "toughbook",
    "vivobook" "pavilion", "elitebook"]

products = [
    "carbon x1",  "x230", "2740p", "xmg", "aspire", "toughbook", "vivobook",
    "pavilion", "elitebook", "latitude"
]

models_by_brand = {
    "lenovo": ["thinkpad"],
    "dell": ["latitude", "inspiron", "vostro", "xps"],
    "acer": ["aspire"],
    "panasonic": ["toughbook"],
    "asus": ["vivobook"],
    "hp": ["pavilion", "elitebook"]

}

brand_by_models = {vv: k for k, v in models_by_brand.items() for vv in v}


class Partitioner():

    def __init__(self, df) -> None:
        self.df = df

    def blocking_step(self) -> None:
        pass

    @staticmethod
    def build(path, data):
        datasets = {
            "X1_large.csv": X1_Partitioner,
            "X2_large.csv": X2_Partitioner,
            "X1.csv": X1_Partitioner,
            "X2.csv": X1_Partitioner,
        }
        return datasets[path](data)

    def _get_candidate_pairs(self) -> list:
        return []

    def get_candidate_pairs(self) -> list:
        return self._get_candidate_pairs()


class X1_Partitioner(Partitioner):
    def __init__(self, df):
        super().__init__(df)

    def partition_by_attribute(self, attributes, parents) -> list:
        buckets = {p: [] for p in attributes}

        assigned_block = []  # remove id to reduce candidate_pairs in future blocking step
        for idx, row in self.df.iterrows():
            title = row['title']
            for k in buckets.keys():
                if k in title and (len(parents) == 0 or parents[k] in title):
                    #  pdb.set_trace()
                    buckets[k].append(row.id)
                    assigned_block.append(idx)
                    break
        self.df = self.df.drop(assigned_block)

        candidate_pairs = set()
        for _, candidates in buckets.items():
            for ix1, id1 in enumerate(candidates):
                for _, id2 in enumerate(candidates[ix1+1:]):
                    pair = (id1, id2) if id1 < id2 else (id2, id1)
                    candidate_pairs.add(pair)

        return list(candidate_pairs)

    def partition_by_product(self) -> list:
        return self.partition_by_attribute(products, [])

    def partition_by_model(self) -> list:
        return self.partition_by_attribute(models, brand_by_models)

    def partition_by_brand(self) -> list:
        return self.partition_by_attribute(brands, [])

    def _get_candidate_pairs(self) -> list:
        candidate_pairs = []

        candidate_pairs.extend(self.partition_by_product())
        pdb.set_trace()
        candidate_pairs.extend(self.partition_by_model())
        candidate_pairs.extend(self.partition_by_brand())

        return candidate_pairs


class X2_Partitioner(Partitioner):
    def __init__(self, df):
        super().__init__(df)

    def _get_candidate_pairs(self) -> list:
        group_dict = self.df.groupby(by=['brand']).groups
        print(len(group_dict))
        block_pairs = set()
        for _, ixs in group_dict.items():
            block = list(self.df.loc[ixs, 'id'])
            for ix1, id1 in enumerate(block):
                for _, id2 in enumerate(block[ix1+1:]):
                    pair = (id1, id2) if id1 < id2 else (id2, id1)
                    block_pairs.add(pair)

        return list(block_pairs)
