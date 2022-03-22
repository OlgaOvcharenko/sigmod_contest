# sigmod_contest

### Steps to create submission file

Install [reprozip](https://docs.reprozip.org/en/0.7.x/install.html) -`pip install reprozip` and then follow the below steps

```python
    reprozip trace python blocking.py
    reprozip pack submission.rpz
```

### Additional libraries

For [Sentence Embeddings](https://github.com/UKPLab/sentence-transformers), install the required
library by running the following cmd `pip install -U sentence-transformers` or `conda install -c conda-forge sentence-transformers
`