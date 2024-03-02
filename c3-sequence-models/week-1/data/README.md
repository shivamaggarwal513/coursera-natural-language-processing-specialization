# Large Movie Review Dataset

- **Download:** [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment)
- Place `aclImdb` folder (NOT `aclImdb_v1`) in `data` folder.
- Run `make_data.py` with inputs according to number of samples you want in each class of each train/test split.
  - Inputs: `train_pos`, `train_neg`, `test_pos`, `test_neg`. Notebook was run with all `1000`.
