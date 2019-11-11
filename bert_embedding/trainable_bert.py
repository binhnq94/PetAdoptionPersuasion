from flair.embeddings import BertEmbeddings, Sentence
import pandas as pd


def read_all_data(list_files=None):
    version_data = 'v6'
    train_fn = f'datasets/190524/converted-{version_data}_train.csv'
    val_fn = f'datasets/190524/converted-{version_data}_val.csv'
    test_fn = f'datasets/190524/converted-{version_data}_test.csv'
    data_files = [train_fn, val_fn, test_fn]
    list_files = data_files if list_files is None else list_files
    out_dict = {}
    for fn in list_files:
        df = pd.read_csv(fn, sep='\t', names=['id', 'document', 'label'], header=None)
        for row in df.itertuples(index=False):
            id_ = int(row.id)
            assert id_ not in out_dict
            out_dict[id_] = row.document
    return out_dict


ID2DOCUMENT = read_all_data()
