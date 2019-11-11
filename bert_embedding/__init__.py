import os
import torch


def load_list_files(layers='-1'):
    data_version = 'v5'
    train_fn = os.path.join('datasets/190524', f'converted-{data_version}_train.csv')
    val_fn = os.path.join('datasets/190524', f'converted-{data_version}_val.csv')
    test_fn = os.path.join('datasets/190524', f'converted-{data_version}_test.csv')
    files = {
        'train': train_fn,
        'val': val_fn,
        'test': test_fn
    }
    list_files = {}
    for kind in ['val', 'test', 'train']:
        folder_out = os.path.join('/mnt/sda4', f'{files[kind]}.layers:{layers}')

        for f_name in os.listdir(folder_out):
            id_ = int(os.path.splitext(f_name)[0])
            f_path = os.path.join(folder_out, f_name)
            list_files[id_] = f_path

    list_files = list_files
    return list_files


BERT_FILES = load_list_files()
print('BERT_FILES', len(BERT_FILES))


def bert_embedding(list_x_id, x, flatten=True):
    out_embeddings = torch.zeros((x.shape[0], x.shape[1], 768))
    if torch.cuda.is_available():
        out_embeddings = out_embeddings.cuda()

    for i, x_id in enumerate(list_x_id):
        f_path = BERT_FILES[x_id.item()]
        list_sens = torch.load(f_path)
        emb = torch.cat(list_sens, dim=0)
        out_embeddings[i][:emb.shape[0]] = emb

    return out_embeddings


def bert_embedding_by_id(list_x_id, _):
    max_token = 0
    list_emb = []
    for x_id in list_x_id:
        f_path = BERT_FILES[x_id]
        list_sens = torch.load(f_path)
        emb = torch.cat(list_sens, dim=0)

        if max_token < emb.shape[0]:
            max_token = emb.shape[0]
        list_emb.append(emb)

    out_embeddings = torch.zeros((len(list_x_id), max_token, 768))
    for i, emb in enumerate(list_emb):
        out_embeddings[i][:emb.shape[0]] = emb

    return out_embeddings.numpy()


def bert_embedding_by_id2(list_x_id, _):
    list_list_sens = []
    for x_id in list_x_id:
        f_path = BERT_FILES[x_id]
        list_sens = torch.load(f_path)
        list_list_sens.append(list_sens)

    max_token = max([sen.shape[0] for doc in list_list_sens for sen in doc])

    max_sen = max([len(doc) for doc in list_list_sens])

    out_embeddings = torch.zeros((len(list_x_id), max_sen, max_token, 768))
    for i, doc in enumerate(list_list_sens):
        for j, sen in enumerate(doc):
            out_embeddings[i][j][:sen.shape[0]] = sen

    return out_embeddings.numpy()
