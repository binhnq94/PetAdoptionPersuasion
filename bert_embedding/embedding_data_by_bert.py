import os
import pandas as pd
import torch
from flair.data import Sentence
from flair.embeddings import BertEmbeddings
from tqdm import tqdm


import re

RE_DUPLICATE_SPACE = re.compile(' +', flags=re.UNICODE)

print(RE_DUPLICATE_SPACE.sub(' ', 'The     quick brown    fox'))


def sent_tokenize(x):
    return x.split('<end>')


def split_long_sentence(list_sen, part_size):
    """Split long sentence."""
    threshold = 250
    # index_to_cut = 250
    out_list_sen = []
    list_number_part = []

    for sen in list_sen:
        num_words = sen.count(' ') + 1
        if num_words > threshold:
            print('num_words', num_words, 'part_size', part_size)
            tokens = sen.split(' ')
            number_part = num_words//part_size
            for i in range(num_words//part_size):
                token_2_add = ' '.join(tokens[i * part_size: (i+1)*part_size])
                out_list_sen.append(token_2_add)

            if num_words % part_size != 0:
                if num_words % part_size > part_size//2:
                    token_2_add = ' '.join(tokens[-(num_words % part_size):])
                    out_list_sen.append(token_2_add)
                    number_part += 1
                else:
                    out_list_sen[-1] += ' ' + ' '.join(tokens[-(num_words % part_size):])

            list_number_part.append(number_part)
        else:
            out_list_sen.append(sen)
            list_number_part.append(1)

    return [Sentence(sen) for sen in out_list_sen], list_number_part


def embedding(list_sen, bert_embedding, count=30, row_id=None, cut_off=False, part_size=100):
    if cut_off:
        list_sentence, list_number_part = split_long_sentence(list_sen, part_size)
        for i in list_number_part:
            if i > 1:
                print('cut_off', row_id, list_number_part)
                break
    else:
        list_sentence, list_number_part = [Sentence(sen) for sen in list_sen], [1 for _ in list_sen]

    for i in range((len(list_sentence) // count) + 1):
        if list_sentence[i * count: (i + 1) * count]:
            bert_embedding.embed(list_sentence[i * count: (i + 1) * count])

    list_sen_embedding = []  # [number_sen x num_word x emb_size]

    start_idx = 0
    for i, number in enumerate(list_number_part):
        sens = list_sentence[start_idx: start_idx + number]
        sen_embedding = torch.stack([token.embedding for sen in sens for token in sen], dim=0)
        assert sen_embedding.shape[1] % 768 == 0
        assert sen_embedding.shape[0] == list_sen[i].count(' ') + 1
        list_sen_embedding.append(sen_embedding)

        start_idx += number

    return list_sen_embedding


def process_a_file(fn, bert_embedding: BertEmbeddings, folder_out):
    """Process a input file."""
    assert folder_out != fn
    if not os.path.isdir(folder_out):
        os.makedirs(folder_out)
    df = pd.read_csv(fn, sep='\t', names=['id', 'document', 'label'], header=None)
    print(fn, df.columns, len(df))
    assert len(df.columns) == 3

    max_sen = 0
    max_number_repeat = 0
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        try:
            document = row.document

            # Do clean \\t for all document.
            document = document.replace('\\t', '')
            document = RE_DUPLICATE_SPACE.sub(' ', document)
            list_sen = sent_tokenize(document)
            list_sen = [sen.strip() for sen in list_sen if sen.strip()]

            count = 40
            repeat = 0
            cut_off = True

            part_size = 100
            while True:
                try:
                    repeat += 1
                    list_sen_embedding = embedding(list_sen, bert_embedding, count, row.id, cut_off=cut_off,
                                                   part_size=part_size)
                    break
                except RuntimeError as e:
                    if count == 1:
                        print('count', count)
                        print('repeat', repeat)
                        print('part_size', part_size)
                        raise e
                        # if part_size == 50:
                        #     print('count', count)
                        #     print('repeat', repeat)
                        #     print('part_size', part_size)
                        #     raise e
                        # else:
                        #     part_size -= 25

                    else:
                        count = count // 2
                        # print('change count', row.id, count)

            if repeat > max_number_repeat:
                max_number_repeat = repeat

            if len(list_sen_embedding) > max_sen:
                max_sen = len(list_sen_embedding)
            torch.save(list_sen_embedding, os.path.join(folder_out, f'{row.id}.pt'))
        except Exception as e:
            print(row.id)
            print(row.document)
            print('len(list_sen)', len(list_sen))
            print('max_sen', max_sen)
            print('max_number_repeat', max_number_repeat)
            raise e

    print('max_number_repeat', max_number_repeat)
    print(f"SAVE data to {folder_out}")


def main(layers='-1'):
    data_version = 'v5'
    bert_embedding = BertEmbeddings(layers=layers)
    train_fn = os.path.join('datasets/190524', f'converted-{data_version}_train.csv')
    val_fn = os.path.join('datasets/190524', f'converted-{data_version}_val.csv')
    test_fn = os.path.join('datasets/190524', f'converted-{data_version}_test.csv')
    files = {
        'train': train_fn,
        'val': val_fn,
        'test': test_fn
    }

    for kind in ['val', 'test', 'train']:
        process_a_file(files[kind], bert_embedding, os.path.join('/mnt/sda4', f'{files[kind]}.layers:{layers}'))


def check_run_all(layers='-1'):
    data_version = 'v5'
    train_fn = os.path.join('datasets/190524', f'converted-{data_version}_train.csv')
    val_fn = os.path.join('datasets/190524', f'converted-{data_version}_val.csv')
    test_fn = os.path.join('datasets/190524', f'converted-{data_version}_test.csv')
    files = {
        'train': train_fn,
        'val': val_fn,
        'test': test_fn
    }
    for kind in ['val', 'test', 'train']:
        fn = files[kind]
        folder_out = os.path.join('/mnt/sda4', f'{files[kind]}.layers:{layers}')
        df = pd.read_csv(fn, sep='\t', names=['id', 'document', 'label'], header=None)
        source_ids = df['id'].values

        dest_ids = os.listdir(folder_out)
        dest_ids = [int(os.path.splitext(id_)[0]) for id_ in dest_ids]
        print(len(source_ids), len(dest_ids))
        assert set(source_ids) == set(dest_ids)


if __name__ == '__main__':
    # main()
    check_run_all()
