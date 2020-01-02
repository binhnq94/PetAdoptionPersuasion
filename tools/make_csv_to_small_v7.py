from os import path
from bs4 import BeautifulSoup
import nltk
import pandas as pd
import os
import tempfile

index_text = 1
index_status = 13

# PREFIX = 'converted-v4'
PREFIX = 'v7'


def stanford_tokenize_sentencesplit(text):
    fd_input_temp, path_input_temp = tempfile.mkstemp(suffix='.txt', prefix='pet_datapoint')

    with open(path_input_temp, 'w', encoding='utf-8') as fo:
        fo.write(text)

    # os.system(f'bash scripts/stanford_scripts.sh {path_input_temp}')

    stream = os.popen(f'bash scripts/stanford_scripts.sh {path_input_temp}')
    output = stream.read()
    # print(output)

    with open(f'{path_input_temp}.conll', encoding='utf-8') as fi:
        output_text = fi.read()

    os.close(fd_input_temp)
    os.unlink(path_input_temp)
    os.unlink(f'{path_input_temp}.conll')

    return output_text.split('\n')


def tokenize_sentencesplit(input_file):
    """

    Args:
        input_file:

    Returns:

    """
    output_file = path.join(path.dirname(input_file), PREFIX + '_' + path.basename(input_file))
    if os.path.exists(output_file):
        raise ValueError(f'File `{output_file}` existed!')

    raw_df = pd.read_csv(input_file, header=None)
    out_df = pd.DataFrame(columns=['id_', 'text', 'status'])

    for row in raw_df.itertuples(index=False):
        text = row._1
        text = text.replace('\t', ' ')
        status = row._13
        if status not in ['Unadopted', 'Adopted']:
            raise Exception('Wrong status: {}'.format(status))

        soup = BeautifulSoup(text, features="html.parser")
        text = soup.get_text()
        text = text.replace('\n', ' ').replace('\t', ' ').replace('\\t', '')

        sentences = stanford_tokenize_sentencesplit(text)

        for sen in sentences:
            num_word = len(sen.split())
            if num_word > 55:
                print(row._0, num_word, sen)

        out_text = '<end>'.join(sentences)
        if out_text:
            # print(out_text)
            out_df = out_df.append({
                'id_': row._0,
                'text': out_text,
                'status': status
            }, ignore_index=True)

    out_df.to_csv(output_file, sep='\t', index=False, header=False)


if __name__ == '__main__':
    import sys

    tokenize_sentencesplit(sys.argv[1])
