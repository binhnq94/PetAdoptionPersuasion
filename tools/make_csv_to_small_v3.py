from os import path
from bs4 import BeautifulSoup
from tools.text_core.clean_data import clean_text, REPLACE_EXPS
import nltk
import pandas as pd


def write_all_col_names(col_names):
    with open('datasets/all_collumns_name.txt', 'w', encoding='utf-8') as fo:
        for name in col_names:
            print(name)
            fo.write("{}\n".format(name))


def process_line(line):
    line = line.strip()

    first_col, line = line.split(',', 1)
    line = line.rsplit(',', 106)
    line.insert(0, first_col)
    return line


index_text = 1
index_status = 13

# PREFIX = 'converted-v4'
PREFIX = 'converted-v6'


def main(input_file):
    output_file = path.join(path.dirname(input_file), PREFIX + '_' + path.basename(input_file))

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
        text = clean_text(text)

        # text = ' '.join(nltk.word_tokenize(text))
        sentences = nltk.sent_tokenize(text)
        # print(sentences)

        tokenized_sentences = []

        for sen in sentences:
            tokens = nltk.word_tokenize(sen, preserve_line=True)
            for t in tokens:
                if len(t) > 20:
                    # print(t)
                    pass
            len_tokens = len(tokens)

            if len_tokens > 180:
                list_split_sen = []
                split_size = 100
                number_split = len_tokens // split_size
                current_split_size = len_tokens // number_split
                print(len_tokens, number_split, current_split_size)

                for x in range(0, number_split - 1):
                    tokenized_sen = ' '.join(tokens[x * current_split_size:(x + 1) * current_split_size])
                    list_split_sen.append(tokenized_sen)

                tokenized_sen = ' '.join(tokens[(number_split - 1) * current_split_size:])
                list_split_sen.append(tokenized_sen)
                assert sum([len(x.split()) for x in list_split_sen]) == len_tokens
            else:
                tokenized_sen = ' '.join(tokens)
                # print(tokenized_sen)
                tokenized_sentences.append(tokenized_sen)
        # print(text, status)

        out_text = '<end>'.join(tokenized_sentences)
        # print(out_text)
        out_df = out_df.append({
            'id_': row._0,
            'text': out_text,
            'status': status
        }, ignore_index=True)

    out_df.to_csv(output_file, sep='\t', index=False, header=False)


if __name__ == '__main__':
    import sys

    main(sys.argv[1])
