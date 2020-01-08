from os import path
from bs4 import BeautifulSoup
from tools.text_core.clean_data import clean_text
import nltk


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


PREFIX = 'converted-v5.5'


def main(input_file):
    output_file = path.join(path.dirname(input_file), PREFIX + '_' + path.basename(input_file))

    with open(input_file, 'r', encoding='utf-8') as fi, \
      open(output_file, 'w', encoding='utf-8') as fo:
        for index, line in enumerate(fi):
            line = process_line(line)
            text = line[index_text]
            text = text.replace('\t', ' ')
            status = line[index_status]
            if status not in ['Unadopted', 'Adopted']:
                raise Exception('Wrong status: {}'.format(status))

            soup = BeautifulSoup(text)
            text = soup.get_text()
            text = text.replace('\n', ' ').replace('\t', ' ')
            text = clean_text(text)

            # text = ' '.join(nltk.word_tokenize(text))
            sentences = nltk.sent_tokenize(text)
            # print(sentences)

            tokenized_sentences = []

            for sen in sentences:
                tokenized_sen = ' '.join(nltk.word_tokenize(sen, preserve_line=True))
                # print(tokenized_sen)
                tokenized_sentences.append(tokenized_sen)
            # print(text, status)

            out_text = '<end>'.join(tokenized_sentences)
            print(out_text)

            fo.write("{}\t{}\t{}\n".format(line[0], out_text, status))
        print(index)


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
