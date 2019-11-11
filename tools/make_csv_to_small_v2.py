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


def main(input_file):
    output_file = path.join(path.dirname(input_file), 'converted-v2_'+path.basename(input_file))

    with open(input_file, 'r', encoding='utf-8') as fi, \
      open(output_file, 'w', encoding='utf-8') as fo:
        for index, line in enumerate(fi):
            line = process_line(line)
            # if index == 0:
            #     write_all_col_names(line)
            #     index_text = line.index('text')
            #     index_status = line.index('status')
            #     print("index_text", index_text, "index_status", index_status)
            #
            # else:
            text = line[index_text]
            text = text.replace('\t', ' ')
            status = line[index_status]
            if status not in ['Unadopted', 'Adopted']:
                raise Exception('Wrong status: {}'.format(status))

            soup = BeautifulSoup(text)
            text = soup.get_text()
            text = text.replace('\n', ' ').replace('\t', ' ')
            text = clean_text(text)
            text = ' '.join(nltk.word_tokenize(text))

            print(text, status)

            fo.write("{}\t{}\t{}\n".format(line[0], text, status))
        print("count", index+1)
        print(output_file)


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
