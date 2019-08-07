import numpy as np
import torch
import os
import pickle

DATA_VERSION = 'v2'
DATA_DIR = 'datasets/190524'
train_path = f'converted-{DATA_VERSION}_train.csv'
validation_path = f'converted-{DATA_VERSION}_val.csv'
test_path = f'converted-{DATA_VERSION}_test.csv'


def get_w2v(embedding_length=200):
    text_field_file = f'baseline_bilstm/models/TEXT-databalance-{DATA_VERSION}.glove-{embedding_length}.pt'
    TEXT = torch.load(text_field_file)

    return TEXT.vocab


VOCAB = get_w2v()


def text2vector(text):
    list_vector = []
    for word in text.split(' '):
        vector = VOCAB.vectors[VOCAB.stoi[word]].numpy()
        list_vector.append(vector)
    return np.mean(list_vector, axis=0)


def read_data(fpath: str):
    data = []
    with open(fpath, encoding='utf-8') as fi:
        for line in fi:
            line = line.strip('\n')
            if line:
                _id, sentence, label = line.split('\t')
                data.append(sentence)
    return data


def save_vectors(fpath):
    out_path = f"{fpath}.w2v"
    fpath = os.path.join(DATA_DIR, fpath)
    data = read_data(fpath)
    vectors = []
    for sen in data:
        vector = text2vector(sen)
        vectors.append(vector)

    vectors = np.vstack(vectors)
    print(vectors.shape)
    np.save(out_path, vectors)


def convert_data2vectors():
    save_vectors(train_path)
    save_vectors(validation_path)
    save_vectors(test_path)


if __name__ == '__main__':
    # vocab = get_w2v()
    # print(vocab.itos)
    # print(vocab.stoi)
    # print(vocab.)
    # a = text2vector("`` You can fill out an adoption application online on our official website.My , name is "
    #                 "Caden.03/05/2013 It 's taken me a long time to get well , but I made it with all the tender "
    #                 "loving care from my foster mom and dad and Dr. Yaro 's who took care of my medical needs . "
    #                 "I have put on some weight , my bloodwork is good , I am neutered and I am ready to go . "
    #                 "I am on my way to my new home in the next week . I am going to live with Cheyenne she was "
    #                 "adopted through the rescue and I am staying in the St. Charles area everyone is happy about "
    #                 "that.Thank-you everyone who sent donations to help with Caden 's medical needs and cared "
    #                 "about him , came to visit him and sent e-mails asking about him . They Dobie Boy is on his "
    #                 "way to his forever home and made against all odds . 01/15/2012 I am finishing up my heartworm "
    #               "treatments today and tomorrow . 11/17/2012 I am looking better and feeling better . I have gained "
    #               "8 pounds so I feel better . My liver counts are n't great and I have been fighting to get rid of "
    #               "all these hookworms and tape worms . I am heavily infested with micro-filariae from the heartworm "
    #               ". Keep your fingers crossed for me , we will do another blood panel in four weeks and see what is "
    #                 "going on . I have the best foster mom and dad they treat me like a King and I just love to give "
    #                 "them kisses . I am a Blue Doberman even though it does n't look like it in the pictures and Dr. "
    #                 "Yaro 's thinks I am only about 3 years old . 11/03/2012 : I arrived this afternoon everyone "
    #                 "treated me so Special on the transport ! Alot of people gave up their day so I could get to "
    #                 "Rescue . Beth made me a coat for the journey up here and I look good , it covers up my skinny "
    #                 "body . They said I was a perfect Doberman and liked to give them lot 's of kisses . "
    #                 "There are good people and I was so lucky to find them ! ``")
    # print(a.shape)
    # print(a)
    convert_data2vectors()
