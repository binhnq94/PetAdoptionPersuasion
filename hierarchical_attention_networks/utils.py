import os
import json
import pickle


def save_args(args, save_dir):
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as fo:
        json.dump(vars(args), fo, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, 'args.pickle'), 'wb') as fo:
        pickle.dump(args, fo)


def load_args(save_dir):
    with open(os.path.join(save_dir, 'config.json'), 'r', encoding='utf-8') as fi:
        print(f"config:\n{fi.read()}")

    with open(os.path.join(save_dir, 'args.pickle'), 'rb') as fi:
        return pickle.load(fi)
