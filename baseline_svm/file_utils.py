import os

CURRENT_DIR = os.path.dirname(__file__)


DATA_VERSION = 'v2'
DATA_DIR = 'datasets/190524'
train_path = f'converted-{DATA_VERSION}_train.csv'
validation_path = f'converted-{DATA_VERSION}_val.csv'
test_path = f'converted-{DATA_VERSION}_test.csv'
