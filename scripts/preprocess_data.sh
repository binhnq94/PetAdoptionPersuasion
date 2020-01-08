set -x
set -e


python -m tools.preprocess_data datasets/190524/train.csv
python -m tools.preprocess_data datasets/190524/val.csv
python -m tools.preprocess_data datasets/190524/test.csv
