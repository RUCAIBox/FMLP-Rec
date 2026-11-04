#!/usr/bin/env bash
# Скачивает MovieLens-1M и готовит data/ML-1M.txt и data/ML-1M_sample.txt.
# Файлы ML-1M не хранятся в репозитории (лицензия GroupLens не разрешает их распространять),
# поэтому каждый участник собирает их у себя этой командой. Результат детерминирован:
# MD5 должен совпасть с docs/data.md.
#
# Запуск из корня репозитория:  bash scripts/get_ml1m.sh
set -euo pipefail

mkdir -p data
if [ ! -f data/ml-1m.zip ]; then
    curl -L -o data/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
fi

python scripts/prepare_ml1m.py --raw data/ml-1m.zip
python scripts/gen_neg_samples.py --data_name ML-1M --force
python scripts/data_stats.py ML-1M
