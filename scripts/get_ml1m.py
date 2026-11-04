"""Скачивает MovieLens-1M и готовит data/ML-1M.txt и data/ML-1M_sample.txt.

Файлы ML-1M не хранятся в репозитории (лицензия GroupLens не разрешает их распространять),
поэтому каждый участник собирает их у себя. Результат детерминирован: MD5 должны совпасть
с docs/data.md.

Работает на Windows, Linux, macOS и в Colab; рабочая папка не важна:
    python scripts/get_ml1m.py
"""
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def run(script, *args):
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), *args], check=True)


def main():
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-1m.zip")
    if not os.path.exists(zip_path):
        print(f"скачиваю {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)

    run("prepare_ml1m.py", "--raw", zip_path)
    run("gen_neg_samples.py", "--data_name", "ML-1M", "--force")
    run("data_stats.py", "ML-1M")


if __name__ == "__main__":
    main()
