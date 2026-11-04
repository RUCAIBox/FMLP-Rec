"""Генерация файла негативов для оценки FMLP-Rec (протокол «1 позитив + 99 негативов»).

Для каждого пользователя из data/<name>.txt выбирается 99 разных объектов, которых нет
в его истории, равномерно из всех объектов 1..max_item. Строки пишутся в том же порядке,
что и в файле данных: "user neg1 ... neg99" → data/<name>_sample.txt.
Так же устроен готовый Beauty_sample.txt из репозитория.

Скрипт подходит для любого датасета в этом формате, в том числе зашумлённых версий.

Пример:
    python scripts/gen_neg_samples.py --data_name ML-1M
"""
import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # корень репозитория


def read_seqs(path):
    users, seqs = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            users.append(parts[0])
            seqs.append([int(x) for x in parts[1:]])
    return users, seqs


def sample_negatives(history, max_item, num_neg, rng):
    allowed = max_item - len(history)
    if allowed < num_neg:
        raise ValueError(f"у пользователя слишком длинная история: доступно {allowed} негативов")
    negs = set()
    while len(negs) < num_neg:
        for x in rng.integers(1, max_item + 1, size=2 * num_neg):
            x = int(x)
            if x not in history and x not in negs:
                negs.add(x)
                if len(negs) == num_neg:
                    break
    return sorted(negs, key=lambda _: rng.random())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_name", required=True)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--num_neg", type=int, default=99)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="перезаписать существующий файл")
    args = ap.parse_args()

    data_path = os.path.join(args.data_dir, f"{args.data_name}.txt")
    out_path = os.path.join(args.data_dir, f"{args.data_name}_sample.txt")
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(f"{out_path} уже существует; добавьте --force, чтобы перезаписать")

    users, seqs = read_seqs(data_path)
    max_item = max(max(s) for s in seqs)
    rng = np.random.default_rng(args.seed)

    with open(out_path, "w", newline="\n") as f:
        for u, s in zip(users, seqs):
            negs = sample_negatives(set(s), max_item, args.num_neg, rng)
            f.write(f"{u} " + " ".join(map(str, negs)) + "\n")

    print(f"{len(users):,} пользователей × {args.num_neg} негативов → {out_path}")


if __name__ == "__main__":
    main()
