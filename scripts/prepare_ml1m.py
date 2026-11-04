"""Подготовка MovieLens-1M в формате FMLP-Rec.

Вход:  ml-1m.zip или распакованный ratings.dat (UserID::MovieID::Rating::Timestamp)
Выход: data/ML-1M.txt           — строка на пользователя: "user item1 item2 ..." (по времени)
       data/ML-1M_item_map.csv  — соответствие новых id фильмов исходным MovieID

Решения по данным:
  * все оценки считаются неявной обратной связью (взаимодействием), как в SASRec / FMLP-Rec;
  * итеративный 5-core: убираем пользователей и фильмы, у которых меньше 5 взаимодействий,
    пока таких не останется;
  * история пользователя сортируется по (timestamp, MovieID) — порядок при одинаковом времени
    фиксирован, поэтому результат не зависит от порядка строк в исходном файле;
  * пользователи нумеруются с 1 по возрастанию исходного UserID, фильмы — с 1 в порядке первого
    появления (0 зарезервирован под паддинг).

Пример:
    python scripts/prepare_ml1m.py
(пути по умолчанию считаются от корня репозитория, рабочая папка не важна)
"""
import argparse
import csv
import io
import os
import zipfile
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # корень репозитория


def read_ratings(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith("ratings.dat"))
            text = io.TextIOWrapper(zf.open(name), encoding="latin-1")
            lines = text.read().splitlines()
    else:
        if os.path.isdir(path):
            path = os.path.join(path, "ratings.dat")
        with open(path, encoding="latin-1") as f:
            lines = f.read().splitlines()
    rows = []
    for line in lines:
        if not line.strip():
            continue
        u, i, r, t = line.split("::")
        rows.append((int(u), int(i), float(r), int(t)))
    return rows


def k_core(rows, k):
    while True:
        uc = Counter(r[0] for r in rows)
        ic = Counter(r[1] for r in rows)
        kept = [r for r in rows if uc[r[0]] >= k and ic[r[1]] >= k]
        if len(kept) == len(rows):
            return kept
        rows = kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "ml-1m.zip"), help="ml-1m.zip, папка ml-1m или ratings.dat")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--name", default="ML-1M")
    ap.add_argument("--min_count", type=int, default=5, help="порог k-core")
    ap.add_argument("--min_rating", type=float, default=0.0, help="оставить оценки >= порога (0 = все)")
    args = ap.parse_args()

    rows = read_ratings(args.raw)
    print(f"прочитано оценок: {len(rows):,}")
    rows = [r for r in rows if r[2] >= args.min_rating]
    rows = k_core(rows, args.min_count)
    print(f"после фильтров:   {len(rows):,}")

    hist = defaultdict(list)
    for u, i, _, t in rows:
        hist[u].append((t, i))

    item_map = {}
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.txt")
    with open(out_path, "w", newline="\n") as f:
        for new_u, u in enumerate(sorted(hist), start=1):
            seq = []
            for _, i in sorted(hist[u]):
                if i not in item_map:
                    item_map[i] = len(item_map) + 1
                seq.append(item_map[i])
            f.write(f"{new_u} " + " ".join(map(str, seq)) + "\n")

    map_path = os.path.join(args.out_dir, f"{args.name}_item_map.csv")
    with open(map_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "movie_id"])
        for movie_id, item_id in sorted(item_map.items(), key=lambda x: x[1]):
            w.writerow([item_id, movie_id])

    print(f"пользователей: {len(hist):,}, фильмов: {len(item_map):,}")
    print(f"записано: {out_path}, {map_path}")


if __name__ == "__main__":
    main()
