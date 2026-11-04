"""Проверка формата и статистика датасетов (печатает таблицу в Markdown для docs/data.md).

Проверяется то, на что опирается код FMLP-Rec:
  * пользователи пронумерованы 1..N подряд, строки в data/<name>_sample.txt идут в том же порядке;
  * id объектов — от 1 до max_item без пропусков (0 — паддинг);
  * в каждой истории не меньше 3 объектов (последний — тест, предпоследний — валидация);
  * в каждой строке негативов 99 разных объектов, ни один не встречается в истории пользователя.
MD5 файлов позволяет убедиться, что у всех участников одинаковые данные.

Пример:
    python scripts/data_stats.py Beauty ML-1M
"""
import argparse
import hashlib
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # корень репозитория


def md5(path):
    # концы строк приводятся к \n: на Windows git может переписать их в \r\n,
    # а содержимое данных от этого не меняется
    with open(path, "rb") as f:
        return hashlib.md5(f.read().replace(b"\r\n", b"\n")).hexdigest()


def check(name, data_dir, num_neg):
    data_path = os.path.join(data_dir, f"{name}.txt")
    sample_path = os.path.join(data_dir, f"{name}_sample.txt")
    seqs, users = [], []
    with open(data_path) as f:
        for line in f:
            p = line.split()
            users.append(int(p[0]))
            seqs.append([int(x) for x in p[1:]])

    errors = []
    if users != list(range(1, len(users) + 1)):
        errors.append("пользователи не пронумерованы 1..N подряд")
    items = {i for s in seqs for i in s}
    max_item = max(items)
    if min(items) < 1:
        errors.append("есть id объекта < 1")
    if len(items) != max_item:
        errors.append(f"id объектов с пропусками: {len(items)} разных, max = {max_item}")
    short = sum(len(s) < 3 for s in seqs)
    if short:
        errors.append(f"{short} пользователей с историей короче 3")

    with open(sample_path) as f:
        samples = [line.split() for line in f]
    if len(samples) != len(seqs):
        errors.append(f"строк в sample-файле {len(samples)}, а пользователей {len(seqs)}")
    for u, s, neg in zip(users, seqs, samples):
        negs = [int(x) for x in neg[1:]]
        if int(neg[0]) != u:
            errors.append(f"порядок строк в sample-файле не совпадает (пользователь {u})")
            break
        if len(negs) != num_neg or len(set(negs)) != num_neg:
            errors.append(f"у пользователя {u} не {num_neg} разных негативов")
            break
        if set(negs) & set(s):
            errors.append(f"у пользователя {u} негатив из истории")
            break

    lens = [len(s) for s in seqs]
    n_inter = sum(lens)
    stats = {
        "name": name,
        "users": len(seqs),
        "items": max_item,
        "inter": n_inter,
        "avg": n_inter / len(seqs),
        "median": statistics.median(lens),
        "min": min(lens),
        "max": max(lens),
        "density": n_inter / (len(seqs) * max_item),
        "md5_data": md5(data_path),
        "md5_sample": md5(sample_path),
    }
    return stats, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--num_neg", type=int, default=99)
    args = ap.parse_args()

    rows, failed = [], False
    for name in args.names:
        st, errors = check(name, args.data_dir, args.num_neg)
        rows.append(st)
        print(f"{name}: " + ("OK" if not errors else "ОШИБКИ: " + "; ".join(errors)))
        failed |= bool(errors)

    print()
    print("| Датасет | Пользователи | Объекты | Взаимодействия | Ср. длина | Медиана | Мин | Макс | Плотность |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['users']:,} | {r['items']:,} | {r['inter']:,} | {r['avg']:.1f} | "
              f"{r['median']:g} | {r['min']} | {r['max']:,} | {r['density']:.2%} |".replace(",", " "))
    print()
    print("| Файл | MD5 |")
    print("|---|---|")
    for r in rows:
        print(f"| {r['name']}.txt | `{r['md5_data']}` |")
        print(f"| {r['name']}_sample.txt | `{r['md5_sample']}` |")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
