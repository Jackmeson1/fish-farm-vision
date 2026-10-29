"""Fetch commercially usable museum specimen photos from Fish-Vista (dead fish lying on their side, white background).

Only images whose own license is CC0 or CC BY are kept (the user cares about commercial use; Fish-Vista's other images
are NC/SA/ND or 'usage conditions apply'). Sampled stratified by family for shape diversity.
  -> data/fishvista/Images/... + data/fishvista/selected.csv (filename, family, species, license, original_url)
"""
import argparse
import csv
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

R = '/root/fish/data/fishvista'
OK = {'CC0 1.0', 'CC0', 'CC BY', 'CC BY 4.0'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    meta = hf_hub_download('imageomics/fish-vista', 'classification_train.csv', repo_type='dataset', local_dir=R)
    rows = [r for r in csv.DictReader(open(meta, encoding='utf-8')) if r['license'].strip() in OK]
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r['family']].append(r)
    rng = random.Random(0)
    for v in by_fam.values():
        rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < min(a.n, len(rows)):  # round-robin over families
        for v in by_fam.values():
            if i < len(v) and len(picked) < a.n:
                picked.append(v[i])
        i += 1

    def get(r):
        try:
            hf_hub_download('imageomics/fish-vista', r['file_name'], repo_type='dataset', local_dir=R)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(get, picked))
    with open(f'{R}/selected.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['file_name', 'family', 'species', 'license', 'original_url'])
        for r, good in zip(picked, ok):
            if good:
                w.writerow([r['file_name'], r['family'], r['standardized_species'], r['license'], r['original_url']])
    print(f'eligible {len(rows)}, families {len(by_fam)}, picked {len(picked)}, downloaded {sum(ok)}')


if __name__ == '__main__':
    main()
