"""Near-duplicate audit between a dataset's test images and its train/valid images (64-bit difference hash).
  phash_audit.py DATASET_ROOT [--out FILE]   DATASET_ROOT holds {train,valid,test}/images
Reports, for each test image, the Hamming distance to its nearest train/valid image; distance <= 6 of 64 bits is
treated as a near-duplicate (consecutive frames of a fixed camera typically land there).
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image


def dhash(path, n=8):
    im = Image.open(path).convert('L').resize((n + 1, n), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out')
    a = ap.parse_args()
    split = {s: sorted(glob.glob(os.path.join(a.root, s, 'images', '*'))) for s in ('train', 'valid', 'test')}
    hashes = {s: np.stack([dhash(p) for p in fs]) if fs else np.zeros((0, 8), np.uint8) for s, fs in split.items()}
    ref = np.concatenate([hashes['train'], hashes['valid']])
    bits = np.unpackbits(ref, axis=1)
    dists = []
    for h in hashes['test']:
        d = (np.unpackbits(h)[None] != bits).sum(1)
        dists.append(int(d.min()))
    dists = np.array(dists)
    res = dict(test_images=len(dists), train_valid_images=len(ref),
               near_dup_le6=int((dists <= 6).sum()), near_dup_le10=int((dists <= 10).sum()),
               share_le6=round(float((dists <= 6).mean()), 3), median_min_distance=float(np.median(dists)),
               histogram={str(k): int((dists == k).sum()) for k in range(0, 16)})
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
