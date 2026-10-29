"""Causal leakage probe (reviewer design): one fixed test set, two training sets of equal size that differ only in
whether frames from the test videos are allowed in.

  test_half  = a random 50% of the frames of the 15 test videos (fixed, the common test set)
  M_clean    = yolo/train85 (8829 frames, training videos only; already built by make_leak.py)
  M_leak     = 8829 frames drawn at random from (training-video frames + the other 50% of test-video frames)
  -> yolo/{test_half,train_leak}/ + data_leak2.yaml, coco_testhalf/test (COCO GT of test_half), leak2_manifest.json
"""
import hashlib
import json
import os
import random

R = '/root/fish/data/brk'


def link_split(name, files, src_split):
    for sub in ('images', 'labels'):
        os.makedirs(f'{R}/yolo/{name}/{sub}', exist_ok=True)
    for n in files:
        stem = os.path.splitext(n)[0]
        for sub, fn in (('images', n), ('labels', stem + '.txt')):
            dst = f'{R}/yolo/{name}/{sub}/{fn}'
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(f'{R}/yolo/{src_split[n]}/{sub}/{fn}'), dst)


def main():
    test = sorted(os.listdir(f'{R}/yolo/test/images'))
    train = sorted(os.listdir(f'{R}/yolo/train/images'))
    rng = random.Random(2)
    shuffled = test[:]
    rng.shuffle(shuffled)
    half = sorted(shuffled[:len(test) // 2])
    other = sorted(shuffled[len(test) // 2:])
    n_train = len(os.listdir(f'{R}/yolo/train85/images'))
    pool = [(n, 'train') for n in train] + [(n, 'test') for n in other]
    leak = rng.sample(pool, n_train)
    src = {n: s for n, s in leak}
    src.update({n: 'test' for n in half})
    link_split('test_half', half, src)
    link_split('train_leak', [n for n, _ in leak], src)
    base = open(f'{R}/yolo/data.yaml').read()
    open(f'{R}/yolo/data_leak2.yaml', 'w').write(base.replace('train: train/images', 'train: train_leak/images'))

    gt = json.load(open(f'{R}/coco/test/_annotations.coco.json'))
    keep = set(half)
    images = [im for im in gt['images'] if im['file_name'] in keep]
    ids = {im['id'] for im in images}
    out = f'{R}/coco_testhalf/test'
    os.makedirs(out, exist_ok=True)
    for im in images:
        dst = f'{out}/{im["file_name"]}'
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(f'{R}/coco/test/{im["file_name"]}'), dst)
    json.dump(dict(images=images, annotations=[a for a in gt['annotations'] if a['image_id'] in ids],
                   categories=gt['categories']), open(f'{out}/_annotations.coco.json', 'w'))
    man = dict(test_half=len(half), train_leak=len(leak),
               train_leak_from_test_videos=sum(1 for _, s in leak if s == 'test'),
               test_half_sha256=hashlib.sha256('\n'.join(half).encode()).hexdigest())
    json.dump(man, open(f'{R}/leak2_manifest.json', 'w'), indent=1)
    print(man)


if __name__ == '__main__':
    main()
