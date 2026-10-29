"""Leakage probe: hold out a random 15% of the *training videos'* frames, so one model can be scored on frames from
videos it has seen (the random-frame protocol) and on the 15 unseen test videos (the video-disjoint protocol).
  -> data/brk/yolo/{train85,seen15}/ + data_leak.yaml, data/brk/coco_seen/test (COCO GT for the seen-video frames)
"""
import json
import os
import random

R = '/root/fish/data/brk'


def main():
    names = sorted(os.listdir(f'{R}/yolo/train/images'))
    random.Random(1).shuffle(names)
    k = len(names) * 15 // 100
    parts = {'seen15': names[:k], 'train85': names[k:]}
    for part, ns in parts.items():
        for sub in ('images', 'labels'):
            os.makedirs(f'{R}/yolo/{part}/{sub}', exist_ok=True)
        for n in ns:
            stem = os.path.splitext(n)[0]
            for sub, fn in (('images', n), ('labels', stem + '.txt')):
                dst = f'{R}/yolo/{part}/{sub}/{fn}'
                if not os.path.lexists(dst):
                    os.symlink(os.path.realpath(f'{R}/yolo/train/{sub}/{fn}'), dst)
    base = open(f'{R}/yolo/data.yaml').read()
    open(f'{R}/yolo/data_leak.yaml', 'w').write(base.replace('train: train/images', 'train: train85/images'))

    gt = json.load(open(f'{R}/coco/train/_annotations.coco.json'))
    keep = set(parts['seen15'])
    images = [im for im in gt['images'] if im['file_name'] in keep]
    ids = {im['id'] for im in images}
    out = f'{R}/coco_seen/test'
    os.makedirs(out, exist_ok=True)
    for im in images:
        dst = f'{out}/{im["file_name"]}'
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(f'{R}/coco/train/{im["file_name"]}'), dst)
    json.dump(dict(images=images, annotations=[a for a in gt['annotations'] if a['image_id'] in ids],
                   categories=gt['categories']), open(f'{out}/_annotations.coco.json', 'w'))
    print({p: len(v) for p, v in parts.items()})


if __name__ == '__main__':
    main()
