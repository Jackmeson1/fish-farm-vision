"""Brackish (Roboflow YOLO export) -> video-disjoint train/val/test split, in YOLO and COCO layouts.

The Roboflow split samples frames at random, so neighbouring frames of one video land in train and test. This script
groups frames by source video, reports that leakage, and re-splits whole videos (seed chosen so every class keeps
roughly its share in test). Outputs (symlinks, no copies):
  OUT/yolo/{train,val,test}/{images,labels} + OUT/yolo/data.yaml (+ train_1pct/train_10pct subsets)
  OUT/coco/{train,valid,test}/ images + _annotations.coco.json   (RF-DETR layout; also used by YOLOX and the evaluator)
"""
import argparse
import collections
import json
import os
import random
import re

from PIL import Image

NAMES = ['crab', 'fish', 'jellyfish', 'shrimp', 'small_fish', 'starfish']
FRAME_RE = re.compile(r'^(?P<video>.+)-(?P<frame>\d{4})_(?:jpg|png)\.rf\.[0-9a-f]+\.(?:jpg|png)$')


def scan(src):
    items = []
    for split in ('train', 'valid', 'test'):
        base = os.path.join(src, 'data', split)
        dirs = [base] if os.path.isdir(os.path.join(base, 'images')) else \
            [os.path.join(base, d) for d in sorted(os.listdir(base))
             if os.path.isdir(os.path.join(base, d, 'images'))]  # train/ is sharded: shard_*/images
        for d in dirs:
            items.extend(scan_dir(d, split))
    return items


def scan_dir(d, split):
    items = []
    img_dir, lbl_dir = os.path.join(d, 'images'), os.path.join(d, 'labels')
    for fn in sorted(os.listdir(img_dir)):
        if not fn.endswith(('.jpg', '.png')):
            continue  # e.g. metadata.jsonl
        m = FRAME_RE.match(fn)
        if not m:
            raise SystemExit(f'unparsed name: {fn}')
        lbl = os.path.join(lbl_dir, os.path.splitext(fn)[0] + '.txt')
        boxes = []
        if os.path.exists(lbl):
            for line in open(lbl):
                p = line.split()
                if len(p) == 5:
                    boxes.append((int(p[0]), *map(float, p[1:])))
        items.append(dict(file=os.path.join(img_dir, fn), lbl=lbl, name=fn, orig_split=split,
                          video=m['video'], frame=int(m['frame']), boxes=boxes))
    return items


def class_counts(items):
    c = collections.Counter()
    for it in items:
        for b in it['boxes']:
            c[b[0]] += 1
    return c


def split_videos(items, seed, fracs=(0.7, 0.1, 0.2)):
    by_vid = collections.defaultdict(list)
    for it in items:
        by_vid[it['video']].append(it)
    vids = sorted(by_vid)
    random.Random(seed).shuffle(vids)
    total = len(items)
    out, acc, k = {'train': [], 'val': [], 'test': []}, 0, 0
    bounds = [fracs[0] * total, (fracs[0] + fracs[1]) * total]
    for v in vids:
        name = 'train' if acc < bounds[0] else ('val' if acc < bounds[1] else 'test')
        out[name].extend(by_vid[v])
        acc += len(by_vid[v])
    return out


def score(split, total_counts):
    """Worst relative deviation of any class's test/val share from its target."""
    worst = 0.0
    for name, target in (('test', 0.2), ('val', 0.1)):
        c = class_counts(split[name])
        for k, n in total_counts.items():
            worst = max(worst, abs(c[k] / n - target) / target)
    return worst


def link(src, dst):
    if not os.path.lexists(dst):
        os.symlink(src, dst)


def write_yolo(out, split):
    root = os.path.join(out, 'yolo')
    for name, items in split.items():
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(root, name, sub), exist_ok=True)
        for it in items:
            link(it['file'], os.path.join(root, name, 'images', it['name']))
            with open(os.path.join(root, name, 'labels', os.path.splitext(it['name'])[0] + '.txt'), 'w') as fh:
                for b in it['boxes']:
                    fh.write('%d %.6f %.6f %.6f %.6f\n' % b)
    yaml = 'path: %s\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n' % root
    yaml += ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES))
    open(os.path.join(root, 'data.yaml'), 'w').write(yaml)


def write_coco(root, split, size_cache):
    for name, items in split.items():
        d = os.path.join(root, 'valid' if name == 'val' else name)
        os.makedirs(d, exist_ok=True)
        images, anns = [], []
        for i, it in enumerate(sorted(items, key=lambda x: x['name']), start=1):
            w, h = size_cache[it['file']]
            link(it['file'], os.path.join(d, it['name']))
            images.append(dict(id=i, file_name=it['name'], width=w, height=h))
            for c, cx, cy, bw, bh in it['boxes']:
                x, y, ww, hh = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
                anns.append(dict(id=len(anns) + 1, image_id=i, category_id=c + 1, bbox=[x, y, ww, hh],
                                 area=ww * hh, iscrowd=0))
        cats = [dict(id=i + 1, name=n, supercategory='animal') for i, n in enumerate(NAMES)]
        json.dump(dict(images=images, annotations=anns, categories=cats),
                  open(os.path.join(d, '_annotations.coco.json'), 'w'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/root/fish/data/brk_git')
    ap.add_argument('--out', default='/root/fish/data/brk')
    ap.add_argument('--seeds', type=int, default=200)
    args = ap.parse_args()

    items = scan(args.src)
    total = class_counts(items)
    vids = collections.defaultdict(set)
    for it in items:
        vids[it['orig_split']].add(it['video'])
    report = dict(images=len(items), videos=len({it['video'] for it in items}),
                  class_counts={NAMES[k]: v for k, v in sorted(total.items())},
                  roboflow_split_videos={k: len(v) for k, v in vids.items()},
                  test_videos_also_in_train=len(vids['test'] & vids['train']),
                  test_images_whose_video_in_train=sum(1 for it in items if it['orig_split'] == 'test'
                                                       and it['video'] in vids['train']),
                  roboflow_test_images=sum(1 for it in items if it['orig_split'] == 'test'))

    best = min(range(args.seeds), key=lambda s: score(split_videos(items, s), total))
    split = split_videos(items, best)
    report['chosen_seed'] = best
    report['worst_class_share_deviation'] = round(score(split, total), 3)
    report['new_split'] = {k: dict(images=len(v), videos=len({i['video'] for i in v}),
                                   classes={NAMES[c]: n for c, n in sorted(class_counts(v).items())})
                           for k, v in split.items()}

    rng = random.Random(0)
    train = sorted(split['train'], key=lambda x: x['name'])
    for pct in (1, 10):
        split[f'train_{pct}pct'] = rng.sample(train, max(1, len(train) * pct // 100))
    report['subsets'] = {k: len(split[k]) for k in ('train_1pct', 'train_10pct')}

    size_cache = {}
    for it in items:
        with Image.open(it['file']) as im:
            size_cache[it['file']] = im.size
    report['image_sizes'] = dict(collections.Counter('%dx%d' % s for s in size_cache.values()))

    write_yolo(args.out, split)
    write_coco(os.path.join(args.out, 'coco'), {k: v for k, v in split.items() if k in ('train', 'val', 'test')},
               size_cache)
    for pct in (1, 10):  # COCO layouts for the low-label RF-DETR runs share valid/test with the full split
        sub = os.path.join(args.out, f'coco_{pct}pct')
        write_coco(sub, {'train': split[f'train_{pct}pct']}, size_cache)
        for s in ('valid', 'test'):
            link(os.path.join(args.out, 'coco', s), os.path.join(sub, s))
    # yaml variants for the low-label YOLO runs
    base = open(os.path.join(args.out, 'yolo', 'data.yaml')).read()
    for pct in (1, 10):
        open(os.path.join(args.out, 'yolo', f'data_{pct}pct.yaml'), 'w').write(
            base.replace('train: train/images', f'train: train_{pct}pct/images'))
    json.dump(report, open(os.path.join(args.out, 'split_report.json'), 'w'), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
