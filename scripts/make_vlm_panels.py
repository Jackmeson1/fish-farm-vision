"""Dead-vs-live judging set for vision-language models, from RAS-MortDB's session-blocked test split (ground-truth
boxes, so this isolates *understanding death* from localisation).

Each sample is one panel: [whole frame with the fish boxed | close-up | same region in the neighbouring capture
(<= 120 min away; grey if none)]. Labels go to a separate file so a judge can be scored blind.
  -> runs/vlm/panels/<id>.jpg, runs/vlm/grids/grid_XX.jpg (6 panels each, numbered), runs/vlm/labels.json,
     runs/vlm/samples.json (id -> source image, box; no label)
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

R = '/root/fish'
GT = f'{R}/data/rasmort/blocked_coco/test'
NAME_RE = re.compile(r'^tl_(?P<s>\d+)_(?P<f>\d+)_(?P<d>\d{8})_(?P<t>\d{6})_')
S = 256  # tile size


def tile(im, box=None, crop=None):
    if crop:
        im = im.crop(crop)
    im = im.copy()
    im.thumbnail((S, S))
    t = Image.new('RGB', (S, S), (40, 40, 40))
    t.paste(im, ((S - im.width) // 2, (S - im.height) // 2))
    if box:  # box in original coords, drawn on the downscaled full frame
        sx = im.width / box[4]
        ox, oy = (S - im.width) // 2, (S - im.height) // 2
        ImageDraw.Draw(t).rectangle([ox + box[0] * sx, oy + box[1] * sx, ox + box[2] * sx, oy + box[3] * sx],
                                    outline=(255, 0, 0), width=2)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=60)
    a = ap.parse_args()
    gt = json.load(open(f'{GT}/_annotations.coco.json'))
    ims = {im['id']: im for im in gt['images']}
    meta = {}
    by_s = defaultdict(list)
    for im in gt['images']:
        m = NAME_RE.match(im['file_name'])
        t = datetime.strptime(m['d'] + m['t'], '%Y%m%d%H%M%S')
        meta[im['id']] = t
        by_s[m['s']].append((t, im['id']))
    nbr = {}
    for v in by_s.values():
        v.sort()
        for k, (t, iid) in enumerate(v):
            cands = [(abs((tj - t).total_seconds()), j) for tj, j in v[max(0, k - 1):k] + v[k + 1:k + 2]]
            cands = [c for c in cands if c[0] <= 7200]
            nbr[iid] = min(cands)[1] if cands else None
    rng = random.Random(0)
    pools = {c: [x for x in gt['annotations'] if x['category_id'] == c] for c in (1, 2)}
    picked = []
    for c in (2, 1):
        rng.shuffle(pools[c])
        picked += [(x, 'dead' if c == 2 else 'live') for x in pools[c][:a.per_class]]
    rng.shuffle(picked)
    out = f'{R}/runs/vlm'
    os.makedirs(f'{out}/panels', exist_ok=True)
    os.makedirs(f'{out}/grids', exist_ok=True)
    labels, samples, panels = {}, {}, []
    for k, (ann, lab) in enumerate(picked):
        sid = f'S{k:03d}'
        im_rec = ims[ann['image_id']]
        im = Image.open(f'{GT}/{im_rec["file_name"]}').convert('RGB')
        x, y, w, h = ann['bbox']
        px, py = 0.3 * w, 0.3 * h
        crop = (max(0, x - px), max(0, y - py), min(im.width, x + w + px), min(im.height, y + h + py))
        full = tile(im, box=(x, y, x + w, y + h, im.width))
        close = tile(im, crop=crop)
        j = nbr.get(ann['image_id'])
        if j is not None:
            nim = Image.open(f'{GT}/{ims[j]["file_name"]}').convert('RGB')
            neigh = tile(nim, crop=crop)
            gap = round(abs((meta[j] - meta[ann['image_id']]).total_seconds()) / 60)
        else:
            neigh, gap = Image.new('RGB', (S, S), (128, 128, 128)), None
        p = Image.new('RGB', (3 * S, S))
        for i, t in enumerate((full, close, neigh)):
            p.paste(t, (i * S, 0))
        p.save(f'{out}/panels/{sid}.jpg', quality=90)
        panels.append((sid, p))
        labels[sid] = lab
        samples[sid] = dict(image=im_rec['file_name'], bbox=ann['bbox'], neighbour_gap_min=gap)
    font = ImageFont.load_default()
    for g in range(0, len(panels), 6):
        grid = Image.new('RGB', (3 * S + 70, 6 * S), 'white')
        for r, (sid, p) in enumerate(panels[g:g + 6]):
            grid.paste(p, (70, r * S))
            ImageDraw.Draw(grid).text((6, r * S + S // 2 - 6), sid, fill='black', font=font)
        grid.save(f'{out}/grids/grid_{g // 6:02d}.jpg', quality=88)
    json.dump(labels, open(f'{out}/labels.json', 'w'))
    json.dump(samples, open(f'{out}/samples.json', 'w'), indent=1)
    print('samples', len(panels), 'grids', (len(panels) + 5) // 6,
          'with neighbour', sum(1 for s in samples.values() if s['neighbour_gap_min'] is not None))


if __name__ == '__main__':
    main()
