"""Video dead-fish pipeline: death is a persistent state, so a single-frame 'looks dead' candidate is only confirmed if
the object also shows no self-motion over time (fish physiology: no tail, fin or gill-cover movement, only passive sway).

  video_dead.py synth  --split {train,test}   insert dead specimens into real Brackish video sequences
                                              (train videos <- train specimens, test videos <- test specimens)
  video_dead.py detect --split S --weights W  per-frame candidates from the single-frame model (both classes)
  video_dead.py verify                        tracks + features; rules calibrated on train videos, scored on test videos

Synthetic dead fish are rendered once per track (same specimen, colour loss and water parameters for the whole clip)
and only sway passively (sinusoidal drift of <= 2 px and <= 2 degrees, period 2-4 s, plus sensor noise). Real live fish,
crabs, starfish and rocks in the clips are untouched and are where false alarms are counted.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_dead import H, W, specimen_cutouts  # noqa: E402

R = '/root/fish'
OUT = f'{R}/data/vid_dead'
FPS = 15.0  # Brackish clips: ~11 s at 30 fps, exported at about every other frame
FRAME_RE = re.compile(r'^(?P<video>.+)-(?P<frame>\d{4})_(?:jpg|png)\.rf\.[0-9a-f]+\.(?:jpg|png)$')


def sequences(split):
    gt = json.load(open(f'{R}/data/brk/coco/{split}/_annotations.coco.json'))
    by_img = defaultdict(list)
    for a in gt['annotations']:
        by_img[a['image_id']].append(a)
    vids = defaultdict(list)
    for im in gt['images']:
        m = FRAME_RE.match(im['file_name'])
        vids[m['video']].append((int(m['frame']), im, by_img.get(im['id'], [])))
    for v in vids.values():
        v.sort(key=lambda x: x[0])
    return vids


def prepare(cut, rng):
    img, m = cut
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.4, 1.0)
    img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    pale = rng.uniform(0, 0.35)
    img = img * (1 - pale) + 200 * pale
    if rng.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    if rng.random() < 0.3:
        img, m = img[::-1], m[::-1]
    L = rng.uniform(0.04, 0.18) * W
    s = L / max(img.shape[:2])
    img = cv2.resize(np.ascontiguousarray(img), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(np.ascontiguousarray(m), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    diag = int(np.hypot(*m.shape)) + 8
    return dict(img=img, m=m, diag=diag, ang=rng.uniform(0, 360),
                x=rng.randint(0, W - diag), y=int(rng.triangular(H * 0.3, H - diag, H - diag)),
                z=rng.uniform(0.3, 2.5), bB=rng.uniform(0.25, 0.8),
                bD=np.array([rng.uniform(0.15, 0.45), rng.uniform(0.1, 0.35), rng.uniform(0.5, 1.2)], np.float32),
                period=rng.uniform(2, 4) * FPS, amp=rng.uniform(0.3, 2.0), aamp=rng.uniform(0.2, 2.0),
                phase=rng.uniform(0, 2 * np.pi))


def render(bg, o, t, rng):
    sway = np.sin(2 * np.pi * t / o['period'] + o['phase'])
    ang = o['ang'] + o['aamp'] * sway
    dx, dy = o['amp'] * sway, 0.5 * o['amp'] * np.cos(2 * np.pi * t / o['period'] + o['phase'])
    h, w = o['m'].shape
    d = o['diag']
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[0, 2] += d / 2 - w / 2
    M[1, 2] += d / 2 - h / 2
    img = cv2.warpAffine(o['img'], M, (d, d), flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(o['m'], M, (d, d), flags=cv2.INTER_NEAREST)
    x, y = int(round(o['x'] + dx)), int(round(o['y'] + dy))
    x, y = min(max(0, x), W - d), min(max(0, y), H - d)
    binf = np.median(bg[: H // 4].reshape(-1, 3), axis=0).astype(np.float32)
    img = img * np.exp(-o['bD'] * o['z']) + binf * (1 - np.exp(-o['bB'] * o['z']))
    k = max(1, int(round(o['z'])))
    img = cv2.GaussianBlur(img, (2 * k + 1, 2 * k + 1), 0) + np.random.default_rng(rng.getrandbits(32)).normal(0, 2.0, img.shape)
    a = cv2.GaussianBlur(m.astype(np.float32) / 255, (5, 5), 0)[..., None]
    roi = bg[y:y + d, x:x + d].astype(np.float32)
    bg[y:y + d, x:x + d] = (roi * (1 - a) + img * a).clip(0, 255).astype(np.uint8)
    ys, xs = np.nonzero(m > 127)
    return [int(x + xs.min()), int(y + ys.min()), int(x + xs.max()), int(y + ys.max())]


def cmd_synth(a):
    rng = random.Random(1 if a.split == 'train' else 2)
    cuts = specimen_cutouts(3000, rng, part=a.split)
    meta = {}
    for v, frames in sorted(sequences(a.split).items()):
        objs = [prepare(rng.choice(cuts), rng) for _ in range(rng.choice([1, 1, 2]))]
        vm = []
        os.makedirs(f'{OUT}/{a.split}/{v}', exist_ok=True)
        for t, (fi, im, anns) in enumerate(frames):
            bg = cv2.resize(cv2.imread(f'{R}/data/brk/coco/{a.split}/{im["file_name"]}'), (W, H),
                            interpolation=cv2.INTER_AREA)
            dead = [render(bg, o, t, rng) for o in objs]
            sx, sy = W / im['width'], H / im['height']
            real = [dict(cat=x['category_id'], box=[x['bbox'][0] * sx, x['bbox'][1] * sy,
                                                    (x['bbox'][0] + x['bbox'][2]) * sx, (x['bbox'][1] + x['bbox'][3]) * sy])
                    for x in anns]
            name = f'{fi:04d}.jpg'
            cv2.imwrite(f'{OUT}/{a.split}/{v}/{name}', bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
            vm.append(dict(frame=fi, file=name, dead=dead, real=real))
        meta[v] = vm
    json.dump(meta, open(f'{OUT}/{a.split}_meta.json', 'w'))
    print(a.split, 'videos', len(meta), 'frames', sum(len(x) for x in meta.values()),
          'dead tracks', sum(len(x[0]['dead']) for x in meta.values()))


def cmd_detect(a):
    from ultralytics import YOLO
    model = YOLO(a.weights)
    meta = json.load(open(f'{OUT}/{a.split}_meta.json'))
    dets = {}
    for v, fr in meta.items():
        files = [f'{OUT}/{a.split}/{v}/{x["file"]}' for x in fr]
        out = []
        for i in range(0, len(files), 32):
            for r in model.predict(files[i:i + 32], imgsz=640, conf=0.1, verbose=False):
                out.append([dict(box=b, cls=int(c), score=float(s)) for b, c, s in
                            zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())])
        dets[v] = out
    json.dump(dets, open(f'{OUT}/{a.split}_dets.json', 'w'))
    print('detected', a.split, sum(len(x) for x in dets.values()), 'frames')


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def link(frame_dets, max_gap=3, thr=0.3):
    tracks, active = [], []
    for t, ds in enumerate(frame_dets):
        active = [tr for tr in active if t - tr[-1][0] <= max_gap + 1]
        used = set()
        for d in sorted(ds, key=lambda d: -d['score']):
            best, bi = 0, -1
            for k, tr in enumerate(active):
                if k not in used:
                    v = iou(d['box'], tr[-1][1]['box'])
                    if v > best:
                        best, bi = v, k
            if best >= thr:
                used.add(bi)
                active[bi].append((t, d))
            else:
                tr = [(t, d)]
                tracks.append(tr)
                active.append(tr)
    return tracks


def motion_ratio(split, v, files, tr):
    """Self-motion inside the box relative to a ring around it, from frame differences at a 2-frame lag."""
    rs = []
    for (t, d) in tr[2::2]:
        t0 = t - 2
        if t0 < 0:
            continue
        a = cv2.imread(f'{OUT}/{split}/{v}/{files[t]}', cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(f'{OUT}/{split}/{v}/{files[t0]}', cv2.IMREAD_GRAYSCALE)
        diff = cv2.GaussianBlur(cv2.absdiff(a, b), (5, 5), 0).astype(np.float32)
        x1, y1, x2, y2 = [int(round(c)) for c in d['box']]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        pw, ph = (x2 - x1) // 2, (y2 - y1) // 2
        X1, Y1, X2, Y2 = max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph)
        inside = diff[y1:y2, x1:x2].mean()
        outer = diff[Y1:Y2, X1:X2]
        ring = (outer.sum() - diff[y1:y2, x1:x2].sum()) / max(1, outer.size - (x2 - x1) * (y2 - y1))
        rs.append((inside + 1.0) / (ring + 1.0))
    return float(np.median(rs)) if rs else float('nan')


def features(split, min_len=15):
    meta = json.load(open(f'{OUT}/{split}_meta.json'))
    dets = json.load(open(f'{OUT}/{split}_dets.json'))
    rows = []
    for v, fr in meta.items():
        files = [x['file'] for x in fr]
        for tr in link(dets[v]):
            if len(tr) < min_len:
                continue
            boxes = [d['box'] for _, d in tr]
            diag = np.median([np.hypot(b[2] - b[0], b[3] - b[1]) for b in boxes])
            c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes])
            disp = float(np.median(np.linalg.norm(np.diff(c, axis=0), axis=1)) / max(diag, 1))
            dead_scores = [d['score'] if d['cls'] == 1 else 0.0 for _, d in tr]
            # ground truth for scoring only: which inserted dead track (if any) this track follows
            hits = [max([iou(d['box'], db) for db in fr[t]['dead']] or [0]) for t, d in tr]
            real = [max([(iou(d['box'], r['box']), r['cat']) for r in fr[t]['real']] or [(0, 0)]) for t, d in tr]
            is_dead = float(np.mean([h >= 0.3 for h in hits])) >= 0.5
            hit_cat = max(set(c for i, c in real if i >= 0.3) or {0}, key=lambda k: sum(1 for i, c in real if c == k and i >= 0.3))
            rows.append(dict(video=v, len=len(tr), t0=tr[0][0], disp=disp, max_dead=max(dead_scores),
                             dead_frac=float(np.mean([s >= 0.5 for s in dead_scores])),
                             motion=motion_ratio(split, v, files, tr), is_dead=is_dead,
                             hit_cat=int(hit_cat), dead_ids=[i for i, db in enumerate(fr[tr[0][0]]['dead'])
                                                             if iou(tr[0][1]['box'], db) >= 0.3]))
    return rows, meta


def score(rows, meta, rule):
    alarms = [r for r in rows if rule(r)]
    n_dead = sum(len(fr[0]['dead']) for fr in meta.values())
    found = set()
    for r in alarms:
        if r['is_dead']:
            found.update((r['video'], i) for i in r['dead_ids'])
    fa = [r for r in alarms if not r['is_dead']]
    frames = sum(len(fr) for fr in meta.values())
    by = defaultdict(int)
    names = {0: 'background', 1: 'crab', 2: 'fish', 3: 'jellyfish', 4: 'shrimp', 5: 'small_fish', 6: 'starfish'}
    for r in fa:
        by[names[r['hit_cat']]] += 1
    return dict(recall=round(len(found) / max(n_dead, 1), 3), dead_tracks=n_dead, false_alarm_tracks=len(fa),
                fa_per_video_minute=round(len(fa) / (frames / FPS / 60), 2), fa_by_object=dict(by))


def cmd_verify(a):
    tr_rows, _ = features('train')
    te_rows, te_meta = features('test')
    dead_tr = [r for r in tr_rows if r['is_dead']]
    # calibrate on TRAIN videos only: keep 95% of synthetic dead tracks
    d0 = float(np.nanpercentile([r['disp'] for r in dead_tr], 95)) if dead_tr else 0.05
    m0 = float(np.nanpercentile([r['motion'] for r in dead_tr], 95)) if dead_tr else 1.5
    rules = {
        'single_frame (any frame dead >= 0.5)': lambda r: r['max_dead'] >= 0.5,
        'appearance vote (>= 50% frames dead)': lambda r: r['dead_frac'] >= 0.5,
        'appearance + still (displacement)': lambda r: r['dead_frac'] >= 0.5 and r['disp'] <= d0,
        'appearance + still + no self-motion': lambda r: r['dead_frac'] >= 0.5 and r['disp'] <= d0 and r['motion'] <= m0,
    }
    res = dict(calibration=dict(disp_max=round(d0, 4), motion_ratio_max=round(m0, 3), train_dead_tracks=len(dead_tr),
                                train_tracks=len(tr_rows)),
               test={k: score(te_rows, te_meta, f) for k, f in rules.items()})
    os.makedirs(f'{R}/runs/video_dead', exist_ok=True)
    json.dump(res, open(f'{R}/runs/video_dead/result.json', 'w'), indent=1)
    json.dump(dict(train=tr_rows, test=te_rows), open(f'{R}/runs/video_dead/tracks.json', 'w'))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['synth', 'detect', 'verify'])
    ap.add_argument('--split', default='test')
    ap.add_argument('--weights')
    a = ap.parse_args()
    {'synth': cmd_synth, 'detect': cmd_detect, 'verify': cmd_verify}[a.cmd](a)


if __name__ == '__main__':
    main()
