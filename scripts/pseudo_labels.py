"""Label-free training sets from a zero-shot teacher's detections on the training videos (flagship experiment).

Variants (thresholds fixed in advance, never tuned on ground truth):
  naive  every teacher box with score >= HIGH
  track  temporal-consistency filter: boxes >= LOW are linked across frames per video (class-agnostic, greedy IoU);
         a track is kept if it has >= MIN_LEN boxes and its best score >= HIGH; only its boxes >= HIGH are written;
         class = score-weighted vote along the track
  fill   recall recovery on top of track: every box of a kept track (>= LOW) plus linearly interpolated boxes for
         frames the track skips (gaps <= MAX_GAP frame indices)
(After the 200-frame smoke test, track was narrowed to high-score boxes so that filtering and recovery are separate
variants; the thresholds were not changed.)
Writes data/brk_pl_<variant>/yolo/{train/{images,labels}, data.yaml} (val = the real val split, used only for the
final printout; training runs with --no-val and keeps the last weights). Pseudo-label precision/recall against the
training ground truth is computed afterwards, for the report only.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import torch
from torchvision.ops import box_iou, nms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

R = '/root/fish'
HIGH, LOW, MIN_LEN, MAX_GAP, LINK_IOU, DEDUP_IOU = 0.35, 0.15, 3, 3, 0.3, 0.6
FRAME_RE = re.compile(r'^(?P<video>.+)-(?P<frame>\d{4})_(?:jpg|png)\.rf\.[0-9a-f]+\.(?:jpg|png)$')


def xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def dedup(ds):
    """One box per object: the teacher answers each class prompt separately, so the same fish can come back twice."""
    if not ds:
        return ds
    b = torch.tensor([xyxy(d['bbox']) for d in ds])
    s = torch.tensor([d['score'] for d in ds])
    return [ds[i] for i in nms(b, s, DEDUP_IOU).tolist()]


def link(frames):
    """frames: [(frame_idx, image_id, [dets])] sorted by frame -> list of tracks, each [(frame_idx, image_id, det)]."""
    tracks, active = [], []
    for fi, iid, ds in frames:
        ds = sorted(ds, key=lambda d: -d['score'])
        active = [t for t in active if fi - t[-1][0] <= MAX_GAP + 1]
        used = set()
        if active and ds:
            iou = box_iou(torch.tensor([xyxy(d['bbox']) for d in ds]),
                          torch.tensor([xyxy(t[-1][2]['bbox']) for t in active]))
            for di in range(len(ds)):
                ti = int(iou[di].argmax())
                if iou[di, ti] >= LINK_IOU and ti not in used:
                    used.add(ti)
                    active[ti].append((fi, iid, ds[di]))
                    ds[di] = None
        for d in ds:
            if d is not None:
                t = [(fi, iid, d)]
                tracks.append(t)
                active.append(t)
    return tracks


def build(dets_path, variant, out_root):
    imgs = json.load(open(f'{R}/data/brk/coco/train/_annotations.coco.json'))['images']  # image list only
    by_img = defaultdict(list)
    for d in json.load(open(dets_path)):
        if d['score'] >= LOW:
            by_img[d['image_id']].append(d)
    labels = defaultdict(list)  # image_id -> [(cat, xyxy)]
    if variant == 'naive':
        for iid, ds in by_img.items():
            labels[iid] = [(d['category_id'], xyxy(d['bbox'])) for d in dedup(ds) if d['score'] >= HIGH]
    else:
        videos = defaultdict(list)
        for im in imgs:
            m = FRAME_RE.match(im['file_name'])
            videos[m['video']].append((int(m['frame']), im['id'], dedup(by_img.get(im['id'], []))))
        frame_to_img = {}
        for v, fr in videos.items():
            fr.sort(key=lambda x: x[0])
            frame_to_img.update({(v, fi): iid for fi, iid, _ in fr})
            for t in link(fr):
                if len(t) < MIN_LEN or max(d['score'] for _, _, d in t) < HIGH:
                    continue
                votes = Counter()
                for _, _, d in t:
                    votes[d['category_id']] += d['score']
                cat = votes.most_common(1)[0][0]
                for fi, iid, d in t:
                    if variant == 'fill' or d['score'] >= HIGH:
                        labels[iid].append((cat, xyxy(d['bbox'])))
                if variant == 'fill':
                    for (f0, _, d0), (f1, _, d1) in zip(t, t[1:]):
                        for f in range(f0 + 1, f1):
                            iid = frame_to_img.get((v, f))
                            if iid is None:
                                continue
                            a = (f - f0) / (f1 - f0)
                            b0, b1 = xyxy(d0['bbox']), xyxy(d1['bbox'])
                            labels[iid].append((cat, [p + a * (q - p) for p, q in zip(b0, b1)]))
    # write YOLO dataset
    root = f'{out_root}/yolo'
    for sub in ('images', 'labels'):
        os.makedirs(f'{root}/train/{sub}', exist_ok=True)
    counts = Counter()
    for im in imgs:
        src = os.path.realpath(f'{R}/data/brk/yolo/train/images/{im["file_name"]}')
        dst = f'{root}/train/images/{im["file_name"]}'
        if not os.path.lexists(dst):
            os.symlink(src, dst)
        w, h = im['width'], im['height']
        with open(f'{root}/train/labels/{os.path.splitext(im["file_name"])[0]}.txt', 'w') as fh:
            for cat, (x1, y1, x2, y2) in labels.get(im['id'], []):
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                counts[cat] += 1
                fh.write('%d %.6f %.6f %.6f %.6f\n' % (cat - 1, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                                                      (x2 - x1) / w, (y2 - y1) / h))
    yaml = open(f'{R}/data/brk/yolo/data.yaml').read()
    yaml = re.sub(r'^path: .*$', f'path: {root}', yaml, flags=re.M)
    yaml = yaml.replace('val: val/images', f'val: {R}/data/brk/yolo/val/images')
    yaml = yaml.replace('test: test/images', f'test: {R}/data/brk/yolo/test/images')
    open(f'{root}/data.yaml', 'w').write(yaml)
    return labels, counts


def quality(labels):
    """Diagnostics only: pseudo-label precision / recall at IoU 0.5 against the training ground truth (class-aware)."""
    gt = json.load(open(f'{R}/data/brk/coco/train/_annotations.coco.json'))
    names = {c['id']: c['name'] for c in gt['categories']}
    g = defaultdict(list)
    for a in gt['annotations']:
        g[a['image_id']].append((a['category_id'], xyxy(a['bbox'])))
    tp, npred, ngt = Counter(), Counter(), Counter()
    for im in gt['images']:
        P, G = labels.get(im['id'], []), g.get(im['id'], [])
        for c, _ in P:
            npred[c] += 1
        for c, _ in G:
            ngt[c] += 1
        if not P or not G:
            continue
        iou = box_iou(torch.tensor([b for _, b in P]), torch.tensor([b for _, b in G]))
        taken = set()
        for pi in iou.max(1).values.argsort(descending=True).tolist():
            for gi in iou[pi].argsort(descending=True).tolist():
                if iou[pi, gi] < 0.5:
                    break
                if gi not in taken and P[pi][0] == G[gi][0]:
                    taken.add(gi)
                    tp[P[pi][0]] += 1
                    break
    out = {}
    for c in sorted(ngt):
        out[names[c]] = dict(pseudo=npred[c], gt=ngt[c], precision=round(tp[c] / max(npred[c], 1), 3),
                             recall=round(tp[c] / max(ngt[c], 1), 3))
    T, Np, Ng = sum(tp.values()), sum(npred.values()), sum(ngt.values())
    out['all'] = dict(pseudo=Np, gt=Ng, precision=round(T / max(Np, 1), 3), recall=round(T / max(Ng, 1), 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dets', default=f'{R}/runs/teacher_gdino/train_dets.json')
    ap.add_argument('--variants', default='naive,track,fill')
    a = ap.parse_args()
    report = {}
    for v in a.variants.split(','):
        labels, _ = build(a.dets, v, f'{R}/data/brk_pl_{v}')
        report[v] = quality(labels)
        print(v, json.dumps(report[v]['all']), flush=True)
    report['settings'] = dict(HIGH=HIGH, LOW=LOW, MIN_LEN=MIN_LEN, MAX_GAP=MAX_GAP, LINK_IOU=LINK_IOU,
                              DEDUP_IOU=DEDUP_IOU, teacher=a.dets)
    json.dump(report, open(f'{R}/runs/pseudo_label_quality.json', 'w'), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
