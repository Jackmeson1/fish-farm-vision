"""Is death a persistent state in real farm data? RAS-MortDB time-lapse (one capture every ~15 min, session-blocked
test split). Two questions:
  1. ground truth: how often does a dead / live fish box reappear at the same place in the neighbouring capture?
  2. detectors: if a 'dead' detection must reappear in a neighbouring capture (<= MAX_GAP) to count, how do dead-fish
     precision / recall / AP change? (hard gate: unconfirmed boxes are dropped; soft gate: their score x 0.2)
  ras_persist.py NAME [NAME ...]    runs/NAME/test_dets.json scored on data/rasmort/blocked_coco/test
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dead_eval import op_point  # noqa: E402
from eval_coco import evaluate  # noqa: E402

R = '/root/fish'
GT = f'{R}/data/rasmort/blocked_coco/test/_annotations.coco.json'
NAME_RE = re.compile(r'^tl_(?P<s>\d+)_(?P<f>\d+)_(?P<d>\d{8})_(?P<t>\d{6})_')
MAX_GAP_MIN, LINK_IOU, LOW = 120, 0.3, 0.25


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    ix, iy = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / u if u > 0 else 0.0


def neighbours(gt):
    meta = {}
    for im in gt['images']:
        m = NAME_RE.match(im['file_name'])
        meta[im['id']] = (m['s'], datetime.strptime(m['d'] + m['t'], '%Y%m%d%H%M%S'))
    by_s = defaultdict(list)
    for iid, (s, t) in meta.items():
        by_s[s].append((t, iid))
    nb = {}
    for v in by_s.values():
        v.sort()
        for k, (t, iid) in enumerate(v):
            nb[iid] = [j for (tj, j) in (v[k - 1:k] + v[k + 1:k + 2])
                       if j != iid and abs((tj - t).total_seconds()) <= MAX_GAP_MIN * 60]
    return nb


def gt_persistence(gt, nb):
    boxes = defaultdict(lambda: defaultdict(list))
    for a in gt['annotations']:
        boxes[a['image_id']][a['category_id']].append(a['bbox'])
    out = {}
    for cat, name in ((2, 'dead'), (1, 'live')):
        n = hit = 0
        for iid, cats in list(boxes.items()):
            if not nb.get(iid):
                continue
            for b in cats.get(cat, []):
                n += 1
                hit += any(iou(b, c) >= 0.5 for j in nb[iid] for c in boxes.get(j, {}).get(cat, []))
        out[name] = dict(boxes_with_neighbour_capture=n, same_place_in_neighbour=round(hit / max(n, 1), 3))
    return out


def gate(dets, nb, soft):
    dead = defaultdict(list)
    for d in dets:
        if d['category_id'] == 2 and d['score'] >= LOW:
            dead[d['image_id']].append(d['bbox'])
    out = []
    for d in dets:
        if d['category_id'] != 2:
            out.append(d)
            continue
        ok = any(iou(d['bbox'], c) >= LINK_IOU for j in nb.get(d['image_id'], []) for c in dead[j])
        if ok:
            out.append(d)
        elif soft:
            out.append(dict(d, score=d['score'] * 0.2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='+')
    a = ap.parse_args()
    gt = json.load(open(GT))
    nb = neighbours(gt)
    res = dict(settings=dict(max_gap_min=MAX_GAP_MIN, link_iou=LINK_IOU, low=LOW),
               test_images=len(gt['images']), images_with_neighbour=sum(1 for v in nb.values() if v),
               gt_persistence=gt_persistence(gt, nb), detectors={})
    for n in a.names:
        p = f'{R}/runs/{n}/test_dets.json'
        if not os.path.exists(p):
            continue
        dets = json.load(open(p))
        row = {}
        for tag, ds in (('as_is', dets), ('soft_gate', gate(dets, nb, True)), ('hard_gate', gate(dets, nb, False))):
            r = evaluate(GT, ds)
            row[tag] = dict(dead_AP=r['per_class']['fish-dead']['AP'], dead_AP50=r['per_class']['fish-dead']['AP50'],
                            at_0_5=op_point(gt, ds), at_0_25=op_point(gt, ds, thr=0.25))
        res['detectors'][n] = row
    os.makedirs(f'{R}/runs/ras_persist', exist_ok=True)
    json.dump(res, open(f'{R}/runs/ras_persist/persist_result.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
