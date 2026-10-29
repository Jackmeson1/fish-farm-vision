"""Collect runs/*/result.json into one table, with two re-scorings of each run's test detections (reviewer requests):
  fish_merged_*  fish + small_fish as one class, duplicates removed by NMS across the two (zero-shot models cannot
                 tell the size classes apart by name, and one fish boxed as both would count as a false positive)
  agnostic_*     every class as one 'animal' class, NMS across classes: finding animals vs naming them
"""
import argparse
import copy
import glob
import json
import os
import sys
from collections import defaultdict

import torch
from torchvision.ops import nms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coco import evaluate  # noqa: E402

R = '/root/fish'
IOU = 0.7


def remap_gt(gt, mapping, keep_names, out_path):
    g = copy.deepcopy(gt)
    for a in g['annotations']:
        a['category_id'] = mapping.get(a['category_id'], a['category_id'])
    g['categories'] = [c for c in g['categories'] if c['name'] in keep_names]
    json.dump(g, open(out_path, 'w'))
    return out_path


def remap_dets(dets, mapping, nms_cats):
    by = defaultdict(list)
    out = []
    for d in dets:
        d = dict(d, category_id=mapping.get(d['category_id'], d['category_id']))
        (by[d['image_id']] if d['category_id'] in nms_cats else out).append(d)
    for ds in by.values():
        b = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in (d['bbox'] for d in ds)], dtype=torch.float32)
        s = torch.tensor([d['score'] for d in ds], dtype=torch.float32)
        out.extend(ds[i] for i in nms(b, s, IOU).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{R}/runs/summary.json')
    a = ap.parse_args()
    gt_path = f'{R}/data/brk/coco/test/_annotations.coco.json'
    gt = json.load(open(gt_path))
    ids = {c['name']: c['id'] for c in gt['categories']}
    fish_map = {ids['small_fish']: ids['fish']}
    mgt = remap_gt(gt, fish_map, {n for n in ids if n != 'small_fish'}, f'{R}/data/brk/coco/test_merged_fish.json')
    agn_map = {cid: ids['fish'] for cid in ids.values()}
    agt = remap_gt(gt, agn_map, {'fish'}, f'{R}/data/brk/coco/test_agnostic.json')
    test_ids = {im['id'] for im in gt['images']}
    rows = []
    for p in sorted(glob.glob(f'{R}/runs/*/result.json')):
        name = os.path.basename(os.path.dirname(p))
        r = json.load(open(p))
        if 'AP' not in r or name.startswith(('smoke', 'uie_')):
            continue
        dets_path = os.path.join(os.path.dirname(p), 'test_dets.json')
        same_test = r.get('test_images') == len(test_ids) and 'coco_' not in str(r.get('data', '')) \
            and not name.endswith(('_on_half', '_seen', '_leak2')) and '_on_' not in name
        if os.path.exists(dets_path) and same_test:
            dets = json.load(open(dets_path))
            mr = evaluate(mgt, remap_dets(dets, fish_map, {ids['fish']}))
            r['fish_merged_AP'], r['fish_merged_AP50'] = mr['per_class']['fish']['AP'], mr['per_class']['fish']['AP50']
            ar = evaluate(agt, remap_dets(dets, agn_map, {ids['fish']}))
            r['agnostic_AP'], r['agnostic_AP50'] = ar['AP'], ar['AP50']
        rows.append(r)
    json.dump(rows, open(a.out, 'w'), indent=1)
    cols = ['name', 'test_images', 'AP', 'AP50', 'APs', 'fish_merged_AP', 'fish_merged_AP50', 'agnostic_AP',
            'agnostic_AP50', 'params_M', 'train_seconds']
    print(' | '.join(cols))
    for r in rows:
        pc = r.get('per_class', {})
        print(' | '.join(f'{r[c]:.3f}' if isinstance(r.get(c), float) else str(r.get(c, '')) for c in cols),
              '|', ' '.join(f"{k}={v['AP']:.3f}" for k, v in pc.items()))


if __name__ == '__main__':
    main()
