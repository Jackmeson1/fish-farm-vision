"""COCO box evaluation shared by every model: GT json + detections json -> overall and per-class AP (json)."""
import argparse
import contextlib
import io
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(gt_path, det_path, img_ids=None):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_path)
        dets = json.load(open(det_path)) if isinstance(det_path, str) else det_path
        if not dets:
            return dict(AP=0.0, AP50=0.0, AP75=0.0, APs=0.0, APm=0.0, APl=0.0, AR100=0.0, per_class={})
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, 'bbox')
        if img_ids is not None:
            ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    out = dict(AP=s[0], AP50=s[1], AP75=s[2], APs=s[3], APm=s[4], APl=s[5], AR100=s[8], per_class={})
    prec = ev.eval['precision']  # [T, R, K, A, M]
    for k, cid in enumerate(ev.params.catIds):
        name = gt.cats[cid]['name']
        p = prec[:, :, k, 0, -1]
        p50 = prec[0, :, k, 0, -1]
        out['per_class'][name] = dict(AP=float(np.mean(p[p > -1])) if (p > -1).any() else float('nan'),
                                      AP50=float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else float('nan'),
                                      n_gt=len(gt.getAnnIds(catIds=[cid], imgIds=img_ids or [])))
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('gt')
    ap.add_argument('dets')
    ap.add_argument('--out')
    a = ap.parse_args()
    r = evaluate(a.gt, a.dets)
    print(json.dumps(r, indent=1))
    if a.out:
        json.dump(r, open(a.out, 'w'), indent=1)
