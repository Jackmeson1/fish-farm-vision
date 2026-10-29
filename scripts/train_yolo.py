"""Train an Ultralytics model on the video-disjoint Brackish split, then write COCO-format test detections and eval."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coco import evaluate  # noqa: E402
from ultralytics import YOLO  # noqa: E402

R = '/root/fish'


def predict_coco(model, coco_dir, imgsz, out_json, batch=32):
    gt = json.load(open(os.path.join(coco_dir, '_annotations.coco.json')))
    dets = []
    files = [(im['id'], os.path.join(coco_dir, im['file_name'])) for im in gt['images']]
    t0 = time.time()
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        res = model.predict([f for _, f in chunk], imgsz=imgsz, conf=0.001, iou=0.6, max_det=300, verbose=False)
        for (img_id, _), r in zip(chunk, res):
            b = r.boxes
            for (x1, y1, x2, y2), c, s in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist()):
                dets.append(dict(image_id=img_id, category_id=int(c) + 1, bbox=[x1, y1, x2 - x1, y2 - y1],
                                 score=float(s)))
    json.dump(dets, open(out_json, 'w'))
    return time.time() - t0, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)            # e.g. yolo26l.pt
    ap.add_argument('--name', required=True)
    ap.add_argument('--data', default=f'{R}/data/brk/yolo/data.yaml')
    ap.add_argument('--coco', default=f'{R}/data/brk/coco')  # GT for the final test eval
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--eval-only', help='weights to evaluate instead of training')
    ap.add_argument('--no-val', action='store_true', help='skip per-epoch validation (small low-label runs)')
    a = ap.parse_args()

    out = f'{R}/runs/{a.name}'
    os.makedirs(out, exist_ok=True)
    if a.eval_only:
        best = a.eval_only
        train_s = None
    else:
        model = YOLO(a.model)
        t0 = time.time()
        model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=12, cache='ram',
                    project=f'{R}/runs', name=a.name, exist_ok=True, seed=0, deterministic=False, plots=True,
                    close_mosaic=5, patience=0, val=not a.no_val)
        train_s = time.time() - t0
        best = f'{out}/weights/best.pt'
        if a.no_val or not os.path.exists(best):
            best = f'{out}/weights/last.pt'
    m = YOLO(best)
    secs, n = predict_coco(m, os.path.join(a.coco, 'test'), a.imgsz, f'{out}/test_dets.json')
    r = evaluate(os.path.join(a.coco, 'test', '_annotations.coco.json'), f'{out}/test_dets.json')
    r.update(name=a.name, model=a.model, imgsz=a.imgsz, epochs=a.epochs, train_seconds=train_s,
             params_M=round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
             test_images=n, predict_seconds=round(secs, 1), weights=best, data=a.data)
    json.dump(r, open(f'{out}/result.json', 'w'), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
