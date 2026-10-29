"""Fine-tune RF-DETR on the video-disjoint Brackish split (COCO layout), then write COCO test detections and eval."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coco import evaluate  # noqa: E402
import rfdetr  # noqa: E402

R = '/root/fish'


def predict_coco(model, coco_dir, out_json, batch=16):
    gt = json.load(open(os.path.join(coco_dir, '_annotations.coco.json')))
    name_to_cat = {c['name']: c['id'] for c in gt['categories']}
    dets, t0 = [], time.time()
    files = [(im['id'], os.path.join(coco_dir, im['file_name'])) for im in gt['images']]
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        res = model.predict([f for _, f in chunk], threshold=0.001, include_source_image=False)
        if not isinstance(res, list):
            res = [res]
        for (img_id, _), d in zip(chunk, res):
            names = d.data.get('class_name')
            for j, ((x1, y1, x2, y2), s) in enumerate(zip(d.xyxy.tolist(), d.confidence.tolist())):
                cat = name_to_cat.get(str(names[j])) if names is not None else int(d.class_id[j]) + 1
                if cat is None:
                    continue
                dets.append(dict(image_id=img_id, category_id=cat, bbox=[x1, y1, x2 - x1, y2 - y1], score=float(s)))
    json.dump(dets, open(out_json, 'w'))
    return time.time() - t0, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='RFDETRMedium')
    ap.add_argument('--name', required=True)
    ap.add_argument('--data', default=f'{R}/data/brk/coco')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--eval-only', help='checkpoint to evaluate instead of training')
    ap.add_argument('--eval-interval', type=int, default=1)
    a = ap.parse_args()

    out = f'{R}/runs/{a.name}'
    os.makedirs(out, exist_ok=True)
    cls = getattr(rfdetr, a.variant)
    train_s = None
    if not a.eval_only:
        model = cls()
        t0 = time.time()
        model.train(dataset_dir=a.data, epochs=a.epochs, batch_size=a.batch, grad_accum_steps=1, output_dir=out,
                    num_workers=12, eval_interval=a.eval_interval)
        train_s = time.time() - t0
    ckpt = a.eval_only or next(os.path.join(out, f) for f in ('checkpoint_best_total.pth', 'checkpoint_best_ema.pth',
                                                              'checkpoint_best_regular.pth', 'checkpoint.pth')
                               if os.path.exists(os.path.join(out, f)))
    model = cls(pretrain_weights=ckpt)
    try:
        model.optimize_for_inference()
    except Exception as e:  # optional speed-up only
        print('optimize_for_inference skipped:', e)
    secs, n = predict_coco(model, os.path.join(R, 'data/brk/coco/test'), f'{out}/test_dets.json')
    r = evaluate(os.path.join(R, 'data/brk/coco/test/_annotations.coco.json'), f'{out}/test_dets.json')
    r.update(name=a.name, model=a.variant, epochs=a.epochs, train_seconds=train_s, checkpoint=ckpt, test_images=n,
             predict_seconds=round(secs, 1), data=a.data)
    json.dump(r, open(f'{out}/result.json', 'w'), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
