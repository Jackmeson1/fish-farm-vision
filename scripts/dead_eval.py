"""Score dead-fish detectors on a COCO set in the house class order (1 fish-alive, 2 fish-dead).
  dead_eval.py yolo W --coco DIR --name N [--src-names Dead,Live]   any Ultralytics model; its class order is remapped
  dead_eval.py gdino --coco DIR --name N                            prompts 'live fish' / 'dead fish'
  dead_eval.py twostage --coco DIR --name N                         Grounding DINO 'fish' + CLIP alive/dead per crop
Reports COCO AP per class plus dead-fish precision / recall at score 0.5 (the operating point a farm would use).
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from df3_zeroshot import CLIP, CLIP_TEXTS, GD, emb, gdino  # noqa: E402
from eval_coco import evaluate  # noqa: E402
from yolo2coco import house_id  # noqa: E402

R = '/root/fish'


def op_point(gt, dets, cat=2, thr=0.5, iou_thr=0.5):
    from torchvision.ops import box_iou
    g, d = {}, {}
    for a in gt['annotations']:
        if a['category_id'] == cat:
            x, y, w, h = a['bbox']
            g.setdefault(a['image_id'], []).append([x, y, x + w, y + h])
    for x in dets:
        if x['category_id'] == cat and x['score'] >= thr:
            bx, by, bw, bh = x['bbox']
            d.setdefault(x['image_id'], []).append([bx, by, bx + bw, by + bh])
    tp = 0
    for iid, P in d.items():
        G = g.get(iid, [])
        if not G:
            continue
        m = box_iou(torch.tensor(P), torch.tensor(G))
        used = set()
        for pi in range(len(P)):
            gi = int(m[pi].argmax())
            if m[pi, gi] >= iou_thr and gi not in used:
                used.add(gi)
                tp += 1
    npred, ngt = sum(map(len, d.values())), sum(map(len, g.values()))
    return dict(precision=round(tp / max(npred, 1), 3), recall=round(tp / max(ngt, 1), 3), n_pred=npred, n_gt=ngt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['yolo', 'gdino', 'twostage'])
    ap.add_argument('weights', nargs='?')
    ap.add_argument('--coco', required=True, help='dir with _annotations.coco.json and images')
    ap.add_argument('--name', required=True)
    ap.add_argument('--src-names', default='fish-alive,fish-dead', help="the model's own class order")
    a = ap.parse_args()
    gt = json.load(open(f'{a.coco}/_annotations.coco.json'))
    dets = []
    if a.kind == 'yolo':
        from ultralytics import YOLO
        m = YOLO(a.weights)
        cmap = [house_id(n) for n in a.src_names.split(',')]
        for im in gt['images']:
            r = m.predict(f'{a.coco}/{im["file_name"]}', imgsz=640, conf=0.001, verbose=False)[0]
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                dets.append(dict(image_id=im['id'], category_id=cmap[int(c)], bbox=[x1, y1, x2 - x1, y2 - y1],
                                 score=float(s)))
    else:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        gp = AutoProcessor.from_pretrained(GD)
        gm = AutoModelForZeroShotObjectDetection.from_pretrained(GD).cuda().eval()
        if a.kind == 'twostage':
            from transformers import CLIPModel, CLIPProcessor
            cp = CLIPProcessor.from_pretrained(CLIP)
            cm = CLIPModel.from_pretrained(CLIP, torch_dtype=torch.float16).cuda().eval()
            texts = [t for c in (1, 2) for t in CLIP_TEXTS[c]]
            with torch.no_grad():
                tf = torch.nn.functional.normalize(
                    emb(cm.get_text_features(**cp(text=texts, return_tensors='pt', padding=True).to('cuda'))).float(), dim=-1)
        for im_rec in gt['images']:
            im = Image.open(f'{a.coco}/{im_rec["file_name"]}').convert('RGB')
            if a.kind == 'gdino':
                for (x1, y1, x2, y2), s, l in zip(*gdino(gp, gm, im, 'live fish. dead fish.')):
                    dets.append(dict(image_id=im_rec['id'], category_id=2 if 'dead' in l else 1,
                                     bbox=[x1, y1, x2 - x1, y2 - y1], score=s))
                continue
            boxes, scores, _ = gdino(gp, gm, im, 'fish.')
            if not boxes:
                continue
            crops = []
            for x1, y1, x2, y2 in boxes:
                pw, ph = (x2 - x1) * 0.15, (y2 - y1) * 0.15
                cx1, cy1 = max(0, min(x1 - pw, im.width - 4)), max(0, min(y1 - ph, im.height - 4))
                crops.append(im.crop((cx1, cy1, max(cx1 + 4, min(im.width, x2 + pw)), max(cy1 + 4, min(im.height, y2 + ph)))))
            with torch.no_grad():
                f = torch.nn.functional.normalize(
                    emb(cm.get_image_features(**cp(images=crops, return_tensors='pt').to('cuda', torch.float16))).float(), dim=-1)
                p = (100 * f @ tf.T).softmax(-1)
            for (x1, y1, x2, y2), s, pa, pd in zip(boxes, scores, p[:, :2].sum(-1).tolist(), p[:, 2:].sum(-1).tolist()):
                for c, pc in ((1, pa), (2, pd)):
                    dets.append(dict(image_id=im_rec['id'], category_id=c, bbox=[x1, y1, x2 - x1, y2 - y1],
                                     score=float(s * pc)))
    out = f'{R}/runs/{a.name}'
    os.makedirs(out, exist_ok=True)
    json.dump(dets, open(f'{out}/test_dets.json', 'w'))
    r = evaluate(f'{a.coco}/_annotations.coco.json', dets)
    r.update(name=a.name, coco=a.coco, images=len(gt['images']), dead_at_0_5=op_point(gt, dets),
             dead_at_0_25=op_point(gt, dets, thr=0.25))
    json.dump(r, open(f'{out}/result.json', 'w'), indent=1)
    print(json.dumps({k: r[k] for k in ('name', 'AP', 'AP50', 'per_class', 'dead_at_0_5', 'dead_at_0_25')}))


if __name__ == '__main__':
    main()
