"""YOLOX checkpoint -> COCO-format test detections (YOLOX's own preproc/postprocess) -> the shared evaluator."""
import argparse
import json
import os
import sys
import time

import cv2
import torch

sys.path.insert(0, '/root/fish/src/YOLOX')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coco import evaluate  # noqa: E402
from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

R = '/root/fish'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default=f'{R}/scripts/yolox_l_brackish.py')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--batch', type=int, default=32)
    a = ap.parse_args()

    exp = get_exp(a.exp)
    model = exp.get_model()
    ckpt = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.cuda().eval().half()
    preproc = ValTransform(legacy=False)
    size = exp.test_size

    test = f'{R}/data/brk/coco/test'
    gt = json.load(open(f'{test}/_annotations.coco.json'))
    files = [(im['id'], os.path.join(test, im['file_name'])) for im in gt['images']]
    dets, t0 = [], time.time()
    for i in range(0, len(files), a.batch):
        chunk = files[i:i + a.batch]
        xs, ratios = [], []
        for _, f in chunk:
            img = cv2.imread(f)
            ratios.append(min(size[0] / img.shape[0], size[1] / img.shape[1]))
            xs.append(torch.from_numpy(preproc(img, None, size)[0]))
        with torch.no_grad():
            out = model(torch.stack(xs).cuda().half())
            out = postprocess(out, exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=False)
        for (img_id, _), r, o in zip(chunk, ratios, out):
            if o is None:
                continue
            o = o.float().cpu()
            for x1, y1, x2, y2, obj, cls_conf, c in o.tolist():
                x1, y1, x2, y2 = x1 / r, y1 / r, x2 / r, y2 / r
                # COCODataset indexes sorted category ids, which are 1..6 here
                dets.append(dict(image_id=img_id, category_id=int(c) + 1, bbox=[x1, y1, x2 - x1, y2 - y1],
                                 score=float(obj * cls_conf)))
    secs = time.time() - t0
    out_dir = f'{R}/runs/{a.name}'
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dets, open(f'{out_dir}/test_dets.json', 'w'))
    r = evaluate(f'{test}/_annotations.coco.json', f'{out_dir}/test_dets.json')
    r.update(name=a.name, model='YOLOX-L', checkpoint=a.ckpt, test_images=len(files), predict_seconds=round(secs, 1),
             params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
    json.dump(r, open(f'{out_dir}/result.json', 'w'), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
