"""Zero-label baselines on the dead-fish proxy set (no dead-fish labels used for training or thresholds).
  direct     Grounding DINO with the prompts 'live fish' / 'dead fish'
  twostage   Grounding DINO 'fish' boxes, then CLIP decides alive vs dead on each crop; every box is emitted once per
             class with score = box score x CLIP probability, so both classes get a ranked list for AP
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coco import evaluate  # noqa: E402

R = '/root/fish'
GD = 'IDEA-Research/grounding-dino-base'
CLIP = 'openai/clip-vit-large-patch14'
CLIP_TEXTS = {1: ['a photo of a live fish swimming', 'a photo of a healthy swimming fish'],
              2: ['a photo of a dead fish', 'a photo of a dead fish lying still']}


def emb(out):
    """CLIP features as a tensor: transformers 5 returns an output object from get_*_features."""
    if torch.is_tensor(out):
        return out
    for k in ('text_embeds', 'image_embeds', 'pooler_output'):
        v = getattr(out, k, None)
        if v is not None:
            return v
    raise TypeError(type(out))


def gdino(proc, model, im, text, thr=0.05):
    inp = proc(images=im, text=text, return_tensors='pt').to('cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        out = model(**inp)
    r = proc.post_process_grounded_object_detection(out, inp['input_ids'], threshold=thr, text_threshold=thr,
                                                    target_sizes=[im.size[::-1]])[0]
    return r['boxes'].float().tolist(), r['scores'].float().tolist(), [str(x) for x in r.get('text_labels', r['labels'])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['direct', 'twostage'])
    ap.add_argument('--split', default='test')
    a = ap.parse_args()
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel, CLIPProcessor
    gp = AutoProcessor.from_pretrained(GD)
    gm = AutoModelForZeroShotObjectDetection.from_pretrained(GD).cuda().eval()
    if a.mode == 'twostage':
        cp = CLIPProcessor.from_pretrained(CLIP)
        cm = CLIPModel.from_pretrained(CLIP, torch_dtype=torch.float16).cuda().eval()
        texts = [t for c in (1, 2) for t in CLIP_TEXTS[c]]
        with torch.no_grad():
            tf = emb(cm.get_text_features(**cp(text=texts, return_tensors='pt', padding=True).to('cuda')))
            tf = torch.nn.functional.normalize(tf.float(), dim=-1)
    d = f'{R}/data/deadfish/coco/{a.split}'
    gt = json.load(open(f'{d}/_annotations.coco.json'))
    dets = []
    for im_rec in gt['images']:
        im = Image.open(f'{d}/{im_rec["file_name"]}').convert('RGB')
        if a.mode == 'direct':
            boxes, scores, labels = gdino(gp, gm, im, 'live fish. dead fish.')
            for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                c = 2 if 'dead' in l else 1
                dets.append(dict(image_id=im_rec['id'], category_id=c, bbox=[x1, y1, x2 - x1, y2 - y1], score=s))
        else:
            boxes, scores, _ = gdino(gp, gm, im, 'fish.')
            if not boxes:
                continue
            crops = []
            for x1, y1, x2, y2 in boxes:
                pw, ph = (x2 - x1) * 0.15, (y2 - y1) * 0.15
                cx1, cy1 = max(0, min(x1 - pw, im.width - 4)), max(0, min(y1 - ph, im.height - 4))
                crops.append(im.crop((cx1, cy1, max(cx1 + 4, min(im.width, x2 + pw)), max(cy1 + 4, min(im.height, y2 + ph)))))
            with torch.no_grad():
                f = emb(cm.get_image_features(**cp(images=crops, return_tensors='pt').to('cuda', torch.float16)))
                f = torch.nn.functional.normalize(f.float(), dim=-1)
                logits = 100 * f @ tf.T  # [n, 4]: two prompts per class, averaged as probabilities
                p = logits.softmax(-1)
                p_alive, p_dead = p[:, :2].sum(-1), p[:, 2:].sum(-1)
            for (x1, y1, x2, y2), s, pa, pd in zip(boxes, scores, p_alive.tolist(), p_dead.tolist()):
                for c, pc in ((1, pa), (2, pd)):
                    dets.append(dict(image_id=im_rec['id'], category_id=c, bbox=[x1, y1, x2 - x1, y2 - y1],
                                     score=float(s * pc)))
    out = f'{R}/runs/df3_zs_{a.mode}_{a.split}'
    os.makedirs(out, exist_ok=True)
    json.dump(dets, open(f'{out}/test_dets.json', 'w'))
    r = evaluate(f'{d}/_annotations.coco.json', f'{out}/test_dets.json')
    r.update(name=f'df3_zs_{a.mode}', split=a.split, images=len(gt['images']))
    json.dump(r, open(f'{out}/result.json', 'w'), indent=1)
    print(json.dumps({k: r[k] for k in ('name', 'split', 'AP', 'AP50', 'per_class')}, indent=1))


if __name__ == '__main__':
    main()
