"""Zero-shot (no Brackish labels) detection on the test split with open-vocabulary detectors; COCO dets + eval."""
import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coco import evaluate  # noqa: E402

R = '/root/fish'
TEST = f'{R}/data/brk/coco/test'
PROMPTS = {'crab': 'crab', 'fish': 'fish', 'jellyfish': 'jellyfish', 'shrimp': 'shrimp', 'small_fish': 'small fish',
           'starfish': 'starfish'}


def images(split='test'):
    # only the image list and category ids are read: no annotations, so the train split stays label-free
    d = TEST if split == 'test' else f'{R}/data/brk/coco/{split}'
    gt = json.load(open(f'{d}/_annotations.coco.json'))
    cat = {c['name']: c['id'] for c in gt['categories']}
    return [(im['id'], os.path.join(d, im['file_name'])) for im in gt['images']], cat


def run_owlv2(files, cat, mid='google/owlv2-large-patch14-ensemble', batch=8):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained(mid)
    model = Owlv2ForObjectDetection.from_pretrained(mid, torch_dtype=torch.float16).cuda().eval()
    names = list(PROMPTS)
    queries = [f'a photo of a {PROMPTS[n]}' for n in names]
    dets = []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        ims = [Image.open(f).convert('RGB') for _, f in chunk]
        inp = proc(text=[queries] * len(ims), images=ims, return_tensors='pt').to('cuda')
        inp['pixel_values'] = inp['pixel_values'].half()
        with torch.no_grad():
            out = model(**inp)
        # OWLv2 pads to a square, so boxes are relative to the padded side
        sizes = [(max(im.size), max(im.size)) for im in ims]
        res = proc.post_process_grounded_object_detection(out, threshold=0.01, target_sizes=sizes)
        for (img_id, _), r in zip(chunk, res):
            for (x1, y1, x2, y2), s, l in zip(r['boxes'].tolist(), r['scores'].tolist(), r['labels'].tolist()):
                dets.append(dict(image_id=img_id, category_id=cat[names[l]], bbox=[x1, y1, x2 - x1, y2 - y1],
                                 score=float(s)))
    return dets


def run_gdino(files, cat, mid='IDEA-Research/grounding-dino-base', batch=8):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).cuda().eval()
    names = list(PROMPTS)
    text = '. '.join(PROMPTS[n] for n in names) + '.'
    phrase_to_cat = {PROMPTS[n]: cat[n] for n in names}
    dets = []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        ims = [Image.open(f).convert('RGB') for _, f in chunk]
        inp = proc(images=ims, text=[text] * len(ims), return_tensors='pt').to('cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            out = model(**inp)
        res = proc.post_process_grounded_object_detection(out, inp['input_ids'], threshold=0.05, text_threshold=0.05,
                                                          target_sizes=[im.size[::-1] for im in ims])
        for (img_id, _), r in zip(chunk, res):
            labels = r.get('text_labels', r.get('labels'))
            for (x1, y1, x2, y2), s, l in zip(r['boxes'].tolist(), r['scores'].tolist(), labels):
                l = str(l).strip()
                # a phrase like "small fish" also contains "fish": take the longest matching prompt
                hits = [p for p in phrase_to_cat if p in l]
                if not hits:
                    continue
                dets.append(dict(image_id=img_id, category_id=phrase_to_cat[max(hits, key=len)],
                                 bbox=[x1, y1, x2 - x1, y2 - y1], score=float(s)))
    return dets


def run_sam3(files, cat, mid=f'{R}/weights/sam3'):
    """SAM 3 (mirror of facebook/sam3, safetensors, sha256 6d06f0a5...): one vision encode per image, one decode per
    class prompt; boxes from post_process_object_detection."""
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained(mid)
    model = Sam3Model.from_pretrained(mid, torch_dtype=torch.bfloat16).cuda().eval()
    names = list(PROMPTS)
    texts = [proc(text=PROMPTS[n], return_tensors='pt').to('cuda') for n in names]
    dets = []
    for img_id, f in files:
        im = Image.open(f).convert('RGB')
        inp = proc(images=im, return_tensors='pt').to('cuda')
        with torch.no_grad():
            ve = model.get_vision_features(pixel_values=inp['pixel_values'].to(torch.bfloat16))
            for n, t in zip(names, texts):
                out = model(vision_embeds=ve, input_ids=t['input_ids'], attention_mask=t.get('attention_mask'))
                r = proc.post_process_object_detection(out, threshold=0.01, target_sizes=[im.size[::-1]])[0]
                for (x1, y1, x2, y2), sc in zip(r['boxes'].float().tolist(), r['scores'].float().tolist()):
                    dets.append(dict(image_id=img_id, category_id=cat[n], bbox=[x1, y1, x2 - x1, y2 - y1],
                                     score=float(sc)))
    return dets


def run_yoloe(files, cat, weights, batch=32):
    from ultralytics import YOLOE
    model = YOLOE(weights)
    names = list(PROMPTS)
    prompts = [PROMPTS[n] for n in names]
    model.set_classes(prompts, model.get_text_pe(prompts))
    dets = []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        res = model.predict([f for _, f in chunk], conf=0.001, imgsz=640, verbose=False, max_det=300)
        for (img_id, _), r in zip(chunk, res):
            b = r.boxes
            for (x1, y1, x2, y2), c, s in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist()):
                dets.append(dict(image_id=img_id, category_id=cat[names[int(c)]], bbox=[x1, y1, x2 - x1, y2 - y1],
                                 score=float(s)))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=['owlv2', 'gdino', 'yoloe', 'sam3'])
    ap.add_argument('--weights', default='yoloe-26l-seg.pt')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--split', default='test', help="'train' = teacher pass for pseudo-labels (no evaluation)")
    a = ap.parse_args()
    files, cat = images(a.split)
    if a.limit:
        files = files[:a.limit]
    t0 = time.time()
    dets = {'owlv2': lambda: run_owlv2(files, cat), 'gdino': lambda: run_gdino(files, cat),
            'yoloe': lambda: run_yoloe(files, cat, a.weights), 'sam3': lambda: run_sam3(files, cat)}[a.model]()
    secs = time.time() - t0
    if a.split != 'test':
        out = f'{R}/runs/teacher_{a.model}'
        os.makedirs(out, exist_ok=True)
        json.dump(dets, open(f'{out}/{a.split}_dets.json', 'w'))
        print(json.dumps(dict(teacher=a.model, split=a.split, images=len(files), dets=len(dets),
                              seconds=round(secs, 1))))
        return
    out = f'{R}/runs/zs_{a.model}'
    os.makedirs(out, exist_ok=True)
    json.dump(dets, open(f'{out}/test_dets.json', 'w'))
    r = evaluate(f'{TEST}/_annotations.coco.json', f'{out}/test_dets.json',
                 img_ids=[i for i, _ in files] if a.limit else None)
    r.update(name=f'zs_{a.model}', test_images=len(files), predict_seconds=round(secs, 1),
             weights=a.weights if a.model == 'yoloe' else None)
    json.dump(r, open(f'{out}/result.json', 'w'), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
