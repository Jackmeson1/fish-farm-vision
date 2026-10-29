"""Vision-language models as dead-vs-live judges on the blind RAS panels, with and without a mortality knowledge base.
  vlm_judge.py MODEL_ID [MODEL_ID ...]   -> runs/vlm/pred_<model>_<prompt>.json and runs/vlm/vlm_result.json
"""
import argparse
import json
import os
import re

import torch
from PIL import Image

R = '/root/fish/runs/vlm'

PANEL = ("The image has three tiles from an underwater camera that looks down at the bottom drain plate of a "
         "recirculating aquaculture tank with trout. Left: the whole frame, one fish outlined by a red box. Middle: a "
         "close-up of that fish. Right: the same region in the neighbouring photo taken 15-120 minutes earlier or "
         "later (grey if there is none).")
BARE = PANEL + " Is the boxed fish dead or alive? Answer with exactly one word: dead or live."
KB = PANEL + """
Mortality checklist used by farm technicians:
- Dead: lying on the tank bottom or drain plate; on its side or belly-up, so the pale or white belly and flank face
  the camera; faded, washed-out colour; body rigid (straight or stiffly bent), fins folded; sharp outline with no
  motion blur; usually still at the same place and pose in the neighbouring photo.
- Live: upright, so from above you see the dark back and the spotted dorsal side; swimming in mid-water above the
  plate; body flexed in a swimming curve; often motion-blurred; at a different place in the neighbouring photo.
- Weigh all cues; one cue alone (for example a pale colour under bright light) is not enough.
Is the boxed fish dead or alive? Answer with exactly one word: dead or live."""
PROMPTS = {'bare': BARE, 'kb': KB}


def parse(text):
    t = text.strip().lower()
    m = re.search(r'\b(dead|live|alive)\b', t)
    return None if not m else ('dead' if m.group(1) == 'dead' else 'live')


def score(pred, labels):
    ids = sorted(labels)
    tp = sum(pred.get(i) == 'dead' and labels[i] == 'dead' for i in ids)
    fp = sum(pred.get(i) == 'dead' and labels[i] == 'live' for i in ids)
    fn = sum(pred.get(i) != 'dead' and labels[i] == 'dead' for i in ids)
    acc = sum(pred.get(i) == labels[i] for i in ids) / len(ids)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(n=len(ids), accuracy=round(acc, 3), dead_precision=round(p, 3), dead_recall=round(r, 3),
                dead_f1=round(2 * p * r / max(p + r, 1e-9), 3), unparsed=sum(pred.get(i) is None for i in ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    a = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    labels = json.load(open(f'{R}/labels.json'))
    ids = sorted(labels)
    res = json.load(open(f'{R}/vlm_result.json')) if os.path.exists(f'{R}/vlm_result.json') else {}
    for mid in a.models:
        proc = AutoProcessor.from_pretrained(mid)
        model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, device_map='cuda').eval()
        short = mid.split('/')[-1]
        for pk, prompt in PROMPTS.items():
            pred, raw = {}, {}
            for sid in ids:
                img = Image.open(f'{R}/panels/{sid}.jpg').convert('RGB')
                msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': prompt}]}]
                inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors='pt').to(model.device)
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=16, do_sample=False)
                txt = proc.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                raw[sid], pred[sid] = txt, parse(txt)
            json.dump(dict(pred=pred, raw=raw), open(f'{R}/pred_{short}_{pk}.json', 'w'), indent=1)
            res[f'{short}/{pk}'] = score(pred, labels)
            print(short, pk, res[f'{short}/{pk}'], flush=True)
            json.dump(res, open(f'{R}/vlm_result.json', 'w'), indent=1)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
