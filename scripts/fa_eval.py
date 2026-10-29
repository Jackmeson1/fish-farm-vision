"""False alarms of a dead-fish detector on real underwater video that contains no dead fish (Brackish test videos:
15 videos, 2743 frames). Every 'dead' box is a false alarm; reported per 1000 frames and as the share of frames with
at least one, at several score thresholds.
  fa_eval.py --yolo WEIGHTS --name N      (class 1 = fish-dead)
  fa_eval.py --gdino --name N             (Grounding DINO prompt 'dead fish')
"""
import argparse
import glob
import json
import os

R = '/root/fish'
THRS = (0.25, 0.35, 0.5, 0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yolo')
    ap.add_argument('--gdino', action='store_true')
    ap.add_argument('--name', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    a = ap.parse_args()
    files = sorted(glob.glob(f'{R}/data/brk/coco/test/*.jpg'))
    scores = []  # per frame: list of dead-fish scores
    if a.yolo:
        from ultralytics import YOLO
        m = YOLO(a.yolo)
        for i in range(0, len(files), 32):
            for r in m.predict(files[i:i + 32], imgsz=a.imgsz, conf=0.2, verbose=False):
                scores.append([float(s) for s, c in zip(r.boxes.conf.tolist(), r.boxes.cls.tolist()) if int(c) == 1])
    else:
        import torch
        from PIL import Image
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        mid = 'IDEA-Research/grounding-dino-base'
        proc = AutoProcessor.from_pretrained(mid)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).cuda().eval()
        for f in files:
            im = Image.open(f).convert('RGB')
            inp = proc(images=im, text='dead fish.', return_tensors='pt').to('cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                out = model(**inp)
            r = proc.post_process_grounded_object_detection(out, inp['input_ids'], threshold=0.2, text_threshold=0.2,
                                                            target_sizes=[im.size[::-1]])[0]
            scores.append([float(s) for s in r['scores'].tolist()])
    res = dict(name=a.name, frames=len(files))
    for t in THRS:
        n = sum(sum(s >= t for s in fr) for fr in scores)
        res[f'fa_per_1000_frames@{t}'] = round(1000 * n / len(files), 1)
        res[f'frames_with_fa@{t}'] = round(sum(any(s >= t for s in fr) for fr in scores) / len(files), 3)
    os.makedirs(f'{R}/runs/{a.name}', exist_ok=True)
    json.dump(res, open(f'{R}/runs/{a.name}/fa_result.json', 'w'), indent=1)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
