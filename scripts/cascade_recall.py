"""Front end of the dead-fish cascade: how many true dead fish does a detector box at all (any class), and how many
candidates per image does that cost? A frontier VLM then only has to judge these candidates."""
import json
import sys

sys.path.insert(0, '/root/fish/scripts')
from ras_persist import iou  # noqa: E402

GT = '/root/fish/data/rasmort/blocked_coco/test/_annotations.coco.json'
gt = json.load(open(GT))
dead = {}
for a in gt['annotations']:
    if a['category_id'] == 2:
        dead.setdefault(a['image_id'], []).append(a['bbox'])
n_dead = sum(map(len, dead.values()))
out = {}
for name in ('ras_y26s_blocked_on_blocked', 'ras_zs_gdino', 'ras_zs_twostage'):
    dets = json.load(open(f'/root/fish/runs/{name}/test_dets.json'))
    for thr in (0.1, 0.25, 0.5):
        by = {}
        for d in dets:
            if d['score'] >= thr:
                by.setdefault(d['image_id'], []).append(d['bbox'])
        # class-agnostic: merge both classes, dedupe identical boxes
        hit = sum(any(iou(b, c) >= 0.5 for c in by.get(i, [])) for i, bs in dead.items() for b in bs)
        cands = sum(len({tuple(round(v) for v in b) for b in v}) for v in by.values()) / len(gt['images'])
        out[f'{name}@{thr}'] = dict(dead_recall_any_class=round(hit / n_dead, 3), candidates_per_image=round(cands, 1))
print(json.dumps(dict(dead_gt=n_dead, images=len(gt['images']), **out), indent=1))
json.dump(out, open('/root/fish/runs/cascade_recall.json', 'w'), indent=1)
