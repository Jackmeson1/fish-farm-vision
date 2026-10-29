"""Evaluator self-check: GT boxes fed back as detections must score AP = 1.0."""
import json
import sys

sys.path.insert(0, '/root/fish/scripts')
from eval_coco import evaluate

gt = '/root/fish/data/brk/coco/test/_annotations.coco.json'
g = json.load(open(gt))
d = [dict(image_id=a['image_id'], category_id=a['category_id'], bbox=a['bbox'], score=1.0) for a in g['annotations']]
r = evaluate(gt, d)
print('GT-as-dets AP', r['AP'], 'AP50', r['AP50'], {k: v['AP'] for k, v in r['per_class'].items()})
