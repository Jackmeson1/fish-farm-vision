#!/bin/bash
# Dead-fish proxy baselines: zero-label (direct prompts, detect-then-CLIP) and supervised YOLO26-S on its own labels.
cd /root/fish
P=venv/bin/python
$P scripts/df3_prep.py
for s in test valid; do
  $P scripts/df3_zeroshot.py direct --split $s
  $P scripts/df3_zeroshot.py twostage --split $s
done
$P scripts/train_yolo.py --model yolo26s.pt --name df3_y26s --data data/deadfish/data.yaml --coco data/deadfish/coco --epochs 50
echo QUEUE_DF3_DONE
