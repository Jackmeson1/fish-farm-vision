#!/bin/bash
# fish2 flagship queue: label-free self-training. Grounding DINO teaches on the training videos (no labels read),
# pseudo-labels in three variants, one YOLO26-S per variant (last weights, no validation-based selection).
cd /root/fish
P=venv/bin/python
while pgrep -f '[z]eroshot.py sam3' >/dev/null; do sleep 30; done
$P scripts/zeroshot.py gdino --split train
$P scripts/pseudo_labels.py
for v in naive track fill; do
  $P scripts/train_yolo.py --model yolo26s.pt --name y26s_pl_$v --data data/brk_pl_$v/yolo/data.yaml --epochs 30 --no-val
done
echo QUEUE2B_DONE
