#!/bin/bash
# fish2 queue: low-label RF-DETR-M after the full RF-DETR-M run.
cd /root/fish
P=venv/bin/python
while pgrep -f '[t]rain_rfdetr.py --variant RFDETRMedium --name rfdm ' >/dev/null; do sleep 30; done
$P scripts/train_rfdetr.py --variant RFDETRMedium --name rfdm_1pct --data data/brk/coco_1pct --epochs 100 --eval-interval 20
$P scripts/train_rfdetr.py --variant RFDETRMedium --name rfdm_10pct --data data/brk/coco_10pct --epochs 30 --eval-interval 10
echo QUEUE2_DONE
