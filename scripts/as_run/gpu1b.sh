#!/bin/bash
# fish1 follow-up queue (reviewer requests), after queue_fish1.sh: causal leakage probe and the 2x2 enhancement grid.
cd /root/fish
P=venv/bin/python
while pgrep -f '[q]ueue_fish1.sh' >/dev/null; do sleep 60; done
$P scripts/make_leak2.py
# causal leakage probe: same test frames, equal-size training sets with / without test-video frames
$P scripts/train_yolo.py --model yolo26s.pt --name y26s_clean_on_half --coco data/brk/coco_testhalf --eval-only runs/y26s_leak/weights/best.pt
$P scripts/train_yolo.py --model yolo26s.pt --name y26s_leak2 --data data/brk/yolo/data_leak2.yaml --coco data/brk/coco_testhalf --epochs 30
# 2x2 enhancement grid: cross-evaluate raw- and enhanced-trained models on the other test images
for m in clahe nafnet; do
  $P scripts/train_yolo.py --model yolo26s.pt --name y26s_raw_on_$m --coco data/brk_$m/coco --eval-only runs/y26s_raw/weights/best.pt
  $P scripts/train_yolo.py --model yolo26s.pt --name y26s_${m}_on_raw --eval-only runs/y26s_$m/weights/best.pt
done
echo QUEUE1B_DONE
