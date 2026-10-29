#!/bin/bash
# fish2: the resolution confound (YOLO at 640 shrinks 1920x1080 frames 3x; Grounding DINO runs at ~1333x750).
cd /root/fish
P=venv/bin/python
$P scripts/train_yolo.py --model yolo26s.pt --name y26s_raw_1280 --imgsz 1280 --batch 16 --epochs 30
$P scripts/train_yolo.py --model yolo26s.pt --name y26s_pl_naive_1280 --data data/brk_pl_naive/yolo/data.yaml --imgsz 1280 --batch 16 --epochs 30 --no-val
echo QUEUE2C_DONE
