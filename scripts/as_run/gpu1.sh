#!/bin/bash
# fish1 queue: runs after the YOLO26-L main run, one GPU job at a time (NAFNet trains alongside until it ends).
cd /root/fish
P=venv/bin/python
while pgrep -f '[t]rain_yolo.py --model yolo26l.pt --name y26l ' >/dev/null; do sleep 30; done
# F: leakage probe, one model scored on unseen videos (coco/test) and on seen-video frames (coco_seen/test)
$P scripts/train_yolo.py --model yolo26s.pt --name y26s_leak --data data/brk/yolo/data_leak.yaml --epochs 30
$P scripts/train_yolo.py --model yolo26s.pt --name y26s_leak_seen --coco data/brk/coco_seen --eval-only runs/y26s_leak/weights/best.pt
# D baseline: raw images
$P scripts/train_yolo.py --model yolo26s.pt --name y26s_raw --epochs 30
# B: low-label YOLO26-L
$P scripts/train_yolo.py --model yolo26l.pt --name y26l_1pct --data data/brk/yolo/data_1pct.yaml --epochs 200 --no-val
$P scripts/train_yolo.py --model yolo26l.pt --name y26l_10pct --data data/brk/yolo/data_10pct.yaml --epochs 60 --no-val
# D: enhanced copies (CLAHE; NAFNet once its weights exist)
bash scripts/make_enhanced.sh clahe
$P scripts/train_yolo.py --model yolo26s.pt --name y26s_clahe --data data/brk_clahe/yolo/data.yaml --coco data/brk_clahe/coco --epochs 30
while [ ! -f runs/uie_nafnet/result.json ]; do sleep 30; done
bash scripts/make_enhanced.sh nafnet
$P scripts/train_yolo.py --model yolo26s.pt --name y26s_nafnet --data data/brk_nafnet/yolo/data.yaml --coco data/brk_nafnet/coco --epochs 30
echo QUEUE1_DONE
