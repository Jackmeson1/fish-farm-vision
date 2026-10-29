#!/bin/bash
# RAS-MortDB (real commercial RAS, dead vs live trout): leakage audit numbers, honest split, zero-label and synthetic
# transfer. Runs on fish4 alongside the synthetic dead-fish queue.
cd /root/fish
P=venv/bin/python
export MPLBACKEND=Agg
$P scripts/yolo2coco.py --src data/rasmort/ds --names Dead,Live --out data/rasmort/coco
$P scripts/ras_resplit.py
$P scripts/yolo2coco.py --src data/rasmort/blocked --names Dead,Live --out data/rasmort/blocked_coco
$P scripts/phash_audit.py data/rasmort/blocked --out runs/rasmort_blocked_phash.json
W=$(find data/rasmort -path '*weights/pytorch/yolo26s_best.pt' | head -1)
# 1) the authors' released YOLO26s on their own (random-frame) test split
$P scripts/dead_eval.py yolo $W --src-names Dead,Live --coco data/rasmort/coco/test --name ras_pub_y26s_on_pubtest
# 2) same recipe trained by us on the published split vs on the session-blocked split (train_yolo's built-in eval
#    assumes the house class order, so RAS models are scored with dead_eval.py, which remaps Dead/Live by name)
printf 'path: %s\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: Dead\n  1: Live\n' \
  "$(realpath data/rasmort/ds)" > data/rasmort/ds_data.yaml
$P scripts/train_yolo.py --model yolo26s.pt --name ras_y26s_pub --data data/rasmort/ds_data.yaml --coco data/rasmort/coco --epochs 50
$P scripts/train_yolo.py --model yolo26s.pt --name ras_y26s_blocked --data data/rasmort/blocked/data.yaml --coco data/rasmort/blocked_coco --epochs 50
$P scripts/dead_eval.py yolo runs/ras_y26s_pub/weights/best.pt --src-names Dead,Live --coco data/rasmort/coco/test --name ras_y26s_pub_on_pubtest
for t in ras_y26s_pub ras_y26s_blocked; do
  $P scripts/dead_eval.py yolo runs/$t/weights/best.pt --src-names Dead,Live --coco data/rasmort/blocked_coco/test --name ${t}_on_blocked
done
# 3) zero-label methods on the honest test split
$P scripts/dead_eval.py gdino --coco data/rasmort/blocked_coco/test --name ras_zs_gdino
$P scripts/dead_eval.py twostage --coco data/rasmort/blocked_coco/test --name ras_zs_twostage
while [ ! -f runs/dead_synth_y26s/result.json ]; do sleep 60; done
$P scripts/dead_eval.py yolo runs/dead_synth_y26s/weights/best.pt --src-names fish-alive,fish-dead \
  --coco data/rasmort/blocked_coco/test --name ras_synth_on_blocked
echo QUEUE_RAS_DONE
