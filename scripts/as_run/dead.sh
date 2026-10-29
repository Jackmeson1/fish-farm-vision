#!/bin/bash
# Dead-fish synthetic track: fetch CC0/CC BY museum specimens, synthesise, train, then score on the real proxy set
# (df3 basin, recall side) and on real underwater video without dead fish (false alarms).
cd /root/fish
P=venv/bin/python
if [ ! -d data/deadfish/coco ]; then
  mkdir -p data/deadfish && curl -sL -o data/deadfish/df3.zip     'https://zenodo.org/records/23565468/files/dead-fish-3_water-spray_v1_yolo.zip?download=1'
  echo "db7c1fb884ca1bd0aa528b173ff19915  data/deadfish/df3.zip" | md5sum -c - || exit 1
  unzip -q -o data/deadfish/df3.zip -d data/deadfish/df3 && $P scripts/df3_prep.py
fi
$P scripts/fv_fetch.py --n 3000
$P scripts/synth_dead.py --n 6000 --val 300
$P scripts/contact_sheet.py data/synth_dead/yolo/val/images runs/synth_sheet.jpg 8
$P scripts/train_yolo.py --model yolo26s.pt --name dead_synth_y26s --data data/synth_dead/data.yaml \
  --coco data/deadfish/coco --epochs 40
$P scripts/fa_eval.py --yolo runs/dead_synth_y26s/weights/best.pt --name dead_synth_y26s
$P scripts/fa_eval.py --gdino --name dead_gdino
echo QUEUE_DEAD_DONE
