#!/bin/bash
# fish4 round 2: two-stage on RAS (CLIP fix), real-data persistence study, then the (simulated) video pipeline.
cd /root/fish
P=venv/bin/python
export MPLBACKEND=Agg
$P scripts/dead_eval.py twostage --coco data/rasmort/blocked_coco/test --name ras_zs_twostage
$P scripts/ras_persist.py ras_y26s_blocked_on_blocked ras_zs_gdino ras_zs_twostage
rm -rf data/vid_dead
bash scripts/queue_video.sh
echo QUEUE4B_DONE
