#!/bin/bash
# fish4: video dead-fish pipeline once the single-frame candidate model is trained.
cd /root/fish
P=venv/bin/python
while [ ! -f runs/dead_synth_y26s/result.json ]; do sleep 60; done
W=runs/dead_synth_y26s/weights/best.pt
$P scripts/video_dead.py synth --split train
$P scripts/video_dead.py synth --split test
$P scripts/video_dead.py detect --split train --weights $W
$P scripts/video_dead.py detect --split test --weights $W
$P scripts/video_dead.py verify
echo QUEUE_VIDEO_DONE
