#!/bin/bash
# fish4 (A100): setup, then the dead-fish synthetic single-frame stage (candidate detector for the video pipeline).
cd /root/fish
bash scripts/setup_box.sh
bash scripts/queue_dead.sh
echo QUEUE4_DONE
