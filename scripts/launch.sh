#!/bin/bash
# launch.sh NAME CMD...: run CMD from /root/fish in the background with a clean env, log to logs/NAME.log
cd /root/fish
name=$1; shift
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONPATH=/root/fish/src/YOLOX PATH=/root/fish/venv/bin:$PATH
nohup "$@" > logs/$name.log 2>&1 < /dev/null &
echo "started $name pid $!"
