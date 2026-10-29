#!/bin/bash
# fish3 (recreated after Colab reclaimed the first one): full setup, then YOLOX-L on a dedicated GPU and its test eval.
cd /root/fish
bash scripts/setup_box.sh
P=/root/fish/venv/bin/python
cd /root/fish/src/YOLOX
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONPATH=/root/fish/src/YOLOX PATH=/root/fish/venv/bin:$PATH
$P tools/train.py -f /root/fish/scripts/yolox_l_brackish.py -expn yoloxl -d 1 -b 32 --fp16 -c /root/fish/weights/yolox_l.pth \
  > /root/fish/logs/yoloxl.log 2>&1
cd /root/fish
O=src/YOLOX/YOLOX_outputs/yoloxl
$P scripts/yolox_eval.py --ckpt $O/best_ckpt.pth --name yoloxl
$P scripts/yolox_eval.py --ckpt $O/latest_ckpt.pth --name yoloxl_last
echo QUEUE3V2_DONE
