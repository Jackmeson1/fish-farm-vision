#!/bin/bash
# Machine setup (tested on a Google Colab VM with an RTX PRO 6000, Blackwell sm_120): venv, data, sources, split.
# Expects this scripts/ folder at /root/fish/scripts.
set -euo pipefail
R=/root/fish
mkdir -p $R/{data,runs,logs,src,weights}
cd $R

uv venv --managed-python --python 3.12 --allow-existing venv
PY=$R/venv/bin/python
uv pip install --python $PY torch torchvision
uv pip install --python $PY ultralytics "rfdetr[train,loggers]" transformers accelerate pycocotools \
  opencv-python-headless "huggingface_hub[hf_xet]" scikit-image loguru tabulate thop ninja timm supervision pandas \
  matplotlib psutil tensorboard tqdm dominate visdom packaging
MPLBACKEND=Agg $PY -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"

# HF anonymous resolves are rate-limited (~600 files/min); git-lfs's batch API fetches many-file repos far faster.
command -v git-lfs >/dev/null || (apt-get update -q >/dev/null && apt-get install -y -q git-lfs >/dev/null)
lfs_clone() {  # repo dir
  [ -d "$2/.git" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 "https://huggingface.co/datasets/$1" "$2"
  git -C "$2" config lfs.concurrenttransfers 64
  git -C "$2" lfs install --local >/dev/null
  for try in 1 2 3 4 5; do
    git -C "$2" lfs pull || true
    # an incomplete pull leaves LFS pointer files (~130 bytes) in place of images
    left=$(find "$2" -path "$2/.git" -prune -o \( -name '*.jpg' -o -name '*.png' \) -size -1000c -print | wc -l)
    [ "$left" = 0 ] && return 0
    echo "lfs_clone $1: $left pointer files left after try $try"
  done
  return 1
}
lfs_clone dronefreak/Brackish $R/data/brk_git & p1=$!
lfs_clone Edddddd8787/UIEB $R/data/uieb_git & p2=$!
wait $p1
wait $p2
[ -d $R/src/YOLOX ] || git clone -q --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git $R/src/YOLOX
[ -d $R/src/CUT ] || git clone -q --depth 1 https://github.com/taesungp/contrastive-unpaired-translation.git $R/src/CUT
sed -i "s/choices='(CUT, cut, FastCUT, fastcut)'/choices=('CUT', 'cut', 'FastCUT', 'fastcut')/" $R/src/CUT/models/cut_model.py
[ -f $R/weights/yolox_l.pth ] || curl -sL -o $R/weights/yolox_l.pth \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth

rm -rf data/brk
$PY -I scripts/prep_brackish.py > logs/prep_brackish.json
mkdir -p data/brk/yolox/annotations
for s in train valid test; do
  ln -sfn $R/data/brk/coco/$s data/brk/yolox/$s
  ln -sf $R/data/brk/coco/$s/_annotations.coco.json data/brk/yolox/annotations/$s.json
done
ln -sf $R/data/brk/coco_1pct/train/_annotations.coco.json data/brk/yolox/annotations/train_1pct.json
$PY scripts/uie.py split

uv pip freeze --python $PY > $R/logs/freeze.txt
date > $R/setup.done
echo SETUP_DONE
