#!/bin/bash
# make_enhanced.sh METHOD: enhanced copy of the video-disjoint Brackish split (images re-encoded, labels linked)
#   -> data/brk_METHOD/yolo/{train,val,test}/{images,labels} + data.yaml, data/brk_METHOD/coco/test (for the evaluator)
set -euo pipefail
m=$1
R=/root/fish
D=$R/data/brk_$m
cd $R
export MPLBACKEND=Agg
for s in train val test; do
  venv/bin/python scripts/uie.py apply --method $m $R/data/brk/yolo/$s/images $D/yolo/$s/images
  ln -sfn $R/data/brk/yolo/$s/labels $D/yolo/$s/labels
done
sed "s#^path: .*#path: $D/yolo#" $R/data/brk/yolo/data.yaml > $D/yolo/data.yaml
mkdir -p $D/coco/test
cp $R/data/brk/coco/test/_annotations.coco.json $D/coco/test/
for f in $D/yolo/test/images/*.jpg; do ln -sf "$f" $D/coco/test/; done
echo "enhanced $m: $(ls $D/yolo/train/images | wc -l) train, $(ls $D/coco/test | wc -l) test files"
