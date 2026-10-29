#!/bin/bash
# RAS-MortDB audit: class names, sessions/dates per split, near-duplicates, COCO conversion, contact sheet.
cd /root/fish
D=$(find data/rasmort -type d -name dataset | head -1)
[ -n "$D" ] || D=$(dirname "$(find data/rasmort -type d -name test | head -1)")
echo "dataset dir: $D"; ls "$D"
Y=$(find "$D" -maxdepth 1 -name '*.y*ml' | head -1); echo "yaml: $Y"; cat "$Y"
for s in train valid test; do
  echo "$s sessions: $(ls $D/$s/images | sed -E 's/^tl_([0-9]+)_.*/\1/' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  echo "$s dates: $(ls $D/$s/images | sed -E 's/^tl_[0-9]+_[0-9]+_([0-9]{8})_.*/\1/' | sort -u | tr '\n' ' ')"
done
ln -sfn "$(realpath $D)" data/rasmort/ds
venv/bin/python scripts/phash_audit.py data/rasmort/ds --out runs/rasmort_phash.json
NAMES=$(venv/bin/python -c "import yaml,sys; d=yaml.safe_load(open('$Y')); n=d['names']; print(','.join(n.values() if isinstance(n,dict) else n))")
echo "names: $NAMES"
venv/bin/python scripts/yolo2coco.py --src data/rasmort/ds --names "$NAMES" --out data/rasmort/coco
venv/bin/python scripts/contact_sheet.py data/rasmort/ds/test/images runs/ras_test_sheet.jpg 8
echo "$NAMES" > data/rasmort/names.txt
