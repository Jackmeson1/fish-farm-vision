#!/bin/bash
# RAS-MortDB v1.1.0 (Zenodo 10.5281/zenodo.21829219, dataset CC BY 4.0, weights AGPL-3.0): resume, verify, unpack.
cd /root/fish/data/rasmort
for i in 1 2 3 4 5; do
  curl -sL -C - -o ras.zip 'https://zenodo.org/records/21829219/files/PA-RRanjan/RAS-MortDB-v1.1.0.zip?download=1'
  echo "69c5da1caf67bcdf56b468a2bdadbdc1  ras.zip" | md5sum -c - && break
done
unzip -q -o ras.zip
find . -maxdepth 4 -type d | head -20
for s in train valid test; do d=$(find . -type d -name $s | head -1); echo "$s: $(ls $d/images | wc -l) images; e.g. $(ls $d/images | head -3 | tr '\n' ' ')"; done
cat $(find . -name data.yaml | head -1)
echo RASMORT_READY
