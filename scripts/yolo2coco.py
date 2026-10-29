"""Generic YOLO-layout dataset -> COCO GT in the house class order (1 fish-alive, 2 fish-dead), with symlinked images.
  yolo2coco.py --src DIR --names Dead,Live --out DIR      (names in the source's class order; matched by keyword)
"""
import argparse
import json
import os

from PIL import Image

HOUSE = ['fish-alive', 'fish-dead']


def house_id(name):
    n = name.lower()
    return 2 if ('dead' in n or 'mort' in n) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--names', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    names = a.names.split(',')
    for split in ('train', 'valid', 'test'):
        img_dir = os.path.join(a.src, split, 'images')
        if not os.path.isdir(img_dir):
            continue
        out = os.path.join(a.out, split)
        os.makedirs(out, exist_ok=True)
        images, anns = [], []
        for i, fn in enumerate(sorted(os.listdir(img_dir)), start=1):
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            p = os.path.join(img_dir, fn)
            w, h = Image.open(p).size
            if not os.path.lexists(os.path.join(out, fn)):
                os.symlink(os.path.abspath(p), os.path.join(out, fn))
            images.append(dict(id=i, file_name=fn, width=w, height=h))
            lbl = os.path.join(a.src, split, 'labels', os.path.splitext(fn)[0] + '.txt')
            for line in open(lbl) if os.path.exists(lbl) else []:
                p5 = line.split()
                if len(p5) < 5:
                    continue
                c, cx, cy, bw, bh = int(p5[0]), *map(float, p5[1:5])
                anns.append(dict(id=len(anns) + 1, image_id=i, category_id=house_id(names[c]), iscrowd=0,
                                 bbox=[(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h], area=bw * w * bh * h))
        json.dump(dict(images=images, annotations=anns, categories=[dict(id=k + 1, name=n) for k, n in enumerate(HOUSE)]),
                  open(os.path.join(out, '_annotations.coco.json'), 'w'))
        print(split, len(images), 'images', len(anns), 'boxes',
              {n: sum(x['category_id'] == k + 1 for x in anns) for k, n in enumerate(HOUSE)})


if __name__ == '__main__':
    main()
