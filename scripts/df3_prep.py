"""Dead-fish proxy set (Zenodo 'dead fish 3', CC BY 4.0, lab basin, top view): YOLO splits -> COCO GT + a clean yaml.
  -> data/deadfish/coco/{train,valid,test}/ (symlinked images + _annotations.coco.json), data/deadfish/data.yaml
"""
import json
import os

from PIL import Image

R = '/root/fish/data/deadfish'
NAMES = ['fish-alive', 'fish-dead']


def main():
    for split in ('train', 'valid', 'test'):
        src = f'{R}/df3/{split}'
        out = f'{R}/coco/{split}'
        os.makedirs(out, exist_ok=True)
        images, anns = [], []
        for i, fn in enumerate(sorted(os.listdir(f'{src}/images')), start=1):
            p = f'{src}/images/{fn}'
            w, h = Image.open(p).size
            dst = f'{out}/{fn}'
            if not os.path.lexists(dst):
                os.symlink(p, dst)
            images.append(dict(id=i, file_name=fn, width=w, height=h))
            lbl = f'{src}/labels/{os.path.splitext(fn)[0]}.txt'
            for line in open(lbl) if os.path.exists(lbl) else []:
                c, cx, cy, bw, bh = map(float, line.split()[:5])
                anns.append(dict(id=len(anns) + 1, image_id=i, category_id=int(c) + 1, iscrowd=0,
                                 bbox=[(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h], area=bw * w * bh * h))
        json.dump(dict(images=images, annotations=anns,
                       categories=[dict(id=k + 1, name=n) for k, n in enumerate(NAMES)]),
                  open(f'{out}/_annotations.coco.json', 'w'))
        print(split, len(images), 'images', len(anns), 'boxes',
              {n: sum(a['category_id'] == k + 1 for a in anns) for k, n in enumerate(NAMES)})
    open(f'{R}/data.yaml', 'w').write(
        f'path: {R}/df3\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: fish-alive\n  1: fish-dead\n')


if __name__ == '__main__':
    main()
