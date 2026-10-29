"""RAS-MortDB, leakage-free: the published 70/20/10 split samples time-lapse frames at random (every test session
also appears in train; 83% of test images have a near-duplicate in train/valid). This builds a time-forward split by
capture session (train on earlier sessions, test on later ones, as a farm would deploy), after dropping the Roboflow
augmented copies so every capture appears once.
  -> data/rasmort/blocked/{train,valid,test}/{images,labels} + data.yaml (names Dead, Live as in the source),
     data/rasmort/blocked_coco/... (house order), blocked_manifest.json
"""
import json
import os
import re
from collections import defaultdict

R = '/root/fish/data/rasmort'
NAME_RE = re.compile(r'^tl_(?P<session>\d+)_(?P<frame>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_')


def main():
    src = f'{R}/ds'
    captures = {}  # base capture name -> (split, file)
    for split in ('train', 'valid', 'test'):
        for fn in sorted(os.listdir(f'{src}/{split}/images')):
            m = NAME_RE.match(fn)
            if not m:
                continue
            base = fn.split('_jpg.rf.')[0]
            captures.setdefault(base, (split, fn, m.groupdict()))  # first copy wins; augmented twins dropped
    by_sess = defaultdict(list)
    for base, (split, fn, g) in captures.items():
        by_sess[g['session']].append((g['date'] + g['time'], base, split, fn))
    order = sorted(by_sess, key=lambda s: min(x[0] for x in by_sess[s]))
    total = sum(len(v) for v in by_sess.values())
    plan, acc = {}, 0
    for s in order:
        plan[s] = 'train' if acc < 0.7 * total else ('valid' if acc < 0.8 * total else 'test')
        acc += len(by_sess[s])
    out = f'{R}/blocked'
    counts = defaultdict(int)
    for s, items in by_sess.items():
        dst = plan[s]
        for sub in ('images', 'labels'):
            os.makedirs(f'{out}/{dst}/{sub}', exist_ok=True)
        for _, base, split, fn in items:
            for sub, name in (('images', fn), ('labels', os.path.splitext(fn)[0] + '.txt')):
                p = f'{src}/{split}/{sub}/{name}'
                q = f'{out}/{dst}/{sub}/{name}'
                if os.path.exists(p) and not os.path.lexists(q):
                    os.symlink(os.path.realpath(p), q)
            counts[dst] += 1
    open(f'{out}/data.yaml', 'w').write(
        f'path: {out}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: Dead\n  1: Live\n')
    man = dict(unique_captures=total, sessions=len(order), per_split=dict(counts),
               sessions_per_split={k: [s for s in order if plan[s] == k] for k in ('train', 'valid', 'test')},
               dates_per_split={k: [min(x[0] for s in order if plan[s] == k for x in by_sess[s])[:8],
                                    max(x[0] for s in order if plan[s] == k for x in by_sess[s])[:8]]
                                for k in ('train', 'valid', 'test')})
    json.dump(man, open(f'{R}/blocked_manifest.json', 'w'), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
