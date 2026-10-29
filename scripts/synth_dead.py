"""Synthetic 'dead fish on the bottom of an underwater scene' training set (no real dead-fish labels anywhere).

  background  a Brackish TRAINING-video frame (never a test video); its real live fish keep their ground-truth boxes as
              'fish-alive'; crabs, starfish and rocks stay unlabeled as static hard negatives; half the frames get a
              procedural net-mesh overlay (aquaculture pen look)
  dead fish   Fish-Vista museum specimens (CC0 / CC BY only; the 80% 'train' part of a fixed hash split), cut out from the white background, rotated, scaled,
              flipped, paled and desaturated (post-mortem colour loss), placed with a bias towards the lower half (sunk)
  water       Akkaynak-Treibitz style per-object rendering: I = J * exp(-bD * z) + Binf * (1 - exp(-bB * z)), with the
              veiling light Binf taken from the background and red attenuating fastest
  -> data/synth_dead/yolo/{train,val}/{images,labels} + data.yaml (classes: 0 fish-alive, 1 fish-dead, as in df3)
"""
import argparse
import csv
import glob
import hashlib
import os
import random

import cv2
import numpy as np

R = '/root/fish'
W, H = 1280, 720


def is_test_specimen(file_name):
    """Fixed 80/20 specimen split by filename hash: test specimens only ever appear in test videos."""
    return int(hashlib.md5(file_name.encode()).hexdigest(), 16) % 5 == 0


def specimen_cutouts(n_max, rng, part='train'):
    rows = [r for r in csv.DictReader(open(f'{R}/data/fishvista/selected.csv', encoding='utf-8'))
            if is_test_specimen(r['file_name']) == (part == 'test')]
    rng.shuffle(rows)
    cuts = []
    for r in rows[:n_max]:
        img = cv2.imread(f'{R}/data/fishvista/{r["file_name"]}')
        if img is None:
            continue
        fg = (img.min(axis=2) < 225).astype(np.uint8)  # white background
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
        if n < 2:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = stats[k, :4]
        if w < 40 or h < 15 or stats[k, cv2.CC_STAT_AREA] < 0.3 * w * h:
            continue
        m = (lab[y:y + h, x:x + w] == k).astype(np.uint8) * 255
        cuts.append((img[y:y + h, x:x + w].copy(), m))
    return cuts


def net_overlay(bg, rng):
    """Rhombic net mesh, slightly darker than the water, partly transparent."""
    mesh = np.zeros((H, W), np.uint8)
    s = rng.randint(28, 90)
    t = rng.randint(1, 3)
    ang = rng.uniform(30, 60)
    for sign in (1, -1):
        dx = s / np.sin(np.radians(ang))
        for x0 in np.arange(-H * 2, W + H * 2, dx):
            x1 = x0 + sign * H / np.tan(np.radians(ang))
            cv2.line(mesh, (int(x0), 0), (int(x1), H), 255, t, cv2.LINE_AA)
    a = (mesh.astype(np.float32) / 255 * rng.uniform(0.25, 0.6))[..., None]
    colour = np.array(bg.reshape(-1, 3).mean(0) * rng.uniform(0.3, 0.6), np.float32)
    return (bg.astype(np.float32) * (1 - a) + colour * a).clip(0, 255).astype(np.uint8)


def paste(bg, cut, rng):
    img, m = cut
    # post-mortem look: desaturate and pale
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.4, 1.0)
    img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    pale = rng.uniform(0, 0.35)
    img = img * (1 - pale) + 200 * pale
    if rng.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    if rng.random() < 0.3:
        img, m = img[::-1], m[::-1]  # upside down (belly-up look in side view)
    # scale to a body length of 3-22% of the frame width, then rotate
    L = rng.uniform(0.03, 0.22) * W
    s = L / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(np.ascontiguousarray(m), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    h, w = m.shape
    ang = rng.uniform(0, 360)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (nw, nh), flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(np.ascontiguousarray(m), M, (nw, nh), flags=cv2.INTER_NEAREST)
    if nw >= W or nh >= H:
        return bg, None
    x = rng.randint(0, W - nw)
    y = int(min(H - nh, max(0, rng.triangular(0, H - nh, H - nh))))  # bias towards the bottom
    # water: veiling light from the background, red attenuating fastest
    binf = np.median(bg[: H // 4].reshape(-1, 3), axis=0).astype(np.float32)
    z = rng.uniform(0.3, 3.0)
    bD = np.array([rng.uniform(0.15, 0.45), rng.uniform(0.1, 0.35), rng.uniform(0.5, 1.2)], np.float32)  # B, G, R
    bB = rng.uniform(0.25, 0.8)
    img = img * np.exp(-bD * z) + binf * (1 - np.exp(-bB * z))
    k = max(1, int(round(z)))
    img = cv2.GaussianBlur(img, (2 * k + 1, 2 * k + 1), 0)
    a = cv2.GaussianBlur(m.astype(np.float32) / 255, (5, 5), 0)[..., None]
    roi = bg[y:y + nh, x:x + nw].astype(np.float32)
    bg[y:y + nh, x:x + nw] = (roi * (1 - a) + img * a).clip(0, 255).astype(np.uint8)
    ys, xs = np.nonzero(m > 127)
    if len(xs) == 0:
        return bg, None
    return bg, (x + xs.min(), y + ys.min(), x + xs.max(), y + ys.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6000)
    ap.add_argument('--val', type=int, default=300)
    a = ap.parse_args()
    rng = random.Random(0)
    cuts = specimen_cutouts(3000, rng)
    print('specimen cutouts', len(cuts), flush=True)
    bgs = sorted(glob.glob(f'{R}/data/brk/yolo/train/images/*.jpg'))
    out = f'{R}/data/synth_dead/yolo'
    for k in range(a.n + a.val):
        split = 'val' if k < a.val else 'train'
        os.makedirs(f'{out}/{split}/images', exist_ok=True)
        os.makedirs(f'{out}/{split}/labels', exist_ok=True)
        f = rng.choice(bgs)
        bg = cv2.resize(cv2.imread(f), (W, H), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            bg = net_overlay(bg, rng)
        lines = []
        for line in open(f.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'):
            c, *box = line.split()
            if c in ('1', '4'):  # Brackish fish, small_fish -> fish-alive
                lines.append('0 ' + ' '.join(box))
        for _ in range(rng.choice([1, 1, 2, 2, 3, 4])):
            bg, b = paste(bg, rng.choice(cuts), rng)
            if b:
                x1, y1, x2, y2 = b
                lines.append('1 %.6f %.6f %.6f %.6f' % ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
        name = f'synth_{k:05d}'
        cv2.imwrite(f'{out}/{split}/images/{name}.jpg', bg, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
        open(f'{out}/{split}/labels/{name}.txt', 'w').write('\n'.join(lines) + ('\n' if lines else ''))
        if k % 1000 == 0:
            print(k, flush=True)
    open(f'{R}/data/synth_dead/data.yaml', 'w').write(
        f'path: {out}\ntrain: train/images\nval: val/images\nnames:\n  0: fish-alive\n  1: fish-dead\n')
    print('done', a.n, a.val)


if __name__ == '__main__':
    main()
