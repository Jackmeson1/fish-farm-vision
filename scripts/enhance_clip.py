"""Classic underwater enhancement (same operators as scripts/uie.py) applied to every frame of one Brackish clip.
  enhance_clip.py SRC_DIR OUT_DIR [MAX_FRAMES]
  -> OUT_DIR/{raw,grayworld_clahe}/0001.jpg at 1280x720 and OUT_DIR/{clahe,msrcr}/0001.jpg at 640x360
"""
import os
import re
import sys

import cv2
import numpy as np


def gray_world(img):
    f = img.astype(np.float32)
    mean = f.reshape(-1, 3).mean(0)
    return np.clip(f * (mean.mean() / (mean + 1e-6)), 0, 255).astype(np.uint8)


def clahe(img, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def msrcr(img, sigmas=(15, 80, 250), alpha=125.0, beta=46.0, gain=192.0, offset=-30.0):
    f = img.astype(np.float64) + 1.0
    msr = sum(np.log10(f) - np.log10(cv2.GaussianBlur(f, (0, 0), s)) for s in sigmas) / len(sigmas)
    crf = beta * (np.log10(alpha * f) - np.log10(f.sum(axis=2, keepdims=True)))
    out = gain * (msr * crf) + offset
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)


def main():
    src, out = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 6
    fr = re.compile(r'-(\d{4})_jpg\.rf\.')
    files = sorted((int(fr.search(f).group(1)), f) for f in os.listdir(src) if fr.search(f))[:cap]
    for d in ('raw', 'grayworld_clahe', 'clahe', 'msrcr'):
        os.makedirs(f'{out}/{d}', exist_ok=True)
    for k, (_, f) in enumerate(files, start=1):
        img = cv2.imread(os.path.join(src, f))
        name = f'{k:04d}.jpg'
        big = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{out}/raw/{name}', big, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # operators run at full resolution (as in the experiment), then resized
        cv2.imwrite(f'{out}/grayworld_clahe/{name}', cv2.resize(clahe(gray_world(img)), (1280, 720),
                    interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
        small = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{out}/clahe/{name}', clahe(small), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(f'{out}/msrcr/{name}', msrcr(small), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out, len(files), 'frames')


if __name__ == '__main__':
    main()
