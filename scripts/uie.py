"""Underwater image enhancement on UIEB: classic operators, a paired NAFNet, and PSNR/SSIM on the held-out 90.

  uie.py split                      800/90 split of UIEB (seed 0) -> /root/fish/data/uieb_split/{train,test}/{raw,ref}
  uie.py classic                    identity / gray-world / CLAHE / MSRCR on test -> runs/uie_<m>/
  uie.py train-nafnet               paired L1 training on train crops -> runs/uie_nafnet/nafnet.pth
  uie.py apply --method M SRC DST   enhance every image under SRC into DST (for the detection ablation)
  uie.py score DIR [--name N]       PSNR/SSIM of DIR/*.png against the test references
"""
import argparse
import glob
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

R = '/root/fish'
SPLIT = f'{R}/data/uieb_split'


# ---------------------------------------------------------------- classic operators (BGR uint8 in/out)
def gray_world(img):
    f = img.astype(np.float32)
    mean = f.reshape(-1, 3).mean(0)
    return np.clip(f * (mean.mean() / (mean + 1e-6)), 0, 255).astype(np.uint8)


def clahe(img, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def msrcr(img, sigmas=(15, 80, 250), alpha=125.0, beta=46.0, gain=192.0, offset=-30.0):
    """Multi-scale Retinex with colour restoration (Jobson et al. 1997), a classic baseline."""
    f = img.astype(np.float64) + 1.0
    msr = sum(np.log10(f) - np.log10(cv2.GaussianBlur(f, (0, 0), s)) for s in sigmas) / len(sigmas)
    crf = beta * (np.log10(alpha * f) - np.log10(f.sum(axis=2, keepdims=True)))
    out = gain * (msr * crf) + offset
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)


CLASSIC = {'identity': lambda x: x, 'grayworld': gray_world, 'clahe': clahe, 'grayworld_clahe': lambda x: clahe(
    gray_world(x)), 'msrcr': msrcr}


# ---------------------------------------------------------------- NAFNet (Chen et al. 2022), compact version
class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / (var + 1e-6).sqrt() * self.w[:, None, None] + self.b[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.c3 = nn.Conv2d(c, c, 1)
        self.c4 = nn.Conv2d(c, 2 * c, 1)
        self.c5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.sg(self.dw(self.c1(self.n1(x))))
        y = self.c3(y * self.sca(y))
        x = x + y * self.beta
        y = self.c5(self.sg(self.c4(self.n2(x))))
        return x + y * self.gamma


class NAFNet(nn.Module):
    def __init__(self, width=32, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2)):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = (nn.ModuleList() for _ in range(4))
        c = width
        for n in enc:
            self.encs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.pad = 2 ** len(enc)

    def forward(self, inp):
        h, w = inp.shape[-2:]
        ph, pw = (-h) % self.pad, (-w) % self.pad
        x = self.intro(F.pad(inp, (0, pw, 0, ph), mode='reflect'))
        skips = []
        for e, d in zip(self.encs, self.downs):
            x = e(x)
            skips.append(x)
            x = d(x)
        x = self.mid(x)
        for u, d, s in zip(self.ups, self.decs, reversed(skips)):
            x = d(u(x) + s)
        return (self.ending(x)[..., :h, :w] + F.pad(inp, (0, pw, 0, ph))[..., :h, :w]).clamp(0, 1)


def load_nafnet(path=f'{R}/runs/uie_nafnet/nafnet.pth'):
    net = NAFNet().cuda().eval()
    net.load_state_dict(torch.load(path, map_location='cuda'))
    return net


@torch.no_grad()
def nafnet_enhance(net, img, max_side=None):
    h, w = img.shape[:2]
    x = img
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        x = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255
    with torch.autocast('cuda', dtype=torch.bfloat16):
        y = net(t).float()
    out = (y[0].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).round().clip(0, 255).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC) if out.shape[:2] != (h, w) else out


# ---------------------------------------------------------------- commands
def cmd_split(_):
    src = f'{R}/data/uieb_git'
    names = sorted(os.listdir(f'{src}/raw-890'))
    assert names == sorted(os.listdir(f'{src}/reference-890')), 'raw/reference names differ'
    random.Random(0).shuffle(names)
    for part, ns in (('train', names[:800]), ('test', names[800:])):
        for kind, d in (('raw', 'raw-890'), ('ref', 'reference-890')):
            os.makedirs(f'{SPLIT}/{part}/{kind}', exist_ok=True)
            for n in ns:
                dst = f'{SPLIT}/{part}/{kind}/{n}'
                if not os.path.lexists(dst):
                    os.symlink(f'{src}/{d}/{n}', dst)
    print('train 800, test', len(names) - 800)


def score_dir(d):
    ps, ss = [], []
    for ref in sorted(glob.glob(f'{SPLIT}/test/ref/*')):
        n = os.path.splitext(os.path.basename(ref))[0]
        out = cv2.imread(f'{d}/{n}.png')
        gt = cv2.imread(ref)
        if out.shape != gt.shape:
            out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        ps.append(peak_signal_noise_ratio(gt, out, data_range=255))
        ss.append(structural_similarity(gt, out, channel_axis=2, data_range=255))
    return dict(PSNR=round(float(np.mean(ps)), 3), SSIM=round(float(np.mean(ss)), 4), n=len(ps))


def cmd_classic(_):
    res = {}
    for m, fn in CLASSIC.items():
        d = f'{R}/runs/uie_{m}'
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        files = sorted(glob.glob(f'{SPLIT}/test/raw/*'))
        for f in files:
            cv2.imwrite(f'{d}/{os.path.splitext(os.path.basename(f))[0]}.png', fn(cv2.imread(f)))
        r = score_dir(d)
        r['sec_per_img'] = round((time.time() - t0) / len(files), 4)
        json.dump(r, open(f'{d}/result.json', 'w'), indent=1)
        res[m] = r
    print(json.dumps(res, indent=1))


def cmd_train_nafnet(a):
    raws = sorted(glob.glob(f'{SPLIT}/train/raw/*'))
    pairs = [(cv2.imread(f)[..., ::-1], cv2.imread(f.replace('/raw/', '/ref/'))[..., ::-1]) for f in raws]
    net = NAFNet().cuda().train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=1e-6)
    rng = np.random.default_rng(0)
    out = f'{R}/runs/uie_nafnet'
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        xb, yb = [], []
        for _ in range(a.batch):
            x, y = pairs[rng.integers(len(pairs))]
            h, w = x.shape[:2]
            s = a.crop
            if min(h, w) < s:
                k = s / min(h, w)
                x = cv2.resize(x, (int(w * k) + 1, int(h * k) + 1))
                y = cv2.resize(y, (int(w * k) + 1, int(h * k) + 1))
                h, w = x.shape[:2]
            i, j = rng.integers(h - s + 1), rng.integers(w - s + 1)
            x, y = x[i:i + s, j:j + s], y[i:i + s, j:j + s]
            if rng.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            k = rng.integers(4)
            xb.append(np.rot90(x, k).copy())
            yb.append(np.rot90(y, k).copy())
        xt = torch.from_numpy(np.stack(xb)).permute(0, 3, 1, 2).float().cuda() / 255
        yt = torch.from_numpy(np.stack(yb)).permute(0, 3, 1, 2).float().cuda() / 255
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.l1_loss(net(xt).float(), yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01 * 100)
        opt.step()
        sched.step()
        if it % 500 == 0:
            print(f'it {it} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
    torch.save(net.state_dict(), f'{out}/nafnet.pth')
    net.eval()
    t1 = time.time()
    files = sorted(glob.glob(f'{SPLIT}/test/raw/*'))
    for f in files:
        cv2.imwrite(f'{out}/{os.path.splitext(os.path.basename(f))[0]}.png', nafnet_enhance(net, cv2.imread(f)))
    r = score_dir(out)
    r.update(iters=a.iters, batch=a.batch, crop=a.crop, train_seconds=round(t1 - t0),
             params_M=round(sum(p.numel() for p in net.parameters()) / 1e6, 2),
             sec_per_img=round((time.time() - t1) / len(files), 4))
    json.dump(r, open(f'{out}/result.json', 'w'), indent=1)
    print(json.dumps(r, indent=1))


def cmd_apply(a):
    net = load_nafnet() if a.method == 'nafnet' else None
    fn = (lambda im: nafnet_enhance(net, im, max_side=a.max_side)) if net else CLASSIC[a.method]
    files = sorted(glob.glob(f'{a.src}/**/*.jpg', recursive=True))
    t0 = time.time()
    for k, f in enumerate(files):
        dst = os.path.join(a.dst, os.path.relpath(f, a.src))
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, fn(cv2.imread(f)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if k % 1000 == 0:
            print(k, len(files), f'{time.time() - t0:.0f}s', flush=True)
    print('done', len(files))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('split')
    sub.add_parser('classic')
    t = sub.add_parser('train-nafnet')
    t.add_argument('--iters', type=int, default=40000)
    t.add_argument('--batch', type=int, default=16)
    t.add_argument('--crop', type=int, default=256)
    p = sub.add_parser('apply')
    p.add_argument('--method', required=True)
    p.add_argument('--max-side', type=int, default=1280)
    p.add_argument('src')
    p.add_argument('dst')
    s = sub.add_parser('score')
    s.add_argument('dir')
    a = ap.parse_args()
    if a.cmd == 'score':
        print(json.dumps(score_dir(a.dir), indent=1))
    else:
        {'split': cmd_split, 'classic': cmd_classic, 'train-nafnet': cmd_train_nafnet, 'apply': cmd_apply}[a.cmd](a)


if __name__ == '__main__':
    main()
