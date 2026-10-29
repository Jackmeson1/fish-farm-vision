"""Draw the README figures as SVG, one light and one dark file each, from the CSV tables in results/ and data/.
  python figures/make_figures.py
"""
import csv
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W = 760
FONT = 'system-ui, -apple-system, &quot;Segoe UI&quot;, Helvetica, Arial, sans-serif'
THEMES = {  # page = GitHub's background, used for the 2px rings around markers
    'light': dict(page='#ffffff', ink='#0b0b0b', ink2='#52514e', muted='#6f6d68', grid='#e1e0d9', axis='#c3c2b7',
                  s1='#2a78d6', s2='#eb6834', box='#f6f6f4', box_line='#d6d5cf'),
    'dark': dict(page='#0d1117', ink='#f0efec', ink2='#c3c2b7', muted='#9a9890', grid='#2c2c2a', axis='#4a4a46',
                 s1='#3987e5', s2='#d95926', box='#161b22', box_line='#30363d'),
}


def rows(path):
    with open(os.path.join(ROOT, path)) as fh:
        return list(csv.DictReader(fh))


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class Svg:
    def __init__(self, h, t):
        self.h, self.t, self.parts = h, t, []

    def text(self, x, y, s, size=13, fill='ink', anchor='start', weight=400):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{self.t[fill]}" '
                          f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke='grid', width=1):
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{self.t[stroke]}" stroke-width="{width}"/>')

    def rect(self, x, y, w, h, fill, rx=0, stroke=None):
        s = f' stroke="{self.t[stroke]}" stroke-width="1"' if stroke else ''
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                          f'fill="{self.t.get(fill, fill)}"{s}/>')

    def hbar(self, x0, y, length, h, fill):
        """Bar growing right from x0: square at the baseline, 4px rounded data end."""
        if length <= 0:
            return
        r = min(4, length, h / 2)
        x1 = x0 + length
        self.parts.append(f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} '
                          f'V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" '
                          f'fill="{self.t[fill]}"/>')

    def raw(self, s):
        self.parts.append(s)

    def title(self, title, sub=None):
        self.text(0, 20, title, size=16, weight=600)
        if sub:
            self.text(0, 41, sub, size=13, fill='ink2')

    def legend(self, y, items, x=0):
        for key, label in items:
            self.rect(x, y - 9, 10, 10, key, rx=2)
            self.text(x + 16, y, label, size=13, fill='ink2')
            x += 16 + 7.2 * len(label) + 22

    def save(self, name, alt):
        body = '\n'.join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{self.h}" viewBox="0 0 {W} {self.h}" '
               f'font-family="{FONT}" role="img" aria-label="{esc(alt)}">\n<title>{esc(alt)}</title>\n{body}\n</svg>\n')
        with open(os.path.join(HERE, name), 'w', encoding='utf-8') as fh:
            fh.write(svg)


def both(fn):
    for mode, t in THEMES.items():
        fn(t, '' if mode == 'light' else '-dark')


# 1. how a random-frame split leaks, with the controlled result
def split_diagram(t, suf):
    s = Svg(372, t)
    s.title('Same model, same test frames: 0.72 or 0.31, depending on the split',
            'Brackish dataset. Each strip is one video; each tick is one frame.')
    s.legend(70, [('s1', 'training frame'), ('s2', 'test frame')])
    rng = random.Random(4)
    fw, gap, n = 13, 3, 26
    x0 = 92

    def strip(y, colours, name):
        s.text(x0 - 10, y + 15, name, size=12, fill='muted', anchor='end')
        for i, c in enumerate(colours):
            s.rect(x0 + i * (fw + gap), y, fw, 20, c, rx=2)

    def panel(y, heading, note, vids, score, score_note):
        s.text(0, y, heading, size=14, weight=600)
        s.text(0, y + 19, note, size=12, fill='ink2')
        for k, cols in enumerate(vids):
            strip(y + 32 + k * 28, cols, f'video {k + 1}')
        sx = x0 + n * (fw + gap) + 26
        s.text(sx, y + 70, f'{score:.2f}', size=34, weight=600)
        s.text(sx, y + 92, score_note, size=12, fill='ink2')

    mixed = [['s2' if rng.random() < 0.3 else 's1' for _ in range(n)] for _ in range(3)]
    panel(104, 'Random frames', 'How these datasets are usually split. A test frame sits 1/15 s from a training frame.',
          mixed, 0.721, 'test score (AP)')
    whole = [['s1'] * n, ['s1'] * n, ['s2'] * n]
    panel(250, 'Whole videos', 'Closer to a new camera or a new day. No video is on both sides.',
          whole, 0.314, 'test score (AP)')
    s.save(f'split_diagram{suf}.svg', 'Diagram: with a random-frame split the score is 0.72, with whole videos kept '
                                      'apart it is 0.31, for the same model and the same test frames.')


# horizontal bar chart
def bars(t, name, title, sub, items, xmax, ticks, fmt, legend=None, ref=None, label_w=318, alt=''):
    row, bh = 30, 18
    top = 62 + (26 if legend else 0)
    h = top + row * len(items) + 34
    s = Svg(h, t)
    s.title(title, sub)
    if legend:
        s.legend(top - 14, legend)
    x0, x1 = label_w, W - 58
    sc = (x1 - x0) / xmax
    yb = top + row * len(items)
    for v in ticks:
        x = x0 + v * sc
        s.line(x, top - 4, x, yb, 'grid')
        s.text(x, yb + 18, fmt(v), size=12, fill='muted', anchor='middle')
    if ref:
        rv, rl = ref
        x = x0 + rv * sc
        s.line(x, top - 8, x, yb, 'axis', 2)
        s.text(x + 5, top - 10, rl, size=12, fill='muted')
    for i, (label, v, key) in enumerate(items):
        y = top + i * row + (row - bh) / 2
        s.text(x0 - 10, y + 13, label, size=13, anchor='end')
        s.hbar(x0, y, v * sc, bh, key)
        s.text(x0 + v * sc + 6, y + 13, fmt(v), size=13, fill='ink2')
    s.line(x0, top - 4, x0, yb, 'axis')
    s.save(name, alt or title)


def detectors(t, suf):
    r = {x['run']: x for x in rows('results/brackish_detection.csv')}
    pick = [('RF-DETR-M', 'rfdetr_m', 's1'), ('Grounding DINO-B, no training', 'zeroshot_grounding_dino_b', 's2'),
            ('YOLOX-L (2021)', 'yolox_l', 's1'), ('YOLO26-L', 'yolo26_l', 's1'),
            ('YOLO26-S taught by Grounding DINO', 'yolo26_s_model_labels', 's2'), ('YOLO26-S', 'yolo26_s', 's1'),
            ('SAM 3, no training', 'zeroshot_sam3', 's2'), ('OWLv2-L, no training', 'zeroshot_owlv2_l', 's2'),
            ('YOLOE-26L, no training', 'zeroshot_yoloe_26l', 's2')]
    items = [(lab, float(r[k]['AP']), key) for lab, k, key in pick]
    bars(t, f'detectors{suf}.svg', 'RF-DETR scored best on videos it had never seen',
         'Detection score (AP, 0 to 1) on 15 held-out Brackish videos, 2,743 frames, 6 kinds of animal',
         items, 0.45, [0, 0.1, 0.2, 0.3, 0.4], lambda v: (f'{v:.3f}' if v not in (0.1, 0.2, 0.3, 0.4) else f'{v:.1f}') if v else '0',
         legend=[('s1', 'trained on 10,387 hand-labelled frames'), ('s2', 'no hand-drawn boxes used')],
         alt='Bar chart of detection scores on unseen videos. RF-DETR-M 0.380, Grounding DINO without training 0.375, '
             'YOLOX-L 0.33, YOLO26-L 0.30, YOLO26-S taught by Grounding DINO 0.30, YOLO26-S 0.29, SAM 3 0.19, '
             'OWLv2 0.06, YOLOE 0.01.')


def label_curve(t, suf):
    r = {x['run']: x for x in rows('results/brackish_detection.csv')}
    series = [('RF-DETR-M', 's1', [float(r[k]['AP']) for k in ('rfdetr_m_1pct', 'rfdetr_m_10pct', 'rfdetr_m')]),
              ('YOLO26-L', 's2', [float(r[k]['AP']) for k in ('yolo26_l_1pct', 'yolo26_l_10pct', 'yolo26_l')])]
    ns = [103, 1038, 10387]
    h = 330
    s = Svg(h, t)
    s.title('A tenth of the labels did as well as all of them',
            'Detection score (AP) on the 15 held-out videos vs. number of hand-labelled training frames')
    s.legend(78, [(k, n) for n, k, _ in series])
    x0, x1, y0, y1 = 56, W - 150, 280, 100
    lo, hi = 0.15, 0.40

    def X(n):
        return x0 + (math.log10(n) - math.log10(ns[0])) / (math.log10(ns[-1]) - math.log10(ns[0])) * (x1 - x0)

    def Y(v):
        return y0 - (v - lo) / (hi - lo) * (y0 - y1)

    for v in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        s.line(x0, Y(v), x1, Y(v), 'grid')
        s.text(x0 - 8, Y(v) + 4, f'{v:.2f}', size=12, fill='muted', anchor='end')
    for n, lab in zip(ns, ('103 (1%)', '1,038 (10%)', '10,387 (all)')):
        s.text(X(n), y0 + 20, lab, size=12, fill='muted', anchor='middle')
    for name, key, vals in series:
        pts = ' '.join(f'{X(n):.1f},{Y(v):.1f}' for n, v in zip(ns, vals))
        s.raw(f'<polyline points="{pts}" fill="none" stroke="{t[key]}" stroke-width="2" '
              f'stroke-linejoin="round" stroke-linecap="round"/>')
        for n, v in zip(ns, vals):
            s.raw(f'<circle cx="{X(n):.1f}" cy="{Y(v):.1f}" r="5" fill="{t[key]}" stroke="{t["page"]}" stroke-width="2"/>')
        s.text(x1 + 12, Y(vals[-1]) + 4, f'{name} {vals[-1]:.2f}', size=13, fill='ink2')
    s.save(f'label_curve{suf}.svg', 'Line chart: RF-DETR-M scores 0.27, 0.38 and 0.38 with 103, 1,038 and 10,387 '
                                    'labelled frames; YOLO26-L scores 0.22, 0.31 and 0.30.')


def judges(t, suf):
    sc = json.load(open(os.path.join(ROOT, 'data/dead-or-alive-120/scores.json')))
    pick = [('Vote of the three large models', 'vote_of_3', 's1'), ('Large model A', 'large_model_a', 's1'),
            ('Large model B', 'large_model_b', 's1'), ('Large model C', 'large_model_c', 's1'),
            ('Qwen3-VL 32B', 'qwen3_vl_32b_checklist', 's2'), ('Qwen3-VL 8B', 'qwen3_vl_8b_checklist', 's2'),
            ('Qwen3-VL 2B', 'qwen3_vl_2b_checklist', 's2')]
    items = [(lab, sc[k]['accuracy'], key) for lab, k, key in pick]
    bars(t, f'judges{suf}.svg', 'Only the large models could tell dead fish from live ones',
         'Share judged correctly, 120 boxed fish (60 dead, 60 alive) from the later RAS-MortDB sessions, same checklist',
         items, 1.0, [0, 0.25, 0.5, 0.75, 1.0], lambda v: f'{v:.0%}', ref=(0.5, 'coin flip'), label_w=250,
         legend=[('s1', 'large commercial models'), ('s2', 'open models that fit on one GPU')],
         alt='Bar chart of dead-or-alive accuracy: vote of three large models 96%, large model A 95%, B 91%, '
             'C 82%, Qwen3-VL 32B 52%, 8B 50%, 2B 50%. Coin flip is 50%.')


def pipeline(t, suf):
    s = Svg(262, t)
    s.title('A dead-fish alarm built from what worked', 'Each stage was measured on the same later-session photos; '
                                                       'the chain as a whole was not run.')
    boxes = [('Camera photo', ['one every 15 minutes', 'over the tank drain'], None),
             ('Small detector', ['boxes every fish,', 'dead or alive'], 'catches 93% of dead fish, ~6 boxes per photo'),
             ('Large model + checklist', ['judges each box;', 'three models vote'], '98% of its "dead" calls are right; finds 93%')]
    bw, bh, y, gap = 178, 96, 66, 40
    xs = [0, bw + gap, 2 * (bw + gap)]
    for (head, lines, stat), x in zip(boxes, xs):
        s.rect(x, y, bw, bh, 'box', rx=8, stroke='box_line')
        s.text(x + 14, y + 26, head, size=14, weight=600)
        for k, ln in enumerate(lines):
            s.text(x + 14, y + 50 + k * 18, ln, size=13, fill='ink2')
        if stat:
            s.rect(x, y + bh + 10, 3, 34, 's1')
            words, cur, out = stat.split(), '', []
            for w_ in words:
                if len(cur) + len(w_) > 24:
                    out.append(cur)
                    cur = w_
                else:
                    cur = (cur + ' ' + w_).strip()
            out.append(cur)
            for k, ln in enumerate(out):
                s.text(x + 12, y + bh + 24 + k * 17, ln, size=12, fill='ink2')
    for x in xs[:-1]:
        ax = x + bw + 6
        s.line(ax, y + bh / 2, ax + gap - 14, y + bh / 2, 'axis', 2)
        s.raw(f'<path d="M{ax + gap - 14},{y + bh / 2 - 5} L{ax + gap - 6},{y + bh / 2} L{ax + gap - 14},{y + bh / 2 + 5} Z" '
              f'fill="{t["axis"]}"/>')
    ex = 3 * bw + 2 * gap + 14
    s.text(ex, y + 30, 'Alarm', size=14, weight=600)
    s.text(ex, y + 50, 'staff or a', size=12, fill='ink2')
    s.text(ex, y + 66, 'robot removes', size=12, fill='ink2')
    s.text(ex, y + 82, 'the fish', size=12, fill='ink2')
    s.line(ex - 14, y + bh / 2, ex - 4, y + bh / 2, 'axis', 2)
    s.text(0, 250, 'Multiplied together: about 87% of dead fish found, about 98% of alarms real. An estimate, not a test.',
           size=13, fill='ink')
    s.save(f'pipeline{suf}.svg', 'Diagram: camera photo, then a small detector that boxes every fish (catches 93% of '
                                 'dead fish), then three large models vote on each box (98% precision, 93% recall), '
                                 'then an alarm. Estimated end to end: 87% of dead fish found, 98% of alarms real.')


if __name__ == '__main__':
    for f in (split_diagram, detectors, label_curve, judges, pipeline):
        both(f)
    print('written to', HERE)
