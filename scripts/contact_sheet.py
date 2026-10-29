"""Grid of images with YOLO boxes drawn (green = class 0, red = class 1), for a quick look at a dataset."""
import glob
import os
import sys

from PIL import Image, ImageDraw

img_dir, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8
files = sorted(glob.glob(f'{img_dir}/*.jpg') + glob.glob(f'{img_dir}/*.png'))[::max(1, len(glob.glob(f"{img_dir}/*.jpg")) // n)][:n]
tiles = []
for f in files:
    im = Image.open(f).convert('RGB')
    w, h = im.size
    d = ImageDraw.Draw(im)
    lbl = f.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    for line in open(lbl) if os.path.exists(lbl) else []:
        c, cx, cy, bw, bh = map(float, line.split()[:5])
        d.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                    outline=(0, 255, 0) if c == 0 else (255, 0, 0), width=max(2, w // 200))
    im.thumbnail((480, 480))
    tiles.append(im)
cols = 4
W, H = 480, 480
sheet = Image.new('RGB', (cols * W, ((len(tiles) + cols - 1) // cols) * H), 'white')
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * W, (i // cols) * H))
sheet.save(out, quality=85)
print(out, len(tiles), [t.size for t in tiles][:3])
