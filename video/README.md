# Explainer video (Remotion)

Source for `../media/fish-farm-vision.mp4` and `../media/preview.gif`.

- `public/frames/` (gitignored) holds Brackish frames processed with gray-world + CLAHE, CLAHE and MSRCR
  (`raw`, `clahe`, `msrcr`, `grayworld_clahe`, 256 frames each at 15 fps). They come from the Brackish test video
  `2019-03-19_18-01-56to2019-03-19_18-02-13_1`. Put its frames in one folder, then run
  `python scripts/enhance_clip.py <that folder> video/public/frames` from the repo root.
- `public/panels/` holds copies of `figures/example_dead.jpg` and `figures/example_alive.jpg`.
- Brackish footage is CC BY-SA 4.0.

```
npm install
npx remotion studio
npx remotion render Main ../media/fish-farm-vision.mp4 --codec=h264 --pixel-format=yuv420p
```
The committed MP4 was re-encoded with ffmpeg (no audio track, limited-range yuv420p). The GIF is the first 8 s,
720 px wide, 13 fps, made with ffmpeg palettegen/paletteuse.
