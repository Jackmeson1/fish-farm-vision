import React from "react";
import {useCurrentFrame} from "remotion";
import {C, Footage, Reveal, Scene, Txt, Vignette, fade} from "./common";

const tiles = [
  {set: "raw", label: "As filmed"},
  {set: "clahe", label: "CLAHE"},
  {set: "grayworld_clahe", label: "Gray-world + CLAHE"},
  {set: "msrcr", label: "MSRCR"},
];
const rows = [
  ["Train and test on raw images", "0.29"],
  ["Train and test on corrected images", "0.29 to 0.30"],
  ["Train on one kind, test on the other", "0.21 to 0.25"],
];

export const Scene2: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  const TW = 360, TH = 203;
  return (
    <Scene duration={duration}>
      <div style={{position: "absolute", left: 60, top: 70, width: TW * 2 + 16}}>
        <div style={{display: "grid", gridTemplateColumns: `${TW}px ${TW}px`, gap: 16}}>
          {tiles.map((t, i) => (
            <div key={t.label} style={{opacity: fade(frame, 0.2 + i * 0.25, 0.8)}}>
              <div style={{position: "relative", width: TW, height: TH, overflow: "hidden", borderRadius: 4}}>
                <div style={{position: "absolute", inset: 0, transform: `scale(${1 + (0.04 * frame) / duration})`}}>
                  <Footage set={t.set} frame={frame} />
                </div>
                <Vignette strength={0.4} />
              </div>
              <Txt size={20} color={C.sub} style={{marginTop: 8}}>{t.label}</Txt>
            </div>
          ))}
        </div>
      </div>
      <Reveal frame={frame} at={0.5} style={{position: "absolute", left: 840, top: 76, width: 390}}>
        <Txt font="serif" size={40}>Easier on the eye. No help to the detector.</Txt>
      </Reveal>
      <div style={{position: "absolute", left: 840, top: 270, width: 390}}>
        <Reveal frame={frame} at={1.6}>
          <Txt size={16} color={C.sub} style={{marginBottom: 6, letterSpacing: 0.4}}>Detection score (AP, 0 to 1)</Txt>
        </Reveal>
        {rows.map(([l, v], i) => (
          <Reveal
            key={l}
            frame={frame}
            at={2.0 + i * 0.7}
            style={{display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, padding: "14px 0", borderTop: `1px solid ${C.line}`}}
          >
            <Txt size={19} color={C.text} style={{flex: 1}}>{l}</Txt>
            <Txt size={26} weight={600} style={{whiteSpace: "nowrap"}}>{v}</Txt>
          </Reveal>
        ))}
      </div>
      <Reveal frame={frame} at={4.0} style={{position: "absolute", left: 60, right: 60, top: 590}}>
        <Txt size={22} color={C.sub}>A learned enhancer (NAFNet) scored PSNR 22.0 vs 17.5 for the raw image, and still did not help.</Txt>
      </Reveal>
    </Scene>
  );
};
