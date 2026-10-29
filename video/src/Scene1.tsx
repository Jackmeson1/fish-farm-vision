import React from "react";
import {useCurrentFrame} from "remotion";
import {C, Footage, Txt, Vignette, fade, prog, Scene} from "./common";

export const Scene1: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  const zoom = 1 + 0.07 * (frame / duration);
  // Divider x: starts at the right edge, sweeps left, then glides back to the centre.
  const sweep = prog(frame, 0.5, 4.2);
  const back = prog(frame, 4.4, 7.0);
  const x = 1280 - sweep * 1180 + back * (640 - 100);
  const dividerO = fade(frame, 0.3, 0.6);
  const labelO = fade(frame, 6.6, 1.0);
  const titleO = fade(frame, 5.0, 1.6);
  const titleY = (1 - prog(frame, 5.0, 6.6)) * 12;
  const topShade = 0.55 * prog(frame, 4.6, 6.4);
  const layer = (set: string, clip?: string) => (
    <div style={{position: "absolute", inset: 0, overflow: "hidden", clipPath: clip}}>
      <div style={{position: "absolute", inset: 0, transform: `scale(${zoom})`}}>
        <Footage set={set} frame={frame} />
      </div>
    </div>
  );
  return (
    <Scene duration={duration}>
      {layer("raw")}
      {layer("grayworld_clahe", `inset(0 0 0 ${x}px)`)}
      <Vignette />
      <div style={{position: "absolute", inset: 0, background: `linear-gradient(to bottom, rgba(11,22,32,${topShade}) 0%, rgba(11,22,32,0) 55%)`}} />
      <div style={{position: "absolute", inset: 0, background: `linear-gradient(to top, rgba(11,22,32,${0.6 * labelO}) 0%, rgba(11,22,32,0) 25%)`}} />
      <div style={{position: "absolute", left: x - 1, top: 0, width: 2, height: 720, background: C.text, opacity: 0.85 * dividerO, boxShadow: "0 0 14px rgba(240,239,236,0.5)"}} />
      <div style={{position: "absolute", left: 0, right: 0, top: 120, textAlign: "center", opacity: titleO, transform: `translateY(${titleY}px)`}}>
        <Txt font="serif" size={68} weight={400} style={{textShadow: "0 2px 24px rgba(0,0,0,0.6)"}}>Fish-farm vision on public data</Txt>
        <Txt size={24} color={C.text} style={{marginTop: 18, textShadow: "0 1px 12px rgba(0,0,0,0.8)", letterSpacing: 0.3}}>
          One day of experiments on public datasets and cloud GPUs
        </Txt>
      </div>
      <div style={{position: "absolute", left: 48, bottom: 40, opacity: labelO}}>
        <Txt size={24} weight={500} style={{textShadow: "0 1px 10px rgba(0,0,0,0.9)"}}>As filmed</Txt>
      </div>
      <div style={{position: "absolute", right: 48, bottom: 40, opacity: labelO}}>
        <Txt size={24} weight={500} style={{textShadow: "0 1px 10px rgba(0,0,0,0.9)", textAlign: "right"}}>Colour-corrected (gray-world + CLAHE)</Txt>
      </div>
    </Scene>
  );
};
