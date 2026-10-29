import React from "react";
import {useCurrentFrame} from "remotion";
import {C, Legend, Reveal, Scene, Txt, fade, prog} from "./common";

const N = 24;
const hash = (i: number, j: number) => {
  const s = Math.sin(i * 127.1 + j * 311.7) * 43758.5453;
  return s - Math.floor(s);
};
const mix = (a: number[], b: number[], t: number) => `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * t)).join(",")})`;
const BLUE = [57, 135, 229], ORANGE = [217, 89, 38], GREY = [40, 58, 74];

export const Scene3: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  const colourIn = prog(frame, 0.8, 2.4); // tiles go from grey to random colours
  const morph = prog(frame, 5.5, 7.0); // random split to whole videos
  const labelA = fade(frame, 0.3, 0.8, 5.5, 0.8);
  const labelB = fade(frame, 6.0, 0.8);
  const numA = fade(frame, 2.4, 1.0, 5.5, 0.8);
  const numB = fade(frame, 6.8, 1.0);
  return (
    <Scene duration={duration}>
      <div style={{position: "absolute", left: 60, top: 70, width: 900, height: 60}}>
        <div style={{position: "absolute", opacity: labelA}}><Txt font="serif" size={40}>Random frames</Txt></div>
        <div style={{position: "absolute", opacity: labelB}}><Txt font="serif" size={40}>Whole videos</Txt></div>
      </div>
      <div style={{position: "absolute", left: 60, top: 170}}>
        {[0, 1, 2].map((r) => (
          <div key={r} style={{display: "flex", alignItems: "center", marginBottom: 30}}>
            <Txt size={18} color={C.sub} style={{width: 80}}>Video {r + 1}</Txt>
            <div style={{display: "flex", gap: 4}}>
              {Array.from({length: N}).map((_, i) => {
                const rand = hash(i, r) < 0.7 ? BLUE : ORANGE;
                const whole = r < 2 ? BLUE : ORANGE;
                const t = Math.min(1, Math.max(0, colourIn * 1.6 - hash(i, r + 9) * 0.6));
                const m = Math.max(0, Math.min(1, morph * 1.5 - (i / N) * 0.5));
                const col = morph > 0 ? mix(rand, whole, m) : mix(GREY, rand, t);
                return <div key={i} style={{width: 30, height: 46, borderRadius: 3, background: col}} />;
              })}
            </div>
          </div>
        ))}
        <Reveal frame={frame} at={2.4} style={{marginLeft: 80, marginTop: 10}}>
          <Legend items={[{color: C.blue, label: "Training frames"}, {color: C.orange, label: "Test frames"}]} />
        </Reveal>
      </div>
      <div style={{position: "absolute", left: 990, top: 190, width: 260}}>
        <div style={{position: "absolute", opacity: numA}}>
          <Txt font="serif" size={120} style={{lineHeight: 1}}>0.72</Txt>
          <Txt size={20} color={C.sub} style={{marginTop: 6}}>Score on the test frames</Txt>
        </div>
        <div style={{position: "absolute", opacity: numB}}>
          <Txt font="serif" size={120} color={C.orange} style={{lineHeight: 1}}>0.31</Txt>
          <Txt size={20} color={C.sub} style={{marginTop: 6}}>Score on the test frames</Txt>
        </div>
      </div>
      <Reveal frame={frame} at={6.8} style={{position: "absolute", left: 60, top: 500}}>
        <Txt font="serif" size={34}>Same model, same test frames. Only the split changed.</Txt>
      </Reveal>
      <Reveal frame={frame} at={9.2} style={{position: "absolute", left: 60, top: 580, width: 1160}}>
        <Txt size={24} color={C.sub}>A trout-farm dataset (RAS-MortDB): 94.7% with a random split, 82.7% when tested on later days.</Txt>
      </Reveal>
    </Scene>
  );
};
