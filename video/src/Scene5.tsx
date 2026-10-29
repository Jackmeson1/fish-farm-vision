import React from "react";
import {Img, staticFile, useCurrentFrame} from "remotion";
import {Bars, Eyebrow, C, Legend, Reveal, Row, Scene, Txt, fade, prog} from "./common";

const S = 1.1;
const crops: [number, number][] = [[0, 256], [325, 443], [581, 699]];

const Panel: React.FC<{src: string}> = ({src}) => (
  <div style={{display: "flex", gap: 12}}>
    {crops.map(([a, b]) => (
      <div key={a} style={{width: (b - a) * S, height: 256 * S, overflow: "hidden", borderRadius: 4, position: "relative"}}>
        <Img src={staticFile(src)} style={{position: "absolute", left: -a * S, top: 0, width: 768 * S, height: 256 * S}} />
      </div>
    ))}
  </div>
);

const rows: Row[] = [
  {label: "Vote of three large models", value: 96, color: C.blue, text: "96%"},
  {label: "Large model A", value: 95, color: C.blue, text: "95%"},
  {label: "Large model B", value: 91, color: C.blue, text: "91%"},
  {label: "Large model C", value: 82, color: C.blue, text: "82%"},
  {label: "Qwen3-VL 32B", value: 52, color: C.orange, text: "52%"},
  {label: "Qwen3-VL 8B", value: 50, color: C.orange, text: "50%"},
  {label: "Qwen3-VL 2B", value: 50, color: C.orange, text: "50%"},
];

const steps = ["Camera photo", "Small detector boxes every fish", "Large models judge each box", "Alarm"];

const Pipeline: React.FC<{frame: number}> = ({frame}) => (
  <>
    <div style={{position: "absolute", left: 55, top: 190, display: "flex", alignItems: "center"}}>
      {steps.map((s, i) => (
        <React.Fragment key={s}>
          {i > 0 && (
            <Reveal frame={frame} at={18.0 + i * 0.6 - 0.3} style={{width: 70, display: "flex", alignItems: "center", justifyContent: "center"}}>
              <div style={{width: 40, height: 2, background: C.sub}} />
              <div style={{width: 0, height: 0, borderTop: "8px solid transparent", borderBottom: "8px solid transparent", borderLeft: `12px solid ${C.sub}`}} />
            </Reveal>
          )}
          <Reveal
            frame={frame}
            at={18.0 + i * 0.6}
            style={{width: 240, height: 130, boxSizing: "border-box", padding: 16, display: "flex", alignItems: "center", justifyContent: "center", textAlign: "center", border: `2px solid ${i === 3 ? C.orange : C.blue}`, borderRadius: 6}}
          >
            <Txt size={24} weight={500}>{s}</Txt>
          </Reveal>
        </React.Fragment>
      ))}
    </div>
    <Reveal frame={frame} at={20.4} style={{position: "absolute", left: 90, right: 90, top: 420, textAlign: "center"}}>
      <Txt font="serif" size={32}>Estimated: about 87% of dead fish found, about 98% of alarms real.</Txt>
    </Reveal>
    <Reveal frame={frame} at={21.6} style={{position: "absolute", left: 90, right: 90, top: 520, textAlign: "center"}}>
      <Txt size={22} color={C.sub}>Two stages measured separately; the full chain has not been run yet.</Txt>
    </Reveal>
  </>
);

export const Scene5: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  const phaseA = fade(frame, 1.5, 1.0, 8.6, 0.8);
  const lineA = fade(frame, 0.3, 0.8, 8.6, 0.8);
  const barsO = fade(frame, 9.4, 0.6, 17.4, 0.6);
  const BX = 60, LW = 330, BW = 600, top = 150, RH = 52;
  const coinX = BX + LW + BW * 0.5;
  return (
    <Scene duration={duration}>
      <Eyebrow frame={frame}>Large models plus a workflow</Eyebrow>
      <div style={{position: "absolute", left: 60, top: 230, width: 480, opacity: lineA}}>
        <Txt font="serif" size={44}>On fish farms, the costly question is which fish died.</Txt>
        <Txt size={18} color={C.sub} style={{marginTop: 24, opacity: phaseA}}>
          Red box: the fish in question. Right: the same spot in the neighbouring photo.
        </Txt>
      </div>
      <div style={{position: "absolute", left: 650, top: 40, opacity: phaseA}}>
        <Txt size={24} weight={600} style={{marginBottom: 6}}>Dead</Txt>
        <Panel src="panels/example_dead.jpg" />
        <Txt size={24} weight={600} style={{marginTop: 18, marginBottom: 6}}>Alive</Txt>
        <Panel src="panels/example_alive.jpg" />
      </div>
      <div style={{position: "absolute", inset: 0, opacity: barsO}}>
        <div style={{position: "absolute", left: 60, top: 50}}>
          <Txt font="serif" size={36}>Share judged correctly</Txt>
          <Txt size={20} color={C.sub} style={{marginTop: 6}}>120 fish, 60 dead and 60 alive</Txt>
        </div>
        <Bars frame={frame} rows={rows} max={100} start={10.0} stagger={0.4} labelW={LW} barW={BW} rowH={RH} top={top} x={BX} />
        <div style={{position: "absolute", left: coinX, top: top - 12, width: 2, height: RH * 7 + 24, background: C.text, opacity: 0.55 * prog(frame, 10.0, 11.0)}} />
        <Txt size={18} color={C.sub} style={{position: "absolute", left: coinX - 90, top: top - 42, width: 180, whiteSpace: "nowrap", textAlign: "center", opacity: prog(frame, 10.0, 11.0)}}>
          Coin flip (50%)
        </Txt>
        <Reveal frame={frame} at={13.0} style={{position: "absolute", left: 60, top: 560}}>
          <Legend items={[{color: C.blue, label: "Large commercial models"}, {color: C.orange, label: "Open models that fit on one GPU"}]} />
        </Reveal>
        <Reveal frame={frame} at={13.6} style={{position: "absolute", left: 60, top: 620}}>
          <Txt font="serif" size={32}>Small models and synthetic training data could not do it.</Txt>
        </Reveal>
      </div>
      <Pipeline frame={frame} />
    </Scene>
  );
};
