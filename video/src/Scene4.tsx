import React from "react";
import {useCurrentFrame} from "remotion";
import {Bars, C, Legend, Reveal, Row, Scene, Txt} from "./common";

const rows: Row[] = [
  {label: "RF-DETR-M", value: 0.38, color: C.blue, text: "0.380"},
  {label: "Grounding DINO, no training", value: 0.375, color: C.orange, text: "0.375"},
  {label: "YOLOX-L (2021)", value: 0.331, color: C.blue, text: "0.331"},
  {label: "YOLO26-L", value: 0.301, color: C.blue, text: "0.301"},
  {label: "YOLO26-S", value: 0.294, color: C.blue, text: "0.294"},
  {label: "SAM 3, no training", value: 0.186, color: C.orange, text: "0.186"},
];

export const Scene4: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  return (
    <Scene duration={duration}>
      <Reveal frame={frame} at={0.2} style={{position: "absolute", left: 60, top: 60}}>
        <Txt font="serif" size={40}>Detectors</Txt>
        <Txt size={22} color={C.sub} style={{marginTop: 6}}>Score on 15 videos no model had seen</Txt>
      </Reveal>
      <Bars frame={frame} rows={rows} max={0.45} start={0.8} stagger={0.45} labelW={400} barW={640} rowH={58} top={170} x={60} />
      <Reveal frame={frame} at={4.2} style={{position: "absolute", left: 60, top: 540}}>
        <Legend items={[{color: C.blue, label: "Trained on 10,387 labelled frames"}, {color: C.orange, label: "No hand-drawn boxes"}]} />
      </Reveal>
      <Reveal frame={frame} at={3.8} out={7.6} style={{position: "absolute", left: 60, top: 610}}>
        <Txt font="serif" size={32}>A 2021 detector still beat the newest YOLO.</Txt>
      </Reveal>
      <Reveal frame={frame} at={8.0} style={{position: "absolute", left: 60, top: 610}}>
        <Txt font="serif" size={32}>10% of the labels scored 0.377. All of them: 0.380.</Txt>
      </Reveal>
    </Scene>
  );
};
