import React from "react";
import {AbsoluteFill, Sequence} from "remotion";
import {C, sec} from "./common";
import {Scene1} from "./Scene1";
import {SceneHeadline} from "./SceneHeadline";
import {Scene2} from "./Scene2";
import {Scene3} from "./Scene3";
import {Scene4} from "./Scene4";
import {Scene5} from "./Scene5";
import {Scene6} from "./Scene6";

const parts = [
  {at: 0, d: 14, C: Scene1},
  {at: 14, d: 13, C: SceneHeadline},
  {at: 27, d: 16, C: Scene3},
  {at: 43, d: 12, C: Scene2},
  {at: 55, d: 12, C: Scene4},
  {at: 67, d: 25, C: Scene5},
  {at: 92, d: 12, C: Scene6},
];
export const TOTAL = sec(104);

export const Main: React.FC = () => (
  <AbsoluteFill style={{background: C.bg}}>
    {parts.map((p, i) => (
      <Sequence key={i} from={sec(p.at)} durationInFrames={sec(p.d)}>
        <p.C duration={sec(p.d)} />
      </Sequence>
    ))}
  </AbsoluteFill>
);
