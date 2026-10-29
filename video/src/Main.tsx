import React from "react";
import {AbsoluteFill, Sequence} from "remotion";
import {C, sec} from "./common";
import {Scene1} from "./Scene1";
import {Scene2} from "./Scene2";
import {Scene3} from "./Scene3";
import {Scene4} from "./Scene4";
import {Scene5} from "./Scene5";
import {Scene6} from "./Scene6";

const parts = [
  {at: 0, d: 14, C: Scene1},
  {at: 14, d: 12, C: Scene2},
  {at: 26, d: 16, C: Scene3},
  {at: 42, d: 12, C: Scene4},
  {at: 54, d: 18, C: Scene5},
  {at: 72, d: 10, C: Scene6},
];
export const TOTAL = sec(82);

export const Main: React.FC = () => (
  <AbsoluteFill style={{background: C.bg}}>
    {parts.map((p, i) => (
      <Sequence key={i} from={sec(p.at)} durationInFrames={sec(p.d)}>
        <p.C duration={sec(p.d)} />
      </Sequence>
    ))}
  </AbsoluteFill>
);
