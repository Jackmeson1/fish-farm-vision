import React from "react";
import {useCurrentFrame} from "remotion";
import {C, Reveal, Scene, Txt} from "./common";

export const SceneHeadline: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  return (
    <Scene duration={duration}>
      <div style={{position: "absolute", left: 90, right: 90, top: 120}}>
        <Reveal frame={frame} at={0.3}>
          <Txt font="serif" size={54}>The costly question on a fish farm: which fish died?</Txt>
        </Reveal>
        <Reveal frame={frame} at={1.8} style={{marginTop: 48}}>
          <Txt size={30}>On 120 fish from one farm, large commercial AI models with a checklist judged dead or alive correctly 82 to 96% of the time.</Txt>
        </Reveal>
        <Reveal frame={frame} at={4.2} style={{marginTop: 30}}>
          <Txt size={30} weight={500} color={C.orange}>Open models that fit on one GPU: 50%, a coin flip.</Txt>
        </Reveal>
        <Reveal frame={frame} at={6.4} style={{marginTop: 44}}>
          <Txt size={22} color={C.sub}>Hand-made tricks did not help. The Bitter Lesson, in small.</Txt>
        </Reveal>
      </div>
    </Scene>
  );
};
