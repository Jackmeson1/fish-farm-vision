import React from "react";
import {useCurrentFrame} from "remotion";
import {C, Reveal, Scene, Txt} from "./common";

export const Scene6: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  return (
    <Scene duration={duration}>
      <div style={{position: "absolute", left: 90, right: 90, top: 190, textAlign: "center"}}>
        <Reveal frame={frame} at={0.4}><Txt font="serif" size={52}>Scale did what hand-made tricks did not.</Txt></Reveal>
        <Reveal frame={frame} at={3.4} style={{marginTop: 70}}><Txt size={34} weight={500}>github.com/Jackmeson1/fish-farm-vision</Txt></Reveal>
        <Reveal frame={frame} at={4.8} style={{marginTop: 60}}>
          <Txt size={18} color={C.sub}>Data: Brackish (Aalborg University, CC BY-SA 4.0), RAS-MortDB (CC BY 4.0).</Txt>
          <Txt size={18} color={C.sub} style={{marginTop: 6}}>Experiments run with Claude Code.</Txt>
          <Txt size={16} color={C.sub} style={{marginTop: 14}}>Video footage adapted from Brackish, shared under CC BY-SA 4.0.</Txt>
        </Reveal>
      </div>
    </Scene>
  );
};
