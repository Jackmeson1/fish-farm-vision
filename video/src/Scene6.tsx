import React from "react";
import {useCurrentFrame} from "remotion";
import {C, Reveal, Scene, Txt} from "./common";

export const Scene6: React.FC<{duration: number}> = ({duration}) => {
  const frame = useCurrentFrame();
  return (
    <Scene duration={duration}>
      <div style={{position: "absolute", left: 90, right: 90, top: 150, textAlign: "center"}}>
        <Reveal frame={frame} at={0.4}><Txt font="serif" size={52}>Small models find. Large models judge.</Txt></Reveal>
        <Reveal frame={frame} at={2.0} style={{marginTop: 28}}><Txt font="serif" size={40} color={C.sub}>Scale did what hand-made tricks did not.</Txt></Reveal>
        <Reveal frame={frame} at={3.8} style={{marginTop: 56}}><Txt size={34} weight={500}>github.com/Jackmeson1/fish-farm-vision</Txt></Reveal>
        <Reveal frame={frame} at={4.8} style={{marginTop: 50}}>
          <Txt size={18} color={C.sub}>Data: Brackish (Aalborg University, CC BY-SA 4.0), RAS-MortDB (CC BY 4.0).</Txt>
          <Txt size={18} color={C.sub} style={{marginTop: 6}}>Experiments run with Claude Code.</Txt>
          <Txt size={16} color={C.sub} style={{marginTop: 14}}>Video footage adapted from Brackish, shared under CC BY-SA 4.0.</Txt>
        </Reveal>
      </div>
    </Scene>
  );
};
