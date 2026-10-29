import React from "react";
import {Easing, Img, interpolate, staticFile, useCurrentFrame} from "remotion";
import {loadFont as loadSerif} from "@remotion/google-fonts/Fraunces";
import {loadFont as loadInter} from "@remotion/google-fonts/Inter";

export const serif = loadSerif("normal", {weights: ["400", "500"], subsets: ["latin"]}).fontFamily;
export const sans = loadInter("normal", {weights: ["400", "500", "600"], subsets: ["latin"]}).fontFamily;

export const C = {
  bg: "#0b1620",
  text: "#f0efec",
  sub: "#c3c2b7",
  blue: "#3987e5",
  orange: "#d95926",
  line: "rgba(240,239,236,0.18)",
};

export const FPS = 30;
export const sec = (s: number) => Math.round(s * FPS);
export const ease = Easing.inOut(Easing.cubic);
export const easeOut = Easing.out(Easing.cubic);

/** 0 to 1 progress between two times (seconds), eased. */
export const prog = (frame: number, from: number, to: number, e = ease) =>
  interpolate(frame, [from * FPS, to * FPS], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: e});

/** Fade in at `from`, optional fade out at `out` (seconds). */
export const fade = (frame: number, from: number, dur = 0.8, out?: number, outDur = 0.6) => {
  const a = prog(frame, from, from + dur);
  const b = out === undefined ? 0 : prog(frame, out, out + outDur);
  return a * (1 - b);
};

const pad = (n: number) => String(n).padStart(4, "0");
/** Ping-pong looping frame path; each source frame shown 2 video frames. */
export const framePath = (set: string, frame: number) => {
  const n = Math.floor(frame / 2) % 510;
  const i = n < 256 ? n : 510 - n;
  return staticFile(`frames/${set}/${pad(i + 1)}.jpg`);
};

export const Footage: React.FC<{set: string; frame: number; style?: React.CSSProperties}> = ({set, frame, style}) => (
  <Img src={framePath(set, frame)} style={{width: "100%", height: "100%", objectFit: "cover", ...style}} />
);

export const Vignette: React.FC<{strength?: number}> = ({strength = 0.55}) => (
  <div
    style={{
      position: "absolute",
      inset: 0,
      background: `radial-gradient(ellipse at center, rgba(11,22,32,0) 45%, rgba(11,22,32,${strength}) 100%)`,
      pointerEvents: "none",
    }}
  />
);

export const Scene: React.FC<{children: React.ReactNode; duration: number}> = ({children, duration}) => {
  const frame = useCurrentFrame();
  const o = Math.min(prog(frame, 0, 0.4), 1 - prog(frame, duration / FPS - 0.4, duration / FPS));
  return <div style={{position: "absolute", inset: 0, background: C.bg, opacity: o}}>{children}</div>;
};

export const Txt: React.FC<{
  children: React.ReactNode;
  size?: number;
  color?: string;
  font?: "serif" | "sans";
  weight?: number;
  style?: React.CSSProperties;
}> = ({children, size = 28, color = C.text, font = "sans", weight = 400, style}) => (
  <div style={{fontFamily: font === "serif" ? serif : sans, fontSize: size, color, fontWeight: weight, lineHeight: 1.25, ...style}}>
    {children}
  </div>
);

/** Small fade+rise wrapper. */
export const Reveal: React.FC<{frame: number; at: number; out?: number; children: React.ReactNode; style?: React.CSSProperties}> = ({
  frame,
  at,
  out,
  children,
  style,
}) => {
  const o = fade(frame, at, 0.8, out);
  const y = (1 - prog(frame, at, at + 0.8)) * 10;
  return <div style={{opacity: o, transform: `translateY(${y}px)`, ...style}}>{children}</div>;
};

export const Legend: React.FC<{items: {color: string; label: string}[]; style?: React.CSSProperties}> = ({items, style}) => (
  <div style={{display: "flex", gap: 36, ...style}}>
    {items.map((it) => (
      <div key={it.label} style={{display: "flex", alignItems: "center", gap: 10}}>
        <div style={{width: 16, height: 16, borderRadius: 3, background: it.color}} />
        <Txt size={20} color={C.sub}>{it.label}</Txt>
      </div>
    ))}
  </div>
);

export type Row = {label: string; value: number; color: string; text: string};

/** Horizontal bar chart with staggered growth. */
export const Bars: React.FC<{
  frame: number;
  rows: Row[];
  max: number;
  start: number;
  stagger?: number;
  x?: number;
  labelW?: number;
  barW?: number;
  rowH?: number;
  top?: number;
}> = ({frame, rows, max, start, stagger = 0.45, x = 60, labelW = 400, barW = 600, rowH = 56, top = 0}) => (
  <div style={{position: "absolute", left: x, top, width: labelW + barW + 140}}>
    {rows.map((r, i) => {
      const p = prog(frame, start + i * stagger, start + i * stagger + 1.2, easeOut);
      const o = prog(frame, start + i * stagger, start + i * stagger + 0.5);
      return (
        <div key={r.label} style={{height: rowH, display: "flex", alignItems: "center", opacity: o}}>
          <Txt size={22} color={C.text} style={{width: labelW, textAlign: "right", paddingRight: 20}}>{r.label}</Txt>
          <div style={{width: barW, height: 30, position: "relative"}}>
            <div style={{width: (r.value / max) * barW * p, height: 30, background: r.color, borderRadius: 3}} />
            <Txt size={24} weight={600} style={{position: "absolute", left: (r.value / max) * barW * p + 12, top: 0, lineHeight: "30px", opacity: p}}>
              {r.text}
            </Txt>
          </div>
        </div>
      );
    })}
  </div>
);
