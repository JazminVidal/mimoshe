import { memo, useEffect, useRef } from "react";
import { PhaseClock } from "./PhaseClock";

export const PhaseIndicator = memo(function PhaseIndicator({ clock }: { clock: PhaseClock }) {
  const textRef = useRef<HTMLSpanElement>(null);
  const trackRef = useRef<HTMLElement>(null);
  const fillRef = useRef<HTMLElement>(null);
  const markerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let animation = 0;
    let lastPhase = -1;
    const paint = (now: number) => {
      const { phase } = clock.sample(now);
      if (phase !== lastPhase) {
        if (textRef.current) textRef.current.textContent = `phase ${Math.round(phase * 360)}°`;
        trackRef.current?.setAttribute("aria-valuenow", (phase * 100).toFixed(2));
        if (fillRef.current) fillRef.current.style.transform = `scaleX(${phase})`;
        if (markerRef.current) markerRef.current.style.transform = `translateX(${phase * 100}%)`;
        lastPhase = phase;
      }
      animation = requestAnimationFrame(paint);
    };
    animation = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(animation);
  }, [clock]);

  return (
    <>
      <span ref={textRef} className="phase-readout">phase 0°</span>
      <i
        ref={trackRef}
        className="phase-track"
        role="progressbar"
        aria-label="Trajectory cycle phase"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={0}
        title="Smooth display of received model steps, delayed by up to one step; freezes when the stream stops"
      >
        <b ref={fillRef} className="phase-fill" />
        <em ref={markerRef} className="phase-marker" />
      </i>
    </>
  );
});
