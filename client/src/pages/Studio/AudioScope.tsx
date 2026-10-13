import { FC, useCallback, useEffect, useRef } from "react";

type AudioScopeProps = {
  analyser: AnalyserNode | null;
  color: string;
  idleLabel: string;
};

export const AudioScope: FC<AudioScopeProps> = ({ analyser, color, idleLabel }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    context.clearRect(0, 0, width, height);
    context.strokeStyle = "rgba(148, 163, 184, 0.12)";
    context.lineWidth = 1;
    for (let line = 1; line < 4; line += 1) {
      const y = (height / 4) * line;
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }

    if (!analyser) {
      context.fillStyle = "rgba(148, 163, 184, 0.55)";
      context.font = "12px IBM Plex Mono, ui-monospace, monospace";
      context.textAlign = "center";
      context.fillText(idleLabel, width / 2, height / 2 + 4);
      animationRef.current = requestAnimationFrame(draw);
      return;
    }

    const samples = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(samples);
    const glow = context.createLinearGradient(0, 0, width, 0);
    glow.addColorStop(0, `${color}55`);
    glow.addColorStop(0.25, color);
    glow.addColorStop(0.75, color);
    glow.addColorStop(1, `${color}55`);
    context.strokeStyle = glow;
    context.shadowBlur = 12;
    context.shadowColor = color;
    context.lineWidth = 2;
    context.beginPath();
    samples.forEach((sample, index) => {
      const x = (index / (samples.length - 1)) * width;
      const y = (sample / 255) * height;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
    context.shadowBlur = 0;
    animationRef.current = requestAnimationFrame(draw);
  }, [analyser, color, idleLabel]);

  useEffect(() => {
    animationRef.current = requestAnimationFrame(draw);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [draw]);

  return <canvas className="audio-scope" ref={canvasRef} aria-label={idleLabel} />;
};
