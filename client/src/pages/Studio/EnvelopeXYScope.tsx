import { FC, useEffect, useRef } from "react";

type EnvelopeXYScopeProps = {
  inputAnalyser: AnalyserNode | null;
  outputAnalyser: AnalyserNode | null;
};

type Point = {
  x: number;
  y: number;
};

function analyserRms(analyser: AnalyserNode | null, samples: Float32Array) {
  if (!analyser) return 0;
  analyser.getFloatTimeDomainData(samples);
  let energy = 0;
  for (const sample of samples) energy += sample * sample;
  return Math.sqrt(energy / samples.length);
}

function normalizeEnvelope(rms: number) {
  return Math.min(1, Math.log1p(rms * 40) / Math.log(5));
}

export const EnvelopeXYScope: FC<EnvelopeXYScopeProps> = ({
  inputAnalyser,
  outputAnalyser,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const inputSamples = new Float32Array(inputAnalyser?.fftSize ?? 2048);
    const outputSamples = new Float32Array(outputAnalyser?.fftSize ?? 2048);
    const points: Point[] = [];
    let inputEnvelope = 0;
    let outputEnvelope = 0;
    let lastSample = 0;
    let animationFrame = 0;

    const resize = () => {
      const bounds = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(bounds.width * scale));
      canvas.height = Math.max(1, Math.round(bounds.height * scale));
      context.setTransform(scale, 0, 0, scale, 0, 0);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    resize();

    const draw = (timestamp: number) => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#0b0f10";
      context.fillRect(0, 0, width, height);

      context.strokeStyle = "#252c2f";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(width / 2, 0);
      context.lineTo(width / 2, height);
      context.moveTo(0, height / 2);
      context.lineTo(width, height / 2);
      context.stroke();

      if (timestamp - lastSample >= 32) {
        const input = normalizeEnvelope(analyserRms(inputAnalyser, inputSamples));
        const output = normalizeEnvelope(analyserRms(outputAnalyser, outputSamples));
        inputEnvelope += (input - inputEnvelope) * (input > inputEnvelope ? 0.42 : 0.08);
        outputEnvelope += (output - outputEnvelope) * (output > outputEnvelope ? 0.42 : 0.08);
        points.push({ x: inputEnvelope, y: outputEnvelope });
        if (points.length > 70) points.shift();
        lastSample = timestamp;
      }

      if (points.length > 1) {
        context.lineWidth = 1;
        points.slice(1).forEach((point, index) => {
          const previous = points[index];
          const opacity = (index + 1) / points.length;
          context.strokeStyle = `rgba(34, 211, 166, ${0.08 + opacity * 0.75})`;
          context.beginPath();
          context.moveTo(4 + previous.x * (width - 8), height - 4 - previous.y * (height - 8));
          context.lineTo(4 + point.x * (width - 8), height - 4 - point.y * (height - 8));
          context.stroke();
        });
      }

      context.fillStyle = "#5d686e";
      context.font = "7px IBM Plex Mono, monospace";
      context.fillText("OUT", 4, 9);
      context.textAlign = "right";
      context.fillText("IN", width - 4, height - 4);
      context.textAlign = "left";
      animationFrame = requestAnimationFrame(draw);
    };

    animationFrame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
    };
  }, [inputAnalyser, outputAnalyser]);

  return (
    <div className="xy-scope">
      <canvas ref={canvasRef} aria-label="Input versus Moshi output envelope XY scope" />
    </div>
  );
};
