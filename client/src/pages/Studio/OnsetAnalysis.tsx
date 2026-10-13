import { FC, useMemo } from "react";
import { InfoTip } from "./InfoTip";

type OnsetAnalysisProps = {
  armed: boolean;
  repetition: string;
  inputOnsets: number[];
  outputOnsets: number[];
  crossDeltas: number[];
  elapsed: number;
  onReset: () => void;
  histogramRange: number;
  histogramBins: number;
};

type HistogramProps = {
  title: string;
  series: {
    label: string;
    values: number[];
    color: string;
  }[];
  signed?: boolean;
  range: number;
  binCount: number;
};

const RASTER_WINDOW_SECONDS = 30;

function intervals(onsets: number[]) {
  return onsets.slice(1).map((onset, index) => onset - onsets[index]);
}

function formatSeconds(value: number) {
  if (Math.abs(value) < 1) return `${Math.round(value * 1000)} ms`;
  return `${value.toFixed(1)} s`;
}

const TimingHistogram: FC<HistogramProps> = ({
  title,
  series,
  signed = false,
  range,
  binCount,
}) => {
  const histogram = useMemo(() => {
    const min = signed ? -range : 0;
    const max = range;
    const bins = series.map((item) => {
      const itemBins = Array<number>(binCount).fill(0);
      item.values.forEach((value) => {
        if (value < min || value > max) return;
        const position = (value - min) / (max - min);
        const index = Math.min(binCount - 1, Math.floor(position * binCount));
        itemBins[index] += 1;
      });
      return itemBins;
    });
    return { bins, min, max, peak: Math.max(1, ...bins.flat()) };
  }, [binCount, range, series, signed]);

  const stairPath = (bins: number[]) => {
    const startX = 5;
    const endX = 115;
    const baseline = 45;
    const binWidth = (endX - startX) / binCount;
    const y = (count: number) => baseline - (count / histogram.peak) * 38;
    let path = `M ${startX} ${baseline} V ${y(bins[0])}`;
    bins.forEach((_, index) => {
      const nextX = startX + (index + 1) * binWidth;
      path += ` H ${nextX}`;
      if (index < bins.length - 1) path += ` V ${y(bins[index + 1])}`;
    });
    return `${path} V ${baseline}`;
  };

  return (
    <div className="timing-histogram">
      <div className="timing-histogram-heading">
        <span className="title-with-help">
          {title}
          <InfoTip text={signed
            ? "Distribution of the latest input onset time minus the latest Moshi output onset time. Negative values mean input occurred first."
            : "Input and Moshi output inter-onset intervals overlaid using identical bins and time range."}
          />
        </span>
        <em>n={series.reduce((total, item) => total + item.values.length, 0)}</em>
      </div>
      <svg viewBox="0 0 120 58" role="img" aria-label={`${title} histogram`}>
        <line x1="5" x2="115" y1="45.5" y2="45.5" className="histogram-axis" />
        {signed && <line x1="60" x2="60" y1="4" y2="46" className="histogram-zero" />}
        {series.map((item, index) => (
          <path
            key={item.label}
            d={stairPath(histogram.bins[index])}
            fill="none"
            stroke={item.color}
            opacity={item.values.length ? 0.9 : 0.18}
            className="histogram-stairs"
          />
        ))}
        <text x="5" y="56">{formatSeconds(histogram.min)}</text>
        <text x="115" y="56" textAnchor="end">{formatSeconds(histogram.max)}</text>
      </svg>
      <div className="histogram-legend">
        {series.map((item) => (
          <span key={item.label}>
            <i style={{ background: item.color }} />
            {item.label} {item.values.length}
          </span>
        ))}
      </div>
    </div>
  );
};

export const OnsetAnalysis: FC<OnsetAnalysisProps> = ({
  armed,
  repetition,
  inputOnsets,
  outputOnsets,
  crossDeltas,
  elapsed,
  onReset,
  histogramRange,
  histogramBins,
}) => {
  const inputIntervals = useMemo(() => intervals(inputOnsets), [inputOnsets]);
  const outputIntervals = useMemo(() => intervals(outputOnsets), [outputOnsets]);
  const windowEnd = Math.max(RASTER_WINDOW_SECONDS, elapsed);
  const windowStart = Math.max(0, windowEnd - RASTER_WINDOW_SECONDS);
  const toRasterX = (time: number) => {
    return 23 + ((time - windowStart) / RASTER_WINDOW_SECONDS) * 92;
  };
  const visibleInputs = inputOnsets.filter((time) => time >= windowStart && time <= windowEnd);
  const visibleOutputs = outputOnsets.filter((time) => time >= windowStart && time <= windowEnd);

  return (
    <details className={`onset-analysis ${armed ? "recording" : ""}`} open>
      <summary className="onset-analysis-heading experiment-summary">
        <div>
          <span className="onset-state-dot" />
          <strong>{armed ? "ONSET RECORDING" : "REPETITION WATCH"}</strong>
          <InfoTip text="Recording arms after the transcript ends with three consecutive copies of the same normalized character sequence." />
        </div>
        <button onClick={(event) => {
          event.preventDefault();
          onReset();
        }}>Re-arm</button>
      </summary>
      <p className="onset-status">
        {armed
          ? <>Triggered by 3× “<b>{repetition}</b>” · {formatSeconds(elapsed)} recorded</>
          : "Waiting for three consecutive copies of the same character sequence."}
      </p>

      <div className="onset-raster">
        <div className="onset-chart-heading">
          <span className="title-with-help">
            ONSET RASTER
            <InfoTip text="Each vertical mark is a silence-to-signal onset. Input is orange and Moshi output is green." />
          </span>
          <em>rolling 30 s</em>
        </div>
        <svg viewBox="0 0 120 28" role="img" aria-label="Input and Moshi output onset raster plot">
          {[23, 46, 69, 92, 115].map((x) => (
            <line key={x} x1={x} x2={x} y1="2" y2="22" className="raster-grid" />
          ))}
          <text x="1" y="9" className="raster-label input">IN</text>
          <text x="1" y="19" className="raster-label output">OUT</text>
          <line x1="23" x2="115" y1="7.5" y2="7.5" className="raster-track" />
          <line x1="23" x2="115" y1="17.5" y2="17.5" className="raster-track" />
          {visibleInputs.map((time, index) => (
            <line
              key={`input-${time}-${index}`}
              x1={toRasterX(time)}
              x2={toRasterX(time)}
              y1="3.5"
              y2="11.5"
              className="raster-spike input"
            />
          ))}
          {visibleOutputs.map((time, index) => (
            <line
              key={`output-${time}-${index}`}
              x1={toRasterX(time)}
              x2={toRasterX(time)}
              y1="13.5"
              y2="21.5"
              className="raster-spike output"
            />
          ))}
          <text x="23" y="27">{formatSeconds(windowStart)}</text>
          <text x="115" y="27" textAnchor="end">{formatSeconds(windowEnd)}</text>
        </svg>
        <div className="onset-counts">
          <span><i className="input" />INPUT {inputOnsets.length}</span>
          <span><i className="output" />OUTPUT {outputOnsets.length}</span>
        </div>
      </div>

      <div className="timing-histograms">
        <TimingHistogram
          title="INPUT / OUTPUT IOI"
          series={[
            { label: "IN", values: inputIntervals, color: "#f97316" },
            { label: "OUT", values: outputIntervals, color: "#22d3a6" },
          ]}
          range={histogramRange}
          binCount={histogramBins}
        />
        <TimingHistogram
          title="LAST IN − LAST OUT"
          series={[{ label: "Δ", values: crossDeltas, color: "#d4dadd" }]}
          signed
          range={histogramRange}
          binCount={histogramBins}
        />
      </div>
      <p className="onset-footnote">
        IOI = inter-onset interval. Cross-stream values update at every new onset.
      </p>
    </details>
  );
};
