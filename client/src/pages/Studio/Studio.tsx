import Recorder from "opus-recorder";
import encoderPath from "opus-recorder/dist/encoderWorker.min.js?url";
import {
  ChangeEvent,
  FC,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { decodeMessage, encodeMessage } from "../../protocol/encoder";
import { WSMessage } from "../../protocol/types";
import { AudioScope } from "./AudioScope";
import { EnvelopeXYScope } from "./EnvelopeXYScope";
import { InfoTip } from "./InfoTip";
import { OnsetAnalysis } from "./OnsetAnalysis";

type ConnectionState = "connecting" | "live" | "offline" | "busy";
type InputMode = "sample" | "microphone" | "system";
type SoundPad = {
  id: string;
  label: string;
  path: string;
};

type SamplingParams = {
  textTemperature: number;
  textTopk: number;
  audioTemperature: number;
  audioTopk: number;
  padBias: number;
};

type OnsetGate = {
  active: boolean;
  lastOnset: number;
};

type MidiMapping = {
  inputId: string;
  inputName: string;
  channel: number;
  controller: number;
};

const DEFAULT_PARAMS: SamplingParams = {
  textTemperature: 0.7,
  textTopk: 25,
  audioTemperature: 0.8,
  audioTopk: 250,
  padBias: 0,
};

const SOUND_PADS: SoundPad[] = [
  { id: "hi", label: "Hi", path: "/sounds/hi.ogg" },
  { id: "hello", label: "Hello", path: "/sounds/hello.ogg" },
  { id: "yes", label: "Yes", path: "/sounds/yes.ogg" },
  { id: "no", label: "No", path: "/sounds/no.ogg" },
  { id: "go", label: "Go", path: "/sounds/go.ogg" },
  { id: "stop", label: "Stop", path: "/sounds/stop.ogg" },
  { id: "okay", label: "Okay", path: "/sounds/okay.ogg" },
  { id: "thanks", label: "Thanks", path: "/sounds/thanks.ogg" },
  { id: "bye", label: "Bye", path: "/sounds/bye.ogg" },
  { id: "nice", label: "Nice", path: "/sounds/nice.ogg" },
];

const RECORDER_OPTIONS = {
  encoderPath,
  encoderFrameSize: 20,
  encoderSampleRate: 24000,
  maxFramesPerPage: 2,
  numberOfChannels: 1,
  recordingGain: 1,
  resampleQuality: 3,
  encoderComplexity: 0,
  encoderApplication: 2049,
  streamPages: true,
};

function buildSocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/api/chat`;
}

function formatTime(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, "0");
  const seconds = Math.floor(totalSeconds % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function loadParams(): SamplingParams {
  try {
    const saved = JSON.parse(localStorage.getItem("moshi-studio-params") ?? "{}");
    return { ...DEFAULT_PARAMS, ...saved };
  } catch {
    return DEFAULT_PARAMS;
  }
}

function loadMidiMappings(): Record<string, MidiMapping> {
  try {
    const saved = JSON.parse(
      localStorage.getItem("moshi-studio-midi-mappings") ?? "{}",
    ) as Record<string, MidiMapping>;
    const exclusive: Record<string, MidiMapping> = {};
    const owners = new Map<string, string>();
    Object.entries(saved).forEach(([target, mapping]) => {
      if (
        !mapping
        || typeof mapping.inputId !== "string"
        || typeof mapping.channel !== "number"
        || typeof mapping.controller !== "number"
      ) {
        return;
      }
      const ccKey = `${mapping.inputId}:${mapping.channel}:${mapping.controller}`;
      const previousOwner = owners.get(ccKey);
      if (previousOwner) delete exclusive[previousOwner];
      exclusive[target] = mapping;
      owners.set(ccKey, target);
    });
    localStorage.setItem("moshi-studio-midi-mappings", JSON.stringify(exclusive));
    return exclusive;
  } catch {
    return {};
  }
}

function findTripleRepetition(value: string) {
  const normalized = value
    .toLocaleLowerCase()
    .normalize("NFKC")
    .replace(/[^\p{L}\p{N}]+/gu, "");
  const maximumLength = Math.min(96, Math.floor(normalized.length / 3));
  for (let length = maximumLength; length >= 2; length -= 1) {
    const end = normalized.slice(-length);
    const middle = normalized.slice(-length * 2, -length);
    const start = normalized.slice(-length * 3, -length * 2);
    if (
      end === middle
      && middle === start
      && new Set(end).size > 1
    ) {
      return end;
    }
  }
  return "";
}

function readRms(analyser: AnalyserNode | null) {
  if (!analyser) return 0;
  const values = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(values);
  return Math.sqrt(values.reduce((sum, value) => {
    return sum + value * value;
  }, 0) / values.length);
}

function detectOnset(
  gate: OnsetGate,
  rms: number,
  now: number,
  attackThreshold: number,
  releaseThreshold: number,
) {
  if (gate.active) {
    if (rms <= releaseThreshold) gate.active = false;
    return false;
  }
  if (rms >= attackThreshold && now - gate.lastOnset >= 0.12) {
    gate.active = true;
    gate.lastOnset = now;
    return true;
  }
  return false;
}

export const Studio: FC = () => {
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [statusMessage, setStatusMessage] = useState("Opening model stream");
  const [inputMode, setInputMode] = useState<InputMode>("sample");
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [sourceActive, setSourceActive] = useState(false);
  const [looping, setLooping] = useState(false);
  const [sampleName, setSampleName] = useState("");
  const [sampleDuration, setSampleDuration] = useState(0);
  const [selectedPadId, setSelectedPadId] = useState("");
  const [loadingPadId, setLoadingPadId] = useState("");
  const [triggerRate, setTriggerRate] = useState(0.5);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [inputGain, setInputGain] = useState(0.85);
  const [outputGain, setOutputGain] = useState(0.9);
  const [autoGainEnabled, setAutoGainEnabled] = useState(true);
  const [autoGainDb, setAutoGainDb] = useState(0);
  const [params, setParams] = useState<SamplingParams>(loadParams);
  const [padTokenId, setPadTokenId] = useState(3);
  const [transcript, setTranscript] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [inputAnalyser, setInputAnalyser] = useState<AnalyserNode | null>(null);
  const [outputAnalyser, setOutputAnalyser] = useState<AnalyserNode | null>(null);
  const [inputLevel, setInputLevel] = useState(0);
  const [outputLevel, setOutputLevel] = useState(0);
  const [onsetRecording, setOnsetRecording] = useState(false);
  const [repeatedSequence, setRepeatedSequence] = useState("");
  const [inputOnsets, setInputOnsets] = useState<number[]>([]);
  const [outputOnsets, setOutputOnsets] = useState<number[]>([]);
  const [crossOnsetDeltas, setCrossOnsetDeltas] = useState<number[]>([]);
  const [onsetElapsed, setOnsetElapsed] = useState(0);
  const [histogramRange, setHistogramRange] = useState(5);
  const [histogramBins, setHistogramBins] = useState(12);
  const [midiMappings, setMidiMappings] = useState<Record<string, MidiMapping>>(loadMidiMappings);
  const [midiLearning, setMidiLearning] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const inputBusRef = useRef<GainNode | null>(null);
  const silenceClockRef = useRef<ConstantSourceNode | null>(null);
  const inputMonitorGainRef = useRef<GainNode | null>(null);
  const outputDetectorRef = useRef<AnalyserNode | null>(null);
  const autoGainRef = useRef<GainNode | null>(null);
  const outputBusRef = useRef<GainNode | null>(null);
  const recorderRef = useRef<any>(null);
  const decoderRef = useRef<Worker | null>(null);
  const sampleBufferRef = useRef<AudioBuffer | null>(null);
  const soundPadBuffersRef = useRef<Map<string, AudioBuffer>>(new Map());
  const soundPadRequestRef = useRef(0);
  const inputStreamRef = useRef<MediaStream | null>(null);
  const loopTimerRef = useRef<number | null>(null);
  const nextLoopTimeRef = useRef(0);
  const triggerRateRef = useRef(triggerRate);
  const playbackRateRef = useRef(playbackRate);
  const nextPlaybackRef = useRef(0);
  const scheduledSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const sessionStartRef = useRef(Date.now());
  const transcriptRef = useRef<HTMLDivElement>(null);
  const paramsRef = useRef(params);
  const connectRef = useRef<() => void>(() => {});
  const reconnectTimerRef = useRef<number | null>(null);
  const allowReconnectRef = useRef(true);
  const onsetRecordingRef = useRef(false);
  const onsetStartRef = useRef(0);
  const repetitionScanStartRef = useRef(0);
  const latestInputOnsetRef = useRef<number | null>(null);
  const latestOutputOnsetRef = useRef<number | null>(null);
  const onsetGatesRef = useRef<{ input: OnsetGate; output: OnsetGate }>({
    input: { active: false, lastOnset: -Infinity },
    output: { active: false, lastOnset: -Infinity },
  });
  const midiAccessRef = useRef<MIDIAccess | null>(null);
  const midiLearningRef = useRef<string | null>(null);
  const midiMappingsRef = useRef(midiMappings);
  const midiTargetHandlersRef = useRef<Map<string, (normalizedValue: number) => void>>(new Map());
  const voicePadMidiHeldRef = useRef<Set<number>>(new Set());
  const midiMessageHandlerRef = useRef<(input: MIDIInput, data: Uint8Array) => void>(
    () => {},
  );

  const secureCapture = window.isSecureContext || window.location.hostname === "localhost";
  triggerRateRef.current = triggerRate;
  playbackRateRef.current = playbackRate;
  midiMappingsRef.current = midiMappings;
  midiLearningRef.current = midiLearning;

  midiMessageHandlerRef.current = (input, data) => {
    const status = data[0] ?? 0;
    if ((status & 0xf0) !== 0xb0 || data.length < 3) return;
    const channel = status & 0x0f;
    const controller = data[1];
    const normalizedValue = data[2] / 127;
    const learningTarget = midiLearningRef.current;

    if (learningTarget) {
      const voicePadMatch = /^voice-pad-(\d+)$/.exec(learningTarget);
      if (voicePadMatch && data[2] !== 127) return;
      const mapping: MidiMapping = {
        inputId: input.id,
        inputName: input.name || "MIDI input",
        channel,
        controller,
      };
      const displacedTargets = Object.entries(midiMappingsRef.current)
        .filter(([target, existing]) => (
          target !== learningTarget
          && existing.inputId === mapping.inputId
          && existing.channel === mapping.channel
          && existing.controller === mapping.controller
        ))
        .map(([target]) => target);
      const nextMappings = { ...midiMappingsRef.current };
      displacedTargets.forEach((target) => delete nextMappings[target]);
      nextMappings[learningTarget] = mapping;
      midiMappingsRef.current = nextMappings;
      localStorage.setItem("moshi-studio-midi-mappings", JSON.stringify(nextMappings));
      setMidiMappings(nextMappings);
      midiTargetHandlersRef.current.get(learningTarget)?.(normalizedValue);
      const learnedPadIndex = voicePadMatch ? Number(voicePadMatch[1]) : null;
      const nextPadIndex = learnedPadIndex === null ? null : learnedPadIndex + 1;
      const nextLearningTarget = nextPadIndex !== null && nextPadIndex < SOUND_PADS.length
        ? `voice-pad-${nextPadIndex}`
        : null;
      midiLearningRef.current = nextLearningTarget;
      setMidiLearning(nextLearningTarget);

      const replacementStatus = displacedTargets.length
        ? ` · replaced ${displacedTargets.join(", ")}`
        : "";
      if (nextPadIndex !== null && nextPadIndex < SOUND_PADS.length) {
        setStatusMessage(
          `MIDI mapped ${SOUND_PADS[learnedPadIndex!].label}`
          + ` · press CC 127 for ${SOUND_PADS[nextPadIndex].label}`
          + replacementStatus,
        );
      } else if (learnedPadIndex !== null) {
        setStatusMessage(`Voice pad MIDI learn complete${replacementStatus}`);
      } else {
        setStatusMessage(
          `MIDI learned · ${mapping.inputName} · CH ${channel + 1} CC ${controller}`
          + replacementStatus,
        );
      }
      return;
    }

    Object.entries(midiMappingsRef.current).forEach(([target, mapping]) => {
      if (
        mapping.inputId === input.id
        && mapping.channel === channel
        && mapping.controller === controller
      ) {
        midiTargetHandlersRef.current.get(target)?.(normalizedValue);
      }
    });
  };

  const attachMidiInputs = useCallback((access: MIDIAccess) => {
    access.inputs.forEach((input) => {
      input.onmidimessage = (event) => {
        if (event.data) midiMessageHandlerRef.current(input, event.data);
      };
    });
  }, []);

  const beginMidiLearn = useCallback(async (target: string) => {
    if (midiLearningRef.current === target) {
      midiLearningRef.current = null;
      setMidiLearning(null);
      setStatusMessage("MIDI learn cancelled");
      return;
    }

    const requestMidiAccess = (
      navigator as unknown as { requestMIDIAccess?: () => Promise<MIDIAccess> }
    ).requestMIDIAccess?.bind(navigator);
    if (!requestMidiAccess) {
      setStatusMessage("Web MIDI requires a supported browser over HTTPS or localhost");
      return;
    }

    try {
      let access = midiAccessRef.current;
      if (!access) {
        access = await requestMidiAccess();
        midiAccessRef.current = access;
        attachMidiInputs(access);
        access.onstatechange = () => attachMidiInputs(access!);
      }
      midiLearningRef.current = target;
      setMidiLearning(target);
      setStatusMessage("MIDI learn armed · move a CC control");
    } catch {
      setStatusMessage("MIDI access was not granted");
    }
  }, [attachMidiInputs]);

  useEffect(() => {
    if (!Object.keys(midiMappingsRef.current).length || midiAccessRef.current) return;
    const requestMidiAccess = (
      navigator as unknown as { requestMIDIAccess?: () => Promise<MIDIAccess> }
    ).requestMIDIAccess?.bind(navigator);
    if (!requestMidiAccess) return;

    let cancelled = false;
    void requestMidiAccess()
      .then((access) => {
        if (cancelled) return;
        midiAccessRef.current = access;
        attachMidiInputs(access);
        access.onstatechange = () => attachMidiInputs(access);
      })
      .catch(() => {
        // Stored mappings remain available if permission is granted later
        // through an explicit MIDI-learn action.
      });
    return () => {
      cancelled = true;
    };
  }, [attachMidiInputs]);

  const registerMidiTarget = useCallback((
    target: string,
    handler: (normalizedValue: number) => void,
  ) => {
    midiTargetHandlersRef.current.set(target, handler);
    return () => {
      midiTargetHandlersRef.current.delete(target);
    };
  }, []);

  useEffect(() => {
    return () => {
      const access = midiAccessRef.current;
      if (!access) return;
      access.inputs.forEach((input) => {
        input.onmidimessage = null;
      });
      access.onstatechange = null;
    };
  }, []);

  const ensureAudioGraph = useCallback(() => {
    if (audioContextRef.current) return audioContextRef.current;
    const context = new AudioContext();
    const inputBus = context.createGain();
    const silenceClock = context.createConstantSource();
    silenceClock.offset.value = 0;
    silenceClock.connect(inputBus);
    silenceClock.start();
    const inputScope = context.createAnalyser();
    inputScope.fftSize = 2048;
    inputScope.smoothingTimeConstant = 0.75;
    inputBus.connect(inputScope);
    const inputMonitorGain = context.createGain();
    const inputPanner = context.createStereoPanner();
    inputMonitorGain.gain.value = 0;
    inputPanner.pan.value = -1;
    inputBus.connect(inputMonitorGain);
    inputMonitorGain.connect(inputPanner);
    inputPanner.connect(context.destination);

    const outputDetector = context.createAnalyser();
    const autoGain = context.createGain();
    const outputLimiter = context.createDynamicsCompressor();
    const outputBus = context.createGain();
    const outputScope = context.createAnalyser();
    const outputPanner = context.createStereoPanner();
    outputDetector.fftSize = 1024;
    outputDetector.smoothingTimeConstant = 0.25;
    autoGain.gain.value = 1;
    outputLimiter.threshold.value = -6;
    outputLimiter.knee.value = 3;
    outputLimiter.ratio.value = 16;
    outputLimiter.attack.value = 0.003;
    outputLimiter.release.value = 0.18;
    outputScope.fftSize = 2048;
    outputScope.smoothingTimeConstant = 0.82;
    outputDetector.connect(autoGain);
    autoGain.connect(outputBus);
    outputBus.connect(outputLimiter);
    outputLimiter.connect(outputScope);
    outputScope.connect(outputPanner);
    outputPanner.pan.value = 1;
    outputPanner.connect(context.destination);

    inputBus.gain.value = 0.85;
    outputBus.gain.value = 0.9;
    audioContextRef.current = context;
    inputBusRef.current = inputBus;
    silenceClockRef.current = silenceClock;
    inputMonitorGainRef.current = inputMonitorGain;
    outputDetectorRef.current = outputDetector;
    autoGainRef.current = autoGain;
    outputBusRef.current = outputBus;
    setInputAnalyser(inputScope);
    setOutputAnalyser(outputScope);

    const decoder = new Worker("/assets/decoderWorker.min.js");
    decoder.postMessage({
      command: "init",
      bufferLength: Math.round((960 * context.sampleRate) / 24000),
      decoderSampleRate: 24000,
      outputBufferSampleRate: context.sampleRate,
      resampleQuality: 0,
    });
    decoder.onmessage = (event: MessageEvent<Float32Array[]>) => {
      const samples = event.data?.[0];
      if (!samples?.length || !outputDetectorRef.current || context.state !== "running") return;
      const buffer = context.createBuffer(1, samples.length, context.sampleRate);
      buffer.copyToChannel(samples, 0);
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(outputDetectorRef.current);
      scheduledSourcesRef.current.add(source);
      source.onended = () => scheduledSourcesRef.current.delete(source);
      const now = context.currentTime;
      if (nextPlaybackRef.current < now || nextPlaybackRef.current - now > 0.5) {
        nextPlaybackRef.current = now + 0.04;
      }
      source.start(nextPlaybackRef.current);
      nextPlaybackRef.current += buffer.duration;
    };
    decoderRef.current = decoder;
    return context;
  }, []);

  const send = useCallback((message: WSMessage) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(encodeMessage(message));
    }
  }, []);

  const sendSamplingParams = useCallback((nextParams: SamplingParams) => {
    send({
      type: "metadata",
      data: {
        sampling: {
          text_temperature: nextParams.textTemperature,
          text_topk: nextParams.textTopk,
          audio_temperature: nextParams.audioTemperature,
          audio_topk: nextParams.audioTopk,
          pad_mult: nextParams.padBias,
        },
      },
    });
  }, [send]);

  const connect = useCallback(() => {
    socketRef.current?.close();
    setConnection("connecting");
    setStatusMessage("Opening model stream");
    const socket = new WebSocket(buildSocketUrl());
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;
    socket.onmessage = (event) => {
      const message = decodeMessage(new Uint8Array(event.data));
      if (message.type === "handshake") {
        setConnection("live");
        setStatusMessage("Model stream live");
        sessionStartRef.current = Date.now();
        setElapsed(0);
        sendSamplingParams(paramsRef.current);
      } else if (message.type === "audio") {
        const data = message.data;
        decoderRef.current?.postMessage({ command: "decode", pages: data }, [data.buffer]);
      } else if (message.type === "text" || message.type === "coloredtext") {
        setTranscript((current) => current + message.data);
      } else if (message.type === "metadata") {
        const metadata = message.data;
        if (
          typeof metadata === "object"
          && metadata !== null
          && "pad_token_id" in metadata
          && typeof metadata.pad_token_id === "number"
        ) {
          setPadTokenId(metadata.pad_token_id);
        }
      } else if (message.type === "error") {
        setStatusMessage(message.data);
      }
    };
    socket.onclose = (event) => {
      setConnection("offline");
      if (event.code === 1001) {
        setStatusMessage("Session moved to a newer tab");
        return;
      }
      setStatusMessage("Stream closed · reconnecting");
      if (allowReconnectRef.current) {
        reconnectTimerRef.current = window.setTimeout(() => {
          connectRef.current();
        }, 1500);
      }
    };
    socket.onerror = () => {
      setConnection("offline");
      setStatusMessage("Could not reach model");
    };
  }, [sendSamplingParams]);
  connectRef.current = connect;

  useEffect(() => {
    allowReconnectRef.current = true;
    ensureAudioGraph();
    connect();
    const timer = window.setInterval(() => {
      setElapsed((Date.now() - sessionStartRef.current) / 1000);
      if (onsetRecordingRef.current && audioContextRef.current) {
        setOnsetElapsed(audioContextRef.current.currentTime - onsetStartRef.current);
      }
    }, 1000);
    return () => {
      allowReconnectRef.current = false;
      window.clearInterval(timer);
      if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
      decoderRef.current?.terminate();
      recorderRef.current?.close();
      inputStreamRef.current?.getTracks().forEach((track) => track.stop());
      if (loopTimerRef.current) window.clearTimeout(loopTimerRef.current);
      try {
        silenceClockRef.current?.stop();
      } catch {
        // The audio context may already have stopped the source.
      }
      audioContextRef.current?.close();
    };
  }, [connect, ensureAudioGraph]);

  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcript]);

  const resetOnsetAnalysis = useCallback(() => {
    onsetRecordingRef.current = false;
    repetitionScanStartRef.current = transcript.length;
    latestInputOnsetRef.current = null;
    latestOutputOnsetRef.current = null;
    onsetGatesRef.current = {
      input: { active: false, lastOnset: -Infinity },
      output: { active: false, lastOnset: -Infinity },
    };
    setOnsetRecording(false);
    setRepeatedSequence("");
    setInputOnsets([]);
    setOutputOnsets([]);
    setCrossOnsetDeltas([]);
    setOnsetElapsed(0);
  }, [transcript.length]);

  const clearOnsetRecords = useCallback(() => {
    latestInputOnsetRef.current = null;
    latestOutputOnsetRef.current = null;
    setInputOnsets([]);
    setOutputOnsets([]);
    setCrossOnsetDeltas([]);
    setOnsetElapsed(0);
    if (onsetRecordingRef.current) {
      onsetStartRef.current = audioContextRef.current?.currentTime ?? performance.now() / 1000;
    }
  }, []);

  useEffect(() => {
    if (onsetRecordingRef.current) return;
    if (transcript.length < repetitionScanStartRef.current) {
      repetitionScanStartRef.current = 0;
    }
    const repeated = findTripleRepetition(transcript.slice(repetitionScanStartRef.current));
    if (!repeated) return;

    const context = audioContextRef.current;
    onsetStartRef.current = context?.currentTime ?? performance.now() / 1000;
    onsetRecordingRef.current = true;
    latestInputOnsetRef.current = null;
    latestOutputOnsetRef.current = null;
    onsetGatesRef.current = {
      input: { active: false, lastOnset: -Infinity },
      output: { active: false, lastOnset: -Infinity },
    };
    setRepeatedSequence(repeated);
    setInputOnsets([]);
    setOutputOnsets([]);
    setCrossOnsetDeltas([]);
    setOnsetElapsed(0);
    setOnsetRecording(true);
  }, [transcript]);

  useEffect(() => {
    if (inputBusRef.current) inputBusRef.current.gain.value = inputGain;
  }, [inputGain]);

  useEffect(() => {
    if (outputBusRef.current) outputBusRef.current.gain.value = outputGain;
  }, [outputGain]);

  useEffect(() => {
    const context = audioContextRef.current;
    const monitor = inputMonitorGainRef.current;
    if (!context || !monitor) return;
    const shouldMonitor = sourceActive && inputMode !== "microphone";
    monitor.gain.cancelScheduledValues(context.currentTime);
    monitor.gain.setTargetAtTime(shouldMonitor ? 1 : 0, context.currentTime, 0.01);
  }, [inputMode, sourceActive]);

  useEffect(() => {
    const updateMeter = () => {
      const inputRms = readRms(inputAnalyser);
      const outputRms = readRms(outputAnalyser);
      const rawOutputRms = readRms(outputDetectorRef.current);
      setInputLevel(Math.min(100, Math.round(inputRms * 240)));
      setOutputLevel(Math.min(100, Math.round(outputRms * 240)));

      const context = audioContextRef.current;
      const automaticGain = autoGainRef.current;
      if (!context || !automaticGain) return;
      if (!autoGainEnabled) {
        automaticGain.gain.setTargetAtTime(1, context.currentTime, 0.03);
        setAutoGainDb(0);
        return;
      }

      // Target roughly -18 dBFS RMS. Hold the last gain through silence so
      // the beginning of the next phrase is not initially too quiet.
      if (rawOutputRms > 0.003) {
        const desiredGain = Math.min(6, Math.max(1, 0.125 / rawOutputRms));
        const isBoosting = desiredGain > automaticGain.gain.value;
        automaticGain.gain.setTargetAtTime(
          desiredGain,
          context.currentTime,
          isBoosting ? 0.28 : 0.05,
        );
        setAutoGainDb(20 * Math.log10(desiredGain));
      }
    };
    const meter = window.setInterval(updateMeter, 100);
    return () => window.clearInterval(meter);
  }, [autoGainEnabled, inputAnalyser, outputAnalyser]);

  useEffect(() => {
    const recordOnset = (kind: "input" | "output", absoluteTime: number) => {
      const relativeTime = Math.max(0, absoluteTime - onsetStartRef.current);
      if (kind === "input") {
        latestInputOnsetRef.current = relativeTime;
        setInputOnsets((current) => [...current.slice(-511), relativeTime]);
      } else {
        latestOutputOnsetRef.current = relativeTime;
        setOutputOnsets((current) => [...current.slice(-511), relativeTime]);
      }

      if (
        latestInputOnsetRef.current !== null
        && latestOutputOnsetRef.current !== null
      ) {
        const delta = latestInputOnsetRef.current - latestOutputOnsetRef.current;
        setCrossOnsetDeltas((current) => [...current.slice(-511), delta]);
      }
    };

    const analyseOnsets = () => {
      if (!onsetRecordingRef.current) return;
      const context = audioContextRef.current;
      if (!context || context.state !== "running") return;
      const now = context.currentTime;
      const inputRms = readRms(inputAnalyser);
      const outputRms = readRms(outputDetectorRef.current);
      if (detectOnset(onsetGatesRef.current.input, inputRms, now, 0.008, 0.004)) {
        recordOnset("input", now);
      }
      if (detectOnset(onsetGatesRef.current.output, outputRms, now, 0.0025, 0.0012)) {
        recordOnset("output", now);
      }
    };

    const onsetMeter = window.setInterval(analyseOnsets, 25);
    return () => window.clearInterval(onsetMeter);
  }, [inputAnalyser]);

  const enableAudio = useCallback(async () => {
    const context = ensureAudioGraph();
    await context.resume();
    setAudioEnabled(context.state === "running");
  }, [ensureAudioGraph]);

  const stopRecorder = useCallback(() => {
    if (recorderRef.current) {
      recorderRef.current.stop();
      recorderRef.current = null;
    }
  }, []);

  const startRecorder = useCallback(async () => {
    const context = ensureAudioGraph();
    await context.resume();
    setAudioEnabled(true);
    if (recorderRef.current || !inputBusRef.current) return;
    const recorder = new Recorder({
      ...RECORDER_OPTIONS,
      sourceNode: inputBusRef.current,
      bufferLength: Math.round((960 * context.sampleRate) / 24000),
    });
    recorder.ondataavailable = (data: Uint8Array) => {
      send({ type: "audio", data });
    };
    recorder.onstart = () => setSourceActive(true);
    recorder.onstop = () => setSourceActive(false);
    recorderRef.current = recorder;
    await recorder.start();
  }, [ensureAudioGraph, send]);

  const stopInput = useCallback(() => {
    if (loopTimerRef.current) {
      window.clearTimeout(loopTimerRef.current);
      loopTimerRef.current = null;
    }
    setLooping(false);
    inputStreamRef.current?.getTracks().forEach((track) => track.stop());
    inputStreamRef.current = null;
    stopRecorder();
    setSourceActive(false);
  }, [stopRecorder]);

  const triggerSample = useCallback(async (scheduledTime?: number) => {
    const context = ensureAudioGraph();
    if (!sampleBufferRef.current || !inputBusRef.current) {
      setStatusMessage("Load an audio sample first");
      return;
    }
    await startRecorder();
    const source = context.createBufferSource();
    source.buffer = sampleBufferRef.current;
    source.playbackRate.value = playbackRateRef.current;
    source.connect(inputBusRef.current);
    source.start(Math.max(context.currentTime, scheduledTime ?? context.currentTime));
  }, [ensureAudioGraph, startRecorder]);

  const selectSoundPad = useCallback(async (pad: SoundPad) => {
    const requestId = ++soundPadRequestRef.current;
    setLoadingPadId(pad.id);
    try {
      let buffer = soundPadBuffersRef.current.get(pad.id);
      if (!buffer) {
        const response = await fetch(pad.path);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const context = ensureAudioGraph();
        buffer = await context.decodeAudioData(await response.arrayBuffer());
        soundPadBuffersRef.current.set(pad.id, buffer);
      }
      if (requestId !== soundPadRequestRef.current) return;
      sampleBufferRef.current = buffer;
      setSelectedPadId(pad.id);
      setSampleName(`${pad.label} · built-in voice pad`);
      setSampleDuration(buffer.duration);
      setStatusMessage(`${pad.label} pad selected · sequence phase preserved`);
      await triggerSample();
    } catch {
      if (requestId === soundPadRequestRef.current) {
        setStatusMessage(`Could not load the ${pad.label} sound pad`);
      }
    } finally {
      if (requestId === soundPadRequestRef.current) {
        setLoadingPadId("");
      }
    }
  }, [ensureAudioGraph, triggerSample]);

  useEffect(() => {
    const unregister = SOUND_PADS.map((pad, index) => {
      return registerMidiTarget(`voice-pad-${index}`, (normalizedValue) => {
        if (normalizedValue >= 1) {
          if (voicePadMidiHeldRef.current.has(index)) return;
          voicePadMidiHeldRef.current.add(index);
          void selectSoundPad(pad);
        } else {
          voicePadMidiHeldRef.current.delete(index);
        }
      });
    });
    return () => unregister.forEach((cleanup) => cleanup());
  }, [registerMidiTarget, selectSoundPad]);

  useEffect(() => {
    const handleSoundPadKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
      const target = event.target;
      if (
        target instanceof HTMLInputElement
        || target instanceof HTMLTextAreaElement
        || target instanceof HTMLSelectElement
        || (target instanceof HTMLElement && target.isContentEditable)
      ) {
        return;
      }
      const padIndex = event.key === "0" ? 9 : Number(event.key) - 1;
      if (!Number.isInteger(padIndex) || padIndex < 0 || padIndex >= SOUND_PADS.length) return;
      event.preventDefault();
      void selectSoundPad(SOUND_PADS[padIndex]);
    };
    window.addEventListener("keydown", handleSoundPadKeyDown);
    return () => window.removeEventListener("keydown", handleSoundPadKeyDown);
  }, [selectSoundPad]);

  useEffect(() => {
    if (!looping) return;

    const context = ensureAudioGraph();
    const scheduleAheadSeconds = 0.04;

    // Start once, then preserve the sequencer phase. Control changes are read
    // from refs after each trigger, so they latch on the next loop without
    // tearing down the timer or firing an extra sample.
    const firstTriggerTime = context.currentTime + 0.01;
    void triggerSample(firstTriggerTime);
    nextLoopTimeRef.current = firstTriggerTime + 1 / triggerRateRef.current;

    const scheduleNextTrigger = () => {
      const scheduledTime = Math.max(
        nextLoopTimeRef.current,
        context.currentTime + 0.005,
      );
      void triggerSample(scheduledTime);
      nextLoopTimeRef.current = scheduledTime + 1 / triggerRateRef.current;
      const delayMs = Math.max(
        0,
        (nextLoopTimeRef.current - context.currentTime - scheduleAheadSeconds) * 1000,
      );
      loopTimerRef.current = window.setTimeout(scheduleNextTrigger, delayMs);
    };

    loopTimerRef.current = window.setTimeout(
      scheduleNextTrigger,
      Math.max(0, (nextLoopTimeRef.current - context.currentTime - scheduleAheadSeconds) * 1000),
    );

    return () => {
      if (loopTimerRef.current) {
        window.clearTimeout(loopTimerRef.current);
        loopTimerRef.current = null;
      }
    };
  }, [ensureAudioGraph, looping, triggerSample]);

  const toggleLoop = useCallback(async () => {
    if (!sampleBufferRef.current) {
      setStatusMessage("Load an audio sample first");
      return;
    }
    await enableAudio();
    setLooping((current) => !current);
  }, [enableAudio]);

  const loadSample = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const context = ensureAudioGraph();
      const buffer = await context.decodeAudioData(await file.arrayBuffer());
      sampleBufferRef.current = buffer;
      setSelectedPadId("");
      setSampleName(file.name);
      setSampleDuration(buffer.duration);
      setStatusMessage(`Sample ready · ${buffer.duration.toFixed(2)} seconds`);
    } catch {
      setStatusMessage("This audio file could not be decoded");
    }
  }, [ensureAudioGraph]);

  const startCapture = useCallback(async (mode: "microphone" | "system") => {
    if (!secureCapture || !navigator.mediaDevices) {
      setStatusMessage("Microphone and system capture require HTTPS or localhost");
      return;
    }
    stopInput();
    setInputMode(mode);
    try {
      await enableAudio();
      const stream = mode === "microphone"
        ? await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        })
        : await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      if (!stream.getAudioTracks().length) {
        stream.getTracks().forEach((track) => track.stop());
        throw new Error("No audio track was shared");
      }
      stream.getVideoTracks().forEach((track) => track.stop());
      inputStreamRef.current = stream;
      const context = ensureAudioGraph();
      const source = context.createMediaStreamSource(stream);
      source.connect(inputBusRef.current!);
      await startRecorder();
      setStatusMessage(mode === "microphone" ? "Microphone feeding Moshi" : "System audio feeding Moshi");
    } catch (error) {
      setSourceActive(false);
      setStatusMessage(error instanceof Error ? error.message : "Audio capture was cancelled");
    }
  }, [enableAudio, ensureAudioGraph, secureCapture, startRecorder, stopInput]);

  const updateParam = useCallback(<K extends keyof SamplingParams>(key: K, value: SamplingParams[K]) => {
    setParams((current) => {
      const next = { ...current, [key]: value };
      paramsRef.current = next;
      localStorage.setItem("moshi-studio-params", JSON.stringify(next));
      sendSamplingParams(next);
      return next;
    });
  }, [sendSamplingParams]);

  const resetContext = useCallback(() => {
    scheduledSourcesRef.current.forEach((source) => {
      try {
        source.stop();
      } catch {
        // A source that ended between iteration and stop needs no cleanup.
      }
    });
    scheduledSourcesRef.current.clear();
    nextPlaybackRef.current = audioContextRef.current?.currentTime ?? 0;
    setTranscript("");
    send({ type: "control", action: "restart" });
  }, [send]);

  const statusTone = connection === "live" ? "live" : connection === "connecting" ? "waiting" : "offline";
  const sourceLabel = useMemo(() => {
    if (!sourceActive) return "Input idle";
    if (inputMode === "sample") return looping ? `${triggerRate.toFixed(2)} triggers / sec` : "Sample armed";
    return inputMode === "microphone" ? "Microphone active" : "System stream active";
  }, [inputMode, looping, sourceActive, triggerRate]);
  const midiControlProps = (target: string) => ({
    midiKey: target,
    midiLearning: midiLearning === target,
    midiMapping: midiMappings[target],
    onMidiLearn: beginMidiLearn,
    registerMidiTarget,
  });
  const voicePadMidiLearning = midiLearning?.startsWith("voice-pad-") ?? false;
  const voicePadMidiMappedCount = SOUND_PADS.filter((_, index) => {
    return Boolean(midiMappings[`voice-pad-${index}`]);
  }).length;
  const beginVoicePadMidiLearn = () => {
    if (voicePadMidiLearning) {
      midiLearningRef.current = null;
      setMidiLearning(null);
      setStatusMessage("Voice pad MIDI learn cancelled");
      return;
    }
    void beginMidiLearn("voice-pad-0");
  };

  return (
    <main className="studio-shell">
      <header className="studio-header">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div>
            <p className="eyebrow">KYUTAI AUDIO SYSTEMS</p>
            <h1>MOSHI <span>LIVE LAB</span></h1>
          </div>
        </div>
        <div className="header-telemetry">
          <div className={`status-pill ${statusTone}`}><i />{statusMessage}</div>
          <div className="session-clock"><span>SESSION</span>{formatTime(elapsed)}</div>
          {!audioEnabled && (
            <button className="primary-button compact" onClick={enableAudio}>Enable sound</button>
          )}
          {connection === "offline" && (
            <button className="ghost-button compact" onClick={connect}>Reconnect</button>
          )}
        </div>
      </header>

      {!secureCapture && (
        <div className="security-banner">
          <span>HTTP MODE</span>
          File samples work now. Browser microphone and system-audio capture need HTTPS or an SSH localhost tunnel.
        </div>
      )}

      <section className="studio-grid">
        <article className="panel input-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-index">01 / SOURCE</p>
              <h2 className="title-with-help">
                Input deck
                <InfoTip text="Choose a built-in pad, uploaded loop, microphone, or shared system audio to feed Moshi." />
              </h2>
            </div>
            <div className={`activity-label ${sourceActive ? "active" : ""}`}><i />{sourceLabel}</div>
          </div>

          <div className="source-tabs" role="tablist" aria-label="Audio input source">
            {(["sample", "microphone", "system"] as InputMode[]).map((mode) => (
              <button
                key={mode}
                className={inputMode === mode ? "selected" : ""}
                onClick={() => {
                  stopInput();
                  setInputMode(mode);
                }}
              >
                {mode === "sample" ? "Sample loop" : mode === "microphone" ? "Microphone" : "System / loop"}
              </button>
            ))}
          </div>

          <div className="scope-wrap">
            <AudioScope analyser={inputAnalyser} color="#f97316" idleLabel="WAITING FOR INPUT" />
            <div className="scope-meta"><span>MONITOR / LEFT</span><span>24 kHz OPUS</span></div>
            <div className="level-meter"><span style={{ width: `${inputLevel}%` }} /></div>
          </div>

          {inputMode === "sample" ? (
            <div className="sample-controls">
              <div className="sound-pad-block">
                <div className="sound-pad-heading">
                  <span className="title-with-help">
                    BUILT-IN VOICE PADS
                    <InfoTip text="Press a pad with the mouse or use keys 1–9 and 0. Selection does not reset the running loop phase." />
                  </span>
                  <div className="sound-pad-heading-actions">
                    <button
                      type="button"
                      className={`midi-learn-button ${voicePadMidiLearning ? "learning" : ""} ${voicePadMidiMappedCount ? "mapped" : ""}`}
                      onClick={beginVoicePadMidiLearn}
                      title={voicePadMidiLearning
                        ? "Learning voice pads from first to last; click to cancel"
                        : `MIDI learn voice pads (${voicePadMidiMappedCount}/${SOUND_PADS.length} mapped)`}
                    >
                      [ml]
                    </button>
                    <small>Mouse down · keys 1–9, 0</small>
                  </div>
                </div>
                <div className="sound-pad-grid">
                  {SOUND_PADS.map((pad, index) => (
                    <button
                      key={pad.id}
                      className={`${selectedPadId === pad.id ? "active" : ""} ${loadingPadId === pad.id ? "loading" : ""}`}
                      onPointerDown={(event) => {
                        if (event.button !== 0) return;
                        event.preventDefault();
                        void selectSoundPad(pad);
                      }}
                      onClick={(event) => {
                        // Keyboard activation of a focused button has detail 0.
                        // Pointer clicks already fired on pointer-down.
                        if (event.detail === 0) void selectSoundPad(pad);
                      }}
                      aria-pressed={selectedPadId === pad.id}
                      aria-keyshortcuts={index === 9 ? "0" : String(index + 1)}
                    >
                      <span>{String(index === 9 ? 0 : index + 1).padStart(2, "0")}</span>
                      <strong>{loadingPadId === pad.id ? "Loading" : pad.label}</strong>
                      <i />
                    </button>
                  ))}
                </div>
                <p className="sound-pad-credit">
                  Voice by <a href="https://commons.wikimedia.org/wiki/User:Dvortygirl" target="_blank" rel="noreferrer">Dvortygirl</a>
                  {" · "}<a href="https://commons.wikimedia.org/" target="_blank" rel="noreferrer">Wikimedia Commons</a>
                  {" · "}<a href="https://creativecommons.org/licenses/by-sa/3.0/" target="_blank" rel="noreferrer">CC BY-SA 3.0</a>
                </p>
              </div>
              <label className="drop-target">
                <input type="file" accept="audio/*" onChange={loadSample} />
                <span className="upload-icon">↥</span>
                <span className="drop-copy">
                  <strong>{sampleName || "Load an audio sample"}</strong>
                  <small>{sampleName ? `${sampleDuration.toFixed(2)} sec · ready to trigger` : "WAV, MP3, OGG, FLAC or M4A"}</small>
                </span>
                <span className="browse-label">{sampleName ? "Replace" : "Browse"}</span>
              </label>
              <div className="transport-row">
                <button className="trigger-button" onClick={() => void triggerSample()} disabled={!sampleName}>
                  <span>▶</span> Trigger once
                </button>
                <button className={`loop-button ${looping ? "active" : ""}`} onClick={toggleLoop} disabled={!sampleName}>
                  <span>↻</span> {looping ? "Stop sequence" : "Run sequence"}
                </button>
              </div>
              <RangeControl label="Trigger rate" value={triggerRate} min={0.1} max={8} step={0.1} unit="Hz" onChange={setTriggerRate} {...midiControlProps("trigger-rate")} />
              <RangeControl label="Playback rate" value={playbackRate} min={0.25} max={2} step={0.05} unit="×" onChange={setPlaybackRate} {...midiControlProps("playback-rate")} />
              <p className="param-hint">Frequency and pitch latch on the next loop · sequence phase stays continuous</p>
            </div>
          ) : (
            <div className="capture-controls">
              <div className="capture-note">
                <span>{inputMode === "microphone" ? "MIC" : "SYS"}</span>
                <p>{inputMode === "microphone"
                  ? "Capture a microphone or a virtual loopback input exposed by your operating system."
                  : "Share a browser tab or system output. Enable audio in the browser share dialog."}</p>
              </div>
              <button
                className={sourceActive ? "danger-button" : "primary-button"}
                onClick={() => sourceActive ? stopInput() : startCapture(inputMode)}
              >
                {sourceActive ? "Stop input" : `Start ${inputMode === "microphone" ? "microphone" : "system capture"}`}
              </button>
              {!secureCapture && <p className="inline-warning">Unavailable on this insecure HTTP address.</p>}
            </div>
          )}

          <RangeControl label="Input level" value={inputGain} min={0} max={1.5} step={0.05} unit="" onChange={setInputGain} {...midiControlProps("input-level")} />
        </article>

        <article className="panel output-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-index">02 / RESPONSE</p>
              <h2 className="title-with-help">
                Moshi output
                <InfoTip text="Moshi audio is routed to the right channel. The XY scope compares smoothed input and output envelopes." />
              </h2>
            </div>
            <div className={`activity-label ${connection === "live" ? "active green" : ""}`}><i />STREAM {connection.toUpperCase()}</div>
          </div>
          <div className="output-visualizers">
            <div className="scope-wrap output-scope">
              <AudioScope analyser={outputAnalyser} color="#22d3a6" idleLabel="WAITING FOR MOSHI" />
              <div className="scope-meta"><span>MOSHI / RIGHT</span><span>BUFFERED LIVE</span></div>
              <div className="level-meter output"><span style={{ width: `${outputLevel}%` }} /></div>
            </div>
            <EnvelopeXYScope inputAnalyser={inputAnalyser} outputAnalyser={outputAnalyser} />
          </div>
          <div className="transcript-heading">
            <div>
              <span className="live-dot" />
              LIVE TRANSCRIPT
              <InfoTip text="Streaming text tokens produced by Moshi. Three repeated normalized sequences arm onset recording." />
            </div>
            <button onClick={() => setTranscript("")}>Clear</button>
          </div>
          <div className="transcript" ref={transcriptRef}>
            {transcript ? <p>{transcript}</p> : (
              <div className="transcript-empty">
                <span>...</span>
                <p>Moshi’s inner monologue will appear here as audio is processed.</p>
              </div>
            )}
          </div>
          <OnsetAnalysis
            armed={onsetRecording}
            repetition={repeatedSequence}
            inputOnsets={inputOnsets}
            outputOnsets={outputOnsets}
            crossDeltas={crossOnsetDeltas}
            elapsed={onsetElapsed}
            onReset={resetOnsetAnalysis}
            histogramRange={histogramRange}
            histogramBins={histogramBins}
          />
        </article>

        <aside className="panel params-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-index">03 / CONTROLS</p>
              <h2 className="title-with-help">
                More controls
                <InfoTip text="Audio playback, onset visualization, and live model sampling controls." />
              </h2>
            </div>
          </div>

          <section className="controls-section audio-output-controls">
            <div className="controls-section-heading">
              <span className="title-with-help">
                Audio output
                <InfoTip text="Controls local Moshi playback only; these settings do not alter model generation." />
              </span>
            </div>
            <button
              className={`agc-control ${autoGainEnabled ? "active" : ""}`}
              role="switch"
              aria-checked={autoGainEnabled}
              onClick={() => setAutoGainEnabled((enabled) => !enabled)}
            >
              <span className="agc-switch"><i /></span>
              <span><strong>Automatic gain</strong><small>Quiet-speech lift with peak limiting</small></span>
              <em>{autoGainEnabled ? `+${autoGainDb.toFixed(1)} dB` : "OFF"}</em>
            </button>
            <RangeControl label="Output level" value={outputGain} min={0} max={1.5} step={0.05} unit="" onChange={setOutputGain} {...midiControlProps("output-level")} />
          </section>

          <section className="controls-section sampling-controls">
            <div className="controls-section-heading">
              <span className="title-with-help">
                Sampling control
                <InfoTip text="Sampling values are sent live and affect upcoming tokens without clearing model context." />
              </span>
              <button className="reset-button" onClick={() => {
                setParams(DEFAULT_PARAMS);
                paramsRef.current = DEFAULT_PARAMS;
                sendSamplingParams(DEFAULT_PARAMS);
                localStorage.setItem("moshi-studio-params", JSON.stringify(DEFAULT_PARAMS));
              }}>Defaults</button>
            </div>
            <RangeControl label="Text temperature" value={params.textTemperature} min={0} max={1.2} step={0.01} unit="" onChange={(value) => updateParam("textTemperature", value)} accent="green" {...midiControlProps("text-temperature")} />
            <RangeControl label="Text top-k" value={params.textTopk} min={1} max={500} step={1} unit="" onChange={(value) => updateParam("textTopk", value)} accent="green" {...midiControlProps("text-top-k")} />
            <RangeControl label="PAD logit bias" value={params.padBias} min={-10} max={10} step={0.05} unit="" onChange={(value) => updateParam("padBias", value)} accent="green" {...midiControlProps("pad-logit-bias")} />
            <p className="param-hint">Token {padTokenId} · positive is quieter · negative is more willing to speak</p>
            <div className="param-divider" />
            <RangeControl label="Audio temperature" value={params.audioTemperature} min={0} max={1.2} step={0.01} unit="" onChange={(value) => updateParam("audioTemperature", value)} accent="orange" {...midiControlProps("audio-temperature")} />
            <RangeControl label="Audio top-k" value={params.audioTopk} min={1} max={500} step={1} unit="" onChange={(value) => updateParam("audioTopk", value)} accent="orange" {...midiControlProps("audio-top-k")} />
            <button
              className="context-reset-button"
              disabled={connection !== "live"}
              onClick={resetContext}
            >
              <span>↺</span>
              <span><strong>Clear model context</strong><small>Keep connection and sampler running</small></span>
            </button>
          </section>

          <section className="controls-section onset-plot-controls">
            <div className="controls-section-heading">
              <span className="title-with-help">
                Onsets
                <InfoTip text="View settings change histogram rendering only. Existing onset measurements remain intact." />
              </span>
            </div>
            <RangeControl label="Range" value={histogramRange} min={0.5} max={10} step={0.5} unit="s" onChange={setHistogramRange} {...midiControlProps("histogram-range")} />
            <RangeControl label="Bins" value={histogramBins} min={4} max={40} step={1} unit="" onChange={setHistogramBins} {...midiControlProps("histogram-bins")} />
            <button
              className="clear-onsets-button"
              onClick={clearOnsetRecords}
              aria-label="Clear onset records"
              title="Clear onset records; keep detector state and plot settings"
            >
              ×
            </button>
          </section>

          <div className="signal-readout">
            <div><span>MODEL</span><strong>MOSHIKO 7B</strong></div>
            <div><span>PAD TOKEN</span><strong>{padTokenId} · &lt;pad&gt;</strong></div>
            <div><span>CODEC</span><strong>MIMI 24K</strong></div>
            <div><span>LATENCY</span><strong>STREAMING</strong></div>
          </div>
        </aside>
      </section>
    </main>
  );
};

type RangeControlProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (value: number) => void;
  accent?: "green" | "orange";
  midiKey?: string;
  midiLearning?: boolean;
  midiMapping?: MidiMapping;
  onMidiLearn?: (target: string) => void;
  registerMidiTarget?: (
    target: string,
    handler: (normalizedValue: number) => void,
  ) => () => void;
};

const RangeControl: FC<RangeControlProps> = ({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
  accent = "orange",
  midiKey,
  midiLearning = false,
  midiMapping,
  onMidiLearn,
  registerMidiTarget,
}) => {
  const percentage = ((value - min) / (max - min)) * 100;
  useEffect(() => {
    if (!midiKey || !registerMidiTarget) return;
    return registerMidiTarget(midiKey, (normalizedValue) => {
      const rawValue = min + normalizedValue * (max - min);
      const quantized = min + Math.round((rawValue - min) / step) * step;
      onChange(Number(Math.min(max, Math.max(min, quantized)).toFixed(6)));
    });
  }, [max, midiKey, min, onChange, registerMidiTarget, step]);

  const midiTitle = midiLearning
    ? "Learning: move a MIDI CC control; click again to cancel"
    : midiMapping
      ? `${midiMapping.inputName} · CH ${midiMapping.channel + 1} CC ${midiMapping.controller}; click to relearn`
      : "MIDI learn: click, then move a MIDI CC control";

  return (
    <div className={`range-control ${accent}`}>
      <span className="range-label-row">
        <span className="range-label">{label}</span>
        {midiKey && onMidiLearn && (
          <button
            type="button"
            className={`midi-learn-button ${midiLearning ? "learning" : ""} ${midiMapping ? "mapped" : ""}`}
            title={midiTitle}
            aria-label={`${label}: ${midiTitle}`}
            onClick={() => onMidiLearn(midiKey)}
          >
            [ml]
          </button>
        )}
      </span>
      <span className="range-value">{step < 1 ? value.toFixed(2) : value}{unit && <em>{unit}</em>}</span>
      <input
        aria-label={label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ "--range-progress": `${percentage}%` } as React.CSSProperties}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span className="range-min">{min}</span>
      <span className="range-max">{max}</span>
    </div>
  );
};
