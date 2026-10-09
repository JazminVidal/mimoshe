import Recorder from "opus-recorder";
import encoderPath from "opus-recorder/dist/encoderWorker.min.js?url";
import {
  ChangeEvent,
  CSSProperties,
  FC,
  ReactNode,
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
  const [histogramBins, setHistogramBins] = useState(8);
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

  const [draining, setDraining] = useState(false);
  const drainTimerRef = useRef<number | null>(null);
  useEffect(() => () => {
    if (drainTimerRef.current) window.clearTimeout(drainTimerRef.current);
  }, []);

  // The monologue goes down the drain (a few stepped frames) before it is cleared.
  const drainThen = useCallback((action: () => void) => {
    if (drainTimerRef.current) window.clearTimeout(drainTimerRef.current);
    setDraining(true);
    drainTimerRef.current = window.setTimeout(() => {
      action();
      setDraining(false);
      drainTimerRef.current = null;
    }, 720);
  }, []);

  const resetDefaults = useCallback(() => {
    setParams(DEFAULT_PARAMS);
    paramsRef.current = DEFAULT_PARAMS;
    sendSamplingParams(DEFAULT_PARAMS);
    localStorage.setItem("moshi-studio-params", JSON.stringify(DEFAULT_PARAMS));
  }, [sendSamplingParams]);

  const statusTone = connection === "live" ? "live" : connection === "connecting" ? "waiting" : "offline";
  const connectionLabel = connection === "live"
    ? "LIVE"
    : connection === "connecting" ? "WAIT" : connection === "busy" ? "BUSY" : "OFF";
  const activePad = SOUND_PADS.find((pad) => pad.id === selectedPadId);
  const sourceLabel = useMemo(() => {
    if (inputMode !== "sample") return sourceActive ? "en vivo" : "inactivo";
    const name = activePad?.label ?? (sampleName ? "sample" : "sin sample");
    return looping ? `${triggerRate.toFixed(2)} Hz · ${name}` : name;
  }, [activePad, inputMode, looping, sampleName, sourceActive, triggerRate]);

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

  // Past lines above, present line below; repeated sentences are marked in pencil.
  const monologue = useMemo(() => {
    const clean = transcript.replace(/\s+/g, " ").trim();
    if (!clean) return { history: [] as string[], current: [] as { text: string; repeated: boolean }[] };
    const sentences = clean.match(/[^.!?…]+[.!?…]+|[^.!?…]+$/g)?.map((part) => part.trim()).filter(Boolean) ?? [clean];
    const clip = (text: string, size: number) => (text.length > size ? `…${text.slice(-size)}` : text);
    const current: string[] = [];
    let budget = CURRENT_LINE_BUDGET;
    for (let index = sentences.length - 1; index >= 0; index -= 1) {
      const sentence = sentences[index];
      if (current.length && sentence.length > budget) break;
      current.unshift(clip(sentence, CURRENT_LINE_BUDGET));
      budget -= sentence.length;
      if (budget <= 0) break;
    }
    const history = sentences
      .slice(Math.max(0, sentences.length - current.length - HISTORY_LINES), sentences.length - current.length)
      .map((sentence) => clip(sentence, 72));
    const normalize = (text: string) => text.toLocaleLowerCase().normalize("NFKC").replace(/[^\p{L}\p{N}]+/gu, "");
    const counts = new Map<string, number>();
    current.forEach((text) => counts.set(normalize(text), (counts.get(normalize(text)) ?? 0) + 1));
    return {
      history,
      current: current.map((text) => ({ text, repeated: (counts.get(normalize(text)) ?? 0) > 1 })),
    };
  }, [transcript]);

  // Onset strip under the monologue: rolling window, one onset time per row.
  const raster = useMemo(() => {
    const windowEnd = Math.max(RASTER_WINDOW_SECONDS, onsetElapsed);
    const windowStart = windowEnd - RASTER_WINDOW_SECONDS;
    const toX = (time: number) => ((time - windowStart) / RASTER_WINDOW_SECONDS) * 100;
    const visible = (times: number[]) => times.filter((time) => time >= windowStart && time <= windowEnd).map(toX);
    let latency: number | null = null;
    for (let index = inputOnsets.length - 1; index >= 0 && latency === null; index -= 1) {
      const start = inputOnsets[index];
      const end = inputOnsets[index + 1] ?? Infinity;
      const reply = outputOnsets.find((time) => time >= start && time < end);
      if (reply !== undefined) latency = reply - start;
    }
    const meanDelta = crossOnsetDeltas.length
      ? crossOnsetDeltas.reduce((total, value) => total + value, 0) / crossOnsetDeltas.length
      : null;
    return {
      windowStart,
      inputs: visible(inputOnsets),
      outputs: visible(outputOnsets),
      lastIn: inputOnsets.length ? `${inputOnsets[inputOnsets.length - 1].toFixed(2)}s` : "—",
      latency: latency === null ? "—" : `${latency.toFixed(2)}s`,
      meanDelta: meanDelta === null ? "sin datos" : `${meanDelta.toFixed(2)} s`,
    };
  }, [crossOnsetDeltas, inputOnsets, onsetElapsed, outputOnsets]);

  const histogram = useMemo(() => {
    const count = Math.round(histogramBins);
    const fill = (onsets: number[]) => {
      const bins = Array<number>(count).fill(0);
      onsets.slice(1).forEach((onset, index) => {
        const interval = onset - onsets[index];
        if (interval < 0 || interval > histogramRange) return;
        bins[Math.min(count - 1, Math.floor((interval / histogramRange) * count))] += 1;
      });
      return bins;
    };
    const inputs = fill(inputOnsets);
    const outputs = fill(outputOnsets);
    const peak = Math.max(1, ...inputs, ...outputs);
    // Last input onset minus last Moshi onset, from −range to +range around zero.
    const deltas = Array<number>(count).fill(0);
    crossOnsetDeltas.forEach((delta) => {
      if (delta < -histogramRange || delta > histogramRange) return;
      const position = (delta + histogramRange) / (histogramRange * 2);
      deltas[Math.min(count - 1, Math.floor(position * count))] += 1;
    });
    const deltaPeak = Math.max(1, ...deltas);
    return {
      intervals: inputs.map((value, index) => ({
        inH: (value / peak) * 100,
        outH: (outputs[index] / peak) * 100,
      })),
      deltas: deltas.map((value) => (value / deltaPeak) * 100),
      inCount: inputOnsets.length,
      outCount: outputOnsets.length,
      deltaCount: crossOnsetDeltas.length,
    };
  }, [crossOnsetDeltas, histogramBins, histogramRange, inputOnsets, outputOnsets]);

  return (
    <main className="studio">
      <div className="studio-frame">
        <header className="studio-header">
          <div className="brand">
            <strong>MOSHI_LIVE_LAB</strong>
            <span title="Modelo cargado">{MODEL_NAME}</span>
          </div>
          <div className="telemetry">
            <span className={`live-pill ${statusTone}`}><i />{connectionLabel}</span>
            <span className="status-text" title={statusMessage}>{statusMessage}</span>
            <span title="Tiempo de sesión">{formatTime(elapsed)}</span>
            <span title="Cuadro de Mimi">12.5 Hz</span>
            <span title="Códec">MIMI 24K</span>
          </div>
          <div className="header-actions">
            {!secureCapture && (
              <span className="http-badge" title="Conexión HTTP: el micrófono y el audio del sistema necesitan HTTPS o un túnel a localhost">
                <Icon d={ICONS.lock} size={14} />HTTP
              </span>
            )}
            {connection === "offline" && (
              <button type="button" className="key key-small" onClick={connect} aria-label="Reconectar" title="Reconectar">
                <Icon d={ICONS.reset} size={18} />
              </button>
            )}
            <button
              type="button"
              className="key key-small"
              aria-pressed={audioEnabled}
              onClick={enableAudio}
              aria-label={audioEnabled ? "Sonido activo" : "Activar sonido"}
              title={audioEnabled ? "Sonido activo" : "Activar sonido"}
            >
              <Icon d={audioEnabled ? ICONS.speakerOn : ICONS.speakerOff} size={18} />
            </button>
          </div>
        </header>

        <section className="cell source" aria-label="Fuente">
          <div className="caption">
            <strong>Escuchá.</strong>
            <span className="caption-tools">
              <small>{sourceLabel}</small>
              {inputMode === "sample" && (
                <button
                  type="button"
                  className={`midi-dot ${voicePadMidiLearning ? "learning" : ""} ${voicePadMidiMappedCount ? "mapped" : ""}`}
                  aria-pressed={voicePadMidiLearning}
                  onClick={beginVoicePadMidiLearn}
                  aria-label={`MIDI learn de los pads (${voicePadMidiMappedCount}/${SOUND_PADS.length})`}
                  title={`MIDI learn de los pads: ${voicePadMidiMappedCount}/${SOUND_PADS.length} asignados`}
                >
                  <MidiGlyph />
                </button>
              )}
            </span>
          </div>
          <div className="source-body">
            <div className="source-tabs" role="tablist" aria-label="Fuente de entrada">
              {SOURCES.map((source) => (
                <button
                  key={source.mode}
                  type="button"
                  role="tab"
                  className="key"
                  aria-selected={inputMode === source.mode}
                  title={source.title}
                  onClick={() => {
                    stopInput();
                    setInputMode(source.mode);
                  }}
                >
                  <Icon d={source.icon} size={22} />
                  <small>{source.code}</small>
                </button>
              ))}
            </div>

            {inputMode === "sample" ? (
              <div className="sample-deck">
                <div className="pads">
                  {SOUND_PADS.map((pad, index) => (
                    <button
                      key={pad.id}
                      type="button"
                      className="pad"
                      aria-pressed={selectedPadId === pad.id}
                      aria-keyshortcuts={index === 9 ? "0" : String(index + 1)}
                      title={`Tecla ${index === 9 ? 0 : index + 1}`}
                      onPointerDown={(event) => {
                        if (event.button !== 0) return;
                        event.preventDefault();
                        void selectSoundPad(pad);
                      }}
                      onClick={(event) => {
                        // Keyboard activation has detail 0; pointer clicks already fired on pointer-down.
                        if (event.detail === 0) void selectSoundPad(pad);
                      }}
                    >
                      <small>{String(index === 9 ? 0 : index + 1).padStart(2, "0")}</small>
                      <i className="pad-mark" style={{ "--r": `${index * 18}deg` } as CSSProperties} aria-hidden="true" />
                      <span>{loadingPadId === pad.id ? "…" : pad.label}</span>
                    </button>
                  ))}
                </div>
                <div className="transport">
                  <button
                    type="button"
                    className="key"
                    onClick={() => void triggerSample()}
                    disabled={!sampleName}
                    aria-label="Disparar una vez"
                    title="Disparar una vez"
                  >
                    <Icon d={ICONS.play} size={28} />
                  </button>
                  <button
                    type="button"
                    className="key"
                    aria-pressed={looping}
                    onClick={toggleLoop}
                    disabled={!sampleName}
                    aria-label={looping ? "Detener secuencia" : "Correr secuencia"}
                    title={looping ? "Detener secuencia" : "Correr secuencia"}
                  >
                    <Icon d={ICONS.loop} size={28} />
                  </button>
                  <label
                    className="key"
                    title={sampleName
                      ? `${sampleName} · ${sampleDuration.toFixed(2)} s · elegir otro`
                      : "Cargar sample: WAV, MP3, OGG, FLAC o M4A"}
                  >
                    <Icon d={ICONS.upload} size={28} />
                    <input type="file" accept="audio/*" onChange={loadSample} className="visually-hidden" aria-label="Cargar sample de audio" />
                  </label>
                </div>
              </div>
            ) : (
              <div className="capture">
                <button
                  type="button"
                  className="key capture-key"
                  aria-pressed={sourceActive}
                  disabled={!secureCapture}
                  onClick={() => (sourceActive ? stopInput() : startCapture(inputMode))}
                  title={sourceActive ? "Detener entrada" : inputMode === "microphone" ? "Abrir micrófono" : "Capturar audio del sistema"}
                >
                  <Icon d={sourceActive ? ICONS.stop : ICONS.record} size={40} />
                  <small>{sourceActive ? "STOP" : inputMode === "microphone" ? "MIC" : "SYS"}</small>
                </button>
                <p className="capture-note">
                  {!secureCapture
                    ? "Requiere HTTPS o localhost."
                    : inputMode === "microphone"
                      ? "Micrófono o entrada virtual de loopback."
                      : "Pestaña del navegador o salida del sistema."}
                </p>
              </div>
            )}
          </div>
        </section>

        <section className="cell monologue-cell" aria-label="Monólogo interior de Moshi">
          <div className="scopes">
            <div className="scope" title={`Entrada · nivel ${inputLevel}%`}>
              <AudioScope analyser={inputAnalyser} variant="input" label="Entrada, canal izquierdo" />
              <span className="scope-tag">IN·L</span>
            </div>
            <div className="scope" title={`Moshi · nivel ${outputLevel}%`}>
              <AudioScope analyser={outputAnalyser} variant="output" label="Moshi, canal derecho" />
              <span className="scope-tag">OUT·R</span>
            </div>
          </div>

          <div className="monologue">
            <div className="monologue-head">
              <span>&gt; INNER_MONOLOGUE</span>
              <button
                type="button"
                className="key key-small"
                onClick={() => drainThen(() => setTranscript(""))}
                disabled={!transcript}
                aria-label="Borrar monólogo"
                title="Borrar monólogo"
              >
                <Icon d={ICONS.backspace} size={18} />
              </button>
            </div>
            {monologue.current.length ? (
              <div className={`monologue-body ${draining ? "draining" : ""}`} ref={transcriptRef}>
                <div className="monologue-history">
                  {monologue.history.map((line, index) => (
                    <span key={`${index}-${line}`}>› {line}</span>
                  ))}
                </div>
                <p className="monologue-current" aria-live="polite">
                  {monologue.current.map((segment, index) => (
                    <span key={`${index}-${segment.text}`} className={segment.repeated ? "repeated" : undefined}>
                      {segment.text}
                    </span>
                  ))}
                  <i className="cursor" aria-hidden="true" />
                </p>
              </div>
            ) : (
              <p className="monologue-current monologue-empty">
                _<i className="cursor" aria-hidden="true" />
              </p>
            )}
          </div>

          <div
            className="onset-strip"
            title={onsetRecording
              ? `Ataques desde la repetición «${repeatedSequence}». Desfase medio entrada − Moshi: ${raster.meanDelta}`
              : "Se empieza a registrar cuando Moshi repite tres veces la misma secuencia"}
          >
            <div className="onset-rows">
              <div className="onset-row" aria-label="Ataques de la entrada">
                {raster.inputs.map((x, index) => (
                  <i key={`in-${index}`} className="tick tick-in" style={{ left: `${x}%` }} />
                ))}
              </div>
              <div className="onset-rule" />
              <div className="onset-row" aria-label="Ataques de Moshi">
                {raster.outputs.map((x, index) => (
                  <i key={`out-${index}`} className="tick tick-out" style={{ left: `${x}%` }} />
                ))}
              </div>
              {!onsetRecording && <span className="onset-wait">esperando ×3</span>}
              <div className="onset-scale" aria-hidden="true">
                <span>{raster.windowStart.toFixed(0)}s</span>
                <span>{(raster.windowStart + RASTER_WINDOW_SECONDS / 2).toFixed(1)}s</span>
                <span>{(raster.windowStart + RASTER_WINDOW_SECONDS).toFixed(0)}s</span>
              </div>
            </div>
            <div className="onset-times">
              <span title="Último ataque de la entrada">{raster.lastIn}</span>
              <span title="Onset de la última respuesta de Moshi: segundos desde el estímulo">{raster.latency}</span>
            </div>
          </div>
        </section>

        <section className="cell compare" aria-label="Análisis">
          <div className="caption">
            <strong>Compará.</strong>
            <small>IN × OUT</small>
          </div>
          <div className="compare-body">
            <EnvelopeXYScope inputAnalyser={inputAnalyser} outputAnalyser={outputAnalyser} />
            <dl className="readout">
              <dt>FRAME</dt><dd>80 ms</dd>
              <dt>PAD</dt><dd>&lt;pad&gt; #{padTokenId}</dd>
            </dl>
            <div className="steppers">
              <Stepper
                code="RANGE"
                title="Rango del histograma (solo cambia la vista)"
                text={`${histogramRange.toFixed(1)}s`}
                onDecrease={() => setHistogramRange((value) => Math.max(0.5, value - 0.5))}
                onIncrease={() => setHistogramRange((value) => Math.min(10, value + 0.5))}
              />
              <Stepper
                code="BINS"
                title="Barras del histograma (solo cambia la vista)"
                text={String(Math.round(histogramBins))}
                onDecrease={() => setHistogramBins((value) => Math.max(4, value - 1))}
                onIncrease={() => setHistogramBins((value) => Math.min(40, value + 1))}
              />
            </div>
            <div className="histogram">
              <div className="histogram-head">
                <span>Δt</span>
                <span className="histogram-tools">
                  <small title="Ataques registrados: entrada · Moshi">in {histogram.inCount} · out {histogram.outCount}</small>
                  <button type="button" className="mini-key" onClick={resetOnsetAnalysis} aria-label="Volver a esperar una repetición" title="Volver a esperar una repetición">
                    <Icon d={ICONS.reset} size={12} />
                  </button>
                  <button type="button" className="mini-key" onClick={clearOnsetRecords} aria-label="Borrar registros de ataques" title="Borrar registros de ataques">
                    <Icon d={ICONS.cross} size={12} />
                  </button>
                </span>
              </div>
              <div
                className="histogram-bars"
                style={{ gridTemplateColumns: `repeat(${histogram.intervals.length}, minmax(0, 1fr))` }}
                title="Intervalos entre ataques consecutivos: trama = entrada, negro = Moshi"
              >
                {histogram.intervals.map((bin, index) => (
                  <div className="histogram-bin" key={index}>
                    <i className="bar-in" style={{ height: `${bin.inH}%` }} />
                    <i className="bar-out" style={{ height: `${bin.outH}%` }} />
                  </div>
                ))}
              </div>
              <div className="histogram-scale"><span>0</span><span>{histogramRange.toFixed(1)}s</span></div>
            </div>
            <div className="histogram">
              <div className="histogram-head">
                <span>IN − OUT</span>
                <small title="Diferencias medidas">n {histogram.deltaCount}</small>
              </div>
              <div
                className="histogram-bars histogram-bars-signed"
                style={{ gridTemplateColumns: `repeat(${histogram.deltas.length}, minmax(0, 1fr))` }}
                title="Último ataque de la entrada menos último ataque de Moshi. A la izquierda del cero, la entrada vino primero"
              >
                {histogram.deltas.map((height, index) => (
                  <div className="histogram-bin" key={index}>
                    <i className="bar-out" style={{ height: `${height}%` }} />
                  </div>
                ))}
                <i className="histogram-zero" aria-hidden="true" />
              </div>
              <div className="histogram-scale">
                <span>−{histogramRange.toFixed(1)}s</span>
                <span>0</span>
                <span>+{histogramRange.toFixed(1)}s</span>
              </div>
            </div>
          </div>
        </section>

        <div className="deck">
          <Module verb="Repetí." code="SEQ" icon={ICONS.seq}>
            <Knob code="RATE" title="Disparos por segundo" value={triggerRate} min={0.1} max={8} step={0.1} decimals={2} onChange={setTriggerRate} {...midiControlProps("trigger-rate")} />
            <Knob code="PITCH" title="Velocidad y altura del sample" value={playbackRate} min={0.25} max={2} step={0.05} decimals={2} unit="×" onChange={setPlaybackRate} {...midiControlProps("playback-rate")} />
          </Module>
          <Module verb="Pensá." code="TXT" icon={ICONS.text}>
            <Knob code="TEMP" title="Temperatura del texto: más alta, más impredecible" value={params.textTemperature} min={0} max={1.2} step={0.01} decimals={2} onChange={(value) => updateParam("textTemperature", value)} {...midiControlProps("text-temperature")} />
            <Knob code="TOP-K" title="Candidatos de texto por paso" value={params.textTopk} min={1} max={500} step={1} decimals={0} onChange={(value) => updateParam("textTopk", value)} {...midiControlProps("text-top-k")} />
            <Knob code="PAD" title={`Token ${padTokenId}. Positivo, más silencio. Negativo, Moshi habla sin que le hablen`} value={params.padBias} min={-10} max={10} step={0.05} decimals={1} bipolar onChange={(value) => updateParam("padBias", value)} {...midiControlProps("pad-logit-bias")} />
          </Module>
          <Module verb="Soná." code="AUD" icon={ICONS.wave}>
            <Knob code="TEMP" title="Temperatura del audio" value={params.audioTemperature} min={0} max={1.2} step={0.01} decimals={2} onChange={(value) => updateParam("audioTemperature", value)} {...midiControlProps("audio-temperature")} />
            <Knob code="TOP-K" title="Candidatos de audio por paso" value={params.audioTopk} min={1} max={500} step={1} decimals={0} onChange={(value) => updateParam("audioTopk", value)} {...midiControlProps("audio-top-k")} />
          </Module>
          <Module verb="Mezclá." code="MIX" icon={ICONS.mix}>
            <Knob code="IN" title="Nivel de entrada" value={inputGain} min={0} max={1.5} step={0.05} decimals={2} onChange={setInputGain} {...midiControlProps("input-level")} />
            <Knob code="OUT" title="Nivel de salida (solo la reproducción local)" value={outputGain} min={0} max={1.5} step={0.05} decimals={2} onChange={setOutputGain} {...midiControlProps("output-level")} />
            <div className="knob" title="Ganancia automática: sube la voz baja y limita los picos">
              <span className="knob-code">AGC</span>
              <button
                type="button"
                className="switch"
                role="switch"
                aria-checked={autoGainEnabled}
                aria-label="Ganancia automática"
                onClick={() => setAutoGainEnabled((enabled) => !enabled)}
              >
                <i />
              </button>
              <span className="knob-value">{autoGainEnabled ? `+${autoGainDb.toFixed(1)}dB` : "OFF"}</span>
              <span className="knob-spacer" />
            </div>
          </Module>
        </div>

        <section className="cell forget" aria-label="Contexto del modelo">
          <div className="caption">
            <strong>Olvidá.</strong>
            <small>CTX</small>
          </div>
          <div className="forget-body">
            <button
              type="button"
              className="key flush"
              disabled={connection !== "live"}
              onClick={() => drainThen(resetContext)}
              aria-label="Borrar el contexto del modelo; la conexión sigue"
              title="Borrar el contexto del modelo: Moshi olvida, la conexión y el sampler siguen"
            >
              <Icon d={ICONS.reset} size={30} />
              <small>FLUSH</small>
            </button>
            <button type="button" className="key defaults" onClick={resetDefaults} title="Valores por defecto del muestreo">
              <small>DEFAULTS</small>
            </button>
          </div>
        </section>
      </div>
    </main>
  );
};

const MODEL_NAME = "moshiko-7b";
const CURRENT_LINE_BUDGET = 72;
const HISTORY_LINES = 4;
const RASTER_WINDOW_SECONDS = 15;

const ICONS = {
  play: "M7 4 L19 12 L7 20 Z",
  loop: "M20 12 A8 8 0 1 1 17.7 6.3 M20 3 V8 H15",
  upload: "M12 16 V4 M6 10 L12 4 L18 10 M4 20 H20",
  reset: "M4 12 A8 8 0 1 0 6.3 6.3 M4 3 V8 H9",
  cross: "M6 6 L18 18 M18 6 L6 18",
  backspace: "M8 5 H21 V19 H8 L2 12 Z M11 9 L17 15 M17 9 L11 15",
  lock: "M5 11 H19 V21 H5 Z M8 11 V7 A4 4 0 0 1 16 7",
  speakerOn: "M3 9 H7 L12 5 V19 L7 15 H3 Z M16 9 A4 4 0 0 1 16 15 M18.5 6.5 A8 8 0 0 1 18.5 17.5",
  speakerOff: "M3 9 H7 L12 5 V19 L7 15 H3 Z M16 9 L22 15 M22 9 L16 15",
  record: "M12 4 A8 8 0 1 1 11.99 4 Z",
  stop: "M6 6 H18 V18 H6 Z",
  seq: "M3 12 H6 M10 12 H13 M17 12 H20 M3 6 V18",
  text: "M4 5 H20 M12 5 V20 M8 20 H16",
  wave: "M2 12 H5 L7 7 L10 17 L13 4 L16 20 L18 12 H22",
  mix: "M6 3 V21 M12 3 V21 M18 3 V21 M3 8 H9 M9 15 H15 M15 10 H21",
  sample: "M2 12 H5 L8 5 L12 19 L15 8 L17 12 H22",
  mic: "M9 3 H15 V13 H9 Z M5 11 A7 7 0 0 0 19 11 M12 18 V22 M8 22 H16",
  system: "M3 4 H21 V16 H3 Z M8 20 H16 M12 16 V20",
};

const SOURCES: { mode: InputMode; code: string; title: string; icon: string }[] = [
  { mode: "sample", code: "LOOP", title: "Sample en loop", icon: ICONS.sample },
  { mode: "microphone", code: "MIC", title: "Micrófono", icon: ICONS.mic },
  { mode: "system", code: "SYS", title: "Audio del sistema o de una pestaña", icon: ICONS.system },
];

const Icon: FC<{ d: string; size: number }> = ({ d, size }) => (
  <svg className="icon" viewBox="0 0 24 24" width={size} height={size} aria-hidden="true">
    <path d={d} />
  </svg>
);

const MidiGlyph: FC = () => (
  <svg viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">
    <circle cx="2.5" cy="6" r="1.2" />
    <circle cx="9.5" cy="6" r="1.2" />
    <circle cx="3.6" cy="3" r="1.2" />
    <circle cx="8.4" cy="3" r="1.2" />
    <circle cx="6" cy="2" r="1.2" />
  </svg>
);

const Module: FC<{ verb: string; code: string; icon: string; children: ReactNode }> = ({
  verb,
  code,
  icon,
  children,
}) => (
  <section className="cell module" aria-label={verb}>
    <div className="caption">
      <strong>{verb}</strong>
      <span className="caption-tools">
        <Icon d={icon} size={14} />
        <small>{code}</small>
      </span>
    </div>
    <div className="module-body">{children}</div>
  </section>
);

const Stepper: FC<{
  code: string;
  title: string;
  text: string;
  onDecrease: () => void;
  onIncrease: () => void;
}> = ({ code, title, text, onDecrease, onIncrease }) => (
  <div className="stepper" title={title}>
    <span>{code}</span>
    <div className="stepper-box">
      <button type="button" onClick={onDecrease} aria-label={`Bajar ${code}`}>−</button>
      <output>{text}</output>
      <button type="button" onClick={onIncrease} aria-label={`Subir ${code}`}>+</button>
    </div>
  </div>
);

function polar(angle: number, radius: number) {
  const radians = (angle * Math.PI) / 180;
  return { x: 32 + radius * Math.sin(radians), y: 32 - radius * Math.cos(radians) };
}

function arcPath(from: number, to: number, radius = 26) {
  if (Math.abs(to - from) < 0.5) return "M0 0";
  const start = polar(from, radius);
  const end = polar(to, radius);
  return `M${start.x.toFixed(2)} ${start.y.toFixed(2)} A${radius} ${radius} 0 ${to - from > 180 ? 1 : 0} 1 ${end.x.toFixed(2)} ${end.y.toFixed(2)}`;
}

type KnobProps = {
  code: string;
  title: string;
  value: number;
  min: number;
  max: number;
  step: number;
  decimals: number;
  unit?: string;
  bipolar?: boolean;
  onChange: (value: number) => void;
  midiKey?: string;
  midiLearning?: boolean;
  midiMapping?: MidiMapping;
  onMidiLearn?: (target: string) => void;
  registerMidiTarget?: (
    target: string,
    handler: (normalizedValue: number) => void,
  ) => () => void;
};

/** A drawn dial: drag up or right to raise, down or left to lower; arrow keys work too. */
const Knob: FC<KnobProps> = ({
  code,
  title,
  value,
  min,
  max,
  step,
  decimals,
  unit = "",
  bipolar = false,
  onChange,
  midiKey,
  midiLearning = false,
  midiMapping,
  onMidiLearn,
  registerMidiTarget,
}) => {
  const dragRef = useRef<{ x: number; y: number; value: number } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const quantize = useCallback((raw: number) => {
    const snapped = min + Math.round((raw - min) / step) * step;
    return Number(Math.min(max, Math.max(min, snapped)).toFixed(6));
  }, [max, min, step]);

  useEffect(() => {
    if (!midiKey || !registerMidiTarget) return;
    return registerMidiTarget(midiKey, (normalizedValue) => {
      onChange(quantize(min + normalizedValue * (max - min)));
    });
  }, [max, midiKey, min, onChange, quantize, registerMidiTarget]);

  const ratio = (value - min) / (max - min);
  const angle = -135 + ratio * 270;
  const origin = bipolar ? 0 : -135;
  const tip = polar(angle, 14);
  const base = polar(angle, 5);
  const text = `${bipolar && value > 0 ? "+" : ""}${value.toFixed(decimals)}${unit}`;
  const midiTitle = midiLearning
    ? "Aprendiendo: mové un control MIDI; clic para cancelar"
    : midiMapping
      ? `${midiMapping.inputName} · CH ${midiMapping.channel + 1} CC ${midiMapping.controller}; clic para reasignar`
      : "MIDI learn: clic y mové un control";

  return (
    <div className="knob" title={title}>
      <span className="knob-code">{code}</span>
      <div
        className="knob-dial"
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = { x: event.clientX, y: event.clientY, value };
          inputRef.current?.focus();
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag) return;
          const delta = (event.clientX - drag.x) - (event.clientY - drag.y);
          onChange(quantize(drag.value + (delta * (max - min)) / 160));
        }}
        onPointerUp={() => {
          dragRef.current = null;
        }}
        onPointerCancel={() => {
          dragRef.current = null;
        }}
      >
        <svg viewBox="0 0 64 64" width="64" height="64" aria-hidden="true">
          <path className="knob-track" d={arcPath(-135, 135)} />
          <path className="knob-arc" d={arcPath(Math.min(origin, angle), Math.max(origin, angle))} />
          <circle className="knob-cap" cx="32" cy="32" r="17" />
          <line className="knob-needle" x1={base.x} y1={base.y} x2={tip.x} y2={tip.y} />
        </svg>
        <input
          ref={inputRef}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          aria-label={title}
          aria-valuetext={text}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      </div>
      <span className="knob-value">{text}</span>
      {midiKey && onMidiLearn ? (
        <button
          type="button"
          className={`midi-dot ${midiLearning ? "learning" : ""} ${midiMapping ? "mapped" : ""}`}
          aria-pressed={midiLearning}
          aria-label={`${code}: ${midiTitle}`}
          title={midiTitle}
          onClick={() => onMidiLearn(midiKey)}
        >
          <MidiGlyph />
        </button>
      ) : (
        <span className="knob-spacer" />
      )}
    </div>
  );
};
