const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const ts = require("typescript");

const filename = path.join(__dirname, "../src/pages/Studio/PhaseClock.ts");
const compiled = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
});
const scope = { exports: {} };
vm.runInNewContext(compiled.outputText, scope);
const { PhaseClock } = scope.exports;
const near = (value, expected) => assert(Math.abs(value - expected) < 1e-9, `${value} != ${expected}`);

test("interpolates between packets and freezes at the received phase", () => {
  const clock = new PhaseClock();
  clock.setFrameRate(12.5);
  clock.reset(0, 0);
  clock.observe(0.08, 80);
  near(clock.sample(80).phase, 0);
  near(clock.sample(120).phase, 0.04);
  near(clock.sample(160).phase, 0.08);
  near(clock.sample(10000).phase, 0.08);
});

test("moves forward across wrap instead of animating backwards", () => {
  const clock = new PhaseClock();
  clock.reset(0.96, 0);
  clock.observe(0.04, 80);
  near(clock.sample(100).unwrapped, 0.98);
  near(clock.sample(140).unwrapped, 1.02);
  near(clock.sample(160).phase, 0.04);
});

test("jittered packets retain visual continuity and duplicate phases do not stall", () => {
  const clock = new PhaseClock();
  clock.reset(0, 0);
  clock.observe(0.08, 80);
  const before = clock.sample(140).unwrapped;
  clock.observe(0.16, 140);
  near(clock.sample(140).unwrapped, before);
  clock.observe(0.16, 170);
  near(clock.sample(220).phase, 0.16);
});

test("batched high-rate ticks retain every wrap", () => {
  const clock = new PhaseClock();
  clock.reset(0, 0);
  for (const phase of [0.48, 0.96, 0.44, 0.92, 0.4]) clock.observe(phase, 80);
  near(clock.sample(120).unwrapped, 1.2);
  near(clock.sample(160).unwrapped, 2.4);
});

test("reset is immediate and invalid packets are ignored", () => {
  const clock = new PhaseClock();
  clock.reset(0.7, 0);
  clock.observe(0.8, 80);
  clock.reset(0, 100);
  for (const phase of [NaN, Infinity, -0.1, 1]) clock.observe(phase, 100);
  near(clock.sample(100).phase, 0);
  near(clock.sample(1000).phase, 0);
});
