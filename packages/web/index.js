// CleanField Voice 1.0 - browser glue.
// Usage:
//   import { attach } from "@cleanfield/web";
//   const { node, detach } = await attach(audioContext, { gain: 1.0 });
//   source.connect(node).connect(audioContext.destination);
const WASM_URL = new URL("./cf_wasm.wasm", import.meta.url).href;
const WORKLET_URL = new URL("./worklet-processor.js", import.meta.url).href;

export async function attach(context, options = {}) {
  if (!context?.audioWorklet) throw new Error("AudioWorklet is required");
  await context.audioWorklet.addModule(WORKLET_URL);
  const response = await fetch(WASM_URL, { credentials: "same-origin" });
  if (!response.ok) throw new Error(`wasm fetch failed: ${response.status}`);
  const bytes = await response.arrayBuffer();
  const node = new AudioWorkletNode(context, "cleanfield-voice", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    channelCount: 1,
    channelCountMode: "explicit",
    channelInterpretation: "speakers",
  });
  node.port.postMessage({ type: "init", wasm: bytes, gain: options.gain ?? 1.0 });
  return { node, detach: () => { try { node.disconnect(); } catch {} node.port.postMessage({ type: "close" }); } };
}

export const version = "1.0.0";
