# @cleanfield/web

CleanField Voice 1.0 noise suppression for the browser: a 19-layer neural model
running in WebAssembly inside an AudioWorklet. No build step, no backend.

```js
import { attach } from '@cleanfield/web';

const { node, detach } = await attach(audioContext, { gain: 1.0 });
source.connect(node).connect(audioContext.destination);
```

- 16 kHz mono, 768-sample window, 256-sample hop, 80 ms algorithmic latency.
- Processes 0-8 kHz; higher bands pass through.
- Allocation-free steady state; the wasm module embeds the model (2.6 MB).
- `attach()` loads the worklet and the wasm, returns the node and a `detach()`.

Source, benchmark numbers and the full engine: https://github.com/CleanfieldVoice/Cleanfield

License: MIT.
