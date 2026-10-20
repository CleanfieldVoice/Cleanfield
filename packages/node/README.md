# @cleanfield/node

CleanField real-time speech enhancement for Node.js: a prebuilt native addon,
no node-gyp and no Rust toolchain required.

```js
const { createVoice } = require('@cleanfield/node');

const denoiser = createVoice();
const mask = denoiser.process(frame);   // Float32Array(513) -> Float32Array(1026)
```

- Bit-exact with the reference implementation; AVX2 with a scalar fallback.
- 16 kHz mono, one instance per stream, allocation-free steady state.
- Prebuilt binaries are downloaded from the matching GitHub release on install
  (win32 x64/arm64, darwin universal, linux x64/arm64).

Source and the C ABI: https://github.com/CleanfieldVoice/Cleanfield

License: MIT.
