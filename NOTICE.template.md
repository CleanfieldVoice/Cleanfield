# NOTICE fragment for integrators

Copy the block below into your own NOTICE / third-party notices file when you
redistribute an application that embeds CleanField.

```
This product includes CleanField, a neural noise-suppression and echo-cancellation
inference engine.

  Engine and model weights: CleanField (MIT License, see the project's LICENSE
  file).

Processing is performed entirely on the user's device. No audio is transmitted
to a server by the engine.
```

Notes:

- Keep the exact engine version (see `cf_abi_version()` and the release tag) in
  your own build metadata.
- The model containers in `models/` are covered by the same MIT license as the
  engine code.
