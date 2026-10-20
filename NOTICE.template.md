# NOTICE fragment for integrators

Copy the block below into your own NOTICE / third-party notices file when you
redistribute an application that embeds CleanField.

```
This product includes CleanField, a neural noise-suppression and echo-cancellation
inference engine.

  Engine: CleanField (MIT License, see the project's LICENSE file).
  Model weights: distributed with the engine as container files; the terms under
  which these weights may be redistributed are stated in the project's
  the project documentation. They are not covered by the engine's MIT grant.

Processing is performed entirely on the user's device. No audio is transmitted
to a server by the engine.
```

Notes:

- Keep the exact engine version (see `cf_abi_version()` and the release tag) in
  your own build metadata.
- If you ship only the engine and not the weights (for example you load a
  container you built yourself), adjust the second bullet accordingly.
