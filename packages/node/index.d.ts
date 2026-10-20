export type BackendPreference = 'auto' | 'scalar' | 'avx2';

/** The backend the engine actually selected at load time. */
export type LiveBackend = 'scalar' | 'avx2';

export interface VoiceOptions {
  /**
   * Path to a `.cfwm` voice container. Defaults to the model bundled with the
   * package, then to `CLEANFIELD_VOICE_MODEL`.
   */
  modelPath?: string;
  /**
   * Backend preference. `auto` picks AVX2+FMA when the CPU supports it and
   * falls back to the scalar reference otherwise. `scalar` / `avx2` never fall
   * back: an unavailable backend throws.
   */
  backend?: BackendPreference;
}

export interface Voice {
  /** Backend the engine loaded with. */
  readonly backend: LiveBackend;
  /**
   * Run one 513-value feature frame and return a new 1026-value mask frame.
   * The returned array is freshly allocated on every call.
   */
  process(frame: Float32Array): Float32Array;
  /** Reset the cross-frame state (history, LSTM cells, quantised state). */
  reset(): void;
  /** Release the engine immediately. Safe to call more than once. */
  close(): void;
}

/**
 * Create one voice engine instance. Each instance owns an independent model
 * and workspace; calls on the same instance must not overlap.
 */
export function createVoice(options?: VoiceOptions): Voice;
