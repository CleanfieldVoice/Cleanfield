//! End-to-end echo cancellation: the linear AEC front end followed by the
//! neural echo model.
//!
//! This crate is deliberately a thin integration layer over `cf-aec` and
//! `cf-echo`: it owns the analysis/synthesis front end, the three-channel
//! feature builder, the mask application and the streaming framing, while both
//! numeric stages stay in their own crates.
//!
//! # Verification boundary
//!
//! The neural model is bit exact. The linear AEC has no oracle and is validated
//! functionally; the STFT front end here is our own (fixed-size `f32` radix-2),
//! so the complete path is **not** bit exact against the former reference
//! pipeline. It is validated on synthetic and real material instead. See
//! `README.md` and `CONTRACT.md`.
//!
//! # Example
//!
//! ```no_run
//! use cf_echo_pipeline::{EchoPipeline, EchoPipelineConfig, FRAME_LEN};
//!
//! let bytes = std::fs::read("models/echo-default.cfwm").unwrap();
//! let mut pipeline = EchoPipeline::new(&bytes, EchoPipelineConfig::default())
//!     .expect("valid echo container");
//! let mic = [0.0f32; FRAME_LEN];
//! let reference = [0.0f32; FRAME_LEN];
//! let out = pipeline.process(&mic, &reference);
//! assert_eq!(out.len(), FRAME_LEN);
//! ```

pub mod fft;
pub mod pipeline;

pub use fft::{Fft, BINS, FFT_SIZE, WINDOW_LEN};
pub use pipeline::{
    periodic_hann, EchoPipeline, EchoPipelineConfig, PipelineError, EPSILON, FEATURE_BINS,
    FEATURE_LEN, FRAME_LEN, HOP_LEN, INPUT_CHANNELS, LATENCY_SAMPLES, LOOKAHEAD_HOPS,
    MAX_BLOCK_LEN, SAMPLE_RATE, SYNTHESIS_SCALE, WINDOW_SIZE,
};
