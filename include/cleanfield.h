/*
 * CleanField - native embedding ABI
 *
 * One opaque handle type, one small set of C entry points, no C++ and no
 * runtime dependency beyond the platform C library.  The library is built by
 *
 *     cargo build --release -p cf-ffi
 *
 * and produces `cleanfield.lib` (static) plus `cleanfield.dll` and
 * `cleanfield.dll.lib` (dynamic) under `target/release/` on MSVC; ELF targets
 * produce `libcleanfield.a` / `libcleanfield.so`.
 *
 * Model data is NOT embedded in the native library.  The host reads a `.cfwm`
 * container into memory and passes it to `cf_model_new` / `cf_voice_new` /
 * `cf_echo_new`.  The bytes are copied into an aligned buffer at load time;
 * after construction no inference call allocates.
 *
 * Typical real-time use:
 *
 *     cf_model *m = cf_echo_new(bytes, len);
 *     if (!m) { ... cf_last_error() ... }
 *     float *in  = malloc(cf_echo_input_len()  * sizeof(float));
 *     float *out = malloc(cf_echo_output_len() * sizeof(float));
 *     // audio callback, one call per 10/16 ms frame:
 *     if (cf_echo_process(m, in, out) != 0) { ... }
 *     cf_echo_free(m);
 *
 * Threading: a handle owns its weights and fixed workspace.  Different handles
 * may be used from different threads concurrently.  One handle must not be
 * used concurrently or re-entrantly.  cf_last_error() is process-global, so
 * read it immediately after a failure (return codes are thread-safe).
 *
 * Buffer contract: `input` must be readable for the reported input length and
 * `output` writable for the reported output length; the two buffers must not
 * overlap.  Pointers stay owned by the caller.
 *
 * Error model: constructors return NULL (code in cf_last_error()); process and
 * reset return 0 on success or the negated CF_ERR_* code.  There is no silent
 * backend fallback: asking for AVX2 on a CPU that cannot run it fails with
 * CF_ERR_BACKEND.
 *
 * ABI revision: 1.  Additions are backward compatible; a breaking change
 * increments cf_abi_version().
 */

#ifndef CLEANFIELD_H
#define CLEANFIELD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CF_ABI_VERSION 1u

/* Container model ids. */
#define CF_MODEL_VOICE 1u
#define CF_MODEL_ECHO 2u

/* Backend selector for the *_with_backend constructors. */
#define CF_BACKEND_AUTO 0u
#define CF_BACKEND_SCALAR 1u
#define CF_BACKEND_AVX2 2u

/* Error codes (also negated as process/reset return values). */
#define CF_OK 0u
#define CF_ERR_ARG 1u       /* null pointer, zero/misaligned length, bad backend id */
#define CF_ERR_CONTAINER 2u /* not a valid .cfwm image */
#define CF_ERR_MODEL 3u     /* wrong model id, failed validation, wrong handle type */
#define CF_ERR_STATE 4u     /* null handle/frame buffer or failed frame */
#define CF_ERR_BACKEND 5u   /* requested backend unavailable on this CPU */

/* Opaque handle. */
typedef struct cf_model cf_model;

/* ------------------------------------------------------------------ */
/* version and diagnostics                                             */
/* ------------------------------------------------------------------ */

/* ABI revision (1). */
uint32_t cf_abi_version(void);

/* Last error code set by a fallible call; 0 when the last call succeeded. */
uint32_t cf_last_error(void);

/* ------------------------------------------------------------------ */
/* unified construction (model id 1 = voice, 2 = echo)                 */
/* ------------------------------------------------------------------ */

/* Load a container from memory with the automatic backend (AVX2 when the CPU
 * supports AVX2 + FMA, otherwise scalar).  `len` must be a multiple of 4.
 * Returns NULL on failure.  The input buffer is not retained. */
cf_model *cf_model_new(uint32_t model_id, const uint8_t *model_bytes, size_t len);

/* Same, with an explicit backend (CF_BACKEND_AUTO/SCALAR/AVX2). */
cf_model *cf_model_new_with_backend(uint32_t model_id,
                                    const uint8_t *model_bytes,
                                    size_t len,
                                    uint32_t backend);

/* Run one frame.  Input/output lengths are per handle:
 * voice 513 -> 1026, echo 1539 -> 513.  Returns 0 or a negated CF_ERR_*. */
int32_t cf_model_process(cf_model *model, const float *input, float *output);

/* Reset the cross-frame state (history, recurrent state, quantised state). */
int32_t cf_model_reset(cf_model *model);

/* Release a handle from any constructor.  NULL is a no-op. */
void cf_model_free(cf_model *model);

/* Container model id (1/2) or 0 on NULL. */
uint32_t cf_model_id(const cf_model *model);

/* Input feature count of the handle, or 0 on NULL. */
uint32_t cf_model_input_len(const cf_model *model);

/* Output value count of the handle, or 0 on NULL. */
uint32_t cf_model_output_len(const cf_model *model);

/* Live backend: CF_BACKEND_SCALAR (1) or CF_BACKEND_AVX2 (2); 0 on NULL. */
uint32_t cf_model_backend(const cf_model *model);

/* Live backend name ("scalar"/"avx2"), NUL-terminated static string, or NULL. */
const char *cf_model_backend_name(const cf_model *model);

/* Resident container bytes of the handle, or 0 on NULL. */
uint32_t cf_model_model_bytes(const cf_model *model);

/* Fixed workspace footprint of the handle in bytes, or 0 on NULL. */
uint32_t cf_model_workspace_bytes(const cf_model *model);

/* ------------------------------------------------------------------ */
/* voice model (513 f32 in, 1026 f32 out)                              */
/* ------------------------------------------------------------------ */

cf_model *cf_voice_new(const uint8_t *model_bytes, size_t len);
cf_model *cf_voice_new_with_backend(const uint8_t *model_bytes,
                                    size_t len,
                                    uint32_t backend);

int32_t cf_voice_process(cf_model *model, const float *input, float *output);
int32_t cf_voice_reset(cf_model *model);
void cf_voice_free(cf_model *model);

uint32_t cf_voice_input_len(void);   /* 513 */
uint32_t cf_voice_output_len(void);  /* 1026 */
uint32_t cf_voice_model_id(void);    /* 1 */
uint32_t cf_voice_workspace_bytes(void);
uint32_t cf_voice_model_bytes(const cf_model *model);

/* ------------------------------------------------------------------ */
/* echo model (3 x 513 f32 in, 513 f32 out)                            */
/* ------------------------------------------------------------------ */

cf_model *cf_echo_new(const uint8_t *model_bytes, size_t len);
cf_model *cf_echo_new_with_backend(const uint8_t *model_bytes,
                                   size_t len,
                                   uint32_t backend);

int32_t cf_echo_process(cf_model *model, const float *input, float *output);
int32_t cf_echo_reset(cf_model *model);
void cf_echo_free(cf_model *model);

uint32_t cf_echo_input_len(void);   /* 1539 */
uint32_t cf_echo_output_len(void);  /* 513 */
uint32_t cf_echo_model_id(void);    /* 2 */
uint32_t cf_echo_workspace_bytes(void);
uint32_t cf_echo_model_bytes(const cf_model *model);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CLEANFIELD_H */
