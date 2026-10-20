/*
 * Tiny C consumer of the CleanField native ABI.
 *
 * Build (from the project root, dynamic link):
 *   cl /nologo /O2 /MD /I include examples\native_embed.c ^
 *      /Fe:artifacts\m6\embed-c.exe /link /LIBPATH:target\release cleanfield.dll.lib
 *
 * Static link additionally needs the Rust runtime's system libraries:
 *   cl /nologo /O2 /MD /I include examples\native_embed.c ^
 *      /Fe:artifacts\m6\embed-c-static.exe /link /LIBPATH:target\release ^
 *      cleanfield.lib ntdll.lib userenv.lib ws2_32.lib dbghelp.lib bcrypt.lib
 *
 * Run from the project root with the DLL next to the executable (or on PATH).
 */

#include "cleanfield.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_file(const char *path, uint8_t **out, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    long n;
    uint8_t *buf;
    size_t got;

    if (!f) {
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    n = ftell(f);
    if (n <= 0) {
        fclose(f);
        return 0;
    }
    rewind(f);
    buf = (uint8_t *)malloc((size_t)n);
    if (!buf) {
        fclose(f);
        return 0;
    }
    got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) {
        free(buf);
        return 0;
    }
    *out = buf;
    *out_len = (size_t)n;
    return 1;
}

static void fill(float *buf, size_t n, unsigned frame)
{
    uint64_t s = 0x9E3779B97F4A7C15ull ^ ((uint64_t)frame * 0x100000001B3ull);
    size_t i;

    for (i = 0; i < n; i++) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        buf[i] = (float)((double)(s >> 40) * (1.0 / 16777216.0));
    }
}

static uint32_t float_bits(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return bits;
}

static int run_model(unsigned model_id, const char *label, const char *path)
{
    uint8_t *bytes = NULL;
    size_t len = 0;
    cf_model *typed;
    cf_model *unified;
    uint32_t in_len;
    uint32_t out_len;
    float *in;
    float *a;
    float *b;
    uint64_t checksum = 0;
    unsigned long compared = 0;
    unsigned long nonzero = 0;
    int pass;
    int f;

    if (!read_file(path, &bytes, &len)) {
        fprintf(stderr, "cannot read %s\n", path);
        return 0;
    }
    typed = (model_id == 1) ? cf_voice_new(bytes, len) : cf_echo_new(bytes, len);
    unified = cf_model_new(model_id, bytes, len);
    if (!typed || !unified) {
        fprintf(stderr, "%s: load failed (code %u)\n", label, cf_last_error());
        free(bytes);
        return 0;
    }
    in_len = cf_model_input_len(typed);
    out_len = cf_model_output_len(typed);
    if (cf_model_id(typed) != model_id) {
        fprintf(stderr, "%s: wrong model id\n", label);
        return 0;
    }
    if ((model_id == 1 && (in_len != 513 || out_len != 1026)) ||
        (model_id == 2 && (in_len != 1539 || out_len != 513))) {
        fprintf(stderr, "%s: wrong frame lengths %u/%u\n", label, in_len, out_len);
        return 0;
    }

    in = (float *)malloc((size_t)in_len * sizeof(float));
    a = (float *)malloc((size_t)out_len * sizeof(float));
    b = (float *)malloc((size_t)out_len * sizeof(float));
    if (!in || !a || !b) {
        fprintf(stderr, "%s: out of memory\n", label);
        return 0;
    }

    for (pass = 0; pass < 2; pass++) {
        int rc;
        if (pass == 1) {
            rc = (model_id == 1) ? cf_voice_reset(typed) : cf_echo_reset(typed);
            if (rc != 0 || cf_model_reset(unified) != 0) {
                fprintf(stderr, "%s: reset failed\n", label);
                return 0;
            }
        }
        for (f = 0; f < 32; f++) {
            uint32_t i;
            int rc_a;
            int rc_b;

            fill(in, (size_t)in_len, (unsigned)f);
            rc_a = (model_id == 1)
                       ? cf_voice_process(typed, in, a)
                       : cf_echo_process(typed, in, a);
            rc_b = cf_model_process(unified, in, b);
            if (rc_a != 0 || rc_b != 0) {
                fprintf(stderr, "%s: process failed (%d/%d, code %u)\n",
                        label, rc_a, rc_b, cf_last_error());
                return 0;
            }
            if (memcmp(a, b, (size_t)out_len * sizeof(float)) != 0) {
                fprintf(stderr, "%s: typed and unified outputs differ\n", label);
                return 0;
            }
            for (i = 0; i < out_len; i++) {
                checksum ^= (uint64_t)float_bits(a[i]);
                checksum *= 0x00000100000001B3ull;
                if (float_bits(a[i]) & 0x7fffffffu) {
                    nonzero++;
                }
                compared++;
            }
        }
    }

    printf("%s: id=%u in=%u out=%u backend=%s workspace=%u B, 64 frames x 2 passes, "
           "%lu values compared (%lu non-zero), checksum=%016llx\n",
           label, model_id, in_len, out_len, cf_model_backend_name(typed),
           cf_model_workspace_bytes(typed), compared, nonzero,
           (unsigned long long)checksum);

    free(in);
    free(a);
    free(b);
    if (model_id == 1) {
        cf_voice_free(typed);
    } else {
        cf_echo_free(typed);
    }
    cf_model_free(unified);
    free(bytes);
    return 1;
}

int main(int argc, char **argv)
{
    const char *echo_path = (argc > 1) ? argv[1] : "models/echo-default.cfwm";
    const char *voice_path = (argc > 2) ? argv[2] : "models/voice-default.cfwm";

    printf("cleanfield native ABI version %u\n", cf_abi_version());
    if (!run_model(2, "echo", echo_path)) {
        return 1;
    }
    if (!run_model(1, "voice", voice_path)) {
        return 1;
    }
    printf("RESULT: C ABI link example passed\n");
    return 0;
}
