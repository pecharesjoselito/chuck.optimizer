/*
 * moebius.c — ASCII art diffusion
 *
 * Named after Jean "Moebius" Giraud, the legendary French comic artist.
 * Moebius drew with pen and ink — lines, dots, density. No color needed.
 * This module does the same: generates images as ASCII density maps.
 *
 * Chuck optimizes. Lee sees. Kirby draws. Moebius draws with text.
 *
 * Denoising Diffusion Probabilistic Model (DDPM) for ASCII art.
 * Trains on grayscale density maps (CIFAR-100 → grayscale → 32×32).
 * Generates 32×32 density maps → maps to ASCII chars → renders PNG.
 *
 * ASCII density ramp: " .:-=+*#%@"
 *   0.0 = space (empty)
 *   1.0 = @ (maximum density)
 *
 * The diffusion process:
 *   Forward:  x_0 → add noise over T steps → x_T (pure Gaussian)
 *   Reverse:  x_T → predict & remove noise over T steps → x_0
 *   Network:  MLP predicts noise ε given (x_t, t)
 *
 * Output formats:
 *   - Terminal: prints ASCII art directly
 *   - .txt: raw ASCII art text file
 *   - .ppm: rendered as image (each char = 8×8 pixel block)
 *
 * Build:
 *   cc -std=c11 -O2 -o moebius moebius.c -lm
 *   ./moebius --data cifar-100-binary              # train
 *   ./moebius --gen 10 --resume moebius.bin        # generate 10 images
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ---- Config ---- */
#define IMG_W        32       /* density map width */
#define IMG_H        32       /* density map height */
#define IMG_SZ       (IMG_W * IMG_H)  /* 1024 */

#define T_STEPS      500      /* diffusion timesteps (training) */
#define T_SAMPLE     50       /* sampling steps (inference, DDIM-like skip) */

#define TIME_DIM     64       /* timestep embedding dim */
#define H1_DIM       512      /* hidden layer 1 */
#define H2_DIM       512      /* hidden layer 2 */
#define IN_DIM       (IMG_SZ + TIME_DIM)  /* 1088 */

#define LR           0.0002f
#define STEPS        100000
#define BATCH        1        /* no accumulation for simplicity */
#define LOG_EVERY    1000
#define SAVE_EVERY   20000

/* ASCII density ramp — 10 levels */
static const char ASCII_RAMP[] = " .:-=+*#%@";
#define N_RAMP       10

/* ---- RNG ---- */
static uint64_t rng_state = 42;
static uint64_t rnext(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static float randf(void) { return (rnext() & 0xFFFFFF) / (float)0xFFFFFF; }
static float randn(void) {
    float u1 = randf() + 1e-10f, u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ---- Noise schedule (linear beta) ---- */
static float beta[T_STEPS];       /* β_t */
static float alpha[T_STEPS];      /* α_t = 1 - β_t */
static float alpha_bar[T_STEPS];  /* ᾱ_t = Π α_s */
static float sqrt_ab[T_STEPS];    /* √ᾱ_t */
static float sqrt_1mab[T_STEPS];  /* √(1 - ᾱ_t) */

static void init_schedule(void) {
    float beta_start = 0.0001f, beta_end = 0.02f;
    for (int t = 0; t < T_STEPS; t++) {
        beta[t] = beta_start + (beta_end - beta_start) * t / (T_STEPS - 1);
        alpha[t] = 1.0f - beta[t];
        alpha_bar[t] = (t == 0) ? alpha[0] : alpha_bar[t-1] * alpha[t];
        sqrt_ab[t] = sqrtf(alpha_bar[t]);
        sqrt_1mab[t] = sqrtf(1.0f - alpha_bar[t]);
    }
}

/* ---- Data: CIFAR-100 → grayscale density [0,1] ---- */
typedef struct { float *imgs; int n; } Data;

static Data load_cifar100_gray(const char *path) {
    Data d = {0, 0};
    FILE *f = fopen(path, "rb");
    if (!f) return d;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int rec_sz = 2 + 32 * 32 * 3;
    d.n = (int)(sz / rec_sz);
    d.imgs = malloc(d.n * IMG_SZ * sizeof(float));
    uint8_t *buf = malloc(rec_sz);
    for (int i = 0; i < d.n; i++) {
        if (fread(buf, 1, rec_sz, f) != (size_t)rec_sz) { d.n = i; break; }
        /* RGB → grayscale, normalize to [0,1] */
        for (int p = 0; p < 1024; p++) {
            float r = buf[2 + p] / 255.0f;
            float g = buf[2 + 1024 + p] / 255.0f;
            float b = buf[2 + 2048 + p] / 255.0f;
            d.imgs[i * IMG_SZ + p] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
    free(buf); fclose(f);
    return d;
}

/* ---- Linear layer ---- */
typedef struct {
    float *w, *b, *dw, *db;
    float *mw, *vw, *mb, *vb;
    int rows, cols;
} Linear;

static Linear linear_new(int rows, int cols) {
    Linear l;
    l.rows = rows; l.cols = cols;
    float scale = sqrtf(2.0f / cols);
    l.w  = malloc(rows * cols * sizeof(float));
    l.b  = calloc(rows, sizeof(float));
    l.dw = calloc(rows * cols, sizeof(float));
    l.db = calloc(rows, sizeof(float));
    l.mw = calloc(rows * cols, sizeof(float));
    l.vw = calloc(rows * cols, sizeof(float));
    l.mb = calloc(rows, sizeof(float));
    l.vb = calloc(rows, sizeof(float));
    for (int i = 0; i < rows * cols; i++) l.w[i] = randn() * scale;
    return l;
}

static void linear_zero_grad(Linear *l) {
    memset(l->dw, 0, l->rows * l->cols * sizeof(float));
    memset(l->db, 0, l->rows * sizeof(float));
}

static void linear_fwd(Linear *l, const float *in, float *out) {
    for (int r = 0; r < l->rows; r++) {
        float s = l->b[r];
        const float *wr = &l->w[r * l->cols];
        for (int c = 0; c < l->cols; c++) s += wr[c] * in[c];
        out[r] = s;
    }
}

static void linear_bwd(Linear *l, const float *in, const float *d_out, float *d_in) {
    for (int r = 0; r < l->rows; r++) {
        l->db[r] += d_out[r];
        float *wr = &l->w[r * l->cols];
        float *dwr = &l->dw[r * l->cols];
        for (int c = 0; c < l->cols; c++) {
            dwr[c] += d_out[r] * in[c];
            if (d_in) d_in[c] += wr[c] * d_out[r];
        }
    }
}

static void linear_adam(Linear *l, float lr, int t) {
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    int n = l->rows * l->cols;
    for (int i = 0; i < n; i++) {
        l->mw[i] = b1*l->mw[i] + (1-b1)*l->dw[i];
        l->vw[i] = b2*l->vw[i] + (1-b2)*l->dw[i]*l->dw[i];
        l->w[i] -= lr * (l->mw[i]/bc1) / (sqrtf(l->vw[i]/bc2) + eps);
    }
    for (int i = 0; i < l->rows; i++) {
        l->mb[i] = b1*l->mb[i] + (1-b1)*l->db[i];
        l->vb[i] = b2*l->vb[i] + (1-b2)*l->db[i]*l->db[i];
        l->b[i] -= lr * (l->mb[i]/bc1) / (sqrtf(l->vb[i]/bc2) + eps);
    }
}

/* ---- SiLU activation ---- */
static void silu_fwd(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}
static void silu_bwd(const float *x_pre, float *grad, int n) {
    for (int i = 0; i < n; i++) {
        float sig = 1.0f / (1.0f + expf(-x_pre[i]));
        grad[i] *= sig + x_pre[i] * sig * (1.0f - sig);
    }
}

/* ---- Denoiser network ---- */
/*
 * ε_θ(x_t, t) — predicts noise given noisy image and timestep
 *
 * Architecture:
 *   [x_t (1024) | time_emb (64)] → Linear(1088, 512) → SiLU
 *   → Linear(512, 512) → SiLU → Linear(512, 1024) → ε
 *
 * Skip connection: layer2 output += layer1 output (residual)
 */
typedef struct {
    Linear time_proj;  /* (TIME_DIM, TIME_DIM) — timestep MLP */
    Linear l1;         /* (H1_DIM, IN_DIM) */
    Linear l2;         /* (H2_DIM, H1_DIM) */
    Linear l3;         /* (IMG_SZ, H2_DIM) */
    /* Skip projection: H1 → H2 (if dims match, could be identity) */
} Denoiser;

static Denoiser denoiser_new(void) {
    Denoiser d;
    d.time_proj = linear_new(TIME_DIM, TIME_DIM);
    d.l1 = linear_new(H1_DIM, IN_DIM);
    d.l2 = linear_new(H2_DIM, H1_DIM);
    d.l3 = linear_new(IMG_SZ, H2_DIM);
    return d;
}

/* Sinusoidal timestep embedding */
static void time_embed(int t, float *out) {
    for (int i = 0; i < TIME_DIM / 2; i++) {
        float freq = expf(-logf(10000.0f) * i / (TIME_DIM / 2));
        out[i]              = sinf(t * freq);
        out[i + TIME_DIM/2] = cosf(t * freq);
    }
}

/* Forward buffers */
static float t_emb_raw[TIME_DIM], t_emb[TIME_DIM];
static float inp[IN_DIM];              /* concat(x_t, time_emb) */
static float h1_pre[H1_DIM], h1[H1_DIM];  /* layer 1 pre/post activation */
static float h2_pre[H2_DIM], h2[H2_DIM];  /* layer 2 pre/post activation */
static float eps_pred[IMG_SZ];         /* predicted noise */

/* Backward buffers */
static float d_eps[IMG_SZ];
static float d_h2[H2_DIM], d_h2_pre[H2_DIM];
static float d_h1[H1_DIM], d_h1_pre[H1_DIM];
static float d_inp[IN_DIM];
static float d_t_emb[TIME_DIM], d_t_emb_raw[TIME_DIM];

static void denoiser_fwd(Denoiser *d, const float *x_t, int t) {
    /* timestep embedding */
    time_embed(t, t_emb_raw);
    linear_fwd(&d->time_proj, t_emb_raw, t_emb);
    silu_fwd(t_emb, TIME_DIM);

    /* concat input */
    memcpy(inp, x_t, IMG_SZ * sizeof(float));
    memcpy(inp + IMG_SZ, t_emb, TIME_DIM * sizeof(float));

    /* layer 1 */
    linear_fwd(&d->l1, inp, h1_pre);
    memcpy(h1, h1_pre, H1_DIM * sizeof(float));
    silu_fwd(h1, H1_DIM);

    /* layer 2 + skip from layer 1 */
    linear_fwd(&d->l2, h1, h2_pre);
    /* residual: since H1_DIM == H2_DIM, add h1 directly */
    for (int i = 0; i < H2_DIM; i++) h2_pre[i] += h1[i];
    memcpy(h2, h2_pre, H2_DIM * sizeof(float));
    silu_fwd(h2, H2_DIM);

    /* output */
    linear_fwd(&d->l3, h2, eps_pred);
}

static void denoiser_bwd(Denoiser *d, const float *x_t, int t) {
    /* d_eps already filled by caller (MSE grad) */

    /* layer 3 backward */
    memset(d_h2, 0, sizeof(d_h2));
    linear_bwd(&d->l3, h2, d_eps, d_h2);

    /* SiLU backward on h2 */
    memcpy(d_h2_pre, d_h2, H2_DIM * sizeof(float));
    silu_bwd(h2_pre, d_h2_pre, H2_DIM);
    /* skip connection: d_h2_pre also flows to h1 */

    /* layer 2 backward */
    memset(d_h1, 0, sizeof(d_h1));
    linear_bwd(&d->l2, h1, d_h2_pre, d_h1);
    /* add skip gradient */
    for (int i = 0; i < H1_DIM; i++) d_h1[i] += d_h2[i];  /* from residual h1 → h2_pre */

    /* SiLU backward on h1 */
    memcpy(d_h1_pre, d_h1, H1_DIM * sizeof(float));
    silu_bwd(h1_pre, d_h1_pre, H1_DIM);

    /* layer 1 backward */
    memset(d_inp, 0, sizeof(d_inp));
    linear_bwd(&d->l1, inp, d_h1_pre, d_inp);

    /* time_proj backward (for completeness — trains the time embedding) */
    memcpy(d_t_emb, d_inp + IMG_SZ, TIME_DIM * sizeof(float));
    /* SiLU was applied to t_emb — need pre-SiLU for backward */
    float t_emb_pre_silu[TIME_DIM];
    time_embed(t, t_emb_pre_silu);
    float t_emb_after_proj[TIME_DIM];
    linear_fwd(&d->time_proj, t_emb_pre_silu, t_emb_after_proj);
    silu_bwd(t_emb_after_proj, d_t_emb, TIME_DIM);
    memset(d_t_emb_raw, 0, sizeof(d_t_emb_raw));
    linear_bwd(&d->time_proj, t_emb_raw, d_t_emb, d_t_emb_raw);
}

/* ---- Diffusion forward process: q(x_t | x_0) ---- */
static void diffuse(const float *x0, int t, float *x_t, float *noise) {
    for (int i = 0; i < IMG_SZ; i++) {
        noise[i] = randn();
        x_t[i] = sqrt_ab[t] * x0[i] + sqrt_1mab[t] * noise[i];
    }
}

/* ---- Sampling (DDPM) ---- */
static void sample(Denoiser *d, float *x) {
    /* start from pure noise */
    for (int i = 0; i < IMG_SZ; i++) x[i] = randn();

    /* step spacing for accelerated sampling */
    int step_size = T_STEPS / T_SAMPLE;

    for (int s = T_SAMPLE - 1; s >= 0; s--) {
        int t = s * step_size;
        denoiser_fwd(d, x, t);

        if (t > 0) {
            float ab_t = alpha_bar[t];
            float ab_prev = (t - step_size >= 0) ? alpha_bar[t - step_size] : 1.0f;
            float beta_t = 1.0f - ab_t / ab_prev;
            float coef1 = 1.0f / sqrtf(1.0f - beta_t);
            float coef2 = beta_t / sqrt_1mab[t];
            float sigma = sqrtf(beta_t);
            for (int i = 0; i < IMG_SZ; i++) {
                float z = randn();
                x[i] = coef1 * (x[i] - coef2 * eps_pred[i]) + sigma * z;
            }
        } else {
            /* t=0: no noise */
            float coef1 = 1.0f / sqrtf(alpha[0]);
            float coef2 = beta[0] / sqrt_1mab[0];
            for (int i = 0; i < IMG_SZ; i++)
                x[i] = coef1 * (x[i] - coef2 * eps_pred[i]);
        }
    }

    /* clamp to [0,1] */
    for (int i = 0; i < IMG_SZ; i++) {
        if (x[i] < 0) x[i] = 0;
        if (x[i] > 1) x[i] = 1;
    }
}

/* ---- Render density map as ASCII ---- */
static void render_ascii(const float *img, FILE *out) {
    /* Double width for aspect ratio (chars are ~2:1 height:width) */
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            int idx = (int)(img[y * IMG_W + x] * (N_RAMP - 1) + 0.5f);
            if (idx < 0) idx = 0;
            if (idx >= N_RAMP) idx = N_RAMP - 1;
            char c = ASCII_RAMP[idx];
            fputc(c, out);
            fputc(c, out);  /* double for aspect ratio */
        }
        fputc('\n', out);
    }
}

/* ---- Render as PPM (each char = 6×8 pixel block) ---- */
#define CHAR_W 6
#define CHAR_H 8
#define PPM_W  (IMG_W * 2 * CHAR_W)  /* doubled width × char width */
#define PPM_H  (IMG_H * CHAR_H)

/* Minimal 6×8 font bitmaps for the density ramp chars */
/* Each char = 8 rows of 6-bit patterns (MSB left) */
static const uint8_t FONT_SPACE[8] = {0,0,0,0,0,0,0,0};
static const uint8_t FONT_DOT[8]   = {0,0,0,0,0,0,0x0C,0};
static const uint8_t FONT_COLON[8] = {0,0,0x0C,0,0,0x0C,0,0};
static const uint8_t FONT_DASH[8]  = {0,0,0,0x3E,0,0,0,0};
static const uint8_t FONT_EQ[8]    = {0,0,0x3E,0,0x3E,0,0,0};
static const uint8_t FONT_PLUS[8]  = {0,0x08,0x08,0x3E,0x08,0x08,0,0};
static const uint8_t FONT_STAR[8]  = {0,0x14,0x08,0x3E,0x08,0x14,0,0};
static const uint8_t FONT_HASH[8]  = {0x14,0x14,0x3E,0x14,0x3E,0x14,0x14,0};
static const uint8_t FONT_PCT[8]   = {0x22,0x04,0x08,0x10,0x22,0,0,0};
static const uint8_t FONT_AT[8]    = {0x1C,0x22,0x2A,0x2E,0x20,0x1C,0,0};

static const uint8_t *FONTS[10] = {
    FONT_SPACE, FONT_DOT, FONT_COLON, FONT_DASH, FONT_EQ,
    FONT_PLUS, FONT_STAR, FONT_HASH, FONT_PCT, FONT_AT
};

static void render_ppm(const float *img, const char *path) {
    uint8_t *pixels = calloc(PPM_W * PPM_H * 3, 1);

    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            int idx = (int)(img[y * IMG_W + x] * (N_RAMP - 1) + 0.5f);
            if (idx < 0) idx = 0; if (idx >= N_RAMP) idx = N_RAMP - 1;
            const uint8_t *glyph = FONTS[idx];
            /* brightness: denser char = brighter pixel (green on black) */
            uint8_t bright = (uint8_t)(idx * 255 / (N_RAMP - 1));

            /* render glyph twice (doubled width) at positions x*2 and x*2+1 */
            for (int cx = 0; cx < 2; cx++) {
                int bx = (x * 2 + cx) * CHAR_W;
                int by = y * CHAR_H;
                for (int gy = 0; gy < CHAR_H; gy++) {
                    for (int gx = 0; gx < CHAR_W; gx++) {
                        int px = bx + gx, py = by + gy;
                        int pi = (py * PPM_W + px) * 3;
                        if (glyph[gy] & (0x20 >> gx)) {
                            /* glyph pixel — green text on black */
                            pixels[pi + 0] = 0;
                            pixels[pi + 1] = bright;
                            pixels[pi + 2] = 0;
                        }
                        /* else black (already zero) */
                    }
                }
            }
        }
    }

    FILE *f = fopen(path, "wb");
    if (f) {
        fprintf(f, "P6\n%d %d\n255\n", PPM_W, PPM_H);
        fwrite(pixels, 1, PPM_W * PPM_H * 3, f);
        fclose(f);
    }
    free(pixels);
}

/* ---- Save/Load ---- */
#define MOE_MAGIC 0x4D4F4542  /* "MOEB" */

static void model_save(Denoiser *d, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    uint32_t magic = MOE_MAGIC;
    fwrite(&magic, 4, 1, f);
    /* save all layer weights */
    Linear *layers[] = {&d->time_proj, &d->l1, &d->l2, &d->l3};
    for (int i = 0; i < 4; i++) {
        fwrite(layers[i]->w, sizeof(float), layers[i]->rows * layers[i]->cols, f);
        fwrite(layers[i]->b, sizeof(float), layers[i]->rows, f);
    }
    fclose(f);
    printf("  saved %s\n", path);
}

static int model_load(Denoiser *d, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic; fread(&magic, 4, 1, f);
    if (magic != MOE_MAGIC) { fclose(f); return -1; }
    Linear *layers[] = {&d->time_proj, &d->l1, &d->l2, &d->l3};
    for (int i = 0; i < 4; i++) {
        fread(layers[i]->w, sizeof(float), layers[i]->rows * layers[i]->cols, f);
        fread(layers[i]->b, sizeof(float), layers[i]->rows, f);
    }
    fclose(f);
    printf("  loaded %s\n", path);
    return 0;
}

/* ---- Main ---- */
int main(int argc, char **argv) {
    setbuf(stdout, NULL);

    const char *data_dir = "cifar-100-binary";
    const char *save_path = "moebius.bin";
    const char *resume_path = NULL;
    int gen_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_dir = argv[++i];
        else if (strcmp(argv[i], "--save") == 0 && i+1 < argc) save_path = argv[++i];
        else if (strcmp(argv[i], "--resume") == 0 && i+1 < argc) resume_path = argv[++i];
        else if (strcmp(argv[i], "--gen") == 0 && i+1 < argc) gen_count = atoi(argv[++i]);
    }

    printf("moebius.c — ASCII art diffusion (Jean Giraud drew with ink)\n");
    printf("  density map: %d×%d = %d values | %d diffusion steps | %d sample steps\n",
           IMG_W, IMG_H, IMG_SZ, T_STEPS, T_SAMPLE);
    printf("  denoiser: [%d+%d] → %d → %d → %d\n", IMG_SZ, TIME_DIM, H1_DIM, H2_DIM, IMG_SZ);

    /* count params */
    int tp = TIME_DIM*TIME_DIM + TIME_DIM
           + H1_DIM*IN_DIM + H1_DIM
           + H2_DIM*H1_DIM + H2_DIM
           + IMG_SZ*H2_DIM + IMG_SZ;
    printf("  params: %d (%.2fM)\n", tp, tp / 1000000.0f);
    printf("  ASCII ramp: \"%s\" (%d levels)\n\n", ASCII_RAMP, N_RAMP);

    init_schedule();
    Denoiser model = denoiser_new();

    if (resume_path) {
        if (model_load(&model, resume_path) < 0)
            fprintf(stderr, "  warning: could not load %s\n", resume_path);
    }

    /* ---- Generation mode ---- */
    if (gen_count > 0) {
        printf("=== GENERATING %d ASCII art images ===\n\n", gen_count);
        float x[IMG_SZ];
        for (int i = 0; i < gen_count; i++) {
            printf("--- image %d ---\n", i);
            sample(&model, x);
            render_ascii(x, stdout);
            /* save as text */
            char path[64];
            snprintf(path, sizeof(path), "moebius_%02d.txt", i);
            FILE *f = fopen(path, "w");
            if (f) { render_ascii(x, f); fclose(f); }
            /* save as PPM */
            snprintf(path, sizeof(path), "moebius_%02d.ppm", i);
            render_ppm(x, path);
            printf("  → %s\n\n", path);
        }
        return 0;
    }

    /* ---- Training mode ---- */
    char train_path[512];
    snprintf(train_path, sizeof(train_path), "%s/train.bin", data_dir);
    printf("  loading data from %s...\n", train_path);
    Data data = load_cifar100_gray(train_path);
    if (data.n == 0) {
        fprintf(stderr, "error: cannot load data from %s\n", train_path);
        return 1;
    }
    printf("  loaded %d images (converted to grayscale)\n\n", data.n);

    printf("=== TRAINING DDPM (%d steps) ===\n\n", STEPS);
    float running_loss = 0; int rn = 0;
    clock_t t0 = clock();

    float x_t[IMG_SZ], noise[IMG_SZ];

    for (int step = 0; step < STEPS; step++) {
        /* random image, random timestep */
        int idx = (int)(rnext() % (uint64_t)data.n);
        int t = (int)(rnext() % T_STEPS);
        float *x0 = &data.imgs[idx * IMG_SZ];

        /* forward diffusion: add noise */
        diffuse(x0, t, x_t, noise);

        /* predict noise */
        linear_zero_grad(&model.time_proj);
        linear_zero_grad(&model.l1);
        linear_zero_grad(&model.l2);
        linear_zero_grad(&model.l3);

        denoiser_fwd(&model, x_t, t);

        /* MSE loss: ||ε - ε_θ||² */
        float loss = 0;
        for (int i = 0; i < IMG_SZ; i++) {
            float diff = eps_pred[i] - noise[i];
            loss += diff * diff;
            d_eps[i] = 2.0f * diff / IMG_SZ;
        }
        loss /= IMG_SZ;

        /* backward */
        denoiser_bwd(&model, x_t, t);

        /* gradient clipping */
        float gnorm = 0;
        Linear *layers[] = {&model.time_proj, &model.l1, &model.l2, &model.l3};
        for (int l = 0; l < 4; l++) {
            int n = layers[l]->rows * layers[l]->cols;
            for (int i = 0; i < n; i++) gnorm += layers[l]->dw[i] * layers[l]->dw[i];
            for (int i = 0; i < layers[l]->rows; i++) gnorm += layers[l]->db[i] * layers[l]->db[i];
        }
        gnorm = sqrtf(gnorm);
        if (gnorm > 1.0f) {
            float scale = 1.0f / gnorm;
            for (int l = 0; l < 4; l++) {
                int n = layers[l]->rows * layers[l]->cols;
                for (int i = 0; i < n; i++) layers[l]->dw[i] *= scale;
                for (int i = 0; i < layers[l]->rows; i++) layers[l]->db[i] *= scale;
            }
        }

        /* Adam step */
        int adam_t = step + 1;
        for (int l = 0; l < 4; l++) linear_adam(layers[l], LR, adam_t);

        running_loss += loss; rn++;

        if (adam_t % LOG_EVERY == 0) {
            float elapsed = (float)(clock() - t0) / CLOCKS_PER_SEC;
            printf("  step %6d/%d | mse %.4f (avg %.4f) | gnorm %.2f | %.1f steps/sec\n",
                   adam_t, STEPS, loss, running_loss / rn, gnorm, adam_t / elapsed);
            running_loss = 0; rn = 0;
        }
        if (adam_t % SAVE_EVERY == 0) model_save(&model, save_path);
    }

    model_save(&model, save_path);

    /* Generate a few samples after training */
    printf("\n  generating samples after training...\n\n");
    float x[IMG_SZ];
    for (int i = 0; i < 3; i++) {
        printf("--- sample %d ---\n", i);
        sample(&model, x);
        render_ascii(x, stdout);
        char path[64];
        snprintf(path, sizeof(path), "moebius_sample_%d.ppm", i);
        render_ppm(x, path);
        printf("  → %s\n\n", path);
    }

    float elapsed = (float)(clock() - t0) / CLOCKS_PER_SEC;
    printf("done. %.1fs total. Moebius draws with text.\n", elapsed);

    free(data.imgs);
    return 0;
}
