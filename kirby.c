/*
 * kirby.c — VQ-VAE visual vocabulary
 *
 * Named after Jack Kirby, the King of Comics.
 * Kirby drew the visual language of Marvel and DC — every hero,
 * every pose, every explosion. He created the vocabulary.
 * This module does the same: 512 visual words that can
 * reconstruct any image.
 *
 * Chuck optimizes. Lee sees. Kirby draws.
 *
 * Patch-based Vector Quantized Variational Autoencoder.
 * Encodes 32×32 RGB images into 16 discrete codes (4×4 grid).
 * Decodes 16 codes back into 32×32 RGB images.
 *
 * The codebook is the shared visual vocabulary:
 *   - Lee (eyes):    image → patches → transformer → class
 *   - Kirby (hands): codes → codebook → decoder → patches → image
 *   - Leo (brain):   generates code sequences via RetNet retention
 *
 * Architecture (per patch, no convolutions):
 *   Encoder: patch(192) → linear → ReLU → linear → z(64)
 *   Quantize: z → nearest codebook entry → code index
 *   Decoder: codebook(64) → linear → ReLU → linear → patch(192)
 *
 * Training: MSE reconstruction + commitment loss
 * Codebook: EMA update (no gradient through codebook)
 *
 * Build:
 *   cc -std=c11 -O2 -o kirby kirby.c -lm
 *   ./kirby --data cifar-100-binary
 *
 * Output: kirby.bin (codebook + decoder + encoder weights)
 *   Feed 16 code indices → decoder → 32×32 RGB image (PPM)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ---- Config ---- */
#define IMG_SIZE     32
#define IMG_CH       3
#define PATCH_SIZE   8
#define PATCH_PX     (PATCH_SIZE * PATCH_SIZE * IMG_CH)  /* 192 */
#define PATCHES_SIDE 4
#define N_PATCHES    16  /* 4×4 grid */

#define N_CODES      1024   /* codebook size — visual vocabulary */
#define Z_DIM        128    /* latent dimension per patch */
#define H1_DIM       672    /* hidden dim layer 1 (encoder/decoder) */
#define H2_DIM       384    /* hidden dim layer 2 (encoder/decoder) */

#define LR           0.001f
#define BETA_COMMIT  0.25f  /* commitment loss weight */
#define EMA_DECAY    0.99f  /* codebook EMA decay */
#define STEPS        50000
#define LOG_EVERY    500
#define SAVE_EVERY   10000

/* ---- RNG ---- */
static uint64_t rng_state = 42;
static uint64_t rnext(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static float randf(void) { return (rnext() & 0xFFFFFF) / (float)0xFFFFFF; }
static float randn(void) {  /* box-muller */
    float u1 = randf() + 1e-10f, u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ---- Data ---- */
typedef struct { float *imgs; int n; } Data;

static Data load_cifar100(const char *path) {
    Data d = {0, 0};
    FILE *f = fopen(path, "rb");
    if (!f) return d;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int rec_sz = 2 + IMG_SIZE * IMG_SIZE * IMG_CH; /* coarse + fine + pixels */
    d.n = (int)(sz / rec_sz);
    d.imgs = malloc(d.n * IMG_SIZE * IMG_SIZE * IMG_CH * sizeof(float));
    uint8_t *buf = malloc(rec_sz);
    for (int i = 0; i < d.n; i++) {
        if (fread(buf, 1, rec_sz, f) != (size_t)rec_sz) { d.n = i; break; }
        /* skip labels (buf[0]=coarse, buf[1]=fine), normalize pixels */
        for (int j = 0; j < IMG_SIZE * IMG_SIZE * IMG_CH; j++)
            d.imgs[i * IMG_SIZE * IMG_SIZE * IMG_CH + j] = buf[2 + j] / 255.0f;
    }
    free(buf); fclose(f);
    return d;
}

/* ---- Weight matrix ---- */
typedef struct {
    float *w;    /* [rows × cols] */
    float *b;    /* [rows] bias */
    float *dw;   /* gradient */
    float *db;
    float *mw, *vw, *mb, *vb;  /* Adam state */
    int rows, cols;
} Linear;

static Linear linear_new(int rows, int cols) {
    Linear l;
    l.rows = rows; l.cols = cols;
    float scale = sqrtf(2.0f / cols);  /* He init */
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

/* forward: out[rows] = w[rows,cols] × in[cols] + b[rows] */
static void linear_fwd(Linear *l, const float *in, float *out) {
    for (int r = 0; r < l->rows; r++) {
        float s = l->b[r];
        for (int c = 0; c < l->cols; c++) s += l->w[r * l->cols + c] * in[c];
        out[r] = s;
    }
}

/* backward: given d_out[rows], compute d_in[cols] and accumulate dw, db */
static void linear_bwd(Linear *l, const float *in, const float *d_out, float *d_in) {
    for (int r = 0; r < l->rows; r++) {
        l->db[r] += d_out[r];
        for (int c = 0; c < l->cols; c++) {
            l->dw[r * l->cols + c] += d_out[r] * in[c];
            if (d_in) d_in[c] += l->w[r * l->cols + c] * d_out[r];
        }
    }
}

/* Adam step */
static void linear_adam(Linear *l, float lr, int t) {
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    int n = l->rows * l->cols;
    for (int i = 0; i < n; i++) {
        l->mw[i] = b1 * l->mw[i] + (1-b1) * l->dw[i];
        l->vw[i] = b2 * l->vw[i] + (1-b2) * l->dw[i] * l->dw[i];
        l->w[i] -= lr * (l->mw[i]/bc1) / (sqrtf(l->vw[i]/bc2) + eps);
    }
    for (int i = 0; i < l->rows; i++) {
        l->mb[i] = b1 * l->mb[i] + (1-b1) * l->db[i];
        l->vb[i] = b2 * l->vb[i] + (1-b2) * l->db[i] * l->db[i];
        l->b[i] -= lr * (l->mb[i]/bc1) / (sqrtf(l->vb[i]/bc2) + eps);
    }
}

/* ---- ReLU ---- */
static void relu_fwd(float *x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}
static void relu_bwd(const float *act, float *grad, int n) {
    for (int i = 0; i < n; i++) if (act[i] <= 0) grad[i] = 0;
}

/* ---- Codebook ---- */
typedef struct {
    float *embed;       /* [N_CODES × Z_DIM] — the visual vocabulary */
    float *ema_count;   /* [N_CODES] — EMA usage count */
    float *ema_sum;     /* [N_CODES × Z_DIM] — EMA embedding sum */
} Codebook;

static Codebook codebook_new(void) {
    Codebook cb;
    cb.embed     = malloc(N_CODES * Z_DIM * sizeof(float));
    cb.ema_count = malloc(N_CODES * sizeof(float));
    cb.ema_sum   = malloc(N_CODES * Z_DIM * sizeof(float));
    /* init: uniform random in [-1/N_CODES, 1/N_CODES] */
    float scale = 1.0f / N_CODES;
    for (int i = 0; i < N_CODES * Z_DIM; i++)
        cb.embed[i] = (randf() * 2 - 1) * scale;
    for (int i = 0; i < N_CODES; i++) cb.ema_count[i] = 1.0f;
    memcpy(cb.ema_sum, cb.embed, N_CODES * Z_DIM * sizeof(float));
    return cb;
}

/* find nearest code, return index */
static int codebook_quantize(Codebook *cb, const float *z) {
    int best = 0; float best_d = 1e30f;
    for (int i = 0; i < N_CODES; i++) {
        float d = 0;
        for (int j = 0; j < Z_DIM; j++) {
            float diff = z[j] - cb->embed[i * Z_DIM + j];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

/* EMA codebook update (no gradient) */
static void codebook_ema_update(Codebook *cb, const float *z, int idx) {
    float gamma = EMA_DECAY;
    cb->ema_count[idx] = gamma * cb->ema_count[idx] + (1 - gamma);
    for (int j = 0; j < Z_DIM; j++)
        cb->ema_sum[idx * Z_DIM + j] = gamma * cb->ema_sum[idx * Z_DIM + j] + (1 - gamma) * z[j];
    /* update embedding */
    float n = cb->ema_count[idx];
    for (int j = 0; j < Z_DIM; j++)
        cb->embed[idx * Z_DIM + j] = cb->ema_sum[idx * Z_DIM + j] / n;
}

/* ---- VQ-VAE Model (3-layer encoder/decoder, ~1M params) ---- */
typedef struct {
    /* Encoder: patch(192) → 672 → 384 → z(128) */
    Linear enc1;  /* (H1_DIM, PATCH_PX) = (672, 192) */
    Linear enc2;  /* (H2_DIM, H1_DIM)   = (384, 672) */
    Linear enc3;  /* (Z_DIM, H2_DIM)    = (128, 384) */
    /* Decoder: z(128) → 384 → 672 → patch(192) */
    Linear dec1;  /* (H2_DIM, Z_DIM)    = (384, 128) */
    Linear dec2;  /* (H1_DIM, H2_DIM)   = (672, 384) */
    Linear dec3;  /* (PATCH_PX, H1_DIM) = (192, 672) */
    /* Codebook */
    Codebook cb;
} VQVAE;

static VQVAE vqvae_new(void) {
    VQVAE m;
    m.enc1 = linear_new(H1_DIM, PATCH_PX);
    m.enc2 = linear_new(H2_DIM, H1_DIM);
    m.enc3 = linear_new(Z_DIM, H2_DIM);
    m.dec1 = linear_new(H2_DIM, Z_DIM);
    m.dec2 = linear_new(H1_DIM, H2_DIM);
    m.dec3 = linear_new(PATCH_PX, H1_DIM);
    m.cb   = codebook_new();
    return m;
}

/* Buffers for forward/backward (per patch) */
static float enc_h1[H1_DIM], enc_h2[H2_DIM], enc_z[Z_DIM];
static float dec_h1[H2_DIM], dec_h2[H1_DIM], dec_out[PATCH_PX];
static float d_dec_out[PATCH_PX], d_dec_h2[H1_DIM], d_dec_h1[H2_DIM], d_z[Z_DIM];
static float d_enc_z[Z_DIM], d_enc_h2[H2_DIM], d_enc_h1[H1_DIM];

/*
 * Train one patch through VQ-VAE.
 * Returns: reconstruction MSE for this patch.
 * Codebook updated via EMA. Encoder/decoder grads accumulated.
 */
static float vqvae_train_patch(VQVAE *m, const float *patch) {
    /* ---- Encoder: 3 layers ---- */
    linear_fwd(&m->enc1, patch, enc_h1);
    relu_fwd(enc_h1, H1_DIM);
    linear_fwd(&m->enc2, enc_h1, enc_h2);
    relu_fwd(enc_h2, H2_DIM);
    linear_fwd(&m->enc3, enc_h2, enc_z);  /* z_e = encoder output */

    /* ---- Quantize ---- */
    int code = codebook_quantize(&m->cb, enc_z);
    float *z_q = &m->cb.embed[code * Z_DIM];

    /* ---- Decoder: 3 layers (uses z_q, straight-through to z_e) ---- */
    linear_fwd(&m->dec1, z_q, dec_h1);
    relu_fwd(dec_h1, H2_DIM);
    linear_fwd(&m->dec2, dec_h1, dec_h2);
    relu_fwd(dec_h2, H1_DIM);
    linear_fwd(&m->dec3, dec_h2, dec_out);

    /* ---- Reconstruction loss: MSE ---- */
    float mse = 0;
    for (int i = 0; i < PATCH_PX; i++) {
        float diff = dec_out[i] - patch[i];
        mse += diff * diff;
        d_dec_out[i] = 2.0f * diff / PATCH_PX;
    }
    mse /= PATCH_PX;

    /* ---- Backward: decoder ---- */
    memset(d_dec_h2, 0, sizeof(d_dec_h2));
    linear_bwd(&m->dec3, dec_h2, d_dec_out, d_dec_h2);
    relu_bwd(dec_h2, d_dec_h2, H1_DIM);
    memset(d_dec_h1, 0, sizeof(d_dec_h1));
    linear_bwd(&m->dec2, dec_h1, d_dec_h2, d_dec_h1);
    relu_bwd(dec_h1, d_dec_h1, H2_DIM);
    memset(d_z, 0, sizeof(d_z));
    linear_bwd(&m->dec1, z_q, d_dec_h1, d_z);

    /* ---- Straight-through + commitment loss ---- */
    for (int j = 0; j < Z_DIM; j++)
        d_enc_z[j] = d_z[j] + 2.0f * BETA_COMMIT * (enc_z[j] - z_q[j]);

    /* ---- Backward: encoder ---- */
    memset(d_enc_h2, 0, sizeof(d_enc_h2));
    linear_bwd(&m->enc3, enc_h2, d_enc_z, d_enc_h2);
    relu_bwd(enc_h2, d_enc_h2, H2_DIM);
    memset(d_enc_h1, 0, sizeof(d_enc_h1));
    linear_bwd(&m->enc2, enc_h1, d_enc_h2, d_enc_h1);
    relu_bwd(enc_h1, d_enc_h1, H1_DIM);
    linear_bwd(&m->enc1, patch, d_enc_h1, NULL);

    /* ---- EMA codebook update ---- */
    codebook_ema_update(&m->cb, enc_z, code);

    return mse;
}

/* Decode a single code index to a patch */
static void vqvae_decode_patch(VQVAE *m, int code, float *out) {
    float *z_q = &m->cb.embed[code * Z_DIM];
    float h1[H2_DIM], h2[H1_DIM];
    linear_fwd(&m->dec1, z_q, h1);
    relu_fwd(h1, H2_DIM);
    linear_fwd(&m->dec2, h1, h2);
    relu_fwd(h2, H1_DIM);
    linear_fwd(&m->dec3, h2, out);
    for (int i = 0; i < PATCH_PX; i++) {
        if (out[i] < 0) out[i] = 0;
        if (out[i] > 1) out[i] = 1;
    }
}

/* Encode an image → 16 code indices */
static void vqvae_encode_image(VQVAE *m, const float *img, int *codes) {
    for (int py = 0; py < PATCHES_SIDE; py++) {
        for (int px = 0; px < PATCHES_SIDE; px++) {
            float patch[PATCH_PX];
            for (int c = 0; c < IMG_CH; c++)
                for (int y = 0; y < PATCH_SIZE; y++)
                    for (int x = 0; x < PATCH_SIZE; x++)
                        patch[c * PATCH_SIZE * PATCH_SIZE + y * PATCH_SIZE + x] =
                            img[c * IMG_SIZE * IMG_SIZE + (py*PATCH_SIZE+y) * IMG_SIZE + px*PATCH_SIZE + x];
            float h1[H1_DIM], h2[H2_DIM], z[Z_DIM];
            linear_fwd(&m->enc1, patch, h1);
            relu_fwd(h1, H1_DIM);
            linear_fwd(&m->enc2, h1, h2);
            relu_fwd(h2, H2_DIM);
            linear_fwd(&m->enc3, h2, z);
            codes[py * PATCHES_SIDE + px] = codebook_quantize(&m->cb, z);
        }
    }
}

/* Decode 16 codes → image, write PPM */
static void vqvae_decode_image(VQVAE *m, const int *codes, const char *path) {
    float img[IMG_CH * IMG_SIZE * IMG_SIZE];
    memset(img, 0, sizeof(img));
    for (int py = 0; py < PATCHES_SIDE; py++) {
        for (int px = 0; px < PATCHES_SIDE; px++) {
            float patch[PATCH_PX];
            vqvae_decode_patch(m, codes[py * PATCHES_SIDE + px], patch);
            for (int c = 0; c < IMG_CH; c++)
                for (int y = 0; y < PATCH_SIZE; y++)
                    for (int x = 0; x < PATCH_SIZE; x++)
                        img[c * IMG_SIZE * IMG_SIZE + (py*PATCH_SIZE+y) * IMG_SIZE + px*PATCH_SIZE + x] =
                            patch[c * PATCH_SIZE * PATCH_SIZE + y * PATCH_SIZE + x];
        }
    }
    /* write PPM */
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", IMG_SIZE, IMG_SIZE);
    for (int y = 0; y < IMG_SIZE; y++)
        for (int x = 0; x < IMG_SIZE; x++) {
            uint8_t rgb[3];
            for (int c = 0; c < 3; c++) {
                float v = img[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x];
                rgb[c] = (uint8_t)(v * 255.0f + 0.5f);
            }
            fwrite(rgb, 1, 3, f);
        }
    fclose(f);
}

/* ---- Save/Load ---- */
#define VQ_MAGIC 0x4C454556  /* "LEEV" */

static void save_linear(FILE *f, Linear *l) {
    fwrite(l->w, sizeof(float), l->rows * l->cols, f);
    fwrite(l->b, sizeof(float), l->rows, f);
}
static void load_linear(FILE *f, Linear *l) {
    fread(l->w, sizeof(float), l->rows * l->cols, f);
    fread(l->b, sizeof(float), l->rows, f);
}

static void vqvae_save(VQVAE *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    uint32_t magic = VQ_MAGIC;
    fwrite(&magic, 4, 1, f);
    fwrite(m->cb.embed, sizeof(float), N_CODES * Z_DIM, f);
    save_linear(f, &m->dec1); save_linear(f, &m->dec2); save_linear(f, &m->dec3);
    save_linear(f, &m->enc1); save_linear(f, &m->enc2); save_linear(f, &m->enc3);
    fclose(f);
    /* compute size */
    long bytes = 4;
    bytes += N_CODES * Z_DIM * 4;
    Linear *layers[] = {&m->dec1, &m->dec2, &m->dec3, &m->enc1, &m->enc2, &m->enc3};
    for (int i = 0; i < 6; i++) bytes += (layers[i]->rows * layers[i]->cols + layers[i]->rows) * 4;
    printf("  saved %s (%.1fMB)\n", path, bytes / (1024.0f * 1024.0f));
}

static int vqvae_load(VQVAE *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic; fread(&magic, 4, 1, f);
    if (magic != VQ_MAGIC) { fclose(f); return -1; }
    fread(m->cb.embed, sizeof(float), N_CODES * Z_DIM, f);
    load_linear(f, &m->dec1); load_linear(f, &m->dec2); load_linear(f, &m->dec3);
    load_linear(f, &m->enc1); load_linear(f, &m->enc2); load_linear(f, &m->enc3);
    fclose(f);
    printf("  loaded %s\n", path);
    return 0;
}

/* ---- Codebook usage stats ---- */
static void codebook_stats(VQVAE *m, Data *data) {
    int usage[N_CODES]; memset(usage, 0, sizeof(usage));
    int n = data->n < 5000 ? data->n : 5000;
    int total_codes = 0;
    for (int i = 0; i < n; i++) {
        int codes[N_PATCHES];
        float *img = &data->imgs[i * IMG_SIZE * IMG_SIZE * IMG_CH];
        vqvae_encode_image(m, img, codes);
        for (int p = 0; p < N_PATCHES; p++) { usage[codes[p]]++; total_codes++; }
    }
    int used = 0;
    for (int i = 0; i < N_CODES; i++) if (usage[i] > 0) used++;
    printf("  codebook: %d/%d codes used (%.0f%%) over %d images\n",
           used, N_CODES, 100.0f * used / N_CODES, n);
    /* find most/least used */
    int max_use = 0, max_idx = 0;
    for (int i = 0; i < N_CODES; i++)
        if (usage[i] > max_use) { max_use = usage[i]; max_idx = i; }
    printf("  most used: code %d (%d times, %.1f%%)\n",
           max_idx, max_use, 100.0f * max_use / total_codes);
}

/* ---- Main ---- */
int main(int argc, char **argv) {
    setbuf(stdout, NULL);

    const char *data_dir = "cifar-100-binary";
    const char *save_path = "kirby.bin";
    const char *resume_path = NULL;
    int gen_mode = 0;  /* --gen: encode→decode sample images, write PPMs */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_dir = argv[++i];
        else if (strcmp(argv[i], "--save") == 0 && i+1 < argc) save_path = argv[++i];
        else if (strcmp(argv[i], "--resume") == 0 && i+1 < argc) resume_path = argv[++i];
        else if (strcmp(argv[i], "--gen") == 0) gen_mode = 1;
    }

    printf("kirby.c — VQ-VAE visual vocabulary (Jack Kirby drew the language)\n");
    printf("  codebook: %d codes × %d dim | hidden: %d → %d\n", N_CODES, Z_DIM, H1_DIM, H2_DIM);
    printf("  patches: %d×%d×%d = %d per patch, %d patches per image\n",
           PATCH_SIZE, PATCH_SIZE, IMG_CH, PATCH_PX, N_PATCHES);

    /* count params */
    int enc_params = H1_DIM*PATCH_PX + H1_DIM + H2_DIM*H1_DIM + H2_DIM + Z_DIM*H2_DIM + Z_DIM;
    int dec_params = H2_DIM*Z_DIM + H2_DIM + H1_DIM*H2_DIM + H1_DIM + PATCH_PX*H1_DIM + PATCH_PX;
    int cb_params = N_CODES * Z_DIM;
    int total = enc_params + dec_params + cb_params;
    printf("  params: %d encoder + %d decoder + %d codebook = %d (%.1fM)\n",
           enc_params, dec_params, cb_params, total, total / 1000000.0f);

    VQVAE model = vqvae_new();

    if (resume_path) {
        if (vqvae_load(&model, resume_path) < 0)
            fprintf(stderr, "  warning: could not load %s, starting fresh\n", resume_path);
    }

    /* Load data */
    char train_path[512];
    snprintf(train_path, sizeof(train_path), "%s/train.bin", data_dir);
    printf("  loading data from %s...\n", train_path);
    Data data = load_cifar100(train_path);
    if (data.n == 0) {
        fprintf(stderr, "error: cannot load data from %s\n", train_path);
        return 1;
    }
    printf("  loaded %d images\n\n", data.n);

    /* --gen mode: just encode→decode and dump PPMs */
    if (gen_mode) {
        printf("=== GENERATE: encode → decode → PPM ===\n");
        for (int i = 0; i < 10; i++) {
            int codes[N_PATCHES];
            float *img = &data.imgs[i * IMG_SIZE * IMG_SIZE * IMG_CH];
            vqvae_encode_image(&model, img, codes);
            printf("  image %d codes:", i);
            for (int p = 0; p < N_PATCHES; p++) printf(" %d", codes[p]);
            printf("\n");
            char path[64];
            snprintf(path, sizeof(path), "vq_recon_%02d.ppm", i);
            vqvae_decode_image(&model, codes, path);
            /* also save original */
            snprintf(path, sizeof(path), "vq_orig_%02d.ppm", i);
            int orig_codes[N_PATCHES]; /* dummy — just write original directly */
            FILE *f = fopen(path, "wb");
            fprintf(f, "P6\n%d %d\n255\n", IMG_SIZE, IMG_SIZE);
            for (int y = 0; y < IMG_SIZE; y++)
                for (int x = 0; x < IMG_SIZE; x++) {
                    uint8_t rgb[3];
                    for (int c = 0; c < 3; c++) {
                        float v = img[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x];
                        rgb[c] = (uint8_t)(v * 255 + 0.5f);
                    }
                    fwrite(rgb, 1, 3, f);
                }
            fclose(f);
            (void)orig_codes;
        }
        codebook_stats(&model, &data);
        printf("\ndone. Open vq_orig_XX.ppm and vq_recon_XX.ppm to compare.\n");
        free(data.imgs);
        return 0;
    }

    /* ---- Training ---- */
    printf("=== TRAINING VQ-VAE (%d steps) ===\n\n", STEPS);
    float running_loss = 0; int rn = 0;
    clock_t t0 = clock();

    for (int step = 0; step < STEPS; step++) {
        int idx = (int)(rnext() % (uint64_t)data.n);
        float *img = &data.imgs[idx * IMG_SIZE * IMG_SIZE * IMG_CH];
        float step_loss = 0;

        /* zero grads */
        linear_zero_grad(&model.enc1); linear_zero_grad(&model.enc2); linear_zero_grad(&model.enc3);
        linear_zero_grad(&model.dec1); linear_zero_grad(&model.dec2); linear_zero_grad(&model.dec3);

        /* process all 16 patches */
        for (int py = 0; py < PATCHES_SIDE; py++) {
            for (int px = 0; px < PATCHES_SIDE; px++) {
                float patch[PATCH_PX];
                for (int c = 0; c < IMG_CH; c++)
                    for (int y = 0; y < PATCH_SIZE; y++)
                        for (int x = 0; x < PATCH_SIZE; x++)
                            patch[c * PATCH_SIZE * PATCH_SIZE + y * PATCH_SIZE + x] =
                                img[c * IMG_SIZE * IMG_SIZE + (py*PATCH_SIZE+y) * IMG_SIZE + px*PATCH_SIZE + x];
                step_loss += vqvae_train_patch(&model, patch);
            }
        }
        step_loss /= N_PATCHES;

        /* Adam update */
        int t = step + 1;
        linear_adam(&model.enc1, LR, t); linear_adam(&model.enc2, LR, t); linear_adam(&model.enc3, LR, t);
        linear_adam(&model.dec1, LR, t); linear_adam(&model.dec2, LR, t); linear_adam(&model.dec3, LR, t);

        running_loss += step_loss; rn++;

        if (t % LOG_EVERY == 0) {
            float elapsed = (float)(clock() - t0) / CLOCKS_PER_SEC;
            printf("  step %5d/%d | mse %.4f (avg %.4f) | %.1f steps/sec\n",
                   t, STEPS, step_loss, running_loss / rn, t / elapsed);
            running_loss = 0; rn = 0;
        }
        if (t % SAVE_EVERY == 0) {
            vqvae_save(&model, save_path);
        }
    }

    vqvae_save(&model, save_path);
    codebook_stats(&model, &data);

    /* Generate a few sample reconstructions */
    printf("\n  generating sample reconstructions...\n");
    for (int i = 0; i < 5; i++) {
        int codes[N_PATCHES];
        float *img = &data.imgs[i * IMG_SIZE * IMG_SIZE * IMG_CH];
        vqvae_encode_image(&model, img, codes);
        char path[64];
        snprintf(path, sizeof(path), "vq_sample_%d.ppm", i);
        vqvae_decode_image(&model, codes, path);
        printf("  wrote %s (codes:", path);
        for (int p = 0; p < N_PATCHES; p++) printf(" %d", codes[p]);
        printf(")\n");
    }

    float elapsed = (float)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\ndone. %.1fs total. codebook + decoder = Lee's drawing hands.\n", elapsed);
    printf("Leo generates 16 code indices → Kirby decodes them → image.\n");

    free(data.imgs);
    return 0;
}
