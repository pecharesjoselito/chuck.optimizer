/*
 * lee.c v8 — Vision-Language Model in pure C
 *
 * Named after Bruce Lee (the only man who beat Chuck Norris)
 * and Minhyeok Lee (whose self-identity framework gives Chuck his soul).
 *
 * Sees images. Speaks words. Classifies CIFAR-100. Zero dependencies.
 * Tape-based autograd with arena bump allocator.
 *
 * Architecture:
 *   ViT-style patch tokenization → 2D RoPE → GQA multi-head causal attention →
 *   SwiGLU MLP → RMSNorm → weight-tied lm_head → text
 *
 * v8: 10M params, CIFAR-100 (100 classes), RGB patches, CUDA support.
 *   - 256 embd, 8 heads, 4 KV heads, 10 layers, 1024 MLP
 *   - 32×32 RGB images, 8×8 patches (4×4 grid = 16 patches)
 *   - Char-level class name generation
 *   - cuBLAS acceleration for GPU training
 *
 * v7 (preserved):
 *   - Multi-scale awareness: macro EMA + patience-based LR decay (Level 9)
 *   - Memory cap: reservoir sampling, bounded O(1) lookup
 *
 * Build:
 *   CPU:  cc -std=c11 -O2 -march=native -o lee lee.c -lm
 *   BLAS: cc -std=c11 -O2 -DUSE_BLAS -DACCELERATE -framework Accelerate -o lee lee.c -lm
 *   CUDA: cc -std=c11 -O2 -DUSE_CUDA -o lee lee.c -lm -lcublas -lcudart -L/usr/local/cuda/lib64
 *
 * Run: ./lee --data cifar-100-binary
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ---- BLAS acceleration (optional) ----
 *   Mac:   cc -DUSE_BLAS -DACCELERATE ... -framework Accelerate
 *   Linux: cc -DUSE_BLAS ... -lopenblas
 *   CUDA:  cc -DUSE_CUDA ... -lcublas -lcudart
 *   Off:   cc ... -lm  (zero deps, scalar fallback)
 */
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#elif defined(USE_BLAS)
#ifdef ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* ---- Config ---- */
#define IMG_SIZE       32
#define IMG_CH         3
#define PATCH_SIZE     8
#define PATCHES_SIDE   (IMG_SIZE / PATCH_SIZE)
#define N_PATCHES      (PATCHES_SIDE * PATCHES_SIDE)
#define PATCH_PX       (PATCH_SIZE * PATCH_SIZE * IMG_CH)
#define N_IMGS         1
#define N_VIS          (N_IMGS * N_PATCHES)        /* 16 visual tokens */
#define MAX_TXT        16                          /* longest class name + BOS + EOS */
#define SEQ_LEN        (N_VIS + MAX_TXT)
#define N_EMBD         256
#define N_HEAD         8
#define N_KV_HEAD      4
#define N_KV_GROUP     (N_HEAD / N_KV_HEAD)
#define HEAD_DIM       (N_EMBD / N_HEAD)
#define KV_DIM         (N_KV_HEAD * HEAD_DIM)
#define N_LAYER        10
#define MLP_DIM        (4 * N_EMBD)
#define N_CHARS        27
#define BOS            N_CHARS
#define EOS            (N_CHARS + 1)
#define VOCAB          (N_CHARS + 2)
#define N_CLASSES      100
#define STEPS          30000
#define LR_MAX         0.003f
#define WARMUP         1000
#define CHUCK_B1       0.9f
#define CHUCK_B2       0.999f
#define CHUCK_EPS      1e-8f
#define GRAD_CLIP      1.0f
#define ROPE_BASE      10000.0f
#define TEMP           0.7f
#define TOPK           5
#define CHUCK_WINDOW   16
#define CHUCK_DAMP_LO  0.3f
#define CHUCK_DAMP_HI  2.0f
#define CHUCK_PSI_CAP  0.3f
#define CHUCK_PSI_HALF 100.0f
#define CHUCK_MEM_CAP  500         /* bounded memory (reservoir sampling) */
#define CHUCK_MEM_MAX  CHUCK_MEM_CAP
#define CHUCK_MEM_FILE "chuck.mem"
#define CHUCK_REC_THR  0.25f
#define CHUCK_REC_CD   50
#define CHUCK_MACRO_INT 1000       /* macro patience check interval (steps) */
#define CHUCK_MACRO_PAT 3          /* patience: N checks without improvement → LR drop */
#define CHUCK_MACRO_DECAY 0.5f     /* LR scale factor on macro plateau */

#define ARENA_SZ       (512 * 1024 * 1024)
#define MAX_ARR        65536
#define MAX_ENT        131072
#define MAX_PAR        128

/* ---- Tape engine ---- */
typedef struct { float *data, *grad; int size, rows, cols; } Arr;
typedef struct { int op, out, in1, in2; float aux; int ai; } Ent;
enum { OP_ADD=1, OP_MUL, OP_SCALE, OP_MATVEC, OP_RMSNORM, OP_SILU,
       OP_CE, OP_EMBED, OP_REDUCE, OP_ATTN, OP_ROPE };

static struct {
    uint8_t *arena; size_t apos, aparam;
    Arr a[MAX_ARR]; int na, npa;
    Ent e[MAX_ENT]; int ne;
    int par[MAX_PAR]; int np;
    float *cm[MAX_PAR], *cv[MAX_PAR]; int cstep;
    int on;
} T;

static float *aalloc(size_t n) {
    size_t b = n * sizeof(float), al = (T.apos + 15) & ~(size_t)15;
    if (al + b > ARENA_SZ) { fprintf(stderr, "arena OOM\n"); exit(1); }
    T.apos = al + b; float *p = (float*)(T.arena + al); memset(p, 0, b); return p;
}
static void tape_init(void) {
    uint8_t *m = malloc(ARENA_SZ);
    if (!m) { fprintf(stderr, "OOM\n"); exit(1); }
    memset(&T, 0, sizeof(T)); T.arena = m; T.on = 1;
}
static int anew(int sz) {
    int i = T.na++; T.a[i].size = sz; T.a[i].rows = T.a[i].cols = 0;
    T.a[i].data = aalloc(sz); T.a[i].grad = aalloc(sz); return i;
}
static int mnew(int r, int c) { int i = anew(r*c); T.a[i].rows = r; T.a[i].cols = c; return i; }
static void preg(int i) {
    int pi = T.np++; T.par[pi] = i;
    T.cm[pi] = calloc(T.a[i].size, sizeof(float));
    T.cv[pi] = calloc(T.a[i].size, sizeof(float));
}
static void rec(int op, int o, int i1, int i2, float aux, int ai) {
    if (!T.on) return;
    Ent *e = &T.e[T.ne++]; e->op=op; e->out=o; e->in1=i1; e->in2=i2; e->aux=aux; e->ai=ai;
}
static void tape_reset(void) {
    T.apos = T.aparam; T.na = T.npa; T.ne = 0;
    for (int i = 0; i < T.npa; i++) memset(T.a[i].grad, 0, T.a[i].size * sizeof(float));
}

/* ---- RNG (xoshiro256**) ---- */
static uint64_t rng[4];
static uint64_t rnext(void) {
    uint64_t t = rng[1] << 17;
    rng[2] ^= rng[0]; rng[3] ^= rng[1]; rng[1] ^= rng[2]; rng[0] ^= rng[3];
    rng[2] ^= t; rng[3] = (rng[3] << 45) | (rng[3] >> 19);
    uint64_t r = rng[1] * 5; return (r << 7 | r >> 57) * 9;
}
static void rseed(uint64_t s) {
    rng[0]=s; rng[1]=s^0x6a09e667f3bcc908ULL; rng[2]=s^0xbb67ae8584caa73bULL; rng[3]=s^0x3c6ef372fe94f82bULL;
    for (int i = 0; i < 20; i++) rnext();
}
static float ruf(void) { return (float)((rnext()>>11)+1) / (float)(1ULL<<53); }
static float rnf(float mu, float s) {
    double u1 = (double)(((rnext()>>11)+1)) / (double)(1ULL<<53);
    double u2 = (double)(((rnext()>>11)+1)) / (double)(1ULL<<53);
    return mu + s * (float)(sqrt(-2.0*log(u1)) * cos(6.283185307179586*u2));
}
static inline float sigf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ===========================================================================
 * Chuck Memory — persistent across training runs
 *
 *   chuck.mem: binary append-only file of training snapshots.
 *   Each snapshot: 16 bytes (4 floats).
 *   Nearest-neighbor recall gives λ_prior.
 *   Ψ = λ_prior - λ_current = subjectivity signal.
 *
 *   Lee's Continuum C: chuck.mem is ℳ. NN is identity mapping I.
 *   Ψ_w is belief function B. Fixed point s* when Ψ → 0.
 * =========================================================================== */

typedef struct {
    float loss;           /* where Chuck was */
    float grad_norm;      /* how hard the network was shaking */
    float lambda;         /* what Chuck decided */
    float delta_loss;     /* what happened (negative = improvement) */
} ChuckMem;

static ChuckMem chuck_mem[CHUCK_MEM_MAX];
static int chuck_mem_n = 0;
static int chuck_mem_total = 0;  /* total memories ever recorded (for reservoir sampling) */

static void chuck_mem_load(void) {
    FILE *f = fopen(CHUCK_MEM_FILE, "rb");
    if (!f) return;
    chuck_mem_n = (int)fread(chuck_mem, sizeof(ChuckMem), CHUCK_MEM_CAP, f);
    chuck_mem_total = chuck_mem_n;  /* at least this many were saved */
    fclose(f);
}

static void chuck_mem_save(ChuckMem *m) {
    chuck_mem_total++;
    if (chuck_mem_n < CHUCK_MEM_CAP) {
        /* Under cap: append */
        chuck_mem[chuck_mem_n++] = *m;
        FILE *f = fopen(CHUCK_MEM_FILE, "ab");
        if (f) { fwrite(m, sizeof(ChuckMem), 1, f); fclose(f); }
    } else {
        /* At cap: reservoir sampling — replace random entry */
        int slot = (int)(rnext() % (uint64_t)chuck_mem_total);
        if (slot < CHUCK_MEM_CAP) {
            chuck_mem[slot] = *m;
            /* Rewrite entire file (500 entries × 16 bytes = 8 KB — trivial) */
            FILE *f = fopen(CHUCK_MEM_FILE, "wb");
            if (f) { fwrite(chuck_mem, sizeof(ChuckMem), chuck_mem_n, f); fclose(f); }
        }
    }
}

/* Nearest neighbor recall: find most similar past state, return its λ.
 * Distance = normalized (loss, grad_norm) difference.
 * Successful memories (negative delta_loss) get 2x weight. */
static float chuck_mem_recall(float loss, float grad_norm) {
    if (chuck_mem_n == 0) return -1.0f;  /* no memory → no prior */
    float best_dist = 1e9f, best_lambda = -1.0f;
    for (int i = 0; i < chuck_mem_n; i++) {
        float dl = (loss - chuck_mem[i].loss) / (fabsf(loss) + 1e-8f);
        float dg = (grad_norm - chuck_mem[i].grad_norm) / (fabsf(grad_norm) + 1e-8f);
        float dist = dl * dl + dg * dg;
        if (chuck_mem[i].delta_loss < 0) dist *= 0.5f;  /* prefer wins */
        if (dist < best_dist) { best_dist = dist; best_lambda = chuck_mem[i].lambda; }
    }
    return best_lambda;
}

/* ---- Self-Awareness: Eyes ---- */

/* SiLU eye: tracks dead neuron ratio */
static struct { int dead, total; float health; } SiLU_eye;

static void silu_eye_reset(void) { SiLU_eye.dead = 0; SiLU_eye.total = 0; }
static void silu_eye_update(void) {
    if (SiLU_eye.total == 0) { SiLU_eye.health = 1.0f; return; }
    SiLU_eye.health = 1.0f - (float)SiLU_eye.dead / SiLU_eye.total;
    SiLU_eye.dead = 0; SiLU_eye.total = 0;
}

/* RMSNorm eye: tracks normalization scale EMA */
static struct { float scale_ema; int init; } Norm_eye;

/* RoPE eye: tracks frequency band utilization */
static struct { float freq_energy[N_EMBD/2]; int calls; float utilization; } RoPE_eye;

static void rope_eye_reset(void) {
    memset(RoPE_eye.freq_energy, 0, sizeof(RoPE_eye.freq_energy));
    RoPE_eye.calls = 0;
}
static void rope_eye_update(void) {
    if (RoPE_eye.calls == 0) return;
    float max_e = 0;
    for (int b = 0; b < HEAD_DIM/2; b++) {
        RoPE_eye.freq_energy[b] /= RoPE_eye.calls;
        if (RoPE_eye.freq_energy[b] > max_e) max_e = RoPE_eye.freq_energy[b];
    }
    int active = 0;
    for (int b = 0; b < HEAD_DIM/2; b++)
        if (RoPE_eye.freq_energy[b] > max_e * 0.01f) active++;
    RoPE_eye.utilization = (HEAD_DIM/2 > 0) ? (float)active / (HEAD_DIM/2) : 1.0f;
    memset(RoPE_eye.freq_energy, 0, sizeof(RoPE_eye.freq_energy));
    RoPE_eye.calls = 0;
}

/* Attention eye: tracks per-head entropy (Level 7) */
static struct {
    float entropy[N_HEAD];      /* per-head attention entropy */
    float entropy_ema[N_HEAD];  /* EMA-smoothed entropy */
    int calls;
    int init;
} Attn_eye;

static void attn_eye_reset(void) { Attn_eye.calls = 0; memset(Attn_eye.entropy, 0, sizeof(Attn_eye.entropy)); }
static void attn_eye_observe(int head, const float *weights, int len) {
    /* Shannon entropy: H = -Σ p × log(p) */
    float H = 0;
    for (int t = 0; t < len; t++) {
        if (weights[t] > 1e-10f) H -= weights[t] * logf(weights[t]);
    }
    Attn_eye.entropy[head] += H;
    Attn_eye.calls++;
}
static void attn_eye_update(void) {
    if (Attn_eye.calls == 0) return;
    int calls_per_head = Attn_eye.calls / N_HEAD;
    if (calls_per_head == 0) calls_per_head = 1;
    for (int h = 0; h < N_HEAD; h++) {
        float avg = Attn_eye.entropy[h] / calls_per_head;
        if (Attn_eye.init) Attn_eye.entropy_ema[h] = 0.95f * Attn_eye.entropy_ema[h] + 0.05f * avg;
        else Attn_eye.entropy_ema[h] = avg;
    }
    Attn_eye.init = 1;
    memset(Attn_eye.entropy, 0, sizeof(Attn_eye.entropy));
    Attn_eye.calls = 0;
}

/* Cross-layer signal flow */
static float act_mag[N_LAYER];

/* 2D position table for RoPE — image patches get (row,col), text gets sequential */
static int pos_row[SEQ_LEN], pos_col[SEQ_LEN];
static void init_positions(void) {
    /* Single image patches: 4×4 grid */
    for (int p = 0; p < N_PATCHES; p++) {
        pos_row[p] = p / PATCHES_SIDE;
        pos_col[p] = p % PATCHES_SIDE;
    }
    /* Text tokens: sequential rows below images, col=0 */
    for (int t = 0; t < MAX_TXT; t++) {
        pos_row[N_VIS + t] = PATCHES_SIDE + t;
        pos_col[N_VIS + t] = 0;
    }
}

/* ---- CUDA state ---- */
#ifdef USE_CUDA
static cublasHandle_t cublas_h;
static float **gpu_params;   /* GPU mirrors of weight matrices */
static float *gpu_scratch;
static int gpu_scratch_sz;

static void cuda_init_params(void) {
    cublasCreate(&cublas_h);
    cublasSetMathMode(cublas_h, CUBLAS_TF32_TENSOR_OP_MATH);
    gpu_params = calloc(T.np, sizeof(float*));
    for (int i = 0; i < T.np; i++) {
        int sz = T.a[T.par[i]].size * (int)sizeof(float);
        cudaMalloc(&gpu_params[i], sz);
        cudaMemcpy(gpu_params[i], T.a[T.par[i]].data, sz, cudaMemcpyHostToDevice);
    }
    gpu_scratch_sz = 0; gpu_scratch = NULL;
    printf("  cuda: %d params uploaded to GPU (cuBLAS TF32)\n", T.np);
}

static float *gpu_ensure_scratch(int n) {
    if (n > gpu_scratch_sz) {
        if (gpu_scratch) cudaFree(gpu_scratch);
        cudaMalloc(&gpu_scratch, n * sizeof(float));
        gpu_scratch_sz = n;
    }
    return gpu_scratch;
}

static void cuda_sync_params(void) {
    for (int i = 0; i < T.np; i++) {
        int sz = T.a[T.par[i]].size * (int)sizeof(float);
        cudaMemcpy(gpu_params[i], T.a[T.par[i]].data, sz, cudaMemcpyHostToDevice);
    }
}

static void cuda_cleanup(void) {
    for (int i = 0; i < T.np; i++) cudaFree(gpu_params[i]);
    free(gpu_params);
    if (gpu_scratch) cudaFree(gpu_scratch);
    cublasDestroy(cublas_h);
}
#endif

/* ---- Forward ops (with awareness tracking) ---- */
static int op_add(int xi, int yi) {
    int n = T.a[xi].size, zi = anew(n);
    for (int i = 0; i < n; i++) T.a[zi].data[i] = T.a[xi].data[i] + T.a[yi].data[i];
    rec(OP_ADD,zi,xi,yi,0,0); return zi;
}
static int op_mul(int xi, int yi) {
    int n = T.a[xi].size, zi = anew(n);
    for (int i = 0; i < n; i++) T.a[zi].data[i] = T.a[xi].data[i] * T.a[yi].data[i];
    rec(OP_MUL,zi,xi,yi,0,0); return zi;
}
static int op_scale(int xi, float s) {
    int n = T.a[xi].size, zi = anew(n);
    for (int i = 0; i < n; i++) T.a[zi].data[i] = T.a[xi].data[i] * s;
    rec(OP_SCALE,zi,xi,-1,s,0); return zi;
}
static int op_mv(int Wi, int xi) {
    int r = T.a[Wi].rows, c = T.a[Wi].cols, zi = anew(r);
#ifdef USE_CUDA
    /* Find param index for GPU weight */
    int pi = -1;
    for (int p = 0; p < T.np; p++) if (T.par[p] == Wi) { pi = p; break; }
    if (pi >= 0) {
        float *d_scratch = gpu_ensure_scratch(c + r);
        float *d_x = d_scratch, *d_z = d_scratch + c;
        cudaMemcpy(d_x, T.a[xi].data, c * sizeof(float), cudaMemcpyHostToDevice);
        float alpha = 1.0f, beta = 0.0f;
        /* row-major W(r×c) × x(c) = z(r) → cublas: W^T in col-major */
        cublasSgemv(cublas_h, CUBLAS_OP_T, c, r, &alpha, gpu_params[pi], c, d_x, 1, &beta, d_z, 1);
        cudaMemcpy(T.a[zi].data, d_z, r * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        for (int i = 0; i < r; i++) { float s = 0; const float *Wr = &T.a[Wi].data[i*c];
            for (int j = 0; j < c; j++) s += Wr[j] * T.a[xi].data[j]; T.a[zi].data[i] = s; }
    }
#elif defined(USE_BLAS)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, r, c,
                1.0f, T.a[Wi].data, c, T.a[xi].data, 1,
                0.0f, T.a[zi].data, 1);
#else
    for (int i = 0; i < r; i++) { float s = 0; const float *Wr = &T.a[Wi].data[i*c];
        for (int j = 0; j < c; j++) s += Wr[j] * T.a[xi].data[j]; T.a[zi].data[i] = s; }
#endif
    rec(OP_MATVEC,zi,Wi,xi,0,0); return zi;
}
static int op_rms(int xi) {
    int n = T.a[xi].size, zi = anew(n); float ms = 0;
    for (int i = 0; i < n; i++) ms += T.a[xi].data[i] * T.a[xi].data[i];
    ms = ms / n + 1e-5f; float sc = 1.0f / sqrtf(ms);
    for (int i = 0; i < n; i++) T.a[zi].data[i] = T.a[xi].data[i] * sc;
    /* Norm eye: track scale */
    if (Norm_eye.init) Norm_eye.scale_ema = 0.99f * Norm_eye.scale_ema + 0.01f * sc;
    else { Norm_eye.scale_ema = sc; Norm_eye.init = 1; }
    rec(OP_RMSNORM,zi,xi,-1,sc,n); return zi;
}
static int op_silu(int xi) {
    int n = T.a[xi].size, zi = anew(n);
    for (int i = 0; i < n; i++) {
        float x = T.a[xi].data[i]; float s = sigf(x);
        T.a[zi].data[i] = x * s;
        /* SiLU eye: track dead zone */
        if (x < -4.0f) SiLU_eye.dead++;
        SiLU_eye.total++;
    }
    rec(OP_SILU,zi,xi,-1,0,0); return zi;
}
static int op_embed(int Wi, int id) {
    int c = T.a[Wi].cols, zi = anew(c);
    memcpy(T.a[zi].data, &T.a[Wi].data[id*c], c * sizeof(float));
    rec(OP_EMBED,zi,Wi,-1,0,id); return zi;
}
static int op_ce(int li, int tgt) {
    int n = T.a[li].size; float mx = T.a[li].data[0];
    for (int i = 1; i < n; i++) if (T.a[li].data[i] > mx) mx = T.a[li].data[i];
    int pi = anew(n); float *p = T.a[pi].data; float s = 0;
    for (int i = 0; i < n; i++) { p[i] = expf(T.a[li].data[i] - mx); s += p[i]; }
    for (int i = 0; i < n; i++) p[i] /= (s + 1e-10f);
    int zi = anew(1); T.a[zi].data[0] = -logf(p[tgt] + 1e-10f);
    rec(OP_CE,zi,li,pi,(float)tgt,n); return zi;
}
/* 2D RoPE: first half of head encodes row, second half encodes column.
 * Image patches get true 2D positions. Text tokens: row=sequential, col=0. */
static int op_rope(int xi, int pos) {
    int n = T.a[xi].size, zi = anew(n);
    memcpy(T.a[zi].data, T.a[xi].data, n * sizeof(float));
    float *d = T.a[zi].data;
    int n_heads = n / HEAD_DIM, half = HEAD_DIM / 2;
    int row = pos_row[pos], col = pos_col[pos];
    for (int h = 0; h < n_heads; h++) {
        /* Row encoding (first half of head) */
        for (int i = 0; i < half; i += 2) {
            float freq = 1.0f / powf(ROPE_BASE, (float)i / (float)half);
            float ang = row * freq, c = cosf(ang), s = sinf(ang);
            int idx = h * HEAD_DIM + i;
            float x0 = d[idx], x1 = d[idx+1];
            d[idx] = x0*c - x1*s; d[idx+1] = x0*s + x1*c;
            float energy = d[idx]*d[idx] + d[idx+1]*d[idx+1];
            if (i/2 < N_EMBD/2) RoPE_eye.freq_energy[i/2] += energy;
        }
        /* Column encoding (second half of head) */
        for (int i = 0; i < half; i += 2) {
            float freq = 1.0f / powf(ROPE_BASE, (float)i / (float)half);
            float ang = col * freq, c = cosf(ang), s = sinf(ang);
            int idx = h * HEAD_DIM + half + i;
            float x0 = d[idx], x1 = d[idx+1];
            d[idx] = x0*c - x1*s; d[idx+1] = x0*s + x1*c;
            float energy = d[idx]*d[idx] + d[idx+1]*d[idx+1];
            if ((half+i)/2 < N_EMBD/2) RoPE_eye.freq_energy[(half+i)/2] += energy;
        }
    }
    RoPE_eye.calls++;
    rec(OP_ROPE,zi,xi,-1,0,pos); return zi;
}
static int op_reduce(int *la, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += T.a[la[i]].data[0];
    int zi = anew(1); T.a[zi].data[0] = s / n;
    int buf = anew(n); for (int i = 0; i < n; i++) ((int*)T.a[buf].data)[i] = la[i];
    rec(OP_REDUCE,zi,buf,-1,0,n); return zi;
}

/* ---- KV cache (GQA: KV_DIM, not N_EMBD) ---- */
static float *kv_k[N_LAYER][SEQ_LEN], *kv_v[N_LAYER][SEQ_LEN];
static int kv_ki[N_LAYER][SEQ_LEN], kv_vi[N_LAYER][SEQ_LEN];
static void kv_clear(void) {
    memset(kv_k,0,sizeof(kv_k)); memset(kv_v,0,sizeof(kv_v));
    memset(kv_ki,0,sizeof(kv_ki)); memset(kv_vi,0,sizeof(kv_vi));
}

/* ---- Backward ---- */
static void backward(int loss) {
    T.a[loss].grad[0] = 1.0f;
    for (int ei = T.ne - 1; ei >= 0; ei--) {
        Ent *e = &T.e[ei];
        Arr *out = &T.a[e->out], *i1 = (e->in1 >= 0) ? &T.a[e->in1] : NULL, *i2 = (e->in2 >= 0) ? &T.a[e->in2] : NULL;
        switch (e->op) {
        case OP_ADD: { int n = out->size;
            for (int i = 0; i < n; i++) { i1->grad[i] += out->grad[i]; i2->grad[i] += out->grad[i]; } break; }
        case OP_MUL: { int n = out->size;
            for (int i = 0; i < n; i++) { i1->grad[i] += out->grad[i]*i2->data[i]; i2->grad[i] += out->grad[i]*i1->data[i]; } break; }
        case OP_SCALE: { int n = out->size; float s = e->aux;
            for (int i = 0; i < n; i++) i1->grad[i] += out->grad[i] * s; break; }
        case OP_MATVEC: { int r = i1->rows, c = i1->cols;
            for (int i = 0; i < r; i++) { float dz = out->grad[i];
                for (int j = 0; j < c; j++) { i1->grad[i*c+j] += dz*i2->data[j]; i2->grad[j] += dz*i1->data[i*c+j]; } } break; }
        case OP_RMSNORM: { int n = e->ai; float sc = e->aux, dot = 0;
            for (int i = 0; i < n; i++) dot += out->grad[i] * out->data[i];
            for (int i = 0; i < n; i++) i1->grad[i] += sc * (out->grad[i] - out->data[i]*dot/n); break; }
        case OP_SILU: { int n = out->size;
            for (int i = 0; i < n; i++) { float sg = sigf(i1->data[i]); i1->grad[i] += out->grad[i]*sg*(1.0f+i1->data[i]*(1.0f-sg)); } break; }
        case OP_CE: { int n = e->ai; int tgt = (int)e->aux; float dl = out->grad[0];
            for (int i = 0; i < n; i++) i1->grad[i] += dl * (i2->data[i] - (i==tgt ? 1.0f : 0.0f)); break; }
        case OP_EMBED: { int id = e->ai, c = i1->cols;
            for (int j = 0; j < c; j++) i1->grad[id*c+j] += out->grad[j]; break; }
        case OP_ROPE: { int n = out->size, pos = e->ai;
            int nh = n / HEAD_DIM, half = HEAD_DIM / 2;
            int row = pos_row[pos], col = pos_col[pos];
            for (int h = 0; h < nh; h++) {
                /* Row backward (first half) */
                for (int i = 0; i < half; i += 2) {
                    float freq = 1.0f / powf(ROPE_BASE, (float)i/(float)half);
                    float ang = row*freq, c = cosf(ang), s = sinf(ang);
                    int idx = h * HEAD_DIM + i;
                    float g0 = out->grad[idx], g1 = out->grad[idx+1];
                    i1->grad[idx] += g0*c + g1*s; i1->grad[idx+1] += -g0*s + g1*c;
                }
                /* Col backward (second half) */
                for (int i = 0; i < half; i += 2) {
                    float freq = 1.0f / powf(ROPE_BASE, (float)i/(float)half);
                    float ang = col*freq, c = cosf(ang), s = sinf(ang);
                    int idx = h * HEAD_DIM + half + i;
                    float g0 = out->grad[idx], g1 = out->grad[idx+1];
                    i1->grad[idx] += g0*c + g1*s; i1->grad[idx+1] += -g0*s + g1*c;
                }
            } break; }
        case OP_ATTN: { /* GQA attention backward */
            int li = (int)e->aux, pos = e->ai;
            float *qd = i1->data, *ag = out->grad, isq = 1.0f / sqrtf((float)HEAD_DIM);
            for (int h = 0; h < N_HEAD; h++) {
                int hs = h * HEAD_DIM;
                int kvh = h / N_KV_GROUP;
                int kvs = kvh * HEAD_DIM;
                float sc[SEQ_LEN], mx = -1e9f;
                for (int t = 0; t <= pos; t++) { float s = 0;
                    for (int d = 0; d < HEAD_DIM; d++) s += qd[hs+d]*kv_k[li][t][kvs+d];
                    sc[t] = s*isq; if (sc[t] > mx) mx = sc[t]; }
                float sm = 0; for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t]-mx); sm += sc[t]; }
                for (int t = 0; t <= pos; t++) sc[t] /= (sm + 1e-10f);
                float dw[SEQ_LEN];
                for (int t = 0; t <= pos; t++) { dw[t] = 0;
                    for (int d = 0; d < HEAD_DIM; d++) dw[t] += kv_v[li][t][kvs+d]*ag[hs+d]; }
                float dot = 0; for (int t = 0; t <= pos; t++) dot += sc[t]*dw[t];
                for (int t = 0; t <= pos; t++) { float ds = sc[t]*(dw[t]-dot);
                    for (int d = 0; d < HEAD_DIM; d++) {
                        /* grad Q: each Q-head gets its own gradient */
                        i1->grad[hs+d] += ds * kv_k[li][t][kvs+d] * isq;
                        /* grad K: multiple Q-heads accumulate to shared KV-head */
                        T.a[kv_ki[li][t]].grad[kvs+d] += ds * qd[hs+d] * isq;
                        /* grad V: multiple Q-heads accumulate to shared KV-head */
                        T.a[kv_vi[li][t]].grad[kvs+d] += sc[t] * ag[hs+d];
                    } }
            } break; }
        case OP_REDUCE: { int n = e->ai; int *idxs = (int*)i1->data; float dl = out->grad[0]/n;
            for (int i = 0; i < n; i++) T.a[idxs[i]].grad[0] += dl; break; }
        }
    }
}

/* ===========================================================================
 * Chuck v4: Self-Aware Optimizer
 *
 *   θ_l -= (α × λ × λ_l × σ) × m̂/(√v̂ + ε) + η
 *
 *   λ   = global self-modulation (loss trend over 16-step window)
 *   λ_l = per-layer self-modulation (gradient norm trend per layer)
 *   σ   = activation health signal (SiLU alive ratio × norm stability)
 *   η   = stagnation noise (only when globally stuck)
 *   α   = base learning rate from cosine schedule
 *
 *   If λ_l = 0 → layer is frozen. Zero compute waste. Chuck decided it's done.
 *   Adam doesn't know which layers are done. Chuck does.
 * =========================================================================== */

/* Per-layer awareness state */
typedef struct {
    float grad_hist[CHUCK_WINDOW];
    float dampen;
    int frozen;
    int pos, full, stag;
} ChuckLayer;

/* Global awareness state */
static struct {
    float hist[CHUCK_WINDOW];
    float dampen, noise, sigma;
    float loss_ema;         /* EMA-smoothed loss (batch noise filter) */
    float gnorm_ema;        /* EMA-smoothed grad norm (for adaptive clip) */
    float psi;              /* Ψ: subjectivity signal (memory - observation) */
    float psi_w;            /* Ψ weight: trust in memory (0 → 0.3) */
    float macro_ema;        /* slow EMA for epoch-scale trend (Level 9) */
    float best_macro;       /* best macro_ema seen (for patience) */
    float lr_scale;         /* macro LR multiplier (patience decay) */
    int macro_stag;         /* macro patience counter */
    int macro_drops;        /* how many times macro decay fired */
    float rec_lambda;       /* λ at last memory recording */
    float rec_loss;         /* loss at last memory recording */
    int rec_frozen[N_LAYER]; /* frozen state at last recording */
    int rec_cd;             /* cooldown counter (steps since last record) */
    int pos, full, stag;
    int global_step;        /* total step counter for macro interval */
} Chuck;

static ChuckLayer CL[N_LAYER];

static void chuck_init(void) {
    memset(&Chuck, 0, sizeof(Chuck));
    Chuck.dampen = 1.0f; Chuck.sigma = 1.0f;
    Chuck.lr_scale = 1.0f; Chuck.best_macro = 1e9f;
    Chuck.rec_lambda = 1.0f; Chuck.rec_loss = 999.0f;
    memset(Chuck.rec_frozen, 0, sizeof(Chuck.rec_frozen));
    Chuck.psi = 0; Chuck.psi_w = 0;
    for (int l = 0; l < N_LAYER; l++) {
        memset(&CL[l], 0, sizeof(ChuckLayer));
        CL[l].dampen = 1.0f;
    }
    Norm_eye.init = 0; Norm_eye.scale_ema = 1.0f;
    SiLU_eye.health = 1.0f;
    RoPE_eye.utilization = 1.0f;
    /* Load persistent memory */
    chuck_mem_load();
    if (chuck_mem_n > 0)
        printf("  chuck: loaded %d memories from %s (Ψ_w=%.2f)\n",
               chuck_mem_n, CHUCK_MEM_FILE,
               fminf(CHUCK_PSI_CAP, (float)chuck_mem_n / ((float)chuck_mem_n + CHUCK_PSI_HALF)));
}

/* Which layer does param pi belong to? -1 = global (patch_proj, wte, w_cls) */
static int param_layer(int pi) {
    if (pi < 2) return -1;  /* 0=patch_proj, 1=wte */
    if (pi >= 2 + N_LAYER * 7) return -1;  /* w_cls = last param */
    return (pi - 2) / 7;     /* 7 params per layer: wq,wk,wv,wo,w1,w3,w2 */
}

static void chuck_step(float lr, float loss) {
    /* ═══ Level 1: Global self-awareness (loss trend) ═══ */
    /* EMA smoothing: filters batch-to-batch noise for mini-batch SGD */
    if (Chuck.loss_ema == 0.0f) Chuck.loss_ema = loss;
    else Chuck.loss_ema = 0.99f * Chuck.loss_ema + 0.01f * loss;
    Chuck.hist[Chuck.pos % CHUCK_WINDOW] = Chuck.loss_ema;
    Chuck.pos++;
    if (Chuck.pos >= CHUCK_WINDOW) Chuck.full = 1;
    if (Chuck.full) {
        int q = CHUCK_WINDOW / 4;
        float recent = 0, old = 0;
        for (int i = 0; i < q; i++) {
            recent += Chuck.hist[(Chuck.pos - 1 - i) % CHUCK_WINDOW];
            old += Chuck.hist[(Chuck.pos - CHUCK_WINDOW + i) % CHUCK_WINDOW];
        }
        recent /= q; old /= q;
        float trend = (recent - old) / (old + 1e-8f);
        if (trend > 0.01f) Chuck.dampen *= 0.95f;        /* loss rising → dampen */
        else if (trend < -0.05f) Chuck.dampen *= 1.05f;   /* loss falling → boost */
        if (fabsf(trend) < 0.001f) {
            Chuck.stag++;
            if (Chuck.stag > 8) { Chuck.noise = 0.001f; Chuck.stag = 0; }
        } else { Chuck.stag = 0; Chuck.noise *= 0.9f; }
        if (Chuck.dampen < CHUCK_DAMP_LO) Chuck.dampen = CHUCK_DAMP_LO;
        if (Chuck.dampen > CHUCK_DAMP_HI) Chuck.dampen = CHUCK_DAMP_HI;
    }

    /* ═══ Level 9: Multi-scale awareness (macro patience) ═══ */
    Chuck.global_step++;
    if (Chuck.macro_ema == 0.0f) Chuck.macro_ema = loss;
    else Chuck.macro_ema = 0.999f * Chuck.macro_ema + 0.001f * loss;

    if (Chuck.global_step % CHUCK_MACRO_INT == 0 && Chuck.global_step > CHUCK_WINDOW) {
        if (Chuck.macro_ema > Chuck.best_macro * 0.999f) {
            Chuck.macro_stag++;
            if (Chuck.macro_stag >= CHUCK_MACRO_PAT) {
                Chuck.lr_scale *= CHUCK_MACRO_DECAY;
                if (Chuck.lr_scale < 0.05f) Chuck.lr_scale = 0.05f;
                Chuck.macro_stag = 0;
                Chuck.macro_drops++;
            }
        } else {
            Chuck.best_macro = Chuck.macro_ema;
            Chuck.macro_stag = 0;
        }
    }

    /* ═══ Level 4: Activation health signal (σ) ═══ */
    silu_eye_update();
    rope_eye_update();
    attn_eye_update();
    Chuck.sigma = 1.0f;
    if (SiLU_eye.health < 0.7f) Chuck.sigma *= SiLU_eye.health / 0.7f;
    if (Norm_eye.scale_ema > 5.0f) Chuck.sigma *= 0.9f;
    if (Norm_eye.scale_ema < 0.2f) Chuck.sigma *= 0.9f;

    /* ═══ Level 7: Attention entropy awareness ═══ */
    if (Attn_eye.init) {
        float h_max = logf((float)(N_VIS + MAX_TXT));  /* max possible entropy */
        for (int hd = 0; hd < N_HEAD; hd++) {
            float ratio = Attn_eye.entropy_ema[hd] / (h_max + 1e-8f);
            if (ratio < 0.1f) Chuck.sigma *= 0.95f;       /* collapsed head → dampen */
            else if (ratio > 0.95f) Chuck.sigma *= 0.98f;  /* fully diffuse → slight dampen */
        }
    }

    /* ═══ Level 2: Per-layer self-awareness (grad norm trend) ═══ */
    float layer_gnorm[N_LAYER];
    memset(layer_gnorm, 0, sizeof(layer_gnorm));
    for (int pi = 0; pi < T.np; pi++) {
        int l = param_layer(pi);
        if (l < 0 || l >= N_LAYER) continue;
        Arr *p = &T.a[T.par[pi]];
        float gn = 0;
        for (int i = 0; i < p->size; i++) gn += p->grad[i] * p->grad[i];
        layer_gnorm[l] += gn;
    }
    for (int l = 0; l < N_LAYER; l++) layer_gnorm[l] = sqrtf(layer_gnorm[l]);

    for (int l = 0; l < N_LAYER; l++) {
        if (CL[l].frozen) continue;
        CL[l].grad_hist[CL[l].pos % CHUCK_WINDOW] = layer_gnorm[l];
        CL[l].pos++;
        if (CL[l].pos >= CHUCK_WINDOW) CL[l].full = 1;
        if (CL[l].full) {
            int q = CHUCK_WINDOW / 4;
            float recent = 0, old = 0;
            for (int i = 0; i < q; i++) {
                recent += CL[l].grad_hist[(CL[l].pos - 1 - i) % CHUCK_WINDOW];
                old += CL[l].grad_hist[(CL[l].pos - CHUCK_WINDOW + i) % CHUCK_WINDOW];
            }
            recent /= q; old /= q;
            float trend = (recent - old) / (old + 1e-8f);
            /* grad norm trending up → layer needs more work → boost */
            if (trend > 0.05f) CL[l].dampen *= 1.05f;
            /* grad norm trending down → layer is settling → dampen */
            else if (trend < -0.05f) CL[l].dampen *= 0.95f;
            /* freeze: near-zero gradient norm for extended period */
            if (layer_gnorm[l] < 0.01f) {
                CL[l].stag++;
                if (CL[l].stag > 8) CL[l].frozen = 1;
            } else { CL[l].stag = 0; }
            if (CL[l].dampen < CHUCK_DAMP_LO) CL[l].dampen = CHUCK_DAMP_LO;
            if (CL[l].dampen > CHUCK_DAMP_HI) CL[l].dampen = CHUCK_DAMP_HI;
        }
    }

    /* ═══ Level 5: Cross-layer signal flow ═══ */
    if (act_mag[0] > 1e-8f) {
        float ratio = act_mag[N_LAYER-1] / (act_mag[0] + 1e-8f);
        for (int l = 1; l < N_LAYER; l++) {
            if (CL[l].frozen) continue;
            float depth = (float)l / (N_LAYER - 1);
            if (ratio < 0.3f) CL[l].dampen *= (1.0f + 0.02f * depth);       /* vanishing → boost deep */
            else if (ratio > 3.0f) CL[l].dampen *= (1.0f - 0.02f * depth);  /* exploding → dampen deep */
            if (CL[l].dampen < CHUCK_DAMP_LO) CL[l].dampen = CHUCK_DAMP_LO;
            if (CL[l].dampen > CHUCK_DAMP_HI) CL[l].dampen = CHUCK_DAMP_HI;
        }
    }

    /* ═══ Level 6: Ψ — Subjectivity (memory vs observation) ═══ */
    float gnorm_sq = 0;
    for (int pi = 0; pi < T.np; pi++) { Arr *p = &T.a[T.par[pi]];
        for (int i = 0; i < p->size; i++) gnorm_sq += p->grad[i] * p->grad[i]; }
    float gnorm = sqrtf(gnorm_sq + 1e-8f);

    Chuck.psi_w = (chuck_mem_n > 0) ?
        fminf(CHUCK_PSI_CAP, (float)chuck_mem_n / ((float)chuck_mem_n + CHUCK_PSI_HALF)) : 0.0f;

    float lambda_psi = Chuck.dampen;  /* default: pure reactive */
    if (chuck_mem_n > 0) {
        float lambda_prior = chuck_mem_recall(loss, gnorm);
        if (lambda_prior > 0) {
            Chuck.psi = lambda_prior - Chuck.dampen;
            lambda_psi = Chuck.dampen + Chuck.psi_w * Chuck.psi;
            if (lambda_psi < CHUCK_DAMP_LO) lambda_psi = CHUCK_DAMP_LO;
            if (lambda_psi > CHUCK_DAMP_HI) lambda_psi = CHUCK_DAMP_HI;
        }
    }

    /* Record memory on regime change */
    Chuck.rec_cd++;
    if (Chuck.full && Chuck.rec_cd >= CHUCK_REC_CD) {
        float delta_loss = loss - Chuck.rec_loss;
        float lambda_shift = fabsf(Chuck.dampen - Chuck.rec_lambda) / (Chuck.rec_lambda + 1e-8f);
        int regime_change = (lambda_shift > CHUCK_REC_THR);  /* λ shifted >25% */
        for (int l = 0; l < N_LAYER && !regime_change; l++)
            if (CL[l].frozen != Chuck.rec_frozen[l]) regime_change = 1;
        if (regime_change) {
            ChuckMem snap = { loss, gnorm, Chuck.dampen, delta_loss };
            chuck_mem_save(&snap);
            Chuck.rec_lambda = Chuck.dampen;
            Chuck.rec_loss = loss;
            Chuck.rec_cd = 0;
            for (int l = 0; l < N_LAYER; l++) Chuck.rec_frozen[l] = CL[l].frozen;
        }
    }

    /* ═══ Apply parameter updates ═══ */
    T.cstep++;
    float bc1 = 1.0f - powf(CHUCK_B1, (float)T.cstep);
    float bc2 = 1.0f - powf(CHUCK_B2, (float)T.cstep);

    /* Adaptive gradient clipping — Chuck controls the leash */
    if (Chuck.gnorm_ema == 0.0f) Chuck.gnorm_ema = gnorm;
    else Chuck.gnorm_ema = 0.97f * Chuck.gnorm_ema + 0.03f * gnorm;
    float adaptive_clip = GRAD_CLIP;
    if (Chuck.gnorm_ema > 1e-8f) {
        adaptive_clip = fmaxf(0.5f, fminf(2.0f, 1.5f * Chuck.gnorm_ema));  /* track gnorm */
        if (gnorm > 3.0f * Chuck.gnorm_ema) adaptive_clip *= 0.5f;          /* anomaly → clamp hard */
    }
    float clip = (gnorm > adaptive_clip) ? adaptive_clip / gnorm : 1.0f;

    for (int pi = 0; pi < T.np; pi++) {
        int l = param_layer(pi);
        /* Frozen layer → skip entirely */
        if (l >= 0 && l < N_LAYER && CL[l].frozen) continue;
        float layer_damp = (l >= 0 && l < N_LAYER) ? CL[l].dampen : 1.0f;
        float eff_lr = lr * lambda_psi * layer_damp * Chuck.sigma * Chuck.lr_scale;

        int idx = T.par[pi]; Arr *p = &T.a[idx];
        float *m = T.cm[pi], *v = T.cv[pi];
        for (int i = 0; i < p->size; i++) { float g = p->grad[i] * clip;
            m[i] = CHUCK_B1*m[i] + (1.0f-CHUCK_B1)*g;
            v[i] = CHUCK_B2*v[i] + (1.0f-CHUCK_B2)*g*g;
            p->data[i] -= eff_lr * (m[i]/bc1) / (sqrtf(v[i]/bc2) + CHUCK_EPS);
            if (Chuck.noise > 0) p->data[i] += Chuck.noise * rnf(0, 1.0f);
        }
    }
}

/* ===========================================================================
 * CIFAR-100 — 100 fine-grained classes, 32×32 RGB images
 *
 *   Binary format: [coarse_label(1B)][fine_label(1B)][pixels(3072B)]
 *   Pixels: 1024 R + 1024 G + 1024 B (channel-first, row-major)
 *   Train: 50000 images, Test: 10000 images
 *
 *   Download: https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
 * =========================================================================== */

static const char *cifar100_names[N_CLASSES] = {
    "apple","aquarium_fish","baby","bear","beaver",
    "bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly",
    "camel","can","castle","caterpillar","cattle",
    "chair","chimpanzee","clock","cloud","cockroach",
    "couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox",
    "girl","hamster","house","kangaroo","keyboard",
    "lamp","lawn_mower","leopard","lion","lizard",
    "lobster","man","maple_tree","motorcycle","mountain",
    "mouse","mushroom","oak_tree","orange","orchid",
    "otter","palm_tree","pear","pickup_truck","pine_tree",
    "plain","plate","poppy","porcupine","possum",
    "rabbit","raccoon","ray","road","rocket",
    "rose","sea","seal","shark","shrew",
    "skunk","skyscraper","snail","snake","spider",
    "squirrel","streetcar","sunflower","sweet_pepper","table",
    "tank","telephone","television","tiger","tractor",
    "train","trout","tulip","turtle","wardrobe",
    "whale","willow_tree","wolf","woman","worm"
};

/* Char-level tokenization: a-z + underscore */
static const char chars[] = "abcdefghijklmnopqrstuvwxyz_";
static int c2id(char c) { for (int i = 0; i < N_CHARS; i++) if (chars[i] == c) return i; return -1; }
static char id2c(int i) { return (i == BOS) ? '^' : (i == EOS) ? '$' : (i >= 0 && i < N_CHARS) ? chars[i] : '?'; }

typedef struct { float *imgs; int *labels; int n; } Data;

static Data load_cifar100(const char *path) {
    Data d = {0};
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return d; }
    /* Count records: file_size / 3074 */
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    int n = (int)(sz / 3074);
    if (n == 0) { fclose(f); return d; }
    d.n = n;
    d.imgs = malloc(n * IMG_SIZE * IMG_SIZE * IMG_CH * sizeof(float));
    d.labels = malloc(n * sizeof(int));
    uint8_t rec[3074];
    for (int i = 0; i < n; i++) {
        if (fread(rec, 1, 3074, f) != 3074) { d.n = i; break; }
        d.labels[i] = rec[1]; /* fine label */
        float *img = &d.imgs[i * IMG_SIZE * IMG_SIZE * IMG_CH];
        /* Convert planar uint8 → channel-first float [0,1] */
        for (int j = 0; j < 3072; j++) img[j] = rec[2 + j] / 255.0f;
    }
    fclose(f);
    return d;
}

/* ---- Model (GQA: wk/wv use KV_DIM) ---- */
typedef struct {
    int patch_proj, wte, w_cls;
    struct { int wq, wk, wv, wo, w1, w3, w2; } L[N_LAYER];
} Model;
static Model M;
static int g_cls_mode = 0;  /* 0=generative, 1=classification head */

static int init_w(int r, int c, float s) {
    int i = mnew(r, c);
    for (int j = 0; j < r*c; j++) T.a[i].data[j] = rnf(0, s);
    preg(i); return i;
}
static void init_model(void) {
    M.patch_proj = init_w(N_EMBD, PATCH_PX, sqrtf(2.0f / PATCH_PX));  /* param 0: (256, 192) */
    M.wte = init_w(VOCAB, N_EMBD, 0.02f);                               /* param 1: (29, 256) */
    for (int i = 0; i < N_LAYER; i++) {
        float s = 0.02f / sqrtf(2.0f * N_LAYER);
        M.L[i].wq = init_w(N_EMBD, N_EMBD, s);             /* param 2+7i+0 */
        M.L[i].wk = init_w(KV_DIM, N_EMBD, s);             /* param 2+7i+1: GQA! */
        M.L[i].wv = init_w(KV_DIM, N_EMBD, s);             /* param 2+7i+2: GQA! */
        M.L[i].wo = init_w(N_EMBD, N_EMBD, s);             /* param 2+7i+3 */
        M.L[i].w1 = init_w(MLP_DIM, N_EMBD, s);            /* param 2+7i+4 */
        M.L[i].w3 = init_w(MLP_DIM, N_EMBD, s);            /* param 2+7i+5 */
        M.L[i].w2 = init_w(N_EMBD, MLP_DIM, s);            /* param 2+7i+6 */
    }
    M.w_cls = init_w(N_CLASSES, N_EMBD, 0.02f);             /* classification head: (100, 256) */
    T.npa = T.na; T.aparam = T.apos;
}

/* ---- GPT step (one position, GQA attention) ---- */
static int gpt_step(int x, int pos, int layer_track) {
    int h = x;
    for (int li = 0; li < N_LAYER; li++) {
        int res = h; h = op_rms(h);
        int qi = op_mv(M.L[li].wq, h);
        int ki = op_mv(M.L[li].wk, h);  /* KV_DIM output */
        int vi = op_mv(M.L[li].wv, h);  /* KV_DIM output */
        int rqi = op_rope(qi, pos);
        int rki = op_rope(ki, pos);      /* RoPE on KV_DIM */
        kv_k[li][pos] = T.a[rki].data; kv_v[li][pos] = T.a[vi].data;
        kv_ki[li][pos] = rki; kv_vi[li][pos] = vi;

        /* GQA multi-head attention */
        int ao = anew(N_EMBD); float *ad = T.a[ao].data;
        for (int h_ = 0; h_ < N_HEAD; h_++) {
            int hs = h_ * HEAD_DIM;         /* Q offset in N_EMBD */
            int kvh = h_ / N_KV_GROUP;      /* which KV head */
            int kvs = kvh * HEAD_DIM;        /* KV offset in KV_DIM */
            float sc[SEQ_LEN], mx = -1e9f; float *qd = T.a[rqi].data;
            for (int t = 0; t <= pos; t++) { float s = 0;
                for (int d = 0; d < HEAD_DIM; d++) s += qd[hs+d]*kv_k[li][t][kvs+d];
                sc[t] = s / sqrtf((float)HEAD_DIM); if (sc[t] > mx) mx = sc[t]; }
            float sm = 0; for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t]-mx); sm += sc[t]; }
            for (int t = 0; t <= pos; t++) sc[t] /= (sm + 1e-10f);
            /* Attention eye: observe entropy of this head's attention distribution */
            if (layer_track && pos > 0) attn_eye_observe(h_, sc, pos + 1);
            for (int d = 0; d < HEAD_DIM; d++) { float v = 0;
                for (int t = 0; t <= pos; t++) v += sc[t]*kv_v[li][t][kvs+d]; ad[hs+d] = v; }
        }
        rec(OP_ATTN, ao, rqi, -1, (float)li, pos);
        h = op_add(res, op_mv(M.L[li].wo, ao));

        /* Track activation magnitude for cross-layer signal */
        if (layer_track) {
            float rms = 0;
            for (int i = 0; i < N_EMBD; i++) rms += T.a[h].data[i] * T.a[h].data[i];
            act_mag[li] = sqrtf(rms / N_EMBD);
        }

        res = h; h = op_rms(h);
        int gate = op_silu(op_mv(M.L[li].w1, h)), up = op_mv(M.L[li].w3, h);
        h = op_add(res, op_mv(M.L[li].w2, op_mul(gate, up)));
    }
    return op_mv(M.wte, op_rms(h));  /* weight-tied lm_head */
}

/* ---- Vision encoder: ViT-style RGB patch tokenization ---- */
static void encode_vis(float *img, int *tok) {
    /* img: channel-first [3][32][32], patches: 8×8×3 = 192 floats */
    for (int py = 0; py < PATCHES_SIDE; py++)
        for (int px = 0; px < PATCHES_SIDE; px++) {
            int pi = anew(PATCH_PX); float *pd = T.a[pi].data;
            for (int c = 0; c < IMG_CH; c++)
                for (int y = 0; y < PATCH_SIZE; y++)
                    for (int x = 0; x < PATCH_SIZE; x++)
                        pd[c*PATCH_SIZE*PATCH_SIZE + y*PATCH_SIZE + x] =
                            img[c*IMG_SIZE*IMG_SIZE + (py*PATCH_SIZE+y)*IMG_SIZE + px*PATCH_SIZE+x];
            tok[py*PATCHES_SIDE + px] = op_mv(M.patch_proj, pi);
        }
}

/* ===========================================================================
 * Checkpoint — save/load model weights + optimizer state
 *
 *   Format: [magic 4B][step 4B][np 4B]
 *           [param_sizes np×4B]
 *           [param_data ...][adam_m ...][adam_v ...]
 *           [Chuck state][ChuckLayer×N_LAYER][chuck_mem]
 * =========================================================================== */
#define CKPT_MAGIC 0x4C454538  /* "LEE8" */

static void ckpt_save(const char *path, int step) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ckpt: cannot write %s\n", path); return; }
    uint32_t magic = CKPT_MAGIC;
    fwrite(&magic, 4, 1, f);
    fwrite(&step, 4, 1, f);
    int np = T.np; fwrite(&np, 4, 1, f);
    /* param sizes */
    for (int i = 0; i < np; i++) { int sz = T.a[T.par[i]].size; fwrite(&sz, 4, 1, f); }
    /* param data — the only thing that matters */
    for (int i = 0; i < np; i++) fwrite(T.a[T.par[i]].data, sizeof(float), T.a[T.par[i]].size, f);
    /* Chuck global state — his soul */
    fwrite(&Chuck, sizeof(Chuck), 1, f);
    /* Chuck per-layer awareness */
    fwrite(CL, sizeof(ChuckLayer), N_LAYER, f);
    /* Chuck memory — persistent experience */
    fwrite(&chuck_mem_n, 4, 1, f);
    fwrite(&chuck_mem_total, 4, 1, f);
    fwrite(chuck_mem, sizeof(ChuckMem), chuck_mem_n, f);
    fclose(f);
    long sz = 0; FILE *chk = fopen(path, "rb");
    if (chk) { fseek(chk, 0, SEEK_END); sz = ftell(chk); fclose(chk); }
    printf("  ckpt: saved step %d → %s (%.1fMB)\n", step, path, sz / (1024.0f * 1024.0f));
}

static int ckpt_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ckpt: cannot read %s\n", path); return -1; }
    uint32_t magic; fread(&magic, 4, 1, f);
    if (magic != CKPT_MAGIC) { fprintf(stderr, "ckpt: bad magic in %s\n", path); fclose(f); return -1; }
    int step, np;
    fread(&step, 4, 1, f);
    fread(&np, 4, 1, f);
    if (np != T.np) { fprintf(stderr, "ckpt: param count mismatch (%d vs %d)\n", np, T.np); fclose(f); return -1; }
    /* verify sizes */
    for (int i = 0; i < np; i++) {
        int sz; fread(&sz, 4, 1, f);
        if (sz != T.a[T.par[i]].size) { fprintf(stderr, "ckpt: param %d size mismatch\n", i); fclose(f); return -1; }
    }
    /* param data */
    for (int i = 0; i < np; i++) fread(T.a[T.par[i]].data, sizeof(float), T.a[T.par[i]].size, f);
    /* Chuck global state — his soul persists */
    fread(&Chuck, sizeof(Chuck), 1, f);
    /* Chuck per-layer awareness */
    fread(CL, sizeof(ChuckLayer), N_LAYER, f);
    /* Chuck memory — he remembers */
    fread(&chuck_mem_n, 4, 1, f);
    fread(&chuck_mem_total, 4, 1, f);
    if (chuck_mem_n > 0) fread(chuck_mem, sizeof(ChuckMem), chuck_mem_n, f);
    fclose(f);
    /* Adam m/v stay zeroed — Chuck's Ψ memory handles the warmup.
     * He's been here before. He knows the landscape. */
    printf("  ckpt: loaded step %d from %s (%d chuck memories, Ψ ready)\n", step, path, chuck_mem_n);
    return step;
}

static float cos_lr(int step, int total) {
    if (step < WARMUP) return LR_MAX * (float)step / WARMUP;
    float p = (float)(step - WARMUP) / (float)(total - WARMUP);
    return LR_MAX * 0.5f * (1.0f + cosf(3.14159265f * p));
}

/* ---- Training ---- */
static const char *g_ckpt_path = "lee.bin";
static int g_start_step = 0;

static void train(Data *data) {
    printf("\n=== TRAINING (%d steps, Chuck v7 — multi-scale + reservoir memory) ===\n", STEPS);
    int tp = 0; for (int i = 0; i < T.np; i++) tp += T.a[T.par[i]].size;
    printf("  %d params (%.1fM) | %d layers | GQA %dQ/%dKV | embd=%d | %d patches (RGB %dx%d) | 2D RoPE | weight-tied\n",
           tp, tp/1000000.0f, N_LAYER, N_HEAD, N_KV_HEAD, N_EMBD, N_PATCHES, PATCH_SIZE, PATCH_SIZE);
    printf("  task: CIFAR-100 image classification → class name (100 classes, char-level)\n");
    printf("  data: %d images\n", data->n);
    if (g_start_step > 0) printf("  resuming from step %d\n", g_start_step);
    printf("\n");
    float rl = 0; int rn = 0;
    for (int step = g_start_step; step < STEPS; step++) {
        int idx = (int)(rnext() % (uint64_t)data->n);
        int label = data->labels[idx];
        const char *name = cifar100_names[label]; int nlen = strlen(name);
        int toks[MAX_TXT]; int nt = 0;
        toks[nt++] = BOS;
        for (int i = 0; i < nlen && nt < MAX_TXT - 1; i++) toks[nt++] = c2id(name[i]);
        toks[nt++] = EOS;
        tape_reset(); kv_clear();
        silu_eye_reset(); rope_eye_reset(); attn_eye_reset();
        /* Encode image patches */
        int vt[N_VIS];
        float *img = &data->imgs[idx * IMG_SIZE * IMG_SIZE * IMG_CH];
        encode_vis(img, vt);
        for (int p = 0; p < N_VIS; p++) gpt_step(vt[p], p, 0);
        /* Autoregressive text loss */
        int la[MAX_TXT]; int nl = 0;
        for (int t = 0; t < nt - 1; t++) {
            int pos = N_VIS + t, te = op_embed(M.wte, toks[t]);
            int lg = gpt_step(te, pos, (t == nt - 2)); /* track signal on last token */
            la[nl++] = op_ce(lg, toks[t+1]);
        }
        int loss = op_reduce(la, nl); backward(loss);
        float lv = T.a[loss].data[0];
        chuck_step(cos_lr(step, STEPS), lv);
#ifdef USE_CUDA
        if ((step + 1) % 10 == 0) cuda_sync_params(); /* sync GPU weights periodically */
#endif
        rl += lv; rn++;
        if ((step+1) % 500 == 0) {
            float elr = cos_lr(step, STEPS) * Chuck.dampen * Chuck.lr_scale;
            printf("  step %5d/%d | loss %.4f (avg %.4f) | lr %.6f\n",
                   step+1, STEPS, lv, rl/rn, elr);
            printf("    chuck: \xce\xbb=%.2f \xce\xa8=%+.2f (\xce\xa8w=%.2f, %d mem) \xcf\x83=%.2f macro=%.2f",
                   Chuck.dampen, Chuck.psi, Chuck.psi_w, chuck_mem_n, Chuck.sigma, Chuck.lr_scale);
            if (Chuck.macro_drops > 0) printf(" (%d drops)", Chuck.macro_drops);
            /* Show first and last 2 layers only (10 layers is too many for one line) */
            for (int l = 0; l < N_LAYER; l++) {
                if (l == 2 && N_LAYER > 5) { printf(" | ..."); l = N_LAYER - 2; continue; }
                if (CL[l].frozen) printf(" | L%d:frz", l);
                else printf(" | L%d:%.2f", l, CL[l].dampen);
            }
            printf("\n    silu: %.0f%% alive | norm: %.1f | rope: %.0f%%",
                   SiLU_eye.health * 100, Norm_eye.scale_ema, RoPE_eye.utilization * 100);
            if (Attn_eye.init) {
                printf(" | attn H:");
                for (int hd = 0; hd < N_HEAD; hd++) printf(" %.2f", Attn_eye.entropy_ema[hd]);
            }
            printf("\n");
            rl = 0; rn = 0;
        }
        /* Auto-save every 5000 steps */
        if ((step+1) % 5000 == 0) ckpt_save(g_ckpt_path, step+1);
    }
    /* Final save */
    ckpt_save(g_ckpt_path, STEPS);
#ifdef USE_CUDA
    cuda_sync_params(); /* final sync */
#endif
}

/* ---- Classification mode: image → class label (no text generation) ---- */
static void train_cls(Data *data) {
    printf("\n=== TRAINING CLS (%d steps, Chuck v7) ===\n", STEPS);
    int tp = 0; for (int i = 0; i < T.np; i++) tp += T.a[T.par[i]].size;
    printf("  %d params (%.2fM) | %d layers | cls head: %d×%d\n",
           tp, tp/1000000.0f, N_LAYER, N_CLASSES, N_EMBD);
    printf("  task: CIFAR-100 → 100 classes (direct classification)\n");
    printf("  data: %d images\n", data->n);
    if (g_start_step > 0) printf("  resuming from step %d\n", g_start_step);
    printf("\n");
    float rl = 0; int rn = 0, rc = 0;
    for (int step = g_start_step; step < STEPS; step++) {
        int idx = (int)(rnext() % (uint64_t)data->n);
        int label = data->labels[idx];
        tape_reset(); kv_clear();
        silu_eye_reset(); rope_eye_reset(); attn_eye_reset();
        /* Encode image patches */
        int vt[N_VIS];
        float *img = &data->imgs[idx * IMG_SIZE * IMG_SIZE * IMG_CH];
        encode_vis(img, vt);
        /* Process all patches through transformer */
        int last_h = -1;
        for (int p = 0; p < N_VIS; p++) last_h = gpt_step(vt[p], p, (p == N_VIS - 1));
        /* Mean pool over all patch positions (use last position's hidden state after attention over all) */
        /* The last position has attended to all previous patches via causal attention */
        int cls_logits = op_mv(M.w_cls, op_rms(last_h));  /* (100,) */
        int loss = op_ce(cls_logits, label);
        backward(loss);
        float lv = T.a[loss].data[0];
        /* Check if prediction is correct */
        float *lg = T.a[cls_logits].data; int pred = 0;
        for (int i = 1; i < N_CLASSES; i++) if (lg[i] > lg[pred]) pred = i;
        if (pred == label) rc++;
        chuck_step(cos_lr(step, STEPS), lv);
#ifdef USE_CUDA
        if ((step + 1) % 10 == 0) cuda_sync_params();
#endif
        rl += lv; rn++;
        if ((step+1) % 500 == 0) {
            float elr = cos_lr(step, STEPS) * Chuck.dampen * Chuck.lr_scale;
            printf("  step %5d/%d | loss %.4f (avg %.4f) | acc %.1f%% | lr %.6f\n",
                   step+1, STEPS, lv, rl/rn, 100.0f*rc/rn, elr);
            printf("    chuck: \xce\xbb=%.2f \xce\xa8=%+.2f (\xce\xa8w=%.2f, %d mem) \xcf\x83=%.2f macro=%.2f",
                   Chuck.dampen, Chuck.psi, Chuck.psi_w, chuck_mem_n, Chuck.sigma, Chuck.lr_scale);
            if (Chuck.macro_drops > 0) printf(" (%d drops)", Chuck.macro_drops);
            for (int l = 0; l < N_LAYER; l++) {
                if (l == 2 && N_LAYER > 5) { printf(" | ..."); l = N_LAYER - 2; continue; }
                if (CL[l].frozen) printf(" | L%d:frz", l);
                else printf(" | L%d:%.2f", l, CL[l].dampen);
            }
            printf("\n");
            rl = 0; rn = 0; rc = 0;
        }
        if ((step+1) % 5000 == 0) ckpt_save(g_ckpt_path, step+1);
    }
    ckpt_save(g_ckpt_path, STEPS);
#ifdef USE_CUDA
    cuda_sync_params();
#endif
}

static void inference_cls(Data *data) {
    printf("\n=== INFERENCE CLS — CIFAR-100 (%d test images) ===\n\n", data->n);
    T.on = 0; int correct = 0, total = 0;
    int per_class_correct[N_CLASSES], per_class_total[N_CLASSES];
    memset(per_class_correct, 0, sizeof(per_class_correct));
    memset(per_class_total, 0, sizeof(per_class_total));
    for (int s = 0; s < data->n; s++) {
        int label = data->labels[s];
        tape_reset(); kv_clear();
        int vt[N_VIS];
        float *img = &data->imgs[s * IMG_SIZE * IMG_SIZE * IMG_CH];
        encode_vis(img, vt);
        int last_h = -1;
        for (int p = 0; p < N_VIS; p++) last_h = gpt_step(vt[p], p, 0);
        int cls_logits = op_mv(M.w_cls, op_rms(last_h));
        float *lg = T.a[cls_logits].data; int pred = 0;
        for (int i = 1; i < N_CLASSES; i++) if (lg[i] > lg[pred]) pred = i;
        int ok = (pred == label); correct += ok; total++;
        per_class_correct[label] += ok; per_class_total[label]++;
        if (s < 20 || (s % 1000 == 0))
            printf("  [%5d] true: %-16s | pred: %-16s %s\n",
                   s, cifar100_names[label], cifar100_names[pred], ok ? "OK" : "MISS");
    }
    printf("\n  accuracy: %d/%d (%.1f%%)\n", correct, total, 100.0f*correct/total);
    /* Top-5 and bottom-5 classes */
    printf("  best classes:");
    for (int rank = 0; rank < 5; rank++) {
        float best_acc = -1; int best_c = 0;
        for (int c = 0; c < N_CLASSES; c++) {
            if (per_class_total[c] == 0) continue;
            float a = (float)per_class_correct[c] / per_class_total[c];
            int taken = 0; (void)taken;
            if (a > best_acc) { best_acc = a; best_c = c; }
        }
        printf(" %s(%.0f%%)", cifar100_names[best_c], best_acc*100);
        per_class_total[best_c] = 0; /* exclude from next iter */
    }
    printf("\n");
    T.on = 1;
}

/* ---- Sampling ---- */
static int sample_topk(float *logits, int vocab, float temp, int topk) {
    float *sc = malloc(vocab * sizeof(float));
    for (int i = 0; i < vocab; i++) sc[i] = logits[i] / temp;
    float mx = sc[0]; for (int i = 1; i < vocab; i++) if (sc[i] > mx) mx = sc[i];
    float *p = malloc(vocab * sizeof(float)); float s = 0;
    for (int i = 0; i < vocab; i++) { p[i] = expf(sc[i] - mx); s += p[i]; }
    for (int i = 0; i < vocab; i++) p[i] /= s;
    int *ti = malloc(topk * sizeof(int)); float *tv = malloc(topk * sizeof(float));
    for (int k = 0; k < topk && k < vocab; k++) { int best = 0; float bv = -1e9f;
        for (int i = 0; i < vocab; i++) { int taken = 0;
            for (int j = 0; j < k; j++) if (ti[j] == i) { taken = 1; break; }
            if (!taken && p[i] > bv) { bv = p[i]; best = i; } }
        ti[k] = best; tv[k] = bv; }
    float ts = 0; for (int k = 0; k < topk; k++) ts += tv[k];
    float r = ruf() * ts, cum = 0; int result = ti[0];
    for (int k = 0; k < topk; k++) { cum += tv[k]; if (cum >= r) { result = ti[k]; break; } }
    free(sc); free(p); free(ti); free(tv);
    return result;
}

/* ---- Inference ---- */
static void inference(Data *data) {
    printf("\n=== INFERENCE — CIFAR-100 classification (temp=%.1f, top-k=%d) ===\n\n", TEMP, TOPK);
    T.on = 0; int correct = 0, total = 0;
    int n_test = data->n < 200 ? data->n : 200; /* test up to 200 samples */
    for (int s = 0; s < n_test; s++) {
        int idx = s;
        int label = data->labels[idx];
        tape_reset(); kv_clear();
        int vt[N_VIS];
        float *img = &data->imgs[idx * IMG_SIZE * IMG_SIZE * IMG_CH];
        encode_vis(img, vt);
        for (int p = 0; p < N_VIS; p++) gpt_step(vt[p], p, 0);
        int tok = BOS; char gen[MAX_TXT+1]; int gl = 0;
        for (int t = 0; t < MAX_TXT - 1; t++) {
            int pos = N_VIS + t, te = op_embed(M.wte, tok);
            int lg = gpt_step(te, pos, 0);
            tok = sample_topk(T.a[lg].data, VOCAB, TEMP, TOPK);
            if (tok == EOS || tok == BOS) break;
            if (gl < MAX_TXT) gen[gl++] = id2c(tok);
        }
        gen[gl] = '\0'; int ok = strcmp(gen, cifar100_names[label]) == 0; correct += ok; total++;
        if (s < 30 || ok) /* print first 30 and all correct */
            printf("  [%3d] true: %-16s | gen: %-16s %s\n", idx, cifar100_names[label], gen, ok ? "OK" : "MISS");
    }
    printf("\n  accuracy: %d/%d (%.1f%%)\n", correct, total, 100.0f*correct/total);

    /* Frozen layer report */
    int frozen = 0;
    for (int l = 0; l < N_LAYER; l++) if (CL[l].frozen) frozen++;
    if (frozen > 0) printf("  chuck: %d/%d layers frozen (compute saved)\n", frozen, N_LAYER);

    T.on = 1;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL); /* unbuffered for background/pipe */
    const char *data_dir = "cifar-100-binary";
    const char *resume_path = NULL;
    const char *save_path = "lee.bin";

    /* Parse CLI */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_dir = argv[++i];
        else if (strcmp(argv[i], "--resume") == 0 && i+1 < argc) resume_path = argv[++i];
        else if (strcmp(argv[i], "--save") == 0 && i+1 < argc) save_path = argv[++i];
        else if (strcmp(argv[i], "--cls") == 0) g_cls_mode = 1;
    }
    g_ckpt_path = save_path;

    printf("lee.c v8 — Vision-Language Model in pure C\n");
    printf("GQA %dQ/%dKV | %d layers | embd=%d | head=%d | mlp=%d | 2D RoPE | SwiGLU\n",
           N_HEAD, N_KV_HEAD, N_LAYER, N_EMBD, HEAD_DIM, MLP_DIM);
    printf("Chuck v7 (multi-scale + reservoir memory) | CIFAR-100 classification\n");
    printf("Named after Bruce Lee and Minhyeok Lee. Chuck sees inside the transformer.\n\n");

    clock_t t0 = clock(); rseed(42);
    init_positions(); tape_init(); chuck_init(); init_model();

    /* Count params */
    int tp = 0; for (int i = 0; i < T.np; i++) tp += T.a[T.par[i]].size;
    printf("  params: %d (%.2fM)\n", tp, tp / 1000000.0f);
    printf("  arena: %dMB allocated, %.1fMB used by params\n",
           ARENA_SZ / (1024*1024), (float)T.aparam / (1024*1024));

    /* Resume from checkpoint */
    if (resume_path) {
        int s = ckpt_load(resume_path);
        if (s >= 0) g_start_step = s;
        else { fprintf(stderr, "failed to load checkpoint, starting fresh\n"); }
    }

#ifdef USE_CUDA
    cuda_init_params();
#endif

    /* Load CIFAR-100 data */
    char train_path[512], test_path[512];
    snprintf(train_path, sizeof(train_path), "%s/train.bin", data_dir);
    snprintf(test_path, sizeof(test_path), "%s/test.bin", data_dir);

    printf("  loading training data from %s...\n", train_path);
    Data train_data = load_cifar100(train_path);
    if (train_data.n == 0) {
        fprintf(stderr, "\nerror: cannot load CIFAR-100 data from %s\n", train_path);
        fprintf(stderr, "download from: https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz\n");
        fprintf(stderr, "extract and pass: ./lee --data /path/to/cifar-100-binary\n");
        return 1;
    }
    printf("  loaded %d training images\n", train_data.n);

    printf("  loading test data from %s...\n", test_path);
    Data test_data = load_cifar100(test_path);
    printf("  loaded %d test images\n\n", test_data.n);

    if (g_cls_mode) {
        printf("  MODE: classification head (--cls)\n\n");
        train_cls(&train_data);
        inference_cls(test_data.n > 0 ? &test_data : &train_data);
    } else {
        train(&train_data);
        if (test_data.n > 0) inference(&test_data);
        else inference(&train_data);
    }

    printf("\ntotal: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
    printf("chuck.mem: %d memories (%.1f KB) | \xce\xa8_w=%.3f\n",
           chuck_mem_n, (float)(chuck_mem_n * (int)sizeof(ChuckMem)) / 1024.0f, Chuck.psi_w);
    if (chuck_mem_n > 0)
        printf("  next run: Chuck starts with experience. \xce\xa8 \xe2\x89\xa0 0. He remembers.\n");
    else
        printf("  first run: Chuck has no memories yet. Pure reactive. Newborn.\n");

    /* Cleanup */
    free(train_data.imgs); free(train_data.labels);
    if (test_data.n > 0) { free(test_data.imgs); free(test_data.labels); }
    for (int i = 0; i < T.np; i++) { free(T.cm[i]); free(T.cv[i]); }
#ifdef USE_CUDA
    cuda_cleanup();
#endif
    free(T.arena);
    printf("\ndone.\n"); return 0;
}
