// ─────────────────────────────────────────────────────────────────────────────
// Online normalizer calculation for softmax  (Milakov & Gimelshein, NVIDIA 2018)
// https://arxiv.org/abs/1805.02867   ·   github.com/NVIDIA/online-softmax
//
// Scope: the three softmax formulations of the paper, and nothing past them.
//   Algorithm 1  Naive softmax          2 loads + 1 store per element (overflows)
//   Algorithm 2  Safe softmax           3 loads + 1 store per element (safe)
//   Algorithm 3  Online softmax         2 loads + 1 store per element (safe)
//
// The whole point of Algorithm 3: it is numerically safe like Algorithm 2 while
// touching memory as little as Algorithm 1, by tracking the running max m and the
// running normalizer d *together* in one pass and rescaling d whenever m grows:
//
//     m_j = max(m_{j-1}, x_j)
//     d_j = d_{j-1} * e^(m_{j-1} - m_j) + e^(x_j - m_j)
//
// Softmax is memory bound, so "one fewer pass over x" is the speedup.
//
// Note on the GPU port: Algorithm 3 as printed is sequential. To run it on a
// block of threads you need the associative merge of two partial (m,d) states
// (paper §3, eq. for combining two chunks) — that merge is implemented here in
// md_merge() because there is no way to use the algorithm on a GPU without it.
// Everything beyond that (the multi-block / split-row reduction variant, the
// fused softmax+top-k kernels) is deliberately NOT in this file.
//
// Build:  nvcc -O3 -arch=native -o softmax softmax.cu
// Run:    ./softmax
// ─────────────────────────────────────────────────────────────────────────────

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

// Total elements held constant across the sweep, so every row width below moves
// the same number of bytes and the timings are directly comparable.
#define TOTAL_ELEMS (16 * 1024 * 1024)

// One block per row; 256 threads stride along the row.
#define BLOCK 256

#define WARMUP 5
#define REPS   30

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(_e));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── The (m, d) running state ────────────────────────────────────────────────
// m : running max of the elements seen so far
// d : running sum of e^(x - m) over those same elements
struct MD { float m; float d; };

// Merge two partial states. This is associative and commutative, which is what
// makes the online algorithm parallelizable: whichever way the elements were
// split between threads, merging the partials gives the same (m, d).
//
// The identity element is (-FLT_MAX, 0), NOT (-inf, 0): merging two identities
// would compute (-inf) - (-inf) = NaN and poison the reduction. -FLT_MAX gives
// exp(0) = 1 and keeps d at 0, which is the behaviour we want for threads that
// were handed no elements at all.
__device__ __forceinline__ MD md_merge(MD a, MD b) {
    MD r;
    r.m = fmaxf(a.m, b.m);
    r.d = a.d * expf(a.m - r.m) + b.d * expf(b.m - r.m);
    return r;
}

// ─── Block-wide reductions (warp shuffle, then across warps in shared) ────────
// Each begins with __syncthreads() because the safe-softmax kernel calls two of
// them back to back and they reuse the same shared staging buffers.

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float warp[BLOCK / 32];
    __shared__ float bcast;
    __syncthreads();

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    if (lane == 0) warp[wid] = v;
    __syncthreads();

    if (wid == 0) {
        v = (lane < BLOCK / 32) ? warp[lane] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
        if (lane == 0) bcast = v;
    }
    __syncthreads();
    return bcast;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp[BLOCK / 32];
    __shared__ float bcast;
    __syncthreads();

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    if (lane == 0) warp[wid] = v;
    __syncthreads();

    if (wid == 0) {
        v = (lane < BLOCK / 32) ? warp[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) bcast = v;
    }
    __syncthreads();
    return bcast;
}

// Same shape, but folding with md_merge instead of max / +.
__device__ __forceinline__ MD block_reduce_md(MD v) {
    __shared__ MD warp[BLOCK / 32];
    __shared__ MD bcast;
    __syncthreads();

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    for (int off = 16; off > 0; off >>= 1) {
        MD other;
        other.m = __shfl_down_sync(0xffffffff, v.m, off);
        other.d = __shfl_down_sync(0xffffffff, v.d, off);
        v = md_merge(v, other);
    }
    if (lane == 0) warp[wid] = v;
    __syncthreads();

    if (wid == 0) {
        v = (lane < BLOCK / 32) ? warp[lane] : MD{-FLT_MAX, 0.f};
        for (int off = 16; off > 0; off >>= 1) {
            MD other;
            other.m = __shfl_down_sync(0xffffffff, v.m, off);
            other.d = __shfl_down_sync(0xffffffff, v.d, off);
            v = md_merge(v, other);
        }
        if (lane == 0) bcast = v;
    }
    __syncthreads();
    return bcast;
}

// ─── Algorithm 1: naive softmax ──────────────────────────────────────────────
// Pass 1: d = Σ e^(x_j).      Pass 2: y_i = e^(x_i) / d.
// x is read from global memory twice; nothing is cached in registers, because
// caching it would defeat the point of counting passes (and for a real row
// length you cannot hold the row anyway).
// e^(x) overflows to +inf for x > ~88.7f, which is exactly the failure mode the
// paper's Algorithm 2 exists to fix — see the overflow demo in main().
__global__ void softmax_naive(const float *__restrict__ x, float *__restrict__ y, int cols) {
    const float *xr = x + (size_t)blockIdx.x * cols;
    float       *yr = y + (size_t)blockIdx.x * cols;

    float d = 0.f;
    for (int j = threadIdx.x; j < cols; j += BLOCK)
        d += expf(xr[j]);
    d = block_reduce_sum(d);

    float inv = 1.f / d;
    for (int j = threadIdx.x; j < cols; j += BLOCK)
        yr[j] = expf(xr[j]) * inv;
}

// ─── Algorithm 2: safe softmax ───────────────────────────────────────────────
// Pass 1: m = max x_j.   Pass 2: d = Σ e^(x_j - m).   Pass 3: y_i = e^(x_i-m)/d.
// Subtracting the max makes every exponent ≤ 0, so nothing overflows — at the
// cost of a third read of x.
__global__ void softmax_safe(const float *__restrict__ x, float *__restrict__ y, int cols) {
    const float *xr = x + (size_t)blockIdx.x * cols;
    float       *yr = y + (size_t)blockIdx.x * cols;

    float m = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += BLOCK)
        m = fmaxf(m, xr[j]);
    m = block_reduce_max(m);

    float d = 0.f;
    for (int j = threadIdx.x; j < cols; j += BLOCK)
        d += expf(xr[j] - m);
    d = block_reduce_sum(d);

    float inv = 1.f / d;
    for (int j = threadIdx.x; j < cols; j += BLOCK)
        yr[j] = expf(xr[j] - m) * inv;
}

// ─── Algorithm 3: online softmax ─────────────────────────────────────────────
// Pass 1: m and d are computed *together*, rescaling d each time m grows.
// Pass 2: y_i = e^(x_i - m) / d.
// Two reads of x instead of three: same traffic as naive, same safety as safe.
__global__ void softmax_online(const float *__restrict__ x, float *__restrict__ y, int cols) {
    const float *xr = x + (size_t)blockIdx.x * cols;
    float       *yr = y + (size_t)blockIdx.x * cols;

    MD s{-FLT_MAX, 0.f};
    for (int j = threadIdx.x; j < cols; j += BLOCK) {
        float xj = xr[j];
        // md_merge(s, {xj, 1}) written out: rescale the old d if the max moved.
        float m_new = fmaxf(s.m, xj);
        s.d = s.d * expf(s.m - m_new) + expf(xj - m_new);
        s.m = m_new;
    }
    s = block_reduce_md(s);

    float inv = 1.f / s.d;
    for (int j = threadIdx.x; j < cols; j += BLOCK)
        yr[j] = expf(xr[j] - s.m) * inv;
}

// ─── CPU reference ───────────────────────────────────────────────────────────
// Safe softmax with a double accumulator — the ground truth for verification.
static void cpu_softmax_ref(const float *x, float *y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float *xr = x + (size_t)i * cols;
        float       *yr = y + (size_t)i * cols;

        float m = -FLT_MAX;
        for (int j = 0; j < cols; j++) m = fmaxf(m, xr[j]);

        double d = 0.0;
        for (int j = 0; j < cols; j++) d += exp((double)xr[j] - m);

        for (int j = 0; j < cols; j++) yr[j] = (float)(exp((double)xr[j] - m) / d);
    }
}

// The literal sequential Algorithm 3, on the CPU. Not used for timing — it is
// here to show the recurrence on its own and to check that it agrees with the
// safe formulation to the last bit or two.
static void cpu_softmax_online(const float *x, float *y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float *xr = x + (size_t)i * cols;
        float       *yr = y + (size_t)i * cols;

        float m = -FLT_MAX, d = 0.f;
        for (int j = 0; j < cols; j++) {
            float m_new = fmaxf(m, xr[j]);
            d = d * expf(m - m_new) + expf(xr[j] - m_new);
            m = m_new;
        }
        for (int j = 0; j < cols; j++) yr[j] = expf(xr[j] - m) / d;
    }
}

// ─── Verification ────────────────────────────────────────────────────────────
// Softmax outputs are in [0,1] and mostly tiny, so compare absolutely.
static bool verify(const float *ref, const float *got, size_t n, float tol, float *max_err_out) {
    float max_err = 0.f;
    size_t bad = (size_t)-1;
    for (size_t i = 0; i < n; i++) {
        if (isnan(got[i]) || isinf(got[i])) { if (bad == (size_t)-1) bad = i; max_err = INFINITY; break; }
        float e = fabsf(ref[i] - got[i]);
        if (e > max_err) { max_err = e; if (e > tol && bad == (size_t)-1) bad = i; }
    }
    if (max_err_out) *max_err_out = max_err;
    return !(max_err > tol);
}

// ─── Timing ──────────────────────────────────────────────────────────────────
typedef void (*kernel_fn)(const float *, float *, int);

static float time_kernel(kernel_fn k, const float *dx, float *dy, int rows, int cols) {
    for (int i = 0; i < WARMUP; i++) k<<<rows, BLOCK>>>(dx, dy, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < REPS; i++) k<<<rows, BLOCK>>>(dx, dy, cols);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms / REPS;
}

// ─── Overflow demo ───────────────────────────────────────────────────────────
// Fill one small matrix with logits around 90. e^90 = +inf in fp32, so naive
// sums to inf and every output becomes inf/inf = NaN (or 0). Safe and online
// are unaffected. This is the paper's motivation for Algorithm 2 in one table.
static void overflow_demo() {
    const int rows = 64, cols = 1024;
    const size_t n = (size_t)rows * cols, bytes = n * sizeof(float);

    float *hx = (float *)malloc(bytes);
    float *hy = (float *)malloc(bytes);
    for (size_t i = 0; i < n; i++) hx[i] = 85.f + 10.f * ((float)rand() / RAND_MAX);

    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice));

    struct { const char *name; kernel_fn k; } ks[3] = {
        {"naive  (Alg 1)", softmax_naive},
        {"safe   (Alg 2)", softmax_safe},
        {"online (Alg 3)", softmax_online},
    };

    printf("── Overflow: logits drawn from [85, 95], e^x overflows fp32 above ~88.7 ──\n\n");
    printf("  %-15s  %12s  %12s  %10s\n", "kernel", "y[0]", "row sum", "finite");
    printf("  %-15s  %12s  %12s  %10s\n", "───────────────", "────────────", "────────────", "──────────");
    for (int t = 0; t < 3; t++) {
        CUDA_CHECK(cudaMemset(dy, 0, bytes));
        ks[t].k<<<rows, BLOCK>>>(dx, dy, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost));

        double sum = 0.0;
        bool finite = true;
        for (int j = 0; j < cols; j++) {
            sum += hy[j];
            if (isnan(hy[j]) || isinf(hy[j])) finite = false;
        }
        printf("  %-15s  %12.3e  %12.6f  %10s\n",
               ks[t].name, hy[0], sum, finite ? "yes" : "NO");
    }
    printf("\n  (a correct softmax row sums to 1.0)\n\n");

    free(hx); free(hy);
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
}

// ─── main ────────────────────────────────────────────────────────────────────
int main(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
    printf("\nGPU: %s  (sm_%d%d, %.0f GB/s peak)\n\n",
           prop.name, prop.major, prop.minor, peak_bw);

    srand(0);

    overflow_demo();

    // ── Sanity check: the sequential Algorithm 3 really does equal Algorithm 2 ──
    {
        const int rows = 16, cols = 2048;
        const size_t n = (size_t)rows * cols;
        float *a = (float *)malloc(n * sizeof(float));
        float *r = (float *)malloc(n * sizeof(float));
        float *o = (float *)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) a[i] = -10.f + 20.f * ((float)rand() / RAND_MAX);
        cpu_softmax_ref(a, r, rows, cols);
        cpu_softmax_online(a, o, rows, cols);
        float err = 0.f;
        verify(r, o, n, 1e-6f, &err);
        printf("── CPU: sequential Algorithm 3 vs safe reference, max abs err %.3e ──\n\n", err);
        free(a); free(r); free(o);
    }

    // ── Benchmark sweep ──
    const int col_sweep[] = {256, 1024, 4096, 16384};
    const int n_sweep = sizeof(col_sweep) / sizeof(col_sweep[0]);

    const size_t max_bytes = (size_t)TOTAL_ELEMS * sizeof(float);
    float *hx  = (float *)malloc(max_bytes);
    float *hy  = (float *)malloc(max_bytes);
    float *ref = (float *)malloc(max_bytes);
    if (!hx || !hy || !ref) { fprintf(stderr, "host alloc failed\n"); return 1; }

    // Benign range: naive does not overflow here, so all three are comparable.
    for (size_t i = 0; i < TOTAL_ELEMS; i++)
        hx[i] = -10.f + 20.f * ((float)rand() / RAND_MAX);

    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, max_bytes));
    CUDA_CHECK(cudaMalloc(&dy, max_bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx, max_bytes, cudaMemcpyHostToDevice));

    struct { const char *name; kernel_fn k; int passes; } ks[3] = {
        {"naive  (Alg 1)", softmax_naive,  3},   // 2 loads + 1 store
        {"safe   (Alg 2)", softmax_safe,   4},   // 3 loads + 1 store
        {"online (Alg 3)", softmax_online, 3},   // 2 loads + 1 store
    };

    printf("── Benchmark: %d elements total, softmax along each row, %d reps ──\n",
           TOTAL_ELEMS, REPS);
    printf("   'passes' counts memory accesses per element as the paper does.\n");
    printf("   GB/s is compulsory traffic (one read + one write) / time.\n\n");

    for (int s = 0; s < n_sweep; s++) {
        int cols = col_sweep[s];
        int rows = TOTAL_ELEMS / cols;
        size_t n = (size_t)rows * cols;

        cpu_softmax_ref(hx, ref, rows, cols);

        printf("  rows = %6d, cols = %6d\n", rows, cols);
        printf("  %-15s  %8s  %7s  %10s  %11s  %10s\n",
               "kernel", "ms", "passes", "GB/s", "max abs err", "correct");
        printf("  %-15s  %8s  %7s  %10s  %11s  %10s\n",
               "───────────────", "────────", "───────", "──────────", "───────────", "──────────");

        float t_ref = 0.f;
        for (int t = 0; t < 3; t++) {
            CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
            float ms = time_kernel(ks[t].k, dx, dy, rows, cols);
            CUDA_CHECK(cudaMemcpy(hy, dy, n * sizeof(float), cudaMemcpyDeviceToHost));

            float err = 0.f;
            bool ok = verify(ref, hy, n, 1e-6f, &err);
            double gbs = (2.0 * n * sizeof(float)) / (ms * 1e-3) / 1e9;

            printf("  %-15s  %8.3f  %7d  %10.1f  %11.3e  %10s\n",
                   ks[t].name, ms, ks[t].passes, gbs, err, ok ? "yes" : "NO");
            if (t == 1) t_ref = ms;   // safe softmax is the baseline to beat
        }
        printf("  → online vs safe: %.2fx\n\n", t_ref / time_kernel(softmax_online, dx, dy, rows, cols));
    }

    free(hx); free(hy); free(ref);
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    return 0;
}
