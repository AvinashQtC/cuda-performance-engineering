#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define M 4096
#define N 4096
#define K 4096

// ─── Tile / register-tile sizes ───────────────────────────────────────────────
// TILE  : shared-memory tile edge (rows and cols covered per block per K-step)
// TM    : rows each thread owns  (1D tiling: TM rows × 1 col)
//                                (2D tiling: TM rows × TN cols)
// TN    : cols each thread owns in 2D tiling (also the float4 vector width)
// Block for 1D: (TILE, TILE/TM) = (32, 4)  → 128 threads
// Block for 2D: (TILE/TN, TILE/TM) = (8, 4) → 32 threads
#define TILE 32
#define TM   8
#define TN   4

// Timed launches per kernel (averaged).
#define REPS 10

// Peak specs used only for the theoretical lower-bound printout below.
// Defaults are NVIDIA L40 datasheet figures; edit these for your own GPU.
#define PEAK_TFLOPS 90.5
#define PEAK_BW_GBS 864.0

// ─── CPU reference ────────────────────────────────────────────────────────────
// O(N^3) triple loop. Only used when CPU_VERIFY is defined (feasible at small
// sizes only — at 4096^3 this takes minutes, so it is off by default).
void cpu_matmul(float *A, float *B, float *C, int m, int kk, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.f;
            for (int p = 0; p < kk; p++)
                sum += A[i*kk + p] * B[p*n + j];
            C[i*n + j] = sum;
        }
}

// ─── Naive kernel, coalesced (col from threadIdx.x) ───────────────────────────
// A warp (fixed threadIdx.y, threadIdx.x = 0..31) sweeps consecutive columns,
// so its reads of B and writes of C are contiguous → one transaction per warp.
__global__ void naive_matmul(float *A, float *B, float *C, int m, int kk, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int p = 0; p < kk; p++)
            sum += A[row*kk + p] * B[p*n + col];
        C[row*n + col] = sum;
    }
}

// ─── Naive kernel, uncoalesced (row from threadIdx.x) ─────────────────────────
// Same math, but now a warp sweeps consecutive *rows*. Its reads of A and writes
// of C land on 32 unrelated cache lines → 32 transactions per warp instead of 1.
// (Assumes M == N so grd_naive covers both mappings.)
__global__ void naive_matmul_uncoalesced(float *A, float *B, float *C, int m, int kk, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int p = 0; p < kk; p++)
            sum += A[row*kk + p] * B[p*n + col];
        C[row*n + col] = sum;
    }
}

// ─── Shared-memory tiled kernel ───────────────────────────────────────────────
__global__ void tiled_matmul(float *A, float *B, float *C, int m, int kk, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.f;

    for (int bk = 0; bk < kk; bk += TILE) {
        int a_col = bk + tx;
        As[ty][tx] = (row < m && a_col < kk) ? A[row*kk + a_col] : 0.f;

        int b_row = bk + ty;
        Bs[ty][tx] = (b_row < kk && col < n) ? B[b_row*n + col] : 0.f;

        __syncthreads();
        for (int p = 0; p < TILE; p++)
            sum += As[ty][p] * Bs[p][tx];
        __syncthreads();
    }

    if (row < m && col < n)
        C[row*n + col] = sum;
}

// ─── 1D register-tiled kernel ─────────────────────────────────────────────────
// Block shape : (TILE, TILE/TM) = (32, 4)  → 128 threads
// threadIdx.x : column inside the TILE-wide output block   [0 .. TILE-1]
// threadIdx.y : which group of TM rows this thread owns    [0 .. TILE/TM-1]
// Each thread:
//   • loads TM rows of As (column tx) and TM rows of Bs (column tx)
//   • accumulates TM partial sums in registers
// Shared memory: As[TILE][TILE], Bs[TILE][TILE]  (same layout as tiled kernel)
__global__ void one_D_TILE(float *A, float *B, float *C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx  = threadIdx.x;                        // col index inside tile
    int ty  = threadIdx.y;                        // row-group index
    int row = blockIdx.y * TILE + ty * TM;        // first global row for this thread
    int col = blockIdx.x * TILE + tx;             // global column

    float acc[TM] = {0.0f};

    for (int bk = 0; bk < K; bk += TILE) {
        // ── Load TM rows of A into As[ty*TM .. ty*TM+TM-1][tx] ──
        for (int i = 0; i < TM; i++) {
            int a_row = row + i;
            int a_col = bk + tx;
            As[ty*TM + i][tx] = (a_row < M && a_col < K) ? A[a_row*K + a_col] : 0.0f;
        }

        // ── Load TM rows of B into Bs[ty*TM .. ty*TM+TM-1][tx] ──
        for (int i = 0; i < TM; i++) {
            int b_row = bk + ty*TM + i;
            Bs[ty*TM + i][tx] = (b_row < K && col < N) ? B[b_row*N + col] : 0.0f;
        }

        __syncthreads();

        // ── Compute: hoist Bs[p][tx] out of the row loop ──
        for (int p = 0; p < TILE; p++) {
            float b_val = Bs[p][tx];              // one smem read reused TM times
            for (int i = 0; i < TM; i++)
                acc[i] += As[ty*TM + i][p] * b_val;
        }

        __syncthreads();
    }

    // ── Store TM results ──
    for (int i = 0; i < TM; i++) {
        int c_row = row + i;
        if (c_row < M && col < N)
            C[c_row*N + col] = acc[i];
    }
}

// ─── 2D register-tiled kernel ─────────────────────────────────────────────────
// Block shape : (TILE/TN, TILE/TM) = (8, 4)  → 32 threads
// threadIdx.x : col-group index   [0 .. TILE/TN-1]
// threadIdx.y : row-group index   [0 .. TILE/TM-1]
// Each thread owns a TM×TN sub-tile of the output.
//
// Shared memory loading mirrors the 1D kernel:
//   thread (ty, tx) fills As[ty*TM .. ty*TM+TM-1][tx*TN .. tx*TN+TN-1]
//   (the full TILE×TILE tile is covered collectively by all TILE/TN × TILE/TM threads)
//
// Padded to TILE+1 on the leading dimension to avoid shared-memory bank conflicts.
//
// Inner loop: for each k-step p, load regA[TM] from As column p,
//             load regB[TN] from Bs row p, then outer-product into acc[TM][TN].
__global__ void reg2D_TILE(float *A, float *B, float *C) {
    __shared__ float As[TILE+1][TILE];
    __shared__ float Bs[TILE][TILE+1];

    int tx  = threadIdx.x;                        // col-group index
    int ty  = threadIdx.y;                        // row-group index
    int row = blockIdx.y * TILE + ty * TM;        // first global row
    int col = blockIdx.x * TILE + tx * TN;        // first global col

    float acc[TM][TN] = {};

    for (int bk = 0; bk < K; bk += TILE) {
        // ── Load As: thread (ty,tx) fills a TM×TN block of As ──
        for (int i = 0; i < TM; i++) {
            int a_row = row + i;
            for (int j = 0; j < TN; j++) {
                int a_col = bk + tx*TN + j;
                As[ty*TM + i][tx*TN + j] =
                    (a_row < M && a_col < K) ? A[a_row*K + a_col] : 0.0f;
            }
        }

        // ── Load Bs: thread (ty,tx) fills a TM×TN block of Bs ──
        for (int i = 0; i < TM; i++) {
            int b_row = bk + ty*TM + i;
            for (int j = 0; j < TN; j++) {
                int b_col = col + j;
                Bs[ty*TM + i][tx*TN + j] =
                    (b_row < K && b_col < N) ? B[b_row*N + b_col] : 0.0f;
            }
        }

        __syncthreads();

        // ── Outer-product over TILE k-steps ──
        for (int p = 0; p < TILE; p++) {
            float regA[TM], regB[TN];

            // Load one column of As (this thread's TM rows) into registers
            for (int i = 0; i < TM; i++)
                regA[i] = As[ty*TM + i][p];

            // Load one row of Bs (this thread's TN cols) into registers
            for (int j = 0; j < TN; j++)
                regB[j] = Bs[p][tx*TN + j];

            // TM×TN outer product — pure register arithmetic
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    // ── Store TM×TN results ──
    for (int i = 0; i < TM; i++) {
        int c_row = row + i;
        for (int j = 0; j < TN; j++) {
            int c_col = col + j;
            if (c_row < M && c_col < N)
                C[c_row*N + c_col] = acc[i][j];
        }
    }
}

// ─── 2D register-tiled kernel, float4 vectorized global loads ────────────────
// Same output tiling as reg2D_TILE (TM×TN per thread, (TILE/TN, TILE/TM) block),
// but each thread pulls a 16-byte float4 out of global memory per row instead of
// TN separate scalar loads. As is stored transposed ([col][row]) so the float4
// write into shared memory lands contiguously.
// Assumes K and N are multiples of TN (=4) so the float4 loads stay in bounds.
__global__ void vect_reg2D_TILE(float *A, float *B, float *C) {
    __shared__ float As[TILE+1][TILE];
    __shared__ float Bs[TILE][TILE+1];

    int tx  = threadIdx.x;                        // col-group index
    int ty  = threadIdx.y;                        // row-group index
    int row = blockIdx.y * TILE + ty * TM;        // first global row
    int col = blockIdx.x * TILE + tx * TN;        // first global col

    float acc[TM][TN] = {};

    for (int bk = 0; bk < K; bk += TILE) {
        // ── Load As: thread (ty,tx) fills a TM×TM block of As ──
        for (int i = 0; i < TM; i++) {
            int a_row = row + i;
            int a_col = bk + tx*TN;

            float4 a_vec =  *((float4*)(A + a_row*K + a_col));

            As[tx*TN+0][ty*TM+i] = a_vec.x;
            As[tx*TN+1][ty*TM+i] = a_vec.y;
            As[tx*TN+2][ty*TM+i] = a_vec.z;
            As[tx*TN+3][ty*TM+i] = a_vec.w;
        }

        // ── Load Bs: thread (ty,tx) fills a TM×TM block of Bs ──
        for (int i = 0; i < TM; i++) {
            int b_row = bk + ty*TM + i;

            float4 b_vec = *((float4*)(B + b_row*N + col));

            Bs[ty*TM+i][tx*TN+0] = b_vec.x;
            Bs[ty*TM+i][tx*TN+1] = b_vec.y;
            Bs[ty*TM+i][tx*TN+2] = b_vec.z;
            Bs[ty*TM+i][tx*TN+3] = b_vec.w;
        }

        __syncthreads();

        // ── Outer-product over TILE k-steps ──
        for (int p = 0; p < TILE; p++) {
            float regA[TM], regB[TN];

            // Load one column of As (this thread's TM rows) into registers
            for (int i = 0; i < TM; i++)
                regA[i] = As[p][ty*TM + i];

            // Load one row of Bs (this thread's TN cols) into registers
            for (int j = 0; j < TN; j++)
                regB[j] = Bs[p][tx*TN + j];

            // TM×TN outer product — pure register arithmetic
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    // ── Store TM×TN results ──
    for (int i = 0; i < TM; i++) {
        int c_row = row + i;
        for (int j = 0; j < TN; j++) {
            int c_col = col + j;
            if (c_row < M && c_col < N)
                C[c_row*N + c_col] = acc[i][j];
        }
    }
}

// ─── 2D register-tiled kernel, float4 loads + cp.async double buffering ──────
// Same TM×TN output tiling as above, but the shared-memory tiles are double
// buffered (As/Bs[2][...]) so the load for k-step bk+TILE is issued via
// cp.async while the compute for k-step bk is still running, overlapping the
// memory and FMA pipelines instead of separating them with __syncthreads().
//
// cp.async is an Ampere (compute capability 8.0+) instruction. On older GPUs the
// device code below is compiled as a no-op stub so the file still builds; main()
// only launches this kernel when the device reports sm_80 or newer.
//
// Assumptions:
// TILE, TM, TN, M, N, K are compile-time constants or macros.
// Recommended: TN = 4 for float4 vectorized loads.
// A is M x K, B is K x N, C is M x N, all row-major.
__global__ void vect_reg2D_TILE_cp_async(float *A, float *B, float *C)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
    __shared__ float As[2][TILE+1][TILE];
    __shared__ float Bs[2][TILE][TILE+1];

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;

    int row = blockIdx.y * TILE + ty * TM;
    int col = blockIdx.x * TILE + tx * TN;

    float acc[TM][TN] = {};

    int curr = 0;
    int next = 1;

    // ============================================================
    // Preload first tile: bk = 0 into buffer curr
    // ============================================================

    for (int i = 0; i < TM; i++)
    {
        int a_row = row + i;
        int a_col = tx * TN;

        if (a_row < M && a_col + 3 < K)
        {
            float *smem_ptr = &As[curr][ty * TM + i][tx * TN];
            float *gmem_ptr = A + a_row * K + a_col;

            unsigned smem =
                static_cast<unsigned>(
                    __cvta_generic_to_shared(smem_ptr));

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"(smem), "l"(gmem_ptr)
            );
        }
    }

    for (int i = 0; i < TM; i++)
    {
        int b_row = ty * TM + i;

        if (b_row < K && col + 3 < N)
        {
            float *smem_ptr = &Bs[curr][ty * TM + i][tx * TN];
            float *gmem_ptr = B + b_row * N + col;

            unsigned smem =
                static_cast<unsigned>(
                    __cvta_generic_to_shared(smem_ptr));

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"(smem), "l"(gmem_ptr)
            );
        }
    }

    // First tile must be fully ready before entering main loop.
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);

    __syncthreads();

    // ============================================================
    // Main loop
    // ============================================================

    for (int bk = 0; bk < K; bk += TILE)
    {
        int next_bk = bk + TILE;

        // ========================================================
        // 1. Issue async preload for next tile into buffer next
        // ========================================================

        if (next_bk < K)
        {
            for (int i = 0; i < TM; i++)
            {
                int a_row = row + i;
                int a_col = next_bk + tx * TN;

                if (a_row < M && a_col + 3 < K)
                {
                    float *smem_ptr =
                        &As[next][ty * TM + i][tx * TN];

                    float *gmem_ptr =
                        A + a_row * K + a_col;

                    unsigned smem =
                        static_cast<unsigned>(
                            __cvta_generic_to_shared(smem_ptr));

                    asm volatile(
                        "cp.async.cg.shared.global [%0], [%1], 16;\n"
                        :
                        : "r"(smem), "l"(gmem_ptr)
                    );
                }
            }

            for (int i = 0; i < TM; i++)
            {
                int b_row = next_bk + ty * TM + i;

                if (b_row < K && col + 3 < N)
                {
                    float *smem_ptr =
                        &Bs[next][ty * TM + i][tx * TN];

                    float *gmem_ptr =
                        B + b_row * N + col;

                    unsigned smem =
                        static_cast<unsigned>(
                            __cvta_generic_to_shared(smem_ptr));

                    asm volatile(
                        "cp.async.cg.shared.global [%0], [%1], 16;\n"
                        :
                        : "r"(smem), "l"(gmem_ptr)
                    );
                }
            }

            asm volatile("cp.async.commit_group;\n" ::);
        }

        // ========================================================
        // 2. Wait BEFORE computing current tile
        // ========================================================
        //
        // If next_bk < K:
        //   We allow one newest group to remain in-flight.
        //   That newest group is the next tile.
        //
        // If this is the last tile:
        //   There is no future tile, so wait_group 0 is safe.
        // ========================================================

        if (next_bk < K)
        {
            asm volatile("cp.async.wait_group 1;\n" ::);
        }
        else
        {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }

        __syncthreads();

        // ========================================================
        // 3. Compute current tile
        // ========================================================

        for (int p = 0; p < TILE; p++)
        {
            float regA[TM];
            float regB[TN];

            for (int i = 0; i < TM; i++)
            {
                regA[i] = As[curr][ty * TM + i][p];
            }

            for (int j = 0; j < TN; j++)
            {
                regB[j] = Bs[curr][p][tx * TN + j];
            }

            for (int i = 0; i < TM; i++)
            {
                for (int j = 0; j < TN; j++)
                {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        // ========================================================
        // 4. Swap buffers
        // ========================================================

        curr ^= 1;
        next ^= 1;
    }

    // ============================================================
    // Store result
    // ============================================================

    for (int i = 0; i < TM; i++)
    {
        int c_row = row + i;

        for (int j = 0; j < TN; j++)
        {
            int c_col = col + j;

            if (c_row < M && c_col < N)
            {
                C[c_row * N + c_col] = acc[i][j];
            }
        }
    }
#else
    // cp.async (Ampere, sm_80+) is unavailable on this architecture.
    (void)A; (void)B; (void)C;
#endif
}

// ─── Timing helper ────────────────────────────────────────────────────────────
float time_kernel(void (*launch)(float*, float*, float*, int, int, int, dim3, dim3),
                  float *dA, float *dB, float *dC,
                  dim3 grid, dim3 block, int reps) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    launch(dA, dB, dC, M, K, N, grid, block);   // warmup
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < reps; i++)
        launch(dA, dB, dC, M, K, N, grid, block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / reps;
}

// No-arg wrappers for the kernels that use compile-time M/N/K
void launch_naive(float *A, float *B, float *C, int m, int kk, int n,
                  dim3 grid, dim3 block) { naive_matmul<<<grid,block>>>(A,B,C,m,kk,n); }
void launch_naive_uncoalesced(float *A, float *B, float *C, int m, int kk, int n,
                  dim3 grid, dim3 block) { naive_matmul_uncoalesced<<<grid,block>>>(A,B,C,m,kk,n); }
void launch_tiled(float *A, float *B, float *C, int m, int kk, int n,
                  dim3 grid, dim3 block) { tiled_matmul<<<grid,block>>>(A,B,C,m,kk,n); }
void launch_1d(float *A, float *B, float *C, int, int, int,
               dim3 grid, dim3 block)   { one_D_TILE<<<grid,block>>>(A,B,C); }
void launch_2d(float *A, float *B, float *C, int, int, int,
               dim3 grid, dim3 block)   { reg2D_TILE<<<grid,block>>>(A,B,C); }
void launch_vect2d(float *A, float *B, float *C, int, int, int,
                    dim3 grid, dim3 block) { vect_reg2D_TILE<<<grid,block>>>(A,B,C); }
void launch_cpasync(float *A, float *B, float *C, int, int, int,
                    dim3 grid, dim3 block) { vect_reg2D_TILE_cp_async<<<grid,block>>>(A,B,C); }

// Relative-tolerance comparison. At K = 4096 each output sums 4096 terms, so
// kernels that accumulate in different orders differ in the low bits — an
// absolute tolerance would false-fail. A real bug produces order-1 relative
// error (or NaN), which this still catches.
bool verify(float *ref, float *got, int size, float rtol) {
    for (int i = 0; i < size; i++) {
        float diff  = fabsf(ref[i] - got[i]);
        float denom = fabsf(ref[i]) + 1e-6f;
        if (diff / denom > rtol) {
            printf("  Mismatch at %d: ref=%.4f got=%.4f\n", i, ref[i], got[i]);
            return false;
        }
    }
    return true;
}

// Theoretical lower bounds for this problem size (hardware-independent counts,
// plus time floors against the PEAK_* specs above).
void print_lower_bounds() {
    double flops       = 2.0 * (double)M * N * K;
    double min_bytes   = ((double)M*K + (double)K*N + (double)M*N) * sizeof(float);
    double naive_bytes = 2.0 * (double)M * N * K * sizeof(float);
    double comp_time   = flops     / (PEAK_TFLOPS * 1e12);   // seconds
    double mem_time    = min_bytes / (PEAK_BW_GBS  * 1e9);   // seconds

    printf("--- Theoretical lower bounds (peak %.1f TFLOP/s, %.0f GB/s) ---\n",
           PEAK_TFLOPS, PEAK_BW_GBS);
    printf("  FLOPs (2*M*N*K)      : %8.2f GFLOP\n", flops / 1e9);
    printf("  Min memory traffic   : %8.2f MB     (read A,B + write C once)\n",
           min_bytes / 1e6);
    printf("  Compute-bound floor  : %8.3f ms\n", comp_time * 1e3);
    printf("  Memory-bound floor   : %8.3f ms\n", mem_time * 1e3);
    printf("  => %s: compute is %.1fx the memory floor\n",
           comp_time > mem_time ? "compute-bound" : "memory-bound",
           comp_time / mem_time);
    printf("  Naive kernel traffic : %8.2f GB     (~%.0fx the floor)\n\n",
           naive_bytes / 1e9, naive_bytes / min_bytes);
}

int main() {
    const int reps = REPS;
    size_t sA = (size_t)M*K*sizeof(float),
           sB = (size_t)K*N*sizeof(float),
           sC = (size_t)M*N*sizeof(float);

    float *hA     = (float*)malloc(sA);
    float *hB     = (float*)malloc(sB);
    float *hC_ref = (float*)malloc(sC);
    float *hC_gpu = (float*)malloc(sC);

    for (int i = 0; i < M*K; i++) hA[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < K*N; i++) hB[i] = (float)rand()/RAND_MAX;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sA); cudaMalloc(&dB, sB); cudaMalloc(&dC, sC);
    cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice);

    dim3 blk_naive(32, 32);
    dim3 grd_naive((N+31)/32, (M+31)/32);

    dim3 blk_tiled(TILE, TILE);
    dim3 grd_tiled((N+TILE-1)/TILE, (M+TILE-1)/TILE);

    dim3 blk_1d(TILE, TILE/TM);
    dim3 grd_1d((N+TILE-1)/TILE, (M+TILE-1)/TILE);

    dim3 blk_2d(TILE/TN, TILE/TM);
    dim3 grd_2d((N+TILE-1)/TILE, (M+TILE-1)/TILE);

    dim3 blk_vect2d(TILE/TN, TILE/TM);
    dim3 grd_vect2d((N+TILE-1)/TILE, (M+TILE-1)/TILE);

    dim3 blk_cpasync(TILE/TN, TILE/TM);
    dim3 grd_cpasync((N+TILE-1)/TILE, (M+TILE-1)/TILE);

    // cp.async needs Ampere (sm_80+); skip it at runtime on older GPUs.
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    bool has_cp_async = (prop.major >= 8);

    printf("\n=== Matmul benchmark  %dx%dx%d  (avg over %d runs) ===\n", M, K, N, reps);
    printf("GPU: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    print_lower_bounds();

    // ── Reference result ──
    // Default: run the naive GPU kernel and use its output as the reference for
    // every other kernel. Define CPU_VERIFY to also validate naive against the
    // CPU triple loop (only feasible at small M/N/K).
    launch_naive(dA,dB,dC, M,K,N, grd_naive, blk_naive);
    cudaDeviceSynchronize();
    cudaMemcpy(hC_ref, dC, sC, cudaMemcpyDeviceToHost);

    bool naive_ok = true;
#ifdef CPU_VERIFY
    {
        float *hC_cpu = (float*)malloc(sC);
        cpu_matmul(hA, hB, hC_cpu, M, K, N);
        naive_ok = verify(hC_cpu, hC_ref, M*N, 1e-2f);
        free(hC_cpu);
    }
#endif

    // ── Stage 1: coalescing comparison (uncoalesced vs coalesced naive) ──
    float t_unco = time_kernel(launch_naive_uncoalesced, dA,dB,dC, grd_naive, blk_naive, reps);
    cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost);
    bool unco_ok = verify(hC_ref, hC_gpu, M*N, 1e-2f);
    float t_co   = time_kernel(launch_naive,             dA,dB,dC, grd_naive, blk_naive, reps);

    printf("--- Coalescing (Stage 1), block (32,32) ---\n");
    printf("  naive uncoalesced (row from threadIdx.x) : %8.3f ms   %s\n",
           t_unco, unco_ok ? "correct" : "WRONG");
    printf("  naive coalesced   (col from threadIdx.x) : %8.3f ms\n", t_co);
    printf("  => coalescing speedup                    : %7.2fx\n\n", t_unco / t_co);

    // ── Full progression ──
    double flops = 2.0 * (double)M * N * K;
    struct { const char *name; float t; bool ok; bool run; } res[6];

    res[0] = {"Naive", t_co, naive_ok, true};   // reuse the coalesced timing above

    { float t = time_kernel(launch_tiled, dA,dB,dC, grd_tiled, blk_tiled, reps);
      cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost);
      res[1] = {"Tiled-smem", t, verify(hC_ref, hC_gpu, M*N, 1e-2f), true}; }

    { float t = time_kernel(launch_1d,    dA,dB,dC, grd_1d,    blk_1d,    reps);
      cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost);
      res[2] = {"Reg1D",      t, verify(hC_ref, hC_gpu, M*N, 1e-2f), true}; }

    { float t = time_kernel(launch_2d,    dA,dB,dC, grd_2d,    blk_2d,    reps);
      cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost);
      res[3] = {"Reg2D",      t, verify(hC_ref, hC_gpu, M*N, 1e-2f), true}; }

    { float t = time_kernel(launch_vect2d, dA,dB,dC, grd_vect2d, blk_vect2d, reps);
      cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost);
      res[4] = {"Vect2D",     t, verify(hC_ref, hC_gpu, M*N, 1e-2f), true}; }

    if (has_cp_async) {
        float t = time_kernel(launch_cpasync, dA,dB,dC, grd_cpasync, blk_cpasync, reps);
        cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost);
        res[5] = {"CpAsync2D", t, verify(hC_ref, hC_gpu, M*N, 1e-2f), true};
    } else {
        res[5] = {"CpAsync2D", 0.f, false, false};
    }

    printf("%-14s  %8s   %14s   %s\n",
           "Kernel", "ms", "GFLOP/s", "Correct");
    printf("%-14s  %8s   %14s   %s\n",
           "──────────────", "────────", "──────────────", "───────");
    for (auto &r : res) {
        if (!r.run) {
            printf("%-14s  %8s   %14s   %s\n", r.name, "-", "-", "skipped (needs sm_80)");
            continue;
        }
        double gf = (flops / (r.t * 1e-3)) / 1e9;
        printf("%-14s  %8.3f   %14.2f   %s\n",
               r.name, r.t, gf, r.ok ? "yes" : "NO");
    }
    printf("\nSpeedup vs Naive:\n");
    for (int i = 1; i < 6; i++)
        if (res[i].run)
            printf("  %-14s  %.2fx\n", res[i].name, res[0].t / res[i].t);
    printf("\n");

    free(hA); free(hB); free(hC_ref); free(hC_gpu);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
