// Online softmax (Milakov & Gimelshein, 2018) — simple version.
// One thread per row, so no block reductions are needed.
//
// Build:  nvcc -O3 -arch=native -o sofmax_1 sofmax_1.cu
// Run:    ./sofmax_1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

#define M 4096      // rows
#define N 4096      // softmax is taken along each row
#define BLOCK 256

// ─── Algorithm 1: naive ──────────────────────────────────────────────────────
// Loop 1: d = sum of e^x.  Loop 2: y = e^x / d.  Two reads of x.
// e^x overflows to inf for x > ~88.7f, so this one is not safe.
template <typename T>
__global__ void naive_softmax(T *x, T *y, int M_, int N_){

    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M_)
    {
        float d = 0;
        for(int i = 0; i < N_; i++){
            int el = row*N_ + i;
            d += expf(static_cast<float>(x[el]));
        }

        for(int i = 0; i < N_; i++){
            int el = row*N_ + i;
            y[el] = static_cast<T>(expf(static_cast<float>(x[el])) / d);
        }
    }
}

// ─── Algorithm 2: safe ───────────────────────────────────────────────────────
// Loop 1: max.  Loop 2: d = sum of e^(x - max).  Loop 3: y = e^(x - max) / d.
// Three reads of x — that extra pass is the price of not overflowing.
template <typename T>
__global__ void safe_softmax(T *x, T *y, int M_, int N_){

    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M_)
    {
        T x_max = -INFINITY;

        for(int i = 0; i < N_; i++){
            int el = row*N_ + i;
            x_max = max(x_max, x[el]);
        }

        float d = 0;
        for(int i = 0; i < N_; i++){
            int el = row*N_ + i;
            d += expf(static_cast<float>(x[el] - x_max));
        }

        for(int i = 0; i < N_; i++){
            int el = row*N_ + i;
            float x_i = expf(static_cast<float>(x[el] - x_max));
            y[el] = static_cast<T>(x_i / d);
        }
    }
}

// ─── Algorithm 3: online ─────────────────────────────────────────────────────
// Loop 1: max and d in one go — whenever the max grows, rescale d to match.
// Loop 2: y = e^(x - max) / d.  Two reads of x, and still safe.
template <typename T>
__global__ void online_softmax(T *x, T *y, int M_, int N_){
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M){
        T x_max = -INFINITY;
        float d = 0;
        for(int i = 0; i<N; i++){
            int el = row*N + i;
            int x_max_old = x_max;
            x_max = max(x_max, x[el]);
            d = d*expf(static_cast<float>(x_max_old - x_max)) + expf(static_cast<float>(x[el]-x_max));
        }
        for(int i = 0; i < N_; i++){
            int el = row*N_ + i;
            float x_i = expf(static_cast<float>(x[el] - x_max));
            y[el] = static_cast<T>(x_i / d);
        }

    }
}
// ─── CPU call function ───────────────────────────────────────────────────────
// Takes ordinary host pointers: allocates on the device, copies in, launches,
// copies back, frees. Returns the kernel time in ms (copies not counted).

typedef void (*kernel_fn)(float *, float *, int, int);

float softmax_gpu(kernel_fn kernel, float *h_x, float *h_y, int rows, int cols){

    size_t bytes = (size_t)rows * cols * sizeof(float);

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    int grid = (rows + BLOCK - 1) / BLOCK;

    kernel<<<grid, BLOCK>>>(d_x, d_y, rows, cols);   // warmup
    cudaDeviceSynchronize();

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    cudaEventRecord(beg);
    kernel<<<grid, BLOCK>>>(d_x, d_y, rows, cols);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, beg, end);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    cudaFree(d_x);
    cudaFree(d_y);
    return ms;
}

// ─── CPU reference ───────────────────────────────────────────────────────────

void softmax_cpu(float *x, float *y, int rows, int cols){
    for(int row = 0; row < rows; row++){
        float x_max = -INFINITY;
        for(int i = 0; i < cols; i++) x_max = fmaxf(x_max, x[row*cols + i]);

        float d = 0;
        for(int i = 0; i < cols; i++) d += expf(x[row*cols + i] - x_max);

        for(int i = 0; i < cols; i++)
            y[row*cols + i] = expf(x[row*cols + i] - x_max) / d;
    }
}

float max_error(float *a, float *b, size_t n){
    float e = 0;
    for(size_t i = 0; i < n; i++){
        if (isnan(b[i]) || isinf(b[i])) return INFINITY;
        e = fmaxf(e, fabsf(a[i] - b[i]));
    }
    return e;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(void){

    size_t n     = (size_t)M * N;
    size_t bytes = n * sizeof(float);

    float *x   = (float *)malloc(bytes);
    float *y   = (float *)malloc(bytes);
    float *ref = (float *)malloc(bytes);

    for(size_t i = 0; i < n; i++) x[i] = -10.f + 20.f * ((float)rand() / RAND_MAX);
    softmax_cpu(x, ref, M, N);

    printf("\nsoftmax over %d rows x %d cols\n\n", M, N);
    printf("%-8s  %8s  %7s  %12s\n", "kernel", "ms", "reads", "max err");
    printf("%-8s  %8s  %7s  %12s\n", "────────", "────────", "───────", "────────────");

    float t;
    t = softmax_gpu(naive_softmax<float>,  x, y, M, N);
    printf("%-8s  %8.3f  %7d  %12.3e\n", "naive",  t, 2, max_error(ref, y, n));

    t = softmax_gpu(safe_softmax<float>,   x, y, M, N);
    printf("%-8s  %8.3f  %7d  %12.3e\n", "safe",   t, 3, max_error(ref, y, n));

    t = softmax_gpu(online_softmax<float>, x, y, M, N);
    printf("%-8s  %8.3f  %7d  %12.3e\n", "online", t, 2, max_error(ref, y, n));

    printf("\n");
    free(x); free(y); free(ref);
    return 0;
}
