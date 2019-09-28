#include <stdint.h>
#include <stdio.h>
#include "binop_comp_cuda_kernel.h"

int GET_BLOCKS(int N){
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void gemm_kernel(float* A, float* B, float* C, int m, int n, int k) {

    // Get the block location
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

    // Csub points to a specific area in C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float Cvalue = 0.0;

    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // A sub is the sub-matrix of A
        float* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Bsub is the sub-matrix of B
        float* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];

        __syncthreads();

        // Does Muliplication computation
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            Cvalue += As[row][j] * Bs[j][col];
        }

        __syncthreads();
    }

    // Assign Cvalue to Csub
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) {
        Csub[row*k+col] = Cvalue;
    }
}


__global__ void im2col_kernel(int n, float* data_im, int height, int width,
                              int ksize_h, int ksize_w, int pad_h, int pad_w,
                              int stride_h, int stride_w, int dilation_h, int dilation_w,
                              int height_col, int width_col, float* data_col) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize_h * ksize_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_h; ++i) {
      for (int j = 0; j < ksize_w; ++j) {
        int h = h_in + i * dilation_h;
        int w = w_in + j * dilation_w;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
          data_im[i * dilation_h * width + j * dilation_w] : 0;
        data_col += height_col * width_col;
      }
    }
  }
}

__forceinline__ __device__ uint32_t encode_val(float* array, int n) {
    uint32_t r = 0;
    for(int i=0; i<ENCODE_BITS && i<n; i++){
        r |= (array[i]>0)<<i;
    }
    return r;
}

__global__ void encode_rows_kernel(float *input, uint32_t* output, int m, int n, int l) {// l = 1+(n-1)/ENCODE_BITS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = n*(i/l)+ENCODE_BITS*(i%l);
    if (i<m*l) output[i] = encode_val(&input[p], n-ENCODE_BITS*(i%l));
}

__global__ void encode_cols_kernel(float *a, uint32_t* b, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int i32 = i*ENCODE_BITS;
    if (j < n && i32 < m) {
        uint32_t r = 0;
        for(int k = 0; j + n * (i32 + k)< m * n && k < ENCODE_BITS; k++){
            r |= (a[j + n * (i32 + k)]>0)<<k;
		}
        b[j + n * i] = r;
    }
}

void binary_gemm_cuda(uint32_t* A, uint32_t* B, float* C, int m, int n, int k, cudaStream_t stream){
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(k/BLOCK_SIZE+1 , m/BLOCK_SIZE+1);
    binary_gemm_kernel <<< gridDim, blockDim, 0, stream >>>(A, B, C, m, n, k);
}

void gemm_cuda(float* A, float* B, float* C, int m, int n, int k, cudaStream_t stream){
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(k/BLOCK_SIZE+1 , m/BLOCK_SIZE+1);
    gemm_kernel <<< gridDim, blockDim, 0, stream >>>(A, B, C, m, n, k);
}

void im2col_cuda(int n, float* data_im, int height, int width,
                 int ksize_h, int ksize_w, int pad_h, int pad_w,
                 int stride_h, int stride_w, int dilation_h, int dilation_w,
                 int height_col, int width_col, float* data_col, cudaStream_t stream){
    im2col_kernel <<< GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream >>> (
        n, data_im, height, width, ksize_h, ksize_w,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w,
        height_col, width_col, data_col
    );
}

void encode_rows_cuda(float* input, uint32_t* output, int m, int n, int l, cudaStream_t stream) {
    encode_rows_kernel <<< GET_BLOCKS(m*l), CUDA_NUM_THREADS, 0, stream >>>(input, output, m, n, l);
}

void encode_cols_cuda(float* input, uint32_t* output, int n, int k, cudaStream_t stream) {
    dim3 blockDim(ENCODE_BITS, ENCODE_BITS, 1);
    dim3 gridDim(k/ENCODE_BITS+1, n/ENCODE_BITS+1, 1);

    encode_cols_kernel <<< gridDim, blockDim, 0, stream >>>(input, output, n, k);
}



