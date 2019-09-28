#include <TH/TH.h>
#include <stdio.h>
#include <stdint.h>
#include "binop_cpu_comp_kernel.h"
#include "matmul.h"

void binary_gemm_cpu(THFloatTensor* a, THFloatTensor* b, THFloatTensor* c, int M, int N, int K) {
    if (c->nDimension != 2 || c->size[0]*c->size[1] < M*K) {
        THFloatTensor_resize2d(c, M, K);
    }
    float *A = THFloatTensor_data(a);
    float *B = THFloatTensor_data(b);
    float *C = THFloatTensor_data(c);
    dgemm_nn(M, K, N, A, N, 1, B, K, 1, C, K, 1);
}

void encode_rows_cpu(THFloatTensor* input, THIntTensor* output) {
    int m = input->size[0];
    int n = input->size[1];
    int l = 1+(n-1)/ENCODE_BIT;

    THIntTensor_resize2d(output, m, l);
    float* a = THFloatTensor_data(input);
    uint32_t* b = (uint32_t*)THIntTensor_data(output);

    encode_rows_cpu_kernel(a, b, m, n);
}

void encode_cols_cpu(THFloatTensor* input, THIntTensor* output) {
    int n = input->size[0];
    int k = input->size[1];
    int l = 1+(n-1)/ENCODE_BIT;

    THIntTensor_resize2d(output, l, k);
    float* a = THFloatTensor_data(input);
    uint32_t* b = (uint32_t*)THIntTensor_data(output);

    encode_cols_cpu_kernel(a, b, n, k);
}

void im2col_cpu(THFloatTensor* columns, THFloatTensor* input,
int kW, int kH, int dW, int dH, int padW, int padH,
int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    int64_t k;
    float *input_data = THFloatTensor_data(input);
    float *columns_data = THFloatTensor_data(columns);

#pragma omp parallel for private(k)
    for(k = 0; k < (int64_t)nInputPlane*kH*kW; k++) {
        int64_t nip = k / (kH*kW);
        int64_t rest = k % (kH*kW);
        int64_t kh = rest / kW;
        int64_t kw = rest % kW;
        int x, y;
        int64_t ix, iy;
        float *dst = columns_data + nip*((size_t)kH*kW*outputHeight*outputWidth) + kh*((size_t)kW*outputHeight*outputWidth) + kw*((size_t)outputHeight*outputWidth);
        float *src = input_data + nip*((size_t)inputHeight*inputWidth);
        if (padW > 0 || padH > 0) {
            int64_t lpad,rpad;
            for(y = 0; y < outputHeight; y++) {
                iy = (int64_t)y*dH - padH + kh;
                if (iy < 0 || iy >= inputHeight) {
                    memset(dst+(size_t)y*outputWidth, 0, sizeof(float)*outputWidth);
                } else {
                    if (dW==1){
                        ix = 0 - padW + kw;
                        lpad = fmaxf(0,padW-kw);
                        rpad = fmaxf(0,padW-(kW-kw-1));
                        if (outputWidth-rpad-lpad <= 0) {
                            memset(dst+(size_t)y*outputWidth, 0, sizeof(float)*outputWidth);
                        } else {
                            if (lpad > 0) memset(dst+(size_t)y*outputWidth, 0, sizeof(float)*lpad);
                            memcpy(dst+(size_t)y*outputWidth+lpad, src+(size_t)iy*inputWidth+ix+lpad, sizeof(float)*(outputWidth-rpad-lpad));
                            if (rpad > 0) memset(dst+(size_t)y*outputWidth + outputWidth - rpad, 0, sizeof(float)*rpad);
                        }
                    }
                    else{
                        for (x=0; x<outputWidth; x++){
                            ix = (int64_t)x*dW - padW + kw;
                            if (ix < 0 || ix >= inputWidth)
                                memset(dst+(size_t)y*outputWidth+x, 0, sizeof(float)*1);
                            else
                                memcpy(dst+(size_t)y*outputWidth+x, src+(size_t)iy*inputWidth+ix, sizeof(float)*(1));
                        }
                    }
                }
            }
        } else {
            for(y = 0; y < outputHeight; y++) {
                iy = (int64_t)y*dH + kh;
                ix = 0 + kw;
                if (dW == 1)
                    memcpy(dst+(size_t)y*outputWidth, src+(size_t)iy*inputWidth+ix, sizeof(float)*outputWidth);
                else{
                    for (x=0; x<outputWidth; x++)
                        memcpy(dst+(size_t)y*outputWidth+x, src+(size_t)iy*inputWidth+ix+(int64_t)x*dW, sizeof(float)*(1));
                }
            }
        }
    }
}

static void BinaryConvolution_cpu_helper(
    THFloatTensor *output, THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *ones, THFloatTensor *columns_t,
    int kW, int kH, int dW, int dH, int padW, int padH,
    int64_t nInputPlane, int64_t inputWidth, int64_t inputHeight,
    int64_t nOutputPlane, int64_t outputWidth, int64_t outputHeight) {
    THFloatTensor *output2d;
    output2d = THFloatTensor_newWithStorage2d(output->storage, output->storageOffset, nOutputPlane, -1, outputHeight*outputWidth, -1);
    THFloatTensor_zero(output2d);
    binary_gemm_cpu(weight, columns_t, output2d, nOutputPlane, kW*kH*nInputPlane, outputHeight*outputWidth);
    if (bias->nDimension) {
        THFloatTensor_addmm(output2d, 1, output2d, 1, bias, ones);
    }
    THFloatTensor_free(output2d);
}

void BinaryConvolution_cpu(
            THFloatTensor *input,
            THFloatTensor *output,
            THFloatTensor *weight,
            THFloatTensor *columns,
            THFloatTensor *bias,
            int kH, int kW,
            int dH, int dW,
            int padH, int padW) {

    input = THFloatTensor_newContiguous(input);
    THIntTensor *bin_col = THIntTensor_new();
    THFloatTensor *ones  = THFloatTensor_new();

    int ndim = input->nDimension;
    int dimf = 0;
    int dimh = 1;
    int dimw = 2;

    if (ndim == 4) {
        dimf++;
        dimh++;
        dimw++;
    }

    int64_t nInputPlane  = input->size[dimf];
    int64_t inputHeight  = input->size[dimh];
    int64_t inputWidth   = input->size[dimw];
    int64_t nOutputPlane = weight->size[0];
    int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
    int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

    if (bias->nDimension ==1) {
        THFloatTensor_resize2d(bias, bias->size[0], 1);
    }

    THFloatTensor_resize2d(ones, 1, outputHeight*outputWidth);
    THFloatTensor_fill(ones, 1);

    int64_t T = input->size[0];
    int64_t t;

    THFloatTensor_resize4d(output, T, nOutputPlane, outputHeight, outputWidth);
    THFloatTensor_resize3d(columns, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THIntTensor_resize3d(bin_col, T, weight->size[0], outputHeight*outputWidth);
#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
        THFloatTensor *input_t = THFloatTensor_newSelect(input, 0, t);
        THFloatTensor *columns_t = THFloatTensor_newSelect(columns, 0, t);
        THIntTensor *bin_col_t = THIntTensor_newSelect(bin_col, 0, t);

        im2col_cpu(
            columns_t, input_t, kW, kH, dW, dH, padW, padH,
            nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight
        );

        THFloatTensor_free(input_t);
        THFloatTensor_free(columns_t);
        THIntTensor_free(bin_col_t);
    }

    for(t = 0; t < T; t++){
        THFloatTensor *output_t = THFloatTensor_newSelect(output, 0, t);
        THFloatTensor *columns_t = THFloatTensor_newSelect(columns, 0, t);

        BinaryConvolution_cpu_helper(
            output_t, weight, bias, ones, columns_t, kW, kH, dW, dH, padW, padH,
            nInputPlane, inputWidth, inputHeight, nOutputPlane, outputWidth, outputHeight
        );

        THFloatTensor_free(output_t);
        THFloatTensor_free(columns_t);
    }
    THFloatTensor_free(input);
    THFloatTensor_free(ones);
    THIntTensor_free(bin_col);
}