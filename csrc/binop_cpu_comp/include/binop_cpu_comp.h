#define ENCODE_BITS 32

void binary_gemm_cpu(THFloatTensor* a, THFloatTensor* b, THFloatTensor* c, int M, int N, int K);

void im2col_cpu(THFloatTensor* columns, THFloatTensor* input,
int kW, int kH, int dW, int dH, int padW, int padH,
int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight);

void encode_rows_cpu(THFloatTensor* input, THIntTensor* output);

void encode_cols_cpu(THFloatTensor* input, THIntTensor* output);

void BinaryConvolution_cpu(
THFloatTensor *input, THFloatTensor *output, THFloatTensor *weight, THFloatTensor *columns,
THFloatTensor *bias,
int kH, int kW, int dH, int dW, int padH, int padW
);