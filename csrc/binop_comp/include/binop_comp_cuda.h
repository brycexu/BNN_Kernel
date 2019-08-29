void binary_gemm(THCudaIntTensor* weight, THCudaIntTensor* columns_binary, THCudaTensor* output_n, int m, int nn, int k);

void im2col(THCudaTensor* data_im, int channels, int height, int width, int ksize_h, int ksize_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, THCudaTensor* data_col);

void encode_rows(THCudaTensor* input, THCudaIntTensor* output);

void encode_cols(THCudaTensor* input, THCudaIntTensor* output);

void BinaryConvolution(
THCudaTensor *input, THCudaTensor *output, THCudaTensor *weight, THCudaTensor *columns,
THCudaTensor *bias, int nInputPlane,
int kH, int kW, int sH, int sW, int padH, int padW);
