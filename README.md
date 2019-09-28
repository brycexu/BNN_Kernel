## Description

In theory, Binarized Neural Networks boast of x32 compression and x10 acceleration on GPUs.
However, all variables are stored and manipulated in the form of float-32 on PyTorch. In this case, if we want to implement 1-bit networks on PyTorch, we have to change the kernel.

This repository contains a wrapped CUDA kernel that supports both 1-bit storage and computation on PyTorch

## How to use

### check the environment
    GCC:     >=5.0
    PyTorch: <1.0
    PyThon:  >=3.5
    CUDA:    >=8.0

### build the kernel
    cd ./csrc/binop
    make
    
### check the API in binop_cuda.h and use
    import torch
    import binop
    ...

## References

### How to wrap a CUDA and use its functions in PyTorch:

[Pytorch Custom CUDA kernel Tutorial](https://github.com/chrischoy/pytorch-custom-cuda-tutorial)

## Examples

You can see the GPU example in [test](https://github.com/brycexu/BNN_Kernel/tree/master/GPU)

You can see the CPU example in [test](https://github.com/brycexu/BNN_Kernel/tree/master/CPU)
