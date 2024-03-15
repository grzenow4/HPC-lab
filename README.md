# HPC-lab
High Performance Computing laboratories combined.

## Lab 01
Modify the [mat_mult.cpp](lab01/mat_mult.cpp) file so that it contains a parallelised implementation of matrix multiplication. You need to fill in the `matrixMultParallel` function.

## Lab 02
Your task is to parallelize both compress and decompress functions, but mind the following:
1. Parallelization should be done inside the [dft.cpp](lab02/dft.cpp) file only. Do not parallelize the utilities.
1. Prepare two versions of `compress` parallelization. Do your best in the first one. In the second one just use the most naive atomic operations.
1. During decompression we require that only the master thread (`threadId == 0`) reads or writes the image data. You may still use other threads to compute sines / cosines values.
1. You may test your solution on the provided [example.bmp](lab02/example.bmp) bitmap or any other, but try to use similar dimensions.

## Lab 03
Implement the stencil operation on the CPU and GPU. 1D Stencil of radius `D > 1` is a function on a vector `X` to obtain a vector `Y` of the same size such that `Y[i] = X[i - D] + X[i - D + 1] + ... + X[i + D]`. Because `i - D` or `i + D` can exceed `X`'s length, you need to think how to solve this issue. In your kernel, have each thread handle one index. You can either use `threadIdx`, `blockIdx` or a combination of both.

Use the template in [lab03/stencil](lab03/stencil/).
