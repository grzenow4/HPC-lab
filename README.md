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
