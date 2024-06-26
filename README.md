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

## Lab 04
In this exercise your task is to optimise a GPU-based raytracer by using a different type of memory than the global memory. Raytracing is one way of rendering a 2D image of objects in a 3D space. Typically, GPUs do this in a process of rasterization. Raytracing is an alternative to that, but it's much more computationally intensive. The idea is simple - we place a camera in a scene and think of the final image as a digital sensor in that camera. Then we shoot rays for each pixel in the sensor (perpendicular to it and parallel to each other) and depending on the hit location, draw an appropriate color, like so:

<center><img src="lab04/raytrace.png" width="663" height="437"></center>

Analyse the code in [lab04/raytrace/raytrace.cu](lab04/raytrace/raytrace.cu) and change it so it stores some of the data in one of the memory types mentioned in this lab. Compare the performance of the original version with your optimised version.

## Lab 05
Let's see how overlapping memory transfers and kernel executions can improve the performance of CUDA applications. In [lab5/streams/single.cu](lab05/streams/single.cu) you can find the code described in section 2 (it uses a single non-default stream). Modify it so it uses 2 streams instead - each stream should handle half of all memory transfers and kernel executions. Compare the results from both versions using NVIDIA Nsight Systems.

## Lab 06
Download the [transpose.cu](lab06/transpose.cu) file. It contains an incorrect and suboptimal implementation of matrix transpose on the GPU. Your task is to use CUDA-GDB to localise any bugs and fix them. Hint: look for bugs in the kernel code. Then, you have to profile the application and find performance bottlenecks. Hint - shared memory may be used as a buffer to better organize global memory accesses. Optimise the code and profile it again.

Image below shows how threads are assigned to the matrix elements:

<center><img src="lab06/transpose.png" width="400" height="511"></center>

## Lab 07
1. Organize `n` processes into a ring: rank `0` sends to rank `1`, rank `1` sends to rank `2`, rank `n−1` sends to rank `0`. A message contains a single `int64` number. First message is `1`; then each process receives the number, multiplies it by its current rank and sends its to the next rank. Rank `0` prints the received message.

1. Your goal is to compute the throughput (in MB/s) and the round-trip latency (in ms) on Okeanos and on the computers in our labs. For each, you should make `N` (e.g. 30) experiments, discard 1-2 minimal and maximal values and then average the remaining ones. For throughput, send large messages (millions of bytes); for latency, send short messages (1-10-100 bytes).

    After completing this, extend your code to compute throughput in function of the length of the message. You can use our jupyter notebook [draw-bandwidth.ipynb](lab07/draw-bandwidth.ipynb) to display results (the data format is: `experiment_sequence_number message_size communication_time`).

## Lab 08
1. Implement functions from [graph-utils-par.cpp](lab08/graph-utils-par.cpp) that (1) partition the matrix; and (2) write the matrix to std-out. Implement a partitioning in which the number of rows stored by each pair of processes differs by at most one. Use [graph-utils-seq.cpp](lab08/graph-utils-seq.cpp), a sequential version, as a template. [generator-par.cpp](lab08/generator-par.cpp) tests your implementation: this should give exactly the same result as [generator-seq.cpp](lab08/generator-seq.cpp).

    Assume that processes are memory-bound, i.e., no process can store more than O(1) blocks. Note that the matrix is generated by a stateful random number generator, thus only a single process should generate the matrix. Similarly, a single process should write the matrix to the std out. (problems to consider at home: How limiting these assumptions are for a large-scale system? Which methods can we use to gain more parallelism here?)

1. Implement the distributed Floyd-Warshall algorithm by completing fixmes in [floyd-warshall-par.cpp](lab08/floyd-warshall-par.cpp). You can use [floyd-warshall-seq.cpp](lab08/floyd-warshall-seq.cpp), a sequential implementation, as a template.

    Compute speed-ups (strong scaling) for different sizes of the graph and 1-16 nodes.

## Lab 09
1. Implement the first, working, distributed version by extending [laplace-par.cpp](lab09/laplace-par.cpp). You can either use synchronous, point-to-point communication or asynchronous communication. Compute speed-ups.
1. Use `MPI_Allreduce` to test the termination condition of the main loop. Is the program significantly faster?
1. Use asynchronous, point-to-point communication to overlap the computation of the "black" fields with the communication of the "white" borders (analogously for the second phase). (alternatively, if your implementation already uses asynchronous communication, reimplement it with synchronous communication). Does this communication-computation overlap make your program faster?
1. (optional) As our algorithm has low computational intensity, one possible optimization is to communicate less frequently by computing redundant boundaries (communication-avoiding algorithm).

## Lab 10
1. Implement your own matrix-matrix multiplication as a function in [blas-dmmmult.cpp](lab10/blas-dmmmult.cpp). Benchmark both versions (use a single node). Try to optimize your implementation using compiler hints.
1. Benchmark a distributed program (e.g., your homework from the previous lab) using `-tasks-per-node=24` (no hyperthreading) and `-tasks-per-node=48` (hyperthreading).
1. Introduce some performance bugs to the Floyd-Warshall code, e.g. a rank that computes more than other ranks. See how it influences the `app2` profiling.

## Lab 11
Use task-based parallelism to implement a parallel version of the n-queens problem: how to place n queens on an n x n checkerboard so that no two queens attack each other. A sequential solution is in [nqueens.cpp](lab11/nqueens.cpp).
- Extend the solution to a parallel version. Avoid the race condition when modifying `partial_board`.
- Start with a version that only counts the number of possible solutions (instead of generating all of them).
- Make sure you will get a decent speed-up compared to the sequential version (unlike our Fibonacci implementation). Think about the "grain size", or when to stop spawning new tasks.
- To return all solutions in a thread-safe way, use `tbb::concurrent_queue<board>`. Measure the performance degradation compared with just counting the solutions.

## Lab 12
1. Rewrite [nqueens.cpp](lab11/nqueens.cpp) from the previous lab to use `parallel_for_each`.
1. Apply `enumerable_thread_specific` to [nqueens.cpp](lab11/nqueens.cpp).
1. Solve the single source shortest path problem in a randomly-initialized graph. Use `tbb::concurrent_hash_map` (or `tbb::concurrent_unordered_map`) that maps a node to its distance from the source; and `tbb::concurrent_priority_queue` to keep a list of nodes to visit. See https://stackoverflow.com/questions/23501591/tbb-concurrent-hash-map-find-insert for example.
