#include <time.h>
#include <stdio.h>

#define RADIUS        300
#define NUM_ELEMENTS  1e6

static void handleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }	
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))

__global__ void stencil_1d(float *in, float *out) {
    int tid = blockIdx.x;
    if (tid < NUM_ELEMENTS) {
        out[tid] = in[tid];
        for (int j = 1; j <= RADIUS; j++) {
            if (tid - j >= 0)
                out[tid] += in[tid - j];
            if (tid + j < NUM_ELEMENTS)
                out[tid] += in[tid + j];
        }
    }
}

void cpu_stencil_1d(float *in, float *out) {
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        out[i] = in[i];
        for (int j = 1; j <= RADIUS; j++) {
            if (i - j >= 0)
                out[i] += in[i - j];
            if (i + j < NUM_ELEMENTS)
                out[i] += in[i + j];
        }
    }
}

int main() {
    float *in, *out, *hostIn, *hostOut, *devIn, *devOut;

    in = reinterpret_cast<float*>(malloc(NUM_ELEMENTS * sizeof(float)));
    out = reinterpret_cast<float*>(malloc(NUM_ELEMENTS * sizeof(float)));
    hostIn = reinterpret_cast<float*>(malloc(NUM_ELEMENTS * sizeof(float)));
    hostOut = reinterpret_cast<float*>(malloc(NUM_ELEMENTS * sizeof(float)));

    for (int i = 0; i < NUM_ELEMENTS; i++) {
        in[i] = hostIn[i] = (float)(rand()) / (float)(rand());
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    cudaMalloc((void**)&devIn, NUM_ELEMENTS * sizeof(float));
    cudaMalloc((void**)&devOut, NUM_ELEMENTS * sizeof(float));

    cudaMemcpy(devIn, hostIn, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    stencil_1d<<<NUM_ELEMENTS, 1>>>(devIn, devOut);

    cudaCheck(cudaPeekAtLastError());

    cudaMemcpy(hostOut, devOut, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop);
    printf("Total GPU execution time:  %3.1f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(devIn);
    cudaFree(devOut);

    struct timespec cpu_start, cpu_stop;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);

    cpu_stencil_1d(in, out);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
    double result = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3 + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;
    printf( "CPU execution time:  %3.1f ms\n", result);

    free(in);
    free(out);
    free(hostIn);
    free(hostOut);

    return 0;
}
