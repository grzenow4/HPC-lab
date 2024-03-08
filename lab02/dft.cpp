#include <complex>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "utils/bmp.cpp"

void compress(const uint32_t valuesCount, const int accuracy,
    const uint8_t *values, float *Xreal, float *Ximag) {
    // values, Xreal and Ximag are values describing single color of single row of bitmap. 
    // This function will be called once per each (color, row) combination.
    for (int k = 0; k < accuracy; k++) {
        for (int i = 0; i < valuesCount; i++) {
            float theta = (2 * M_PI * k * i) / valuesCount;
            Xreal[k] += values[i] * cos(theta);
            Ximag[k] -= values[i] * sin(theta);
        }
    }
}

void decompress(const uint32_t valuesCount, const int accuracy,
    uint8_t *values, const float *Xreal, const float *Ximag) {
    // values, Xreal and Ximag are values describing single color of single row of bitmap.
    // This function will be called once per each (color, row) combination.
    std::vector<float> rawValues(valuesCount, 0);

    for (int i = 0; i < valuesCount; i++) {
        for (int k = 0; k < accuracy; k++) {
            float theta = (2 * M_PI * k * i) / valuesCount;
            rawValues[i] += Xreal[k] * cos(theta) + Ximag[k] * sin(theta);
        }
        values[i] = rawValues[i] / valuesCount;
    }
}

void compressPar(const uint32_t valuesCount, const int accuracy,
    const uint8_t *values, float *Xreal, float *Ximag) {
    #pragma omp for collapse(2) reduction(+:Xreal[:accuracy]) reduction(-:Ximag[:accuracy])
    for (int k = 0; k < accuracy; k++) {
        for (int i = 0; i < valuesCount; i++) {
            float theta = (2 * M_PI * k * i) / valuesCount;
            Xreal[k] += values[i] * cos(theta);
            Ximag[k] -= values[i] * sin(theta);
        }
    }
}

void decompressPar(const uint32_t valuesCount, const int accuracy,
    uint8_t *values, const float *Xreal, const float *Ximag) {
    std::vector<float> rawValues(valuesCount, 0);
    std::vector<std::vector<float>> sines(2, std::vector<float>(valuesCount, 0));
    std::vector<std::vector<float>> cosines(2, std::vector<float>(valuesCount, 1));

    for (int k = 1; k <= accuracy; k++) {
        #pragma omp master
        for (int i = 0; i < valuesCount; i++) {
            rawValues[i] += Xreal[k - 1] * cosines[(k - 1) % 2][i] + Ximag[k - 1] * sines[(k - 1) % 2][i];
            values[i] = rawValues[i] / valuesCount;
        }

        if (k == accuracy)
            break;

        #pragma omp for nowait
        for (int i = 0; i < valuesCount; i++) {
            float theta = (2 * M_PI * k * i) / valuesCount;
            sines[k % 2][i] = sin(theta);
            cosines[k % 2][i] = cos(theta);
        }
    }
}

int main() {
    BMP bmp;
    bmp.read("example.bmp");

    size_t accuracy = 64; // We are interested in values from range [8; 32]

    // bmp.{compress,decompress} will run provided function on every bitmap row and color.
    float compressTime = bmp.compress(compressPar, accuracy);
    float decompressTime = bmp.decompress(decompressPar);

    printf("Compress time: %.2lfs\nDecompress time: %.2lfs\nTotal: %.2lfs\n", 
    compressTime, decompressTime, compressTime + decompressTime);

    bmp.write("example_result.bmp");

    return 0;
}
