/*
 * A template for the 2016 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki
 * Further modifications by Krzysztof Rzadca 2018
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define MAX_BYTES 10000000
#define N 30

char buf[MAX_BYTES];

int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);

    double startTime;
    double endTime;
    double executionTime;

    int numProcesses, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    struct timespec spec;
    clock_gettime(CLOCK_REALTIME, &spec);
    srand(spec.tv_nsec); // use nsec to have a different value across different processes

    if (numProcesses != 2) return 0;

    for (int bytes = 1; bytes <= MAX_BYTES; bytes *= 10) {
        for (int i = 0; i < N; i++) {
            if (myRank == 0) {
                startTime = MPI_Wtime();
                MPI_Send(buf, bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                endTime = MPI_Wtime();
                executionTime = endTime - startTime;
                printf("%d %d %f\n", i, bytes, executionTime);
            } else { // myRank == 1
                MPI_Recv(buf, bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();

    return 0;
}
