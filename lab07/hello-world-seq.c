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

int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);

    int numProcesses, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    struct timespec spec;
    clock_gettime(CLOCK_REALTIME, &spec);
    srand(spec.tv_nsec); // use nsec to have a different value across different processes

    if (myRank == 0) {
        int buf[1] = {1};
        MPI_Send(buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(buf, 1, MPI_INT, numProcesses - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Result = %d\n", buf[0]);
    } else {
        int buf[1];
        MPI_Recv(buf, 1, MPI_INT, myRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        buf[0] *= myRank;
        if (myRank == numProcesses - 1) {
            MPI_Send(buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Send(buf, 1, MPI_INT, myRank + 1, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    return 0;
}
