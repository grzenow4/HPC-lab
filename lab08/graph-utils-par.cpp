/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include <cassert>
#include <mpi.h>
#include "graph-base.h"
#include "graph-utils.h"

int getFirstGraphRowOfProcess(int numVertices, int numProcesses, int myRank) {
    int q = numVertices / numProcesses;
    int r = numVertices % numProcesses;
    if (myRank < r) {
        return myRank * (q + 1);
    }
    return myRank * q + r;
}

Graph* createAndDistributeGraph(int numVertices, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);

    auto graph = allocateGraphPart(
            numVertices,
            getFirstGraphRowOfProcess(numVertices, numProcesses, myRank),
            getFirstGraphRowOfProcess(numVertices, numProcesses, myRank + 1)
    );

    if (graph == nullptr) {
        return nullptr;
    }

    assert(graph->numVertices > 0 && graph->numVertices == numVertices);
    assert(graph->firstRowIdxIncl >= 0 && graph->lastRowIdxExcl <= graph->numVertices);

    int curProc = 0;
    for (int i = 0; i < graph->numVertices; i++) {
        if (curProc < numProcesses - 1 && getFirstGraphRowOfProcess(graph->numVertices, numProcesses, curProc + 1) == i) {
            curProc++;
        }

        if (myRank == 0) {
            if (myRank == curProc) {
                initializeGraphRow(graph->data[i], i, graph->numVertices);
            } else {
                initializeGraphRow(graph->extraRow, i, graph->numVertices);
                MPI_Send(graph->extraRow, graph->numVertices, MPI_INT, curProc, 0, MPI_COMM_WORLD);
            }
        } else if (myRank == curProc) {
            MPI_Recv(graph->data[i - graph->firstRowIdxIncl], graph->numVertices, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    return graph;
}

void collectAndPrintGraph(Graph* graph, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);
    assert(graph->numVertices > 0);
    assert(graph->firstRowIdxIncl >= 0 && graph->lastRowIdxExcl <= graph->numVertices);

    int curProc = 0;
    for (int i = 0; i < graph->numVertices; i++) {
        if (curProc < numProcesses - 1 && getFirstGraphRowOfProcess(graph->numVertices, numProcesses, curProc + 1) == i) {
            curProc++;
        }

        if (myRank == 0) {
            if (myRank == curProc) {
                printGraphRow(graph->data[i], i, graph->numVertices);
            } else {
                MPI_Recv(graph->extraRow, graph->numVertices, MPI_INT, curProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printGraphRow(graph->extraRow, i, graph->numVertices);
            }
        } else if (myRank == curProc) {
            MPI_Send(graph->data[i - graph->firstRowIdxIncl], graph->numVertices, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
}

void destroyGraph(Graph* graph, int numProcesses, int myRank) {
    freeGraphPart(graph);
}
