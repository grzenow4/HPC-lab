// Solves the n-quees puzzle on an n x x checkerboard.
//
// This sequential implementation is to be extended with TBB to get a
// parallel implementation.
//
// HPC course, MIM UW
// Krzysztof Rzadca, LGPL

#include "tbb/tbb.h"
#include <iostream>
#include <list>
#include <vector>
#include <cmath>


// Indexed by column. Value is the row where the queen is placed,
// or -1 if no queen.
typedef std::vector<int> Board;


void pretty_print(const Board& board) {
    for (int row = 0; row < (int) board.size(); row++) {
        for (const auto& loc : board) {
            if (loc == row)
                std::cout << "*";
            else
                std::cout << " ";
        }
        std::cout << std::endl;
    }
}


// Checks the location of queen in column 'col' against queens in cols [0, col).
bool check_col(Board& board, int col_prop) {
    int row_prop = board[col_prop];
    int col_queen = 0;
    for (auto i = board.begin();
         (i != board.end()) && (col_queen < col_prop);
         ++i, ++col_queen) {
        int row_queen = *i;
        if (row_queen == row_prop) {
            return false;
        }
        if (abs(row_prop - row_queen) == col_prop - col_queen) {
            return false;
        }
    }
    return true;
}


void initialize(Board& board, int size) {
    board.reserve(size);
    for (int col = 0; col < size; ++col)
        board.push_back(-1);
}


// Solves starting from a partially-filled board (up to column col).
void recursive_solve(Board& partial_board, int col, std::list<Board>& solutions) {
    // std::cout << "rec solve col " << col << std::endl;
    // pretty_print(b_partial);
    
    int b_size = partial_board.size();
    if (col == b_size) {
        solutions.push_back(partial_board);
    }
    else {
        for (int tested_row = 0; tested_row < b_size; tested_row++) {
            partial_board[col] = tested_row;
            if (check_col(partial_board, col))
                recursive_solve(partial_board, col + 1, solutions);
        }
    }
}


void parallel_solve(Board partial_board, int col, tbb::task_group& g, tbb::concurrent_queue<Board>& solutions) {
    int b_size = partial_board.size();
    if (col == b_size) {
        solutions.push(partial_board);
    }
    else {
        for (int tested_row = 0; tested_row < b_size; tested_row++) {
            partial_board[col] = tested_row;
            if (check_col(partial_board, col))
                g.run([partial_board, col, &g, &solutions] {
                    parallel_solve(partial_board, col + 1, g, solutions);
                });
        }
    }
}


void parallel_solve_2(Board partial_board, int col, tbb::enumerable_thread_specific<std::list<Board>>& solutions) {
    std::pair<Board, int> init_p = {partial_board, col};
    tbb::parallel_for_each(&init_p, &init_p + 1, [&solutions]
                          (std::pair<Board, int> p, tbb::feeder<std::pair<Board, int>>& feeder) {
            Board board = p.first;
            int col = p.second;
            int b_size = board.size();
            if (col == b_size) {
                solutions.local().push_back(board);
            } else {
                for (int tested_row = 0; tested_row < b_size; tested_row++) {
                    board[col] = tested_row;
                    if (check_col(board, col)) {
                        feeder.add({board, col + 1});
                    }
                }
            }
        }
    );
}


int main() {
    const int board_size = 13;
    Board board{};
    initialize(board, board_size);
    std::list<Board> solutions{};
    tbb::concurrent_queue<Board> par_solutions;
    tbb::enumerable_thread_specific<std::list<Board>> par_solutions_2;

    tbb::tick_count seq_start_time = tbb::tick_count::now();
    recursive_solve(board, 0, solutions);
    tbb::tick_count seq_end_time = tbb::tick_count::now();
    double seq_time = (seq_end_time - seq_start_time).seconds();
    std::cout << "seq time: " << seq_time << "[s]" << std::endl;

    tbb::tick_count par_start_time = tbb::tick_count::now();
    tbb::task_group g;
    parallel_solve(board, 0, g, par_solutions);
    g.wait();
    tbb::tick_count par_end_time = tbb::tick_count::now();
    double par_time = (par_end_time - par_start_time).seconds();
    std::cout << "par time: " << par_time << "[s]" << std::endl;

    tbb::tick_count par_start_time_2 = tbb::tick_count::now();
    parallel_solve_2(board, 0, par_solutions_2);
    tbb::tick_count par_end_time_2 = tbb::tick_count::now();
    double par_time_2 = (par_end_time_2 - par_start_time_2).seconds();
    std::cout << "par2 time: " << par_time_2 << "[s]" << std::endl;

    size_t count = 0;
    while (!par_solutions.empty()) {
        Board solution;
        if (par_solutions.try_pop(solution)) {
            count++;
        }
    }

    size_t count_2 = 0;
    for (auto sol: par_solutions_2) {
        count_2 += sol.size();
    }

    std::cout << "solution count: " << solutions.size() << std::endl;
    std::cout << "par solution count: " << count << std::endl;
    std::cout << "par2 solution count: " << count_2 << std::endl;

    // for (const auto& sol : solutions) {
    //     pretty_print(sol);
    //     std::cout << std::endl;
    // }
}
