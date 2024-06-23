#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"

// 函数来检查输入参数是否有效
int isValidInput(int input) {
    // 检查是否大于0
    if (input <= 0) {
        return 0; // 无效：不是大于0
    }
    
    // 检查是否小于或等于3000万
    if (input > 30000000) {
        return 0; // 无效：太大
    }
    
    return 1; // 有效
}

int main(int argc, char** argv  ) {
    struct Matrix *A, *B, *C;

    size_t rows1 = 16;
    size_t cols1 = 16;
    size_t cols2 = 16;

    
    if(argc == 2){
        int input = atoi(argv[1]);
        if (!isValidInput(input)) {
            fprintf(stderr, "Invalid input: must be a positive number less than or equal to 3,000,000.\n");
            return EXIT_FAILURE;
        }
        rows1 = atoi(argv[1]);
        cols1 = atoi(argv[1]);
        cols2 = atoi(argv[1]);
    }

    GEMM_Print("your job look like:\n M: %ld, N: %ld, K: %ld\n", rows1,cols1,cols2);

    A = generate_matrix(rows1, cols1);
    B = generate_matrix(cols1, cols2);
    C = generate_matrix(rows1, cols2);

    // print_matrix(A);
    // print_matrix(B);

    GEMM_Print("Generate matrix successfully.\n");

    double time1 = HPL_timer_walltime();
    sgemm_reinforce(A, B, C);
    double time2 = HPL_timer_walltime();

    GEMM_Print("Time elapsed: %f seconds.\n", time2 - time1);

    // print_matrix(C);

    freeMatrix(A);
    freeMatrix(B);
    freeMatrix(C);

    return 0;
}

