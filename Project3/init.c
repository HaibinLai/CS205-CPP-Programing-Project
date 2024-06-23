#include "../gemm.h"
// #include <cblas.h>

// void driver_OpenBlAS(int rows1, int cols2, int cols1, float* A, float* B, float* C ) {
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows1, cols2, cols1, 1.0, A, cols1, B, cols2, 0.0, C, cols2);
// }

void driver_plain(int rows1, int cols2, int cols1, float* A, float* B, float* C) {
    matmul_plain(rows1, cols2, cols1, A, B, C);
}

void driver_reinforce(int rows1, int cols2, int cols1, float* A, float* B, float* C,int type) {
    if(type == TinyMatrix){
        SGEMM_AVX2(rows1, cols2, cols1, A, B, C);
        return;
    }
    else if(type == SmallMatrix){
        SGEMM_AVX512(rows1, cols2, cols1, A, B, C);
        return;
    }else if (type == MediumMatrix)
    {
        SGEMM_AVX512_32_Kernel(rows1, cols2, cols1, A, B, C);
        return;
    }else if (type == LargeMatrix)
    {
        SGEMM_AVX512_256_kernel(rows1, cols2, cols1, A, B, C);
        return;
    }
    
    SGEMM_AVX512_256_kernel(rows1, cols2, cols1, A, B, C);
    // SGEMM_AVX2_openMP(rows1, cols2, cols1, A, B, C);
    //gemm_avx512( rows1,cols2,cols1, A, B, C);
}

// gcc -o Mul mul.c -fopenmp -DPRINT -O2 -mavx2
