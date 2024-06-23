#include <stdio.h>
#define GEMM_Print printf
#ifndef GEMM_H
#define GEMM_H

#define TinyMatrix 111
#define SmallMatrix 200
#define MediumMatrix 1200
#define LargeMatrix 10000
#define HugeMatrix 12000

struct Matrix
{
    size_t rows;
    size_t cols;
    size_t ld;
    short  isTran;
    float * data;
};

// interface /////////////////////////////////////////
// BLAS
void sgemm_OpenBLAS(struct Matrix *A, struct Matrix *B, struct Matrix *C);
void sgemm_IntelMKL(struct Matrix *A, struct Matrix *B, struct Matrix *C);
void sgemm_GBLAS(struct Matrix *A, struct Matrix *B, struct Matrix *C);
void sgemm_plain(struct Matrix *A, struct Matrix *B, struct Matrix *C);
void sgemm_reinforce(struct Matrix *A, struct Matrix *B, struct Matrix *C);
// MATRIX
struct Matrix* create_matrix(size_t rows, size_t cols);
struct Matrix* generate_matrix(size_t rows, size_t cols);
void print_matrix(struct Matrix *mat);
void freeMatrix(struct Matrix *mat);
// TIME
double HPL_timer_walltime();


// driver ////////////////////////////////////////////
// void driver_OpenBlAS(int rows1, int cols2, int cols1, float* A, float* B, float* C );
void driver_IntelMKL(int rows1, int cols2, int cols1, float* A, float* B, float* C );
void driver_SGEMM_GB(int rows1, int cols2, int cols1, float* A, float* B, float* C );
void driver_plain(int rows1, int cols2, int cols1, float* A, float* B, float* C);
void driver_reinforce(int rows1, int cols2, int cols1, float* A, float* B, float* C,int type);

// Kernel ////////////////////////////////////////////
void SGEMM_GB       (int rows1, int cols2, int cols1, float* A, float* B, float* C );
void matmul_plain   (int rows1, int cols2, int cols1, float* A, float* B, float* C);
void SGEMM_AVX2     (int rows1, int cols2, int cols1, float *A, float *B, float *C);
void SGEMM_AVX512_32_Kernel(int rows1, int cols2, int cols1, float* A, float* B, float* C);
void SGEMM_AVX512_256_kernel(int rows1, int cols2, int cols1, float* A, float* B, float* C);

void gemm_avx512(int rows1, int cols2, int cols1, float* A, float* B, float* C);
void SGEMM_AVX512(int rows1, int cols2, int cols1, float *A, float *B, float *C);
#endif
