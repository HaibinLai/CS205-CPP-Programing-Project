#include "../gemm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/mman.h>


struct Matrix* create_matrix(size_t rows, size_t cols) {
    struct Matrix *mat = (struct Matrix *)malloc(sizeof(struct Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->ld = cols;  // 假设不进行矩阵转置
    mat->isTran = 0;  // 假设不进行矩阵转置
    mat->data = (float *)mmap(NULL, rows * cols * sizeof(float), PROT_READ |
                     PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (mat->data == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    return mat;
}

struct Matrix* generate_matrix(size_t rows, size_t cols) {
    struct Matrix *mat = (struct Matrix *)malloc(sizeof(struct Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->ld = cols;  // 假设不进行矩阵转置
    mat->isTran = 0;  // 假设不进行矩阵转置
    mat->data = (float *)mmap(NULL, rows * cols * sizeof(float), PROT_READ |
                     PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (mat->data == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    // 为矩阵的每个元素分配一个随机的float值
    for (size_t i = 0; i < rows * cols; i++) {
        mat->data[i] = (float)rand() / RAND_MAX;
    }

    return mat;
}

void freeMatrix(struct Matrix *mat) {
    munmap(mat->data, mat->rows * mat->cols * sizeof(float));
    free(mat);
}


/////////////////////////
// 输出矩阵
void print_matrix(struct Matrix *mat) {
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            printf("%f ", mat->data[i * mat->cols + j]);
        }
        printf("\n");
    }
}

////////////////////////////////////////

void sgemm_OpenBLAS(struct Matrix *A, struct Matrix *B, struct Matrix *C) {
    if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
        fprintf(stderr, "Invalid matrix dimensions\n");
        exit(1);
    }
    // driver_OpenBlAS(A->rows, B->cols, A->cols, A->data, B->data, C->data);
}

void sgemm_plain(struct Matrix *A, struct Matrix *B, struct Matrix *C) {
    if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
        fprintf(stderr, "Invalid matrix dimensions\n");
        exit(1);
    }
    driver_plain(A->rows, B->cols, A->cols, A->data, B->data, C->data);
}


const size_t tiny = 20;


void sgemm_reinforce(struct Matrix *A, struct Matrix *B, struct Matrix *C) {
    if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
        fprintf(stderr, "Invalid matrix dimensions\n");
        exit(1);
    }

    int type = 0;
    register size_t  max = A->rows > B->cols ? A->rows : B->cols;
    max = max > A->cols ? max : A->cols;

    if(max < tiny){
        type = TinyMatrix;
        driver_reinforce(A->rows, B->cols, A->cols, A->data, B->data, C->data,type);
        return;
    }else if (max < 200){
        type = SmallMatrix;
    }else if (max < MediumMatrix){
        type = MediumMatrix;
    }else if (max < LargeMatrix){
        type = LargeMatrix;
    }else{
        type = HugeMatrix;
    }

    driver_reinforce(A->rows, B->cols, A->cols, A->data, B->data, C->data,type);
}