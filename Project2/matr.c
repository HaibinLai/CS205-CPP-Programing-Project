#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
// #define SIZE 1024

/*
    串行实现矩阵乘法
*/
void matrixMultiplication(float *matrix1, float *matrix2, float *result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                result[i * size + j] += matrix1[i * size + k] * matrix2[k * size + j];
            }
        }
    }
}

/*
	OpenMP实现矩阵乘法
*/
void matrixMultiplicationInOpenMP(float *matrix1, float *matrix2, 
														float *result, int size) {
    #pragma omp parallel for collapse(2) // 将两个内层循环并行化，同时减少显式的线程同步
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i * size + j] = 0;
            #pragma omp simd // 单指令多数据流，使矩阵加法操作在SIMD指令集上并行化
            for (int k = 0; k < size; k++) {
                result[i * size + j] += matrix1[i * size + k] * matrix2[k * size + j];
            }
        }
    }
}


/*
    Matrix Reader
*/
void readMatrixFromFile(char *fileName, float *matrix, int size) {
    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fscanf(file, "%f", &matrix[i * size + j]);
        }
    }

    fclose(file);
}

/*
    Matrix Writer
*/
void writeMatrixToFile(char *fileName, float *matrix, int size) {
    FILE *file = fopen(fileName, "w");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fprintf(file, "%f ", matrix[i * size + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main(int argc, char *argv[]) {

    int size;
    printf("input the size of Matrix: ");
    scanf("%d", &size);


    // int size = SIZE;
    float *matrix1 = malloc(size * size * sizeof(float));
    float *matrix2 = malloc(size * size * sizeof(float));
    float *result = malloc(size * size * sizeof(float));

// 创建文件名
    char filename1[20];
    sprintf(filename1, "data/%dDAT1.txt", size);

    char filename2[20];
    sprintf(filename2, "data/%dDAT2.txt", size);



    // 从文件中读取矩阵
    clock_t Readstart = clock();
    // double Readstart = omp_get_wtime();
    readMatrixFromFile(filename1, matrix1, size);
    readMatrixFromFile(filename2, matrix2, size);
    // double Readend = omp_get_wtime();
    clock_t Readend = clock();

    double cpu_time_used_Read = ((double)(Readend - Readstart)) / CLOCKS_PER_SEC * 1000;
    printf("%.4f\n", cpu_time_used_Read);

    if (argc == 2) {
        if (strcmp(argv[1], "s") == 0) {
            // 执行矩阵乘法并计时
            // double start = omp_get_wtime();
            clock_t start = clock();
            // 执行矩阵乘法
            matrixMultiplicationInOpenMP(matrix1, matrix2, result, size);
            // 结束计时
            // double end = omp_get_wtime();
            clock_t end = clock();
            double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
            printf("%.8f\n", cpu_time_used);

			clock_t Readstart = clock();
            // double Readstart = omp_get_wtime();
			writeMatrixToFile("Result/resultO3.txt", result, size);

            // double Readend = omp_get_wtime();
			clock_t Readend = clock();

			double cpu_time_used_Write = ((double)(Readend - Readstart)) / CLOCKS_PER_SEC * 1000;
			printf("%.4f\n", cpu_time_used_Write);
        }
    }

    // 释放动态分配的内存
    free(matrix1);
    free(matrix2);
    free(result);

    return 0;
}
