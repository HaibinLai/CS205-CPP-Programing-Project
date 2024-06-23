#include "matr.h"

#include<time.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const size_t rows = 3;
    const int cols = 3;
    const int GPU_device_0 = 1;
    const int GPU_device_1 = 1;

    ////////////////////// CREATING TEST ///////////////////////////
    // Create two matrices
    Matrix<float> A(rows, cols,GPU_device_0);
    Matrix<float> B(rows, cols,GPU_device_1);

    // Initialize matrices A and B
    for (float i = 0; i < rows; ++i) {
        for (float j = 0; j < cols; ++j) {
            A(i, j) =i+j;
            B(i, j) =0;
        }
    }


    std::cout << "A:\n";
    A.print();


    float a = 6;
    float b = 10;

    std::cout << "\na:" << a << std::endl;
    std::cout<< "b:" << b << std::endl;
    /////////////////////////// Print TEST /////////////////////////////////


    ////////////////////////// SPEED TEST //////////////////////////
    // clock_t start1, end1; 
    // start1 = clock();

    B = a * A + b;

    // end1 = clock();
    // std::cout << "\na*A+b: finish\n";
    // printf("time=%f\n", (double)(end1 - start1) / CLOCKS_PER_SEC);

    std::cout << "\nB:\n";
    B.print();

    std::cout << "\nA:\n";
    A.print();

    ///////////////////////// MUL GPU TEST /////////////////////
    clock_t start2, end2; 
    start2 = clock();

    const size_t rows2 = 4096;
    const int cols2 = 4096;
    const int GPU_device_2 = 0;
    const int GPU_device_3 = 0;

    // Create two matrices
    Matrix<float> C(rows2, cols2, GPU_device_2);
    Matrix<float> D(rows2, cols2, GPU_device_3);

    // Initialize matrices A and B
    for (float i = 0; i < rows2; ++i) {
        for (float j = 0; j < cols2; ++j) {
            C(i, j) =i+j;
            D(i, j) =i-j;
        }
    }
    cudaSetDevice(GPU_device_2); // Ñ¡ÔñGPU

    cudaEvent_t start, stop;
	float esp_time_gpu;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    cudaEventRecord(start, 0);// start

    // your CUDA program
    // Ö´ÐÐ¾ØÕó³Ë·¨

    // for(int i = 0 ; i< 100 ;i++){
        // Matrix E = C.matmul(D);
    // }
    Matrix E = C.matmul(D);
    

	cudaEventRecord(stop, 0);// stop

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&esp_time_gpu, start, stop);
	printf("Time for the kernel: %f ms\n", esp_time_gpu);

    end2 = clock();
    std::cout << "\nA*B: finish\n";
    printf("time=%f\n", (double)(end2 - start2) / CLOCKS_PER_SEC);


    ///////////////////////// CONSTRUCTOR TEST ///////////////////



    // A += E;
    // E.print();
    std::cout<< "end of main\n";

    return 0;
}