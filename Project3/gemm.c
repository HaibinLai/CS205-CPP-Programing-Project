#include "../gemm.h"
#include <emmintrin.h> // 包含SSE指令集
#include <immintrin.h>
#include <omp.h>

void matmul_plain(int rows1, int cols2, int cols1, float* A, float* B, float* C) {

    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            C[i * cols2 + j] = 0;
            for (int k = 0; k < cols1; k++) {
                C[i * cols2 + j] += A[i * cols1 + k] * B[k * cols2 + j];
            }
        }
    }
}

void SGEMM_AVX2(int rows1, int cols2, int cols1, float *A, float *B, float *C) {

    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < cols1; k += 8) {
                __m256 a = _mm256_loadu_ps(&A[i * cols2 + k]);
                __m256 b = _mm256_loadu_ps(&B[k * cols1 + j]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
            }
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            C[i * cols2 + j] = _mm_cvtss_f32(sum128);
        }
    }
}



void SGEMM_AVX512(int rows1, int cols2, int cols1, float *A, float *B, float *C) {
    for (int j = 0; j < cols2; j++) { //  make B's cache usage more regular
        for (int i = 0; i < rows1; i++) { // Then iterate over rows of A
            __m512 sum = _mm512_setzero_ps();
            for (int l = 0; l < cols1; l += 16) { 
                __m512 a = _mm512_loadu_ps(&A[i * cols2 + l]);
                __m512 b = _mm512_loadu_ps(&B[j * cols1 + l]);
                sum = _mm512_fmadd_ps(a, b, sum);
            }
            _mm512_storeu_ps(&C[i * cols2 + j], sum);
        }
    }
}


void SGEMM_AVX512_32_Kernel(int rows1, int cols2, int cols1, float* A, float* B, float* C) {
    #pragma omp parallel for
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j += 4) {
            __m512 sum0 = _mm512_setzero_ps();//16
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();

            for (int k = 0; k < cols1; k += 32) {
                // Load columns of A
                __m512 a0 = _mm512_loadu_ps(&A[i * cols2 + k]);
                __m512 a1 = _mm512_loadu_ps(&A[i * cols2 + k + 16]);
                
                // Load columns of B
                __m512 b0 = _mm512_loadu_ps(&B[k * cols1 + j]);
                __m512 b1 = _mm512_loadu_ps(&B[k * cols1 + j + 16]);
                
                // Perform matrix multiplication
                __m512 temp0 = _mm512_mul_ps(a0, b0);
                __m512 temp1 = _mm512_mul_ps(a0, b1);
                __m512 temp2 = _mm512_mul_ps(a1, b0);
                __m512 temp3 = _mm512_mul_ps(a1, b1);
                sum0 = _mm512_add_ps(sum0, temp0);
                sum1 = _mm512_add_ps(sum1, temp1);
                sum2 = _mm512_add_ps(sum2, temp2);
                sum3 = _mm512_add_ps(sum3, temp3);
            }
            // Store the results in C
            _mm512_storeu_ps(&C[i * cols2 + j], sum0);
            _mm512_storeu_ps(&C[i * cols2 + j + 16], sum1);
            _mm512_storeu_ps(&C[i * cols2 + j + 32], sum2);
            _mm512_storeu_ps(&C[i * cols2 + j + 48], sum3);
        }
    }
}

void SGEMM_AVX512_256_kernel(int rows1, int cols2, int cols1, float* A, float* B, float* C) {
    #pragma omp parallel for
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j += 16) {
            __m512 sum0 = _mm512_setzero_ps();//16
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();

            __m512 sum4 = _mm512_setzero_ps();//16
            __m512 sum5 = _mm512_setzero_ps();
            __m512 sum6 = _mm512_setzero_ps();
            __m512 sum7 = _mm512_setzero_ps();

            __m512 sum8 = _mm512_setzero_ps();//16
            __m512 sum9 = _mm512_setzero_ps();
            __m512 sum10 = _mm512_setzero_ps();
            __m512 sum11 = _mm512_setzero_ps();

            __m512 sum12 = _mm512_setzero_ps();//16
            __m512 sum13 = _mm512_setzero_ps();
            __m512 sum14 = _mm512_setzero_ps();
            __m512 sum15 = _mm512_setzero_ps();

            for (int k = 0; k < cols1; k += 64) {
                // Load columns of A
                __m512 a0 = _mm512_loadu_ps(&A[i * cols1 + k]);
                __m512 a1 = _mm512_loadu_ps(&A[i * cols1 + k + 16]);
                __m512 a2 = _mm512_loadu_ps(&A[i * cols1 + k + 32]);
                __m512 a3 = _mm512_loadu_ps(&A[i * cols1 + k + 48]);
                
                // Load columns of B
                __m512 b0 = _mm512_loadu_ps(&B[k * cols2 + j]);
                __m512 b1 = _mm512_loadu_ps(&B[k * cols2 + j + 16]);
                __m512 b2 = _mm512_loadu_ps(&A[i * cols1 + k + 32]);
                __m512 b3 = _mm512_loadu_ps(&A[i * cols1 + k + 48]);

                // Perform matrix multiplication
                __m512 temp0 = _mm512_mul_ps(a0, b0);
                __m512 temp1 = _mm512_mul_ps(a0, b1);
                __m512 temp2 = _mm512_mul_ps(a0, b2);
                __m512 temp3 = _mm512_mul_ps(a0, b3);

                __m512 temp4 = _mm512_mul_ps(a1, b0);
                __m512 temp5 = _mm512_mul_ps(a1, b1);
                __m512 temp6 = _mm512_mul_ps(a1, b2);
                __m512 temp7 = _mm512_mul_ps(a1, b3);

                __m512 temp8 = _mm512_mul_ps(a2, b0);
                __m512 temp9 = _mm512_mul_ps(a2, b1);
                __m512 temp10 = _mm512_mul_ps(a2, b2);
                __m512 temp11 = _mm512_mul_ps(a2, b3);

                __m512 temp12 = _mm512_mul_ps(a3, b0);
                __m512 temp13 = _mm512_mul_ps(a3, b1);
                __m512 temp14 = _mm512_mul_ps(a3, b2);
                __m512 temp15 = _mm512_mul_ps(a3, b3);

                sum0 = _mm512_add_ps(sum0, temp0);
                sum1 = _mm512_add_ps(sum1, temp1);
                sum2 = _mm512_add_ps(sum2, temp2);
                sum3 = _mm512_add_ps(sum3, temp3);
                sum4 = _mm512_add_ps(sum4, temp4);
                sum5 = _mm512_add_ps(sum5, temp5);
                sum6 = _mm512_add_ps(sum6, temp6);
                sum7 = _mm512_add_ps(sum7, temp7);

                sum8 = _mm512_add_ps(sum8, temp8);
                sum9 = _mm512_add_ps(sum9, temp9);
                sum10 = _mm512_add_ps(sum10, temp10);
                sum11 = _mm512_add_ps(sum11, temp11);
                sum12 = _mm512_add_ps(sum12, temp12);
                sum13 = _mm512_add_ps(sum13, temp13);
                sum14 = _mm512_add_ps(sum14, temp14);
                sum15 = _mm512_add_ps(sum15, temp15);
            }

            // Store the results in C
            _mm512_storeu_ps(&C[i * cols2 + j], sum0);
            _mm512_storeu_ps(&C[i * cols2 + j + 16], sum1);
            _mm512_storeu_ps(&C[i * cols2 + j + 32], sum2);
            _mm512_storeu_ps(&C[i * cols2 + j + 48], sum3);
            _mm512_storeu_ps(&C[i * cols2 + j + 64], sum4);
            _mm512_storeu_ps(&C[i * cols2 + j + 80], sum5);
            _mm512_storeu_ps(&C[i * cols2 + j + 96], sum6);
            _mm512_storeu_ps(&C[i * cols2 + j + 112], sum7);

            _mm512_storeu_ps(&C[i * cols2 + j + 128], sum8);
            _mm512_storeu_ps(&C[i * cols2 + j + 144], sum9);
            _mm512_storeu_ps(&C[i * cols2 + j + 160], sum10);
            _mm512_storeu_ps(&C[i * cols2 + j + 176], sum11);
            _mm512_storeu_ps(&C[i * cols2 + j + 192], sum12);
            _mm512_storeu_ps(&C[i * cols2 + j + 208], sum13);
            _mm512_storeu_ps(&C[i * cols2 + j + 224], sum14);
            _mm512_storeu_ps(&C[i * cols2 + j + 240], sum15);
        }
    }
}






const int BLOCK_SIZE = 1024;
const int BLOCK_SIZE2 = 256;

// inline static void block_avx512_32x4(int n, int K, float* A, float* B, float* C)
// {
// 	__m512d c0000_0700,c0800_1500, c1600_2300, c2400_3100,
// 		c0001_0701, c0801_1501, c1601_2301, c2401_3101,
// 		c0002_0702, c0802_1502, c1602_2302, c2402_3102,
// 		c0003_0703, c0803_1503, c1603_2303, c2403_3103;

// 	__m512d a0x_7x, a8x_15x, a16x_23x, a24x_31x,
// 		bx0, bx1, bx2, bx3;

// 	float* c0001_0701_ptr = C + n;
// 	float* c0002_0702_ptr = C + n * 2;
// 	float* c0003_0703_ptr = C + n * 3;

// 	c0000_0700 = _mm512_load_pd(C);
// 	c0800_1500 = _mm512_load_pd(C + 8);
// 	c1600_2300 = _mm512_load_pd(C + 16);
// 	c2400_3100 = _mm512_load_pd(C + 24);

// 	c0001_0701 = _mm512_load_pd(c0001_0701_ptr);
// 	c0801_1501 = _mm512_load_pd(c0001_0701_ptr + 8);
// 	c1601_2301 = _mm512_load_pd(c0001_0701_ptr + 16);
// 	c2401_3101 = _mm512_load_pd(c0001_0701_ptr + 24);

// 	c0002_0702 = _mm512_load_pd(c0002_0702_ptr);
// 	c0802_1502 = _mm512_load_pd(c0002_0702_ptr + 8);
// 	c1602_2302 = _mm512_load_pd(c0002_0702_ptr + 16);
// 	c2402_3102 = _mm512_load_pd(c0002_0702_ptr + 24);

// 	c0003_0703 = _mm512_load_pd(c0003_0703_ptr);
// 	c0803_1503 = _mm512_load_pd(c0003_0703_ptr + 8);
// 	c1603_2303 = _mm512_load_pd(c0003_0703_ptr + 16);
// 	c2403_3103 = _mm512_load_pd(c0003_0703_ptr + 24);

// 	for (int x = 0; x < K; ++x)
// 	{
// 		a0x_7x = _mm512_load_pd(A);
// 		a8x_15x = _mm512_load_pd(A + 8);
// 		a16x_23x = _mm512_load_pd(A + 16);
// 		a24x_31x = _mm512_load_pd(A + 24);
// 		A += 32;

// 		bx0 = _mm512_broadcastsd_pd(_mm_load_sd(B++));
// 		bx1 = _mm512_broadcastsd_pd(_mm_load_sd(B++));
// 		bx2 = _mm512_broadcastsd_pd(_mm_load_sd(B++));
// 		bx3 = _mm512_broadcastsd_pd(_mm_load_sd(B++));


// 		c0000_0700 = _mm512_add_pd(_mm512_mul_pd(a0x_7x, bx0), c0000_0700);
// 		c0800_1500 = _mm512_add_pd(_mm512_mul_pd(a8x_15x, bx0), c0800_1500);
// 		c1600_2300 = _mm512_add_pd(_mm512_mul_pd(a16x_23x, bx0), c1600_2300);
// 		c2400_3100 = _mm512_add_pd(_mm512_mul_pd(a24x_31x, bx0), c2400_3100);

// 		c0001_0701 = _mm512_add_pd(_mm512_mul_pd(a0x_7x, bx1), c0001_0701);
// 		c0801_1501 = _mm512_add_pd(_mm512_mul_pd(a8x_15x, bx1), c0801_1501);
// 		c1601_2301 = _mm512_add_pd(_mm512_mul_pd(a16x_23x, bx1), c1601_2301);
// 		c2401_3101 = _mm512_add_pd(_mm512_mul_pd(a24x_31x, bx1), c2401_3101);

// 		c0002_0702 = _mm512_add_pd(_mm512_mul_pd(a0x_7x, bx2), c0002_0702);
// 		c0802_1502 = _mm512_add_pd(_mm512_mul_pd(a8x_15x, bx2), c0802_1502);
// 		c1602_2302 = _mm512_add_pd(_mm512_mul_pd(a16x_23x, bx2), c1602_2302);
// 		c2402_3102 = _mm512_add_pd(_mm512_mul_pd(a24x_31x, bx2), c2402_3102);

// 		c0003_0703 = _mm512_add_pd(_mm512_mul_pd(a0x_7x, bx3), c0003_0703);
// 		c0803_1503 = _mm512_add_pd(_mm512_mul_pd(a8x_15x, bx3), c0803_1503);
// 		c1603_2303 = _mm512_add_pd(_mm512_mul_pd(a16x_23x, bx3), c1603_2303);
// 		c2403_3103 = _mm512_add_pd(_mm512_mul_pd(a24x_31x, bx3), c2403_3103);
// 	}
// 	_mm512_storeu_pd(C, c0000_0700);
// 	_mm512_storeu_pd(C + 8, c0800_1500);
// 	_mm512_storeu_pd(C + 16, c1600_2300);
// 	_mm512_storeu_pd(C + 24, c2400_3100);

// 	_mm512_storeu_pd(c0001_0701_ptr, c0001_0701);
// 	_mm512_storeu_pd(c0001_0701_ptr + 8, c0801_1501);
// 	_mm512_storeu_pd(c0001_0701_ptr + 16, c1601_2301);
// 	_mm512_storeu_pd(c0001_0701_ptr + 24, c2401_3101);

// 	_mm512_storeu_pd(c0002_0702_ptr, c0002_0702);
// 	_mm512_storeu_pd(c0002_0702_ptr + 8, c0802_1502);
// 	_mm512_storeu_pd(c0002_0702_ptr + 16, c1602_2302);
// 	_mm512_storeu_pd(c0002_0702_ptr + 24, c2402_3102);

// 	_mm512_storeu_pd(c0003_0703_ptr, c0003_0703);
// 	_mm512_storeu_pd(c0003_0703_ptr + 8, c0803_1503);
// 	_mm512_storeu_pd(c0003_0703_ptr + 16, c1603_2303);
// 	_mm512_storeu_pd(c0003_0703_ptr + 24, c2403_3103);
// }

// static inline void copy_avx512_b(int lda, const int K, float* b_src, float* b_dest) {
// 	float* b_ptr0, * b_ptr1, * b_ptr2, * b_ptr3;
// 	b_ptr0 = b_src;
// 	b_ptr1 = b_ptr0 + lda;
// 	b_ptr2 = b_ptr1 + lda;
// 	b_ptr3 = b_ptr2 + lda;

// 	for (int i = 0; i < K; ++i)
// 	{
// 		*b_dest++ = *b_ptr0++;
// 		*b_dest++ = *b_ptr1++;
// 		*b_dest++ = *b_ptr2++;
// 		*b_dest++ = *b_ptr3++;
// 	}
// }

// static inline void copy_avx512_a(int lda, const int K, float* a_src, float* a_dest) {
// 	for (int i = 0; i < K; ++i)
// 	{
// 		memcpy(a_dest, a_src, 32 * 8);
// 		a_dest += 32;
// 		a_src += lda;
// 	}
// }

// static inline void do_block_avx512(int lda, int M, int N, int K, float* A, float* B, float* C)
// {
// 	float* A_block, * B_block;
// 	A_block = (float*)_mm_malloc(M * K * sizeof(float), 32);
// 	B_block = (float*)_mm_malloc(K * N * sizeof(float), 32);

// 	float* a_ptr, * b_ptr, * c;

// 	const int Nmax = N - 3;
// 	int Mmax = M - 32;

// 	int i = 0, j = 0, p = 0;

// 	for (j = 0; j < Nmax; j += 4)
// 	{
// 		b_ptr = &B_block[j * K];
// 		copy_avx512_b(lda, K, B + j * lda, b_ptr); // 将 B 展开
// 		for (i = 0; i < Mmax; i += 32) {
// 			a_ptr = &A_block[i * K];
// 			if (j == 0) copy_avx512_a(lda, K, A + i, a_ptr); // 将 A 展开
// 			c = C + i + j * lda;
// 			block_avx512_32x4(lda, K, a_ptr, b_ptr, c);
// 		}
// 	}
// 	_mm_free(A_block);
// 	_mm_free(B_block);
// }

//void gemm_avx512(int lda, float* A, float* B, float* C)
//{
//#pragma omp parallel for
//	for (int j = 0; j < lda; j += BLOCK_SIZE) {    // j i k 序 内存读写更快
//		for (int i = 0; i < lda; i += BLOCK_SIZE) {
//			for (int k = 0; k < lda; k += BLOCK_SIZE) {
//				// 大分块里小分块
//				for (int jj = j; jj < j + BLOCK_SIZE; jj += BLOCK_SIZE2)
//					for (int ii = i; ii < i + BLOCK_SIZE; ii += BLOCK_SIZE2)
//						for (int kk = k; kk < k + BLOCK_SIZE; kk += BLOCK_SIZE2){
  //                          // do_block_avx512
                            //     (lda, BLOCK_SIZE2, BLOCK_SIZE2, BLOCK_SIZE2, 
                            //         A + ii + kk * lda, B + kk + jj * lda, C + ii + jj * lda);
//                        }
//							
//			}
//		}
//	}
//}



//#define BLOCK_SIZE 256

//void gemm_avx512(int m, int n, int k, float *A,  float *B,float *C) {
    // 遍历矩阵C的每个块
  //  for (int i = 0; i < m; i += BLOCK_SIZE) {
    //    for (int j = 0; j < n; j += BLOCK_SIZE) {
      //      #pragma omp parallel for collapse(2) // 在两个维度上并行化
        //    for (int kk = 0; kk < k; kk += BLOCK_SIZE) {
          //      for (int ii = i; ii < (i + BLOCK_SIZE); ++ii) {
            //        #pragma omp parallel for
              //      for (int jj = j; jj < (j + BLOCK_SIZE); ++jj) {
                //        float* a_ptr = &A[ii * k + kk];
                  //      float* b_ptr = &B[kk * n + jj];
                    //    float* c_ptr = &C[ii * n + jj];
  //                     SGEMM_AVX512_256_kernel(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE, a_ptr,b_ptr,c_ptr );
      //              }
    //            }
//            }
//        }
//    }
//}
//
//
//

#define BLOCK_SIZE 256

void gemm_avx512(int M, int N, int K, float *A,  float *B,float *C) {
    // 遍历矩阵C的每个块
    #pragma omp parallel for
    for (int j = 0; j < M; j += BLOCK_SIZE) {
        for (int i = 0; i < N; i += BLOCK_SIZE) {
            for (int k = 0; k< K; k += BLOCK_SIZE) {
                // big kernel
                for (int jj = j; jj < (j + BLOCK_SIZE); jj += BLOCK_SIZE) {
                    for (int ii = i; ii < (i + BLOCK_SIZE); ii += BLOCK_SIZE) {
                        for(int kk = k; kk < k + BLOCK_SIZE; kk += BLOCK_SIZE){
                            // small kernel
                            SGEMM_AVX512_256_kernel(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE, 
                            A + ii + kk * M, B + kk + jj * N, C + ii + jj * K );
                        }
                    }
                }
            }
        }
    }
}

