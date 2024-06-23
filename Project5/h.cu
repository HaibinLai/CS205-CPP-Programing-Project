#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

/**
  @defgroup cuda CUDA-accelerated GPU MAT
  @{
    @defgroup cudacore Core part
    @{
      @defgroup Initialization and Information
      @defgroup cudacore_struct Data Structures
    @}
  @}
 */


#define uchar unsigned char

#define MAT_8   1
#define MAT_8U  3
#define MAT_16  2
#define MAT_16I 5
#define MAT_32  4
#define MAT_64  8

#define BIG_LIMIT 1000000

///////////////////////// More jobs can be done /////////////////////////////
// 1. 智能指针 from https://github.com/roostaiyan/CudaSharedPtr
    
// task: 1.改ref_count 2.改=,==

    // int 	get_nrows ()                        done
    //  int 	get_ncols ()                    done
    //  int * 	shape ()                        done
    //  void 	print_shape ()                  done
    //  bool 	shape_equals (Matrix &other)    done
    //  void 	print ()                        done
    //  matrix  copy ()                         done
      
    //       matrix ()
    //       Construct a new matrix object. More...
      
    //       matrix (int nrows, int ncols)      done
    //       Construct a new matrix object by nrows and ncols. 
      
    //       matrix (int nrows, int ncols, T fill) done
    //       Construct a new matrix object. More...
      
    //       matrix (const matrix< T > &other)
    //       Copy constructor. More...
      
    //       ~matrix ()
    //       Destroy the matrix object when no refs left. More...
      

    //       Make a hard copy of the caller. More...
      
    //  matrix< T > & 	operator= (const matrix< T > &other)
    //       Assignment operator. More...
      
      


      
    //  T * 	operator[] (int i)
    //       Index operator. More...
      
    //  T & 	operator() (int i, int j)
    //       Another way to get an element by two indexes. More...
      
    //  bool 	operator== (matrix< T > &other)
    //       Judge if two matrices are equal. More...
      
    //  bool 	operator!= (matrix< T > &other)
    //       Judge if two matrices are not equal. More...
      
    //  matrix< T > 	operator+ (matrix< T > &other)
    //       Matrix addition. More...
      
    //  matrix< T > 	operator- (matrix< T > &other)
    //       Matrix subtraction. More...
      
    //  matrix< T > 	operator* (matrix< T > &other)
    //       Matrix multiplication. More...
      
    //  matrix< T > 	operator* (T coef)
    //       Multiply whole matrix by a number. More...
      
    //  matrix< T > 	operator^ (int expo)
    //       Matrix power. More...
      
    //  matrix< T > & 	operator*= (matrix< T > &other)
    //       Multiplication assignment by another matrix. More...
      
    //  matrix< T > & 	operator*= (T coef)
    //       Multiplication assignment by a number. More...
      
    //  matrix< T > & 	operator+= (matrix< T > &other)
    //       Addition assignment. More...
      
    //  matrix< T > & 	operator-= (matrix< T > &other)
    //       Subtraction assignment. More...
      
    //  matrix< T > 	multiply_elements (matrix< T > &other)
    //       Multiply the elements of two matrices. More...
      
    //  matrix< T > 	submatrix_ROI (int row_start, int row_end, int col_start, int col_end)
    //       Get a submatrix with ROI concept. More...
      
    //  matrix< T > 	submatrix_cpy (int row_start, int row_end, int col_start, int col_end)
    //       Create a submatrix by hard copy. More...
      
    //  matrix< T > 	submatrix (int row_start, int row_end, int col_start, int col_end)
    //       Alias of submatrix_ROI. More...
      
    //  void 	adjust_ROI (int row_start, int row_end, int col_start, int col_end)
    //       Adjust the location of ROI. More...
      
    //  Static Public Member Functions
    //  static matrix< T > 	create_row_vec (int ncols, T fill)
    //       Create a row vector. More...
      
    //  static matrix< T > 	create_col_vec (int nrows, T fill)
    //       Create a column vector. More...
      
      
    //  static matrix< T > 	multiply_matrix (matrix< T > &m1, matrix< T > &m2) done
      
      
    //  static matrix< T > 	merge_matrix (matrix< T > &C11, matrix< T > &C12, matrix< T > &C21, matrix< T > &C22)
    //       Merge four submatrices. More...
      
    //  Private Attributes
    //  int 	nrows                           done
      
    //  int 	ncols                           done
    //       Number of columns.
      
    //  T * 	data                            done??
      
    //  const matrix< T > * 	parent_matrix
    //       Pointer of a submatrix's parent. More...
      
    //  int * 	ref_count
    //       Count the number of matrices that share the same data. More...
      
    //  Friends
    //  template<typename U >
    //  matrix< U > 	operator* (int coef, matrix< U > &m)
    //       Multiply an interger by a matrix. More...


////////////////////// 这些核函数将在Project5中得到优化  ////////////////////////////////////////
template <typename T>
__global__
void matrixAddKernel(T *a, T *b, T *result, size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < rows * cols; i += stride) {
        result[i] = a[i] + b[i];
    }
}

template <typename T>
__global__
void matrixSubtractKernel(T *a, T *b, T *result, size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < rows * cols; i += stride) {
        result[i] = a[i] - b[i];
    }
}

#define BLOCK_X 16
#define BLOCK_Y 16

#define TILE_X 128
#define TILE_X_4 32
#define TILE_Y 128
#define TILE_Y_4 32

#define TILE_K 16

#define WPTN 8
#define WPTM 8
#define WPTN_4  2

// // be faster!  https://github.com/njuhope/cuda_sgemm/blob/master/gemm.cu
// __global__ void gemm_kernel_NN(
//     const float* __restrict__ A,
//     const float* __restrict__ B,
//     float4* __restrict__ C,
//     float alpha, float beta,
//     int M, int N, int K)
// {
//     __shared__ float4 smem_a[2][TILE_K * TILE_Y_4];
//     __shared__ float4 smem_b[2][TILE_K * TILE_X_4];

//     int tx = threadIdx.x % 16;
//     int ty = threadIdx.x / 16;

//     int tx4 = threadIdx.x % 4;
//     int ty4 = threadIdx.x / 4;

//     int tx32 = threadIdx.x % 32;
//     int ty32 = threadIdx.x / 32;

//     //! every thread block read TILE_Y rows of A
//     //! every 4 thread read a row of A with TILE_K  elements
//     //! every thread read 4 elements
//     const float* pA = (A + K * TILE_Y * blockIdx.y + ty4 * K + tx4 * 4);
//     //! every thread block read TILE_X columns of B
//     //! every 32 thread read a row of B with TILE_X elements
//     //! every thread read 4 elements
//     const float* pB = (B + TILE_X * blockIdx.x + ty32 * N + tx32 * 4);

//     //! every thread block write TILE_Y/4 rows of C, TILE_X_4 * 4(float4)
//     //! columns of C
//     float4* pC = C + TILE_Y * blockIdx.y * N / 4 + TILE_X_4 * blockIdx.x;

//     int sts_a_offset = tx4 * 4 * TILE_Y + ty4;
//     int sts_b_offset = ty32 * TILE_X_4 + tx32;

//     float4 f4_zero = make_float4(0.f, 0.f, 0.f, 0.f);
//     bool valid_ld_a_0 = ((blockIdx.y * TILE_Y + ty4) < M) && ((tx4 * 4) < K);
//     bool valid_ld_a_1 = ((blockIdx.y * TILE_Y + ty4 + 64) < M) && ((tx4 * 4) < K); 
//     bool valid_ld_b_0 = ((blockIdx.x * TILE_X + tx32 * 4) < N) && (ty32 < K);
//     bool valid_ld_b_1 = ((blockIdx.x * TILE_X + tx32 * 4) < N) && ((ty32 + 8) < K);

//     float4 ldg_a_reg[2];
//     float4 ldg_b_reg[2];

//     ldg_a_reg[0] = valid_ld_a_0 ? *(const float4*)pA : f4_zero;
//     ldg_a_reg[1] = valid_ld_a_1 ? *(const float4*)(pA + 64 * K) : f4_zero;
//     ldg_b_reg[0] = valid_ld_b_0 ? *(const float4*)(pB + 0 * N) : f4_zero;
//     ldg_b_reg[1] = valid_ld_b_1 ? *(const float4*)(pB + 8 * N) : f4_zero;

//     float4 c[WPTM][WPTN_4] = { { f4_zero } };

//     *((float*)&smem_a[0][0] + sts_a_offset + 0 * TILE_Y + 0) = ldg_a_reg[0].x;
//     *((float*)&smem_a[0][0] + sts_a_offset + 1 * TILE_Y + 0) = ldg_a_reg[0].y;
//     *((float*)&smem_a[0][0] + sts_a_offset + 2 * TILE_Y + 0) = ldg_a_reg[0].z;
//     *((float*)&smem_a[0][0] + sts_a_offset + 3 * TILE_Y + 0) = ldg_a_reg[0].w;
//     *((float*)&smem_a[0][0] + sts_a_offset + 0 * TILE_Y + 64) = ldg_a_reg[1].x;
//     *((float*)&smem_a[0][0] + sts_a_offset + 1 * TILE_Y + 64) = ldg_a_reg[1].y;
//     *((float*)&smem_a[0][0] + sts_a_offset + 2 * TILE_Y + 64) = ldg_a_reg[1].z;
//     *((float*)&smem_a[0][0] + sts_a_offset + 3 * TILE_Y + 64) = ldg_a_reg[1].w;

//     smem_b[0][sts_b_offset + 0] = ldg_b_reg[0];
//     smem_b[0][sts_b_offset + 8 * TILE_X_4] = ldg_b_reg[1];

//     __syncthreads();

//     int i = 0;
//     int write_stage_idx = 1;

//     float4 reg_a[2][2];
//     float4 reg_b[2][2];

//     reg_a[0][0] = smem_a[0][0 + ty];
//     reg_a[0][1] = smem_a[0][16 + ty];
//     reg_b[0][0] = smem_b[0][0 + tx];
//     reg_b[0][1] = smem_b[0][16 + tx];

//     do
//     {
//         i += 16;
//         valid_ld_a_0 = (valid_ld_a_0 && ((tx4 * 4 + i) < K));
//         valid_ld_a_1 = (valid_ld_a_1 && ((tx4 * 4 + i) < K));
//         valid_ld_b_0 = (valid_ld_b_0 && ((ty32 + i) < K));
//         valid_ld_b_1 = (valid_ld_b_1 && ((ty32 + 8 + i) < K));

//         ldg_a_reg[0] = (valid_ld_a_0) ? *(const float4*)(pA + i + 0) : f4_zero;
//         ldg_a_reg[1] = (valid_ld_a_1) ? *(const float4*)(pA + i + 64 * K) : f4_zero;
//         ldg_b_reg[0] = (valid_ld_b_0) ? *(const float4*)(pB + (i + 0) * N) : f4_zero;
//         ldg_b_reg[1] = (valid_ld_b_1) ? *(const float4*)(pB + (i + 8) * N) : f4_zero;

//         int load_stage_idx = write_stage_idx ^ 1;

// #pragma unroll
//         for (int j = 0; j < TILE_K - 1; j++)
//         {
//             reg_a[(j + 1) % 2][0] = smem_a[load_stage_idx][(j + 1) *  TILE_Y_4 + 0 + ty];
//             reg_a[(j + 1) % 2][1] = smem_a[load_stage_idx][(j + 1) *  TILE_Y_4 + 16 + ty];
//             reg_b[(j + 1) % 2][0] = smem_b[load_stage_idx][(j + 1) *  TILE_X_4 + 0 + tx];
//             reg_b[(j + 1) % 2][1] = smem_b[load_stage_idx][(j + 1) *  TILE_X_4 + 16 + tx];
//             c[0][0].x += reg_a[j % 2][0].x * reg_b[j % 2][0].x;
//             c[0][0].y += reg_a[j % 2][0].x * reg_b[j % 2][0].y;
//             c[0][0].z += reg_a[j % 2][0].x * reg_b[j % 2][0].z;
//             c[0][0].w += reg_a[j % 2][0].x * reg_b[j % 2][0].w;
//             c[0][1].x += reg_a[j % 2][0].x * reg_b[j % 2][1].x;
//             c[0][1].y += reg_a[j % 2][0].x * reg_b[j % 2][1].y;
//             c[0][1].z += reg_a[j % 2][0].x * reg_b[j % 2][1].z;
//             c[0][1].w += reg_a[j % 2][0].x * reg_b[j % 2][1].w;
//             c[1][0].x += reg_a[j % 2][0].y * reg_b[j % 2][0].x;
//             c[1][0].y += reg_a[j % 2][0].y * reg_b[j % 2][0].y;
//             c[1][0].z += reg_a[j % 2][0].y * reg_b[j % 2][0].z;
//             c[1][0].w += reg_a[j % 2][0].y * reg_b[j % 2][0].w;
//             c[1][1].x += reg_a[j % 2][0].y * reg_b[j % 2][1].x;
//             c[1][1].y += reg_a[j % 2][0].y * reg_b[j % 2][1].y;
//             c[1][1].z += reg_a[j % 2][0].y * reg_b[j % 2][1].z;
//             c[1][1].w += reg_a[j % 2][0].y * reg_b[j % 2][1].w;
//             c[2][0].x += reg_a[j % 2][0].z * reg_b[j % 2][0].x;
//             c[2][0].y += reg_a[j % 2][0].z * reg_b[j % 2][0].y;
//             c[2][0].z += reg_a[j % 2][0].z * reg_b[j % 2][0].z;
//             c[2][0].w += reg_a[j % 2][0].z * reg_b[j % 2][0].w;
//             c[2][1].x += reg_a[j % 2][0].z * reg_b[j % 2][1].x;
//             c[2][1].y += reg_a[j % 2][0].z * reg_b[j % 2][1].y;
//             c[2][1].z += reg_a[j % 2][0].z * reg_b[j % 2][1].z;
//             c[2][1].w += reg_a[j % 2][0].z * reg_b[j % 2][1].w;
//             c[3][0].x += reg_a[j % 2][0].w * reg_b[j % 2][0].x;
//             c[3][0].y += reg_a[j % 2][0].w * reg_b[j % 2][0].y;
//             c[3][0].z += reg_a[j % 2][0].w * reg_b[j % 2][0].z;
//             c[3][0].w += reg_a[j % 2][0].w * reg_b[j % 2][0].w;
//             c[3][1].x += reg_a[j % 2][0].w * reg_b[j % 2][1].x;
//             c[3][1].y += reg_a[j % 2][0].w * reg_b[j % 2][1].y;
//             c[3][1].z += reg_a[j % 2][0].w * reg_b[j % 2][1].z;
//             c[3][1].w += reg_a[j % 2][0].w * reg_b[j % 2][1].w;
//             c[4][0].x += reg_a[j % 2][1].x * reg_b[j % 2][0].x;
//             c[4][0].y += reg_a[j % 2][1].x * reg_b[j % 2][0].y;
//             c[4][0].z += reg_a[j % 2][1].x * reg_b[j % 2][0].z;
//             c[4][0].w += reg_a[j % 2][1].x * reg_b[j % 2][0].w;
//             c[4][1].x += reg_a[j % 2][1].x * reg_b[j % 2][1].x;
//             c[4][1].y += reg_a[j % 2][1].x * reg_b[j % 2][1].y;
//             c[4][1].z += reg_a[j % 2][1].x * reg_b[j % 2][1].z;
//             c[4][1].w += reg_a[j % 2][1].x * reg_b[j % 2][1].w;
//             c[5][0].x += reg_a[j % 2][1].y * reg_b[j % 2][0].x;
//             c[5][0].y += reg_a[j % 2][1].y * reg_b[j % 2][0].y;
//             c[5][0].z += reg_a[j % 2][1].y * reg_b[j % 2][0].z;
//             c[5][0].w += reg_a[j % 2][1].y * reg_b[j % 2][0].w;
//             c[5][1].x += reg_a[j % 2][1].y * reg_b[j % 2][1].x;
//             c[5][1].y += reg_a[j % 2][1].y * reg_b[j % 2][1].y;
//             c[5][1].z += reg_a[j % 2][1].y * reg_b[j % 2][1].z;
//             c[5][1].w += reg_a[j % 2][1].y * reg_b[j % 2][1].w;
//             c[6][0].x += reg_a[j % 2][1].z * reg_b[j % 2][0].x;
//             c[6][0].y += reg_a[j % 2][1].z * reg_b[j % 2][0].y;
//             c[6][0].z += reg_a[j % 2][1].z * reg_b[j % 2][0].z;
//             c[6][0].w += reg_a[j % 2][1].z * reg_b[j % 2][0].w;
//             c[6][1].x += reg_a[j % 2][1].z * reg_b[j % 2][1].x;
//             c[6][1].y += reg_a[j % 2][1].z * reg_b[j % 2][1].y;
//             c[6][1].z += reg_a[j % 2][1].z * reg_b[j % 2][1].z;
//             c[6][1].w += reg_a[j % 2][1].z * reg_b[j % 2][1].w;
//             c[7][0].x += reg_a[j % 2][1].w * reg_b[j % 2][0].x;
//             c[7][0].y += reg_a[j % 2][1].w * reg_b[j % 2][0].y;
//             c[7][0].z += reg_a[j % 2][1].w * reg_b[j % 2][0].z;
//             c[7][0].w += reg_a[j % 2][1].w * reg_b[j % 2][0].w;
//             c[7][1].x += reg_a[j % 2][1].w * reg_b[j % 2][1].x;
//             c[7][1].y += reg_a[j % 2][1].w * reg_b[j % 2][1].y;
//             c[7][1].z += reg_a[j % 2][1].w * reg_b[j % 2][1].z;
//             c[7][1].w += reg_a[j % 2][1].w * reg_b[j % 2][1].w;
//         }

//         //! the last iter K, write the global data to shared memory which will
//         //! be used in the next iteration
//         if(i < K) {
//             *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 0 * TILE_Y + 0) = ldg_a_reg[0].x;
//             *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 1 * TILE_Y + 0) = ldg_a_reg[0].y;
//             *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 2 * TILE_Y + 0) = ldg_a_reg[0].z;
//             *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 3 * TILE_Y + 0) = ldg_a_reg[0].w;
//             *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 0 * TILE_Y + 64) = ldg_a_reg[1].x;
//             *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 1 * TILE_Y + 64) = ldg_a_reg[1].y;
//             *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 2 * TILE_Y + 64) = ldg_a_reg[1].z;
//             *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 3 * TILE_Y + 64) = ldg_a_reg[1].w;

//             smem_b[write_stage_idx][sts_b_offset + 0] = ldg_b_reg[0];
//             smem_b[write_stage_idx][sts_b_offset + 8 * TILE_X_4] = ldg_b_reg[1];
//             __syncthreads();
//             write_stage_idx ^= 1;
//         }

//         //! load data from shared memory to register for the next TILE_K
//         //! iteration
//         reg_a[0][0] = smem_a[load_stage_idx ^ 1][0 + ty];
//         reg_a[0][1] = smem_a[load_stage_idx ^ 1][16 + ty];
//         reg_b[0][0] = smem_b[load_stage_idx ^ 1][0 + tx];
//         reg_b[0][1] = smem_b[load_stage_idx ^ 1][16 + tx];

//         //! compute the last TILE_K-1 iteration, the register data is load ahead
//         c[0][0].x += reg_a[1][0].x * reg_b[1][0].x;
//         c[0][0].y += reg_a[1][0].x * reg_b[1][0].y;
//         c[0][0].z += reg_a[1][0].x * reg_b[1][0].z;
//         c[0][0].w += reg_a[1][0].x * reg_b[1][0].w;
//         c[0][1].x += reg_a[1][0].x * reg_b[1][1].x;
//         c[0][1].y += reg_a[1][0].x * reg_b[1][1].y;
//         c[0][1].z += reg_a[1][0].x * reg_b[1][1].z;
//         c[0][1].w += reg_a[1][0].x * reg_b[1][1].w;
//         c[1][0].x += reg_a[1][0].y * reg_b[1][0].x;
//         c[1][0].y += reg_a[1][0].y * reg_b[1][0].y;
//         c[1][0].z += reg_a[1][0].y * reg_b[1][0].z;
//         c[1][0].w += reg_a[1][0].y * reg_b[1][0].w;
//         c[1][1].x += reg_a[1][0].y * reg_b[1][1].x;
//         c[1][1].y += reg_a[1][0].y * reg_b[1][1].y;
//         c[1][1].z += reg_a[1][0].y * reg_b[1][1].z;
//         c[1][1].w += reg_a[1][0].y * reg_b[1][1].w;
//         c[2][0].x += reg_a[1][0].z * reg_b[1][0].x;
//         c[2][0].y += reg_a[1][0].z * reg_b[1][0].y;
//         c[2][0].z += reg_a[1][0].z * reg_b[1][0].z;
//         c[2][0].w += reg_a[1][0].z * reg_b[1][0].w;
//         c[2][1].x += reg_a[1][0].z * reg_b[1][1].x;
//         c[2][1].y += reg_a[1][0].z * reg_b[1][1].y;
//         c[2][1].z += reg_a[1][0].z * reg_b[1][1].z;
//         c[2][1].w += reg_a[1][0].z * reg_b[1][1].w;
//         c[3][0].x += reg_a[1][0].w * reg_b[1][0].x;
//         c[3][0].y += reg_a[1][0].w * reg_b[1][0].y;
//         c[3][0].z += reg_a[1][0].w * reg_b[1][0].z;
//         c[3][0].w += reg_a[1][0].w * reg_b[1][0].w;
//         c[3][1].x += reg_a[1][0].w * reg_b[1][1].x;
//         c[3][1].y += reg_a[1][0].w * reg_b[1][1].y;
//         c[3][1].z += reg_a[1][0].w * reg_b[1][1].z;
//         c[3][1].w += reg_a[1][0].w * reg_b[1][1].w;
//         c[4][0].x += reg_a[1][1].x * reg_b[1][0].x;
//         c[4][0].y += reg_a[1][1].x * reg_b[1][0].y;
//         c[4][0].z += reg_a[1][1].x * reg_b[1][0].z;
//         c[4][0].w += reg_a[1][1].x * reg_b[1][0].w;
//         c[4][1].x += reg_a[1][1].x * reg_b[1][1].x;
//         c[4][1].y += reg_a[1][1].x * reg_b[1][1].y;
//         c[4][1].z += reg_a[1][1].x * reg_b[1][1].z;
//         c[4][1].w += reg_a[1][1].x * reg_b[1][1].w;
//         c[5][0].x += reg_a[1][1].y * reg_b[1][0].x;
//         c[5][0].y += reg_a[1][1].y * reg_b[1][0].y;
//         c[5][0].z += reg_a[1][1].y * reg_b[1][0].z;
//         c[5][0].w += reg_a[1][1].y * reg_b[1][0].w;
//         c[5][1].x += reg_a[1][1].y * reg_b[1][1].x;
//         c[5][1].y += reg_a[1][1].y * reg_b[1][1].y;
//         c[5][1].z += reg_a[1][1].y * reg_b[1][1].z;
//         c[5][1].w += reg_a[1][1].y * reg_b[1][1].w;
//         c[6][0].x += reg_a[1][1].z * reg_b[1][0].x;
//         c[6][0].y += reg_a[1][1].z * reg_b[1][0].y;
//         c[6][0].z += reg_a[1][1].z * reg_b[1][0].z;
//         c[6][0].w += reg_a[1][1].z * reg_b[1][0].w;
//         c[6][1].x += reg_a[1][1].z * reg_b[1][1].x;
//         c[6][1].y += reg_a[1][1].z * reg_b[1][1].y;
//         c[6][1].z += reg_a[1][1].z * reg_b[1][1].z;
//         c[6][1].w += reg_a[1][1].z * reg_b[1][1].w;
//         c[7][0].x += reg_a[1][1].w * reg_b[1][0].x;
//         c[7][0].y += reg_a[1][1].w * reg_b[1][0].y;
//         c[7][0].z += reg_a[1][1].w * reg_b[1][0].z;
//         c[7][0].w += reg_a[1][1].w * reg_b[1][0].w;
//         c[7][1].x += reg_a[1][1].w * reg_b[1][1].x;
//         c[7][1].y += reg_a[1][1].w * reg_b[1][1].y;
//         c[7][1].z += reg_a[1][1].w * reg_b[1][1].z;
//         c[7][1].w += reg_a[1][1].w * reg_b[1][1].w;
        
//     } while (i < K);

// #pragma unroll
//     for (int wm = 0; wm < WPTM; wm++){
// #pragma unroll
//         for (int wn = 0; wn < WPTN_4; wn++){
//             c[wm][wn].x *= alpha;
//             c[wm][wn].y *= alpha;
//             c[wm][wn].z *= alpha;
//             c[wm][wn].w *= alpha;
//         }
//     }

// #pragma unroll
//     for (int wm = 0; wm < 4; wm++){
// #pragma unroll
//         for (int wn = 0; wn < WPTN_4; wn++){
//             if (((blockIdx.y * TILE_Y + ty * 4 + wm) < M) 
//                 && ((blockIdx.x * TILE_X + wn * 64 + tx * 4) < N)) {
//                 if (beta != 0) {
//                     float4 vec4c = *(pC + ((ty * 4 + wm) * N / 4 + wn * 16 + tx));
//                     vec4c.x = vec4c.x * beta + c[wm][wn].x;
//                     vec4c.y = vec4c.y * beta + c[wm][wn].y;
//                     vec4c.z = vec4c.z * beta + c[wm][wn].z;
//                     vec4c.w = vec4c.w * beta + c[wm][wn].w;
//                     *(pC + (ty * 4 + wm) * N / 4 + wn * 16 + tx) = vec4c;
//                 } else {
//                     *(pC + (ty * 4 + wm) * N / 4 + wn * 16 + tx) = c[wm][wn];
//                 }
//             }
//         }
//     }

// #pragma unroll
//     for (int wm = 0; wm < 4; wm++){
// #pragma unroll
//         for (int wn = 0; wn < WPTN_4; wn++){
//             if (((blockIdx.y * TILE_Y + 64 + ty * 4 + wm) < M) 
//                 && ((blockIdx.x * TILE_X + wn * 64 + tx * 4) < N)) {
//                 if (beta != 0) {
//                     float4 vec4c = *(pC + ((64 + ty * 4 + wm) * N / 4 + wn * 16 + tx));
//                     vec4c.x = vec4c.x * beta + c[wm + 4][wn].x;
//                     vec4c.y = vec4c.y * beta + c[wm + 4][wn].y;
//                     vec4c.z = vec4c.z * beta + c[wm + 4][wn].z;
//                     vec4c.w = vec4c.w * beta + c[wm + 4][wn].w;
//                     *(pC + (64 + ty * 4 + wm) * N / 4 + wn * 16 + tx) = vec4c;
//                 } else {
//                     *(pC + (64 + ty * 4 + wm) * N / 4 + wn * 16 + tx) = c[wm + 4][wn];
//                 }
//             }
//         }
//     }
// }

/**
@brief GPU mul kernel
*/
template <typename T>
__global__   
void matrixMulKernel(T *a, T *b, T *result, size_t aRows, size_t aCols, size_t bCols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;// 尝试了改变顺序，但是都很大

    // if (row < aRows && col < bCols) {
        T sum = 0;
        for (size_t k = 0; k < aCols; ++k) {
            sum += a[row * aCols + k] * b[k * bCols + col];
        }
        result[row * bCols + col] = sum;
    // }
}

template <typename T>
__global__ void matrixScalarMulKernel(T *matrix, T *result, int rows, int cols, T scalar) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        result[index] = matrix[index] * scalar;
    }
}


template <typename T>
__global__ void matrixAddScalarKernel(T *matrix, T *result, int rows, int cols, T scalar) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        result[index] = matrix[index] + scalar;
    }
}


__global__ void sgemmKernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}


// CUDA kernel: aA + b
template <typename T>
__global__ void matrixScalarAddKernel(T *A, T *C, int rows, int cols, T a, T b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        C[index] = a * A[index] + b;
    }
}

template <typename T>
void matrixScalarAdd(T *A, T *C, int rows, int cols, T a, T b) {
    // Define kernel launch configuration
    dim3 blockDim(32, 32);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matrixScalarAddKernel<<<gridDim, blockDim>>>(A, C, rows, cols, a, b);

    // Synchronize threads
    cudaDeviceSynchronize();
}


template <typename T>
__global__ void LUDecomposition(T *A, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Perform LU decomposition
    for (int k = 0; k < n - 1; k++) {
        if (tid >= k + 1 && tid < n) {
            A[tid * n + k] /= A[k * n + k];
            for (int i = k + 1; i < n; i++) {
                if (tid >= i && tid < n) {
                    A[tid * n + i] -= A[tid * n + k] * A[k * n + i];
                }
            }
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void matrixAddWithConstantKernel(T* A,  T* C, int size, T c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        C[tid] = A[tid]  + c;
    }
}

template<typename T>
__global__ void scalarMatrixMulKernel(const T* A, T* C, int size, T scalar) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        C[tid] = scalar * A[tid];
    }
}

///////////////////////////////////////////

// 检查CUDA函数的错误
#define CUDA_CHECK(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 检查cuFFT函数的错误
#define CUFFT_CHECK(call) \
do { \
    cufftResult_t status = call; \
    if (status != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error in file '%s' in line %i\n", \
                __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


#define CHECK(call)                                                         \
    do                                                                      \
    {                                                                       \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess)                                      \
        {                                                                   \
            printf("CUDA Error\n");                                         \
            printf("    File:   %s\n", __FILE__);                           \
            printf("    Line:   %d\n", __LINE__);                           \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)
///////////////////////////////////////////////////////////////////////


/** @brief Base MAT class for GPU memory with reference counting.

Its limitations:

-   no arbitrary dimensions support (only 2D)

@note In contrast with Mat, in most cases GpuMat::isContinuous() == false . This means that rows are
aligned to a size depending on the hardware. Single-row GpuMat is always a continuous matrix.

@note You are not recommended to leave static or global GpuMat variables allocated, that is, to rely
on its destructor. The destruction order of such variables and CUDA context is undefined. GPU memory
release function returns error if the CUDA context has been destroyed before.

Some member functions are described as a "Blocking Call" while some are described as a
"Non-Blocking Call". Blocking functions are synchronous to host. It is guaranteed that the GPU
operation is finished when the function returns. However, non-blocking functions are asynchronous to
host. Those functions may return even if the GPU operation is not finished.

Compared to their blocking counterpart, non-blocking functions accept Stream as an additional
argument. If a non-default stream is passed, the GPU operation may overlap with operations in other
streams.

@sa Mat
 */

template <typename T>
class Matrix {
private:
    T *data;

    size_t rows;
    size_t cols;
    
    int* ref_count;

    /**
    @brief which GPU device
    */
    int device;

    const Matrix<T>* parent_matrix;

public:

    /////////////////////////////////// 构造&析构函数 ////////////////////////////
    /**
    *@brief 
    @param Constructor for the first time 
    */ 
    Matrix(size_t rows, size_t cols, int device = 0)
        : rows(rows), cols(cols),  device(device) {
        
        // std::cout << "hi from Matrix constructor";

        checkRowsCols(rows,cols);

        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if(device >= deviceCount){
            std::cerr << "Invalid device num";
        }
        cudaSetDevice(device); // 选择 GPU

        ref_count = new int(1);
        // data_type = MAT_32;
        if (*ref_count==1) { // Allocate memory for the matrix
            // std::cout << "\n Manage Memory for Matrix! \n";
            cudaMallocManaged(&data, rows * cols * sizeof(T));
        }


        parent_matrix = nullptr;
    }


    // Destructor
    ~Matrix() {
        *(ref_count) -= 1;
        if (*(ref_count) == 0 && data != nullptr) {

            if(parent_matrix == nullptr){
                cudaError_t cudaStatus = cudaFree(data);
                if (cudaStatus != cudaSuccess) {
                    std::cerr << "CUDA Free Error: " << cudaGetErrorString(cudaStatus) << std::endl;
                }
                free(ref_count);
                ref_count = nullptr;
            }else{
                cudaError_t cudaStatus = cudaFree(parent_matrix->data);
                if (cudaStatus != cudaSuccess) {
                    std::cerr << "CUDA Free Error: " << cudaGetErrorString(cudaStatus) << std::endl;
                }
                free(ref_count);
                ref_count = nullptr;
            }
            // std::cout << "hi from destructor\n";
        }
    }


    /**
    * @brief
    @param 拷贝构造函数，增加引用计数
    */ 
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {
        // std::cout<< "Hi from copy constructor\n";
        device = other.device;
        ref_count = (other.ref_count);
        (*ref_count) += 1;
    }
    

    // Getters
    size_t getRows() const noexcept{ return rows; }
    size_t getCols() const noexcept{ return cols; }
    size_t getSize() const noexcept{ return rows*cols; }
    T* getData() const noexcept{ return data;}
    T getRolColData(size_t row, size_t col) const {
        checkRowsCols(row,col); 
        return data[row*cols + col];
    }


    
    // Output operator
    friend std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix) {
        for (size_t i = 0; i < matrix.rows; ++i) {
            for (size_t j = 0; j < matrix.cols; ++j) {
                os << matrix(i, j) << ' ';
            }
            os << '\n';
        }
        return os;
    }



    /////////////////////// 运算符重载 //////////////////////////////////////

    /**
    @brief 重载（）：给矩阵赋值
    @param row, col
    @see const T& operator()
    */ 
    __device__ __host__  T & operator()(size_t row, size_t col) {

        // 检查行号和列号是否超出矩阵范围或者是负数
        if (row >= rows || col >= cols){
            row = rows - 1;
            col = cols - 1;
        }
        
        return data[row * cols + col];
    }


    __device__ __host__ const T& operator()(size_t row, size_t col) const {

        // 检查行号和列号是否超出矩阵范围或者是负数
        if (row >= rows || col >= cols){
            row = rows - 1;
            col = cols - 1;
        }
    
        return data[row * cols + col];
    }
    

    // Matrix addition
    Matrix<T> operator+(const Matrix<T>& other) {
        if (rows != other.rows || cols != other.cols) {
            std::cerr << "Matrix dimensions do not match for addition!\n";
            // Return an empty matrix
            return Matrix(0, 0, false);
        }

        Matrix<T> result(rows, cols);

        // Launch CUDA kernel for addition
        matrixAddKernel<<<(cols + 255) / 256, 256>>>(data, other.data, result.data, rows, cols);
        cudaDeviceSynchronize();
        
        // 这里会被copy和释放
        return result;
    }


        /**
    @brief 矩阵加法：A += c
    
    @param c 要添加的常数
    */
    Matrix<T>& operator+=(const  T c) {
        // 计算线程块和网格大小
        int blockSize = 256;
        int numBlocks = (cols * rows + blockSize - 1) / blockSize;

        // 调用CUDA kernel 执行矩阵加法
        matrixAddWithConstantKernel<<<numBlocks, blockSize>>>(data, data, rows * cols, c);
        cudaDeviceSynchronize();

        return *this; // 返回修改后的当前对象
    }

            /**
    @brief 矩阵加法：A += c
    
    @param c 要添加的常数
    */
    Matrix<T>& operator+(const  T c) {
        int blockSize = 256;
        int numBlocks = (cols * rows + blockSize - 1) / blockSize;
        matrixAddWithConstantKernel<<<numBlocks, blockSize>>>(data, data, rows * cols, c);
        cudaDeviceSynchronize();
        return *this; 
    }


    template<typename U>
    friend Matrix<U> operator*(const U& scalar, const Matrix<U>& mat);
    // Matrix<T> operator*(const T& scalar, const Matrix<T>& mat) {
    //     Matrix<T> result(mat.rows, mat.cols);
    //     int blockSize = 256;
    //     int numBlocks = (mat.rows * mat.cols + blockSize - 1) / blockSize;

    //     // 调用CUDA kernel 执行常数与矩阵的乘法
    //     scalarMatrixMulKernel<<<numBlocks, blockSize>>>(mat.data, result.data, mat.rows * mat.cols, scalar);
    //     cudaDeviceSynchronize();

    //     return result;
    // }




    /**
    @brief 
    @param Matrix A += B 
    */
    Matrix<T>& operator+=(const Matrix<T>& other) {
        if (rows != other.rows || cols != other.cols) {
            std::cerr << "Matrix dimensions do not match for addition!\n";
            // You may choose to handle this mismatch differently, such as throwing an exception
            return *this; // Return the current object without modification
        }
    
        // Launch CUDA kernel for addition
        matrixAddKernel<<<(cols + 255) / 256, 256>>>(data, other.data, data, rows, cols);
        cudaDeviceSynchronize();
    
        return *this; // Return the modified current object
    }
    

    /**
    @brief 
    @param Matrix subtraction 
    */ 
    Matrix<T> operator-(const Matrix<T>& other) {
        if (rows != other.rows || cols != other.cols) {
            std::cerr << "Matrix dimensions do not match for subtraction!\n";
            // Return an empty matrix
            return Matrix(0, 0, false);
        }

        Matrix<T> result(rows, cols);

        // Launch CUDA kernel for subtraction
        matrixSubtractKernel<<<( cols + 255) / 256, 256>>>
                (data, other.data, result.data, rows, cols);
        cudaDeviceSynchronize();

        // result.ref_count += 1;

        return result;
    }

    /**
    @brief 
    * @param Matrix multiplication
    */ 
    Matrix operator*(const Matrix& other) {
        if (cols != other.rows) {
            std::cerr << "Matrix dimensions do not match for multiplication!\n";
            return Matrix(0, 0, false);// Return an empty matrix
        }

        Matrix result(rows, other.cols);

        // Launch CUDA kernel for matrix multiplication
        dim3 blockDim(32, 32);
        dim3 gridDim((result.getCols() + blockDim.x - 1) / blockDim.x, (result.getRows() + blockDim.y - 1) / blockDim.y);
        matrixMulKernel<<<gridDim, blockDim>>>
                (data, other.data, result.data, rows, cols, other.cols);
        cudaDeviceSynchronize();

        // *result.ref_count += 1;

        return result;
    }

    /**
    @brief Set our Matrix as the ROI (Region of Interest) of other
    @param startRow,startCol,roiRows,roiCols
    */ 
    void SetParentMatrix(const Matrix& other){
        const Matrix* par = &other;  // 获取给定矩阵的父矩阵指针
        // 找到给定矩阵的根矩阵（没有父矩阵的矩阵）
        while (par->parent_matrix != nullptr) {
            par = par->parent_matrix;
        }
        // 将当前矩阵的父矩阵指针设置为根矩阵
        parent_matrix = par;

    }

    // 设置ROI（Region of Interest）
    void setROIas(size_t startRow, size_t startCol,size_t numRows,size_t numCols, const Matrix& other) {  
        
        if (data != nullptr && *ref_count==0) {
            cudaFree(data);
            free(ref_count);
        }
        
        const Matrix* par = &other;  // 获取给定矩阵的父矩阵指针

        // 找到给定矩阵的根矩阵（没有父矩阵的矩阵）
        while (par->parent_matrix != nullptr) {
            par = par->parent_matrix;
        }

        // 将当前矩阵的父矩阵指针设置为根矩阵
        parent_matrix = par;

        if (parent_matrix == nullptr) {
            throw std::logic_error("Cannot set ROI on a non-parent matrix");
        }
        
        // 合法性检查
        if (startRow + numRows > parent_matrix->getRows() || startCol + numCols > parent_matrix->getCols()) {
            throw std::out_of_range("ROI out of bounds of parent matrix");
        }
        *(parent_matrix->ref_count) +=1;
        ref_count = parent_matrix->ref_count;

        // 计算数据偏移量
        size_t offset = startRow * parent_matrix->getCols() + startCol;
        // 设置新的数据指针
        data = parent_matrix->getData() + offset;
        // 更新行数和列数
        rows = numRows;
        cols = numCols;
        // 父矩阵指针保持不变
    }



    /**
    *@brief 
    *
    * @param Matrix& other
    *浅拷贝
    */ 
    Matrix& operator=(const Matrix& other) {
        if (this != &other) { // 检查自我赋值
            (*ref_count)--;
            if (data != nullptr && *ref_count==0) {
                cudaFree(data);
                free(ref_count);
            }
            // std::cout << "Hi from =\n";
            rows = other.rows;
            cols = other.cols;
            data = other.data;
            ref_count = other.ref_count;
            (*ref_count)++;
        }
        return *this;
    }

    /*重载 == 运算符
    */ 
    bool operator==(const Matrix& other) const {
        return isEqual(other);
    }



    ///////////////////////////////// 运算函数 ////////////////////

    void multiply(const Matrix& other, Matrix* result_ptr) {
        if (cols != other.rows) {
            std::cerr << "Matrix dimensions do not match for multiplication!\n";
            // Handle error condition here
            return;
        }
    
        // Launch CUDA kernel for matrix multiplication
        dim3 blockDim(32, 32);
        dim3 gridDim((result_ptr->getCols() + blockDim.x - 1) / blockDim.x, (result_ptr->getRows() + blockDim.y - 1) / blockDim.y);
        matrixMulKernel<<<gridDim, blockDim>>>
                (data, other.data, result_ptr->data, rows, cols, other.cols);
        cudaDeviceSynchronize();
    }

    void multiply(const Matrix& other, Matrix* result_ptr, int type_mul) {
        if (cols != other.rows) {
            std::cerr << "Matrix dimensions do not match for multiplication!\n";
            // Handle error condition here
            return;
        }
    
        // Launch CUDA kernel for matrix multiplication
        dim3 blockDim(48, 48);
        dim3 gridDim((result_ptr->getCols() + blockDim.x - 1) / blockDim.x, (result_ptr->getRows() + blockDim.y - 1) / blockDim.y);
        matrixMulKernel<<<gridDim, blockDim>>>
                (data, other.data, result_ptr->data, rows, cols, other.cols);
        cudaDeviceSynchronize();
    }
    
    void changeGPU(int device_num) {
        // 在GPU num上分配内存用于存储目标矩阵
        T* deviceDestMatrix;
        cudaSetDevice(device_num); // 选择GPU
        cudaMalloc(&deviceDestMatrix, rows * cols * sizeof(T));
    
        // 将源矩阵数据从GPU0复制到GPU1
        cudaSetDevice(device); // 选择原来的GPU

        // from device to device_num
        cudaMemcpyPeer(deviceDestMatrix, device_num, data, device, rows * cols * sizeof(T));
    
        
        // 释放device上的目标矩阵内存
        cudaSetDevice(device_num); // 选择GPU1
        cudaFree(deviceDestMatrix);
    }

    // 判断两个矩阵是否相等
    bool isEqual(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            return false; // 如果维度不同，直接返回 false
        }

        for (size_t i = 0; i < rows * cols; ++i) {
            // 比较每个元素的值，考虑到浮点数比较可能存在精度问题，使用浮点数误差范围进行比较
            // 由于其他元素都是整数，我们的判断是可以的
            if (std::fabs(data[i] - other.data[i]) > 1e-6) {
                return false; // 如果有任何一个元素不相等，则返回 false
            }
        }

        return true; // 所有元素都相等，返回 true
    }


    Matrix matmul(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }

        int m = rows;
        int n = other.cols;
        int k = cols;

        Matrix result(m, n);

        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alpha = 1.0;
        const float beta = 0.0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, data, m, other.data, k, &beta, result.data, m);

        cublasDestroy(handle);
        return result;
    }

    Matrix matmul_ops(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }

        int m = rows;
        int n = other.cols;
        int k = cols;

        Matrix result(m, n);

        const float alpha = 1.0;
        const float beta = 0.0;
        OptsGemm(m,n,k,data, other.data,result.data,alpha,beta);
       

        return result;
    }

    // void OptsGemm(int m, int n, int k, float* d_A, float* d_B, float* d_C,
    //             float alpha, float beta) {

    //     constexpr int BLOCK_DIM = 128;
    //     // subm, subn, subk
    //     dim3 block(256);
    //     dim3 grid((m + BLOCK_DIM - 1) / BLOCK_DIM, (n + BLOCK_DIM - 1) / BLOCK_DIM);

    //     gemm_kernel_NN<<<grid, block>>>(d_A, d_B, (float4*)d_C, alpha, beta, m, n,
    //                                 k);
    // }



    //////////////////////////// 工具函数 ////////////////////////////

    /**
     * @brief 
     * 
     * @param row 
     * @param col 
     * @return false if row,col are bigger than 1000000 or smaller than 0
     */
    bool checkRowsCols(size_t row, size_t col) const{
        // std::cout << row << col;
        if(row > BIG_LIMIT || col > BIG_LIMIT ){
            throw "row,col out of limit! check if they are bigger than 1000000 or smaller than 0";
            return false;
        }
        return true;
    }

    /**
    * @brief 
    @param print matrix
    */
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << data[i * cols + j] << " ";
            }
            std::cout << "\n";
        }
    }

    bool shape_equals (const Matrix& other) const{
        if(other.getRows()==rows && other.getCols()== cols){
            return true;
        }
        return false;
    }
    
    /**
    * @brief 
    @param 私有辅助函数，用于确保数据的唯一性
    */ 
    void ensure_unique() {
        if (*ref_count > 1) {
            (*ref_count)--;
            deepCopy(data,cols*rows); // 创建数据的深拷贝
            ref_count = new int(1); // 重置引用计数
        }
    }    

    /**
    @brief 深拷贝函数，用于在CUDA设备之间复制数据
    */ 
    void deepCopy(const T *src_data, size_t size) const {
        cudaMalloc((T**)&src_data, size * sizeof(T)); // 在设备上分配新的内存
        cudaMemcpy(data, src_data, size * sizeof(T), cudaMemcpyDeviceToDevice); // 将数据从源地址复制到新分配的设备内存上
    }

    /**
    @brief 深拷贝函数，用于在CUDA设备之间复制数据
    */ 
    void deepCopy(const T *src_data, size_t size, int device) {
        cudaSetDevice(device);
        cudaMalloc((T**)&src_data, size * sizeof(T)); // 在设备上分配新的内存
        cudaMemcpy(data, src_data, size * sizeof(T), cudaMemcpyDeviceToDevice); // 将数据从源地址复制到新分配的设备内存上
    }

    /**
    @brief 上传数据到 GPU
    @param 上传到GPU上的数据的指针
    */ 
    void upload(T* device_data) {
        cudaSetDevice(device);
        cudaMalloc((void**)&device_data, rows * cols * sizeof(T)); // 在 GPU 上分配内存
        cudaMemcpy(device_data, data, rows * cols * sizeof(T), cudaMemcpyHostToDevice); // 复制到设备
        if (data != nullptr && ref_count == 0) { // 把这个矩阵重新写了
            cudaFree(data);
        }
        data = device_data; // 更新数据指针
    }

    /**
    @brief 从 GPU 下载数据到 CPU
    @param 返回主机上的数据
    @public
    */ 
    T* download() {
        T* host_data = new T[rows * cols]; // 在主机上分配内存
        cudaMemcpy(host_data, data, rows * cols * sizeof(T), cudaMemcpyDeviceToHost); // 复制到主机
        return host_data;
    }    
};


template<typename T>
Matrix<T> operator*(const T& scalar, const Matrix<T>& mat) {
    Matrix<T> result(mat.rows,mat.cols);
    int blockSize = 256;
    int numBlocks = (mat.rows * mat.cols + blockSize - 1) / blockSize;

    scalarMatrixMulKernel<<<numBlocks, blockSize>>>(mat.data, result.data, mat.rows * mat.cols, scalar);
    cudaDeviceSynchronize();

    return result;
}