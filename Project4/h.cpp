#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

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


// be faster!  https://github.com/njuhope/cuda_sgemm/blob/master/gemm.cu
/**
@brief GPU mul kernel
*/
template <typename T>
__global__   
void matrixMulKernel(T *a, T *b, T *result, size_t aRows, size_t aCols, size_t bCols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;// 尝试了改变顺序，但是都很大

    if (row < aRows && col < bCols) {
        T sum = 0;
        for (size_t k = 0; k < aCols; ++k) {
            sum += a[row * aCols + k] * b[k * bCols + col];
        }
        result[row * bCols + col] = sum;
    }
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
        
        std::cout << "hi from Matrix constructor";

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
            std::cout << "\n Manage Memory for Matrix! \n";
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
        std::cout<< "Hi from copy constructor\n";
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
            std::cout << "Hi from =\n";
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


