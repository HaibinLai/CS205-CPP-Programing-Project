import java.io.*;
import java.util.Scanner;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Matrix {


    // 从文件中读取矩阵数据
    public static float[][] readMatrixFromFile(String filename, int n) {
        float[][] matrix = new float[n][n];

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            int row = 0;
            while ((line = br.readLine()) != null && row < n) {
                String[] values = line.trim().split("\\s+");
                for (int col = 0; col < n; col++) {
                    matrix[row][col] = Float.parseFloat(values[col]);
                }
                row++;
            }
        } catch (IOException e) {
            System.out.println(e.getLocalizedMessage());
        }

        return matrix;
    }

    // 执行矩阵相乘
    public static float[][] multiplyMatrices(float[][] matrix1, float[][] matrix2, int n) {
        float[][] result = new float[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }

        return result;
    }

    // 多线程实现的矩阵乘法计算任务
    public static class MatrixMultiplicationTask implements Runnable {
        private float[][] matrix1;
        private float[][] matrix2;
        private float[][] result;
        private int startRow;
        private int endRow;

        public MatrixMultiplicationTask(float[][] matrix1, float[][] matrix2, float[][] result, int startRow, int endRow) {
            this.matrix1 = matrix1;
            this.matrix2 = matrix2;
            this.result = result;
            this.startRow = startRow;
            this.endRow = endRow;
        }

        @Override
        public void run() {
            int size = matrix1.length;

            for (int i = startRow; i < endRow; i++) {
                for (int j = 0; j < size; j++) {
                    for (int k = 0; k < size; k++) {
                        result[i][j] += matrix1[i][k] * matrix2[k][j];
                    }
                }
            }
        }
    }

    // 使用多线程进行矩阵乘法计算
    public static float[][] multiplyMatricesFast(float[][] matrix1, float[][] matrix2, int numThreads) throws InterruptedException {
        int size = matrix1.length;
        float[][] result = new float[size][size];

        // 计算每个线程需要处理的行数
        int rowsPerThread = size / numThreads;

        // 创建线程数组
        Thread[] threads = new Thread[numThreads];

        // 启动线程并进行计算
        for (int i = 0; i < numThreads; i++) {
            int startRow = i * rowsPerThread;
            int endRow = (i == numThreads - 1) ? size : (startRow + rowsPerThread);

            threads[i] = new Thread(new MatrixMultiplicationTask(matrix1, matrix2, result, startRow, endRow));
            threads[i].start();
        }

        // 等待所有线程完成
        for (int i = 0; i < numThreads; i++) {
            threads[i].join();
        }

        return result;
    }

    // 打印矩阵
    public static void printMatrix(float[][] matrix) {
        for (float[] row : matrix) {
            for (float value : row) {
                System.out.print(value + " ");
            }
            System.out.println();
        }
    }

    public static void writeMatrix(float[][] matrix,String filePath) throws IOException {
        File file = new File(filePath);
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))) {
            for (float[] row : matrix) {
                for (float value : row) {
                    bw.write(String.valueOf(value));
                    bw.write(" ");
                }
                bw.newLine();
            }
        }
    }

        public static void main(String[] args) throws InterruptedException {{
            Scanner scan = new Scanner(System.in);
            System.out.println("Enter the size of the matrix:");
            int n = scan.nextInt(); // 矩阵的大小

            String filePath1 = "/home/asc648/haibin/CPPLearning/Project2/Excutable/data/data/" +
                    ""+n+"DAT1.txt";
            String filePath2 = "/home/asc648/haibin/CPPLearning/Project2/Excutable/data/data/" +
                    ""+n+"DAT2.txt";

            String filePathResult = "/home/asc648/haibin/CPPLearning/Project2/Excutable/data/data/2" +
                    "MatrixJAVA_" +
                    "100" +
                    "_dat.txt";


            long stimeRead = System.nanoTime();
            // 读取第一个矩阵
            float[][] matrix1 = readMatrixFromFile(filePath1, n);
            // 读取第二个矩阵
            float[][] matrix2 = readMatrixFromFile(filePath2, n);

            long etimeRead = System.nanoTime();

            // 开始时间
            long stime = System.nanoTime();

            // 定义使用的线程数
            // int numThreads = 4; // 修改为你需要的线程数

            // // 计算矩阵乘积
            // float[][] result = multiplyMatricesFast(matrix1, matrix2, numThreads);
            

            // 执行矩阵相乘
            float[][] result = multiplyMatrices(matrix1, matrix2, n);

            
            // 结束时间
            long etime = System.nanoTime();

            // 打印结果
            //        System.out.println("Result of Matrix Multiplication:");
            //        printMatrix(result);

            try {
                long stimeWrite = System.nanoTime();
                writeMatrix(result,filePathResult);
                long etimeWrite = System.nanoTime();

                long a = (etimeWrite - stimeWrite);

                double m2 = (double) (etimeWrite - stimeRead) / 1000000;
                System.out.printf("%f\n", m2);

                // 计算执行时间
                double Read = (double) (etimeRead - stimeRead);
                double m3 = (double) Read / 1000000;
                System.out.printf("%f\n", m3);



                double time = (double) (etime - stime);
                double m4 = (double) time / 1000000;
                System.out.printf("%f\n", m4);



                double m = (double) a / 1000000;
                System.out.printf("%f\n", m);


            } catch (IOException e) {
                throw new RuntimeException(e);
            }

        }
    }
}
