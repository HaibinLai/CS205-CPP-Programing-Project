/* 
 * -- Hai Bin Machine(HBM) 5100               
 *    HBM 5100 - 1.1.2 - March 10, 2024                          
 *    Haibin Lai                                                
 *    Southern University of Science and Technology, Shenzhen                                
 *    CS205 C/C++ Project01                                 
 *    
 * -- Key Features:
 *      Security & Cryptography: 
 *          RSA Encryption
 *      Acceleration:
 *          OpenMP Programmming
 *          CUDA (in .cu file)
 *      Cryptography:
 *          GMP & PBC library                  
 *                                                     
 * -- Copyright notice and Licensing terms:                             
 *      MIT licensse
 * ---------------------------------------------------------------------
 */ 

/******************** How to Compile the file: ************************************

# Compile the source code
gcc src/haibin_calculator.c -o haibinCalculator -lm 

# -lm is used to link the math library, 
    since I use the pow function in my code for Karatsuba Algorithm
# -o haibinCalculator is used to name the output file
# -DUSE_LONG_INPUT is used to activate the long input, which is used to test the long input
# -DDEBUG_PRINT is used to activate the debug print, which is used to print the debug message
# -lgmp is used to link the GMP library, 
    since I use the GMP library in my code for big inteager calculation
# -DACTIVATE_MESSAGE is used to activate the message, which is used to print the message
    that the library we use 
# -DDEBUG_PRINT is used to activate the debug print, which is used to print the debug message
# -fopenmp is used to activate the OpenMP, which is used to accelerate the program

# if you want to try Crytogrophy, please add -DUSE_CRYPTO after -DACTIVATE_MESSAGE
# However it is not work in calculation, since I have not implemented the crypto part in my code

What's more:
# MPI is a good thing, I will try it next time

*************************************************************************************
*/


// TODO: 
// 1. 修复浮点数除法的精度问题               2024/3/2解决
// 2. 修复乘法的 * 问题                     2024/2/29解决
// 3. 大整数的问题                          2024/3/5解决
// 4. 科学计数法的问题                      2024/3/8解决
// 5. 内存泄露问题                          2024/3/10解决

//精度问题：不是算的问题，而是打印的问题

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>       // for sqrt() used in Karatsuba Algorithm
#include <omp.h>

/*
GMP 大整数库，你可以使用它进行比较
*/
#ifdef USE_GMP
#include <gmp.h>
#endif

// 检查是否定义了宏USE_CRYPTO
// 在本次Project中，我们并没有实际应用任何密码学库的功能，我仅仅是在学习宏和大整数计算在密码学中的应用
#ifdef USE_CRYPTO
#include <sodium.h>
#include <pbc/pbc.h>
#define CRYPTO_AVAILABLE 1
    #ifdef ACTIVATE_MESSAGE
    #pragma message("Using Sodium cryptography library.")
    #pragma message("Using pbc cryptography library.")
    #endif
#include <sodium/crypto_secretbox.h>
#else
#define CRYPTO_AVAILABLE 0
    #ifdef ACTIVATE_MESSAGE
    #pragma message("Not using cryptography library.")
    #endif
#endif

// check which math library is used
#ifdef __USE_ISOC99
    #define MATH_LIBRARY "ISO C99"
#elif defined(_ISOC99_SOURCE)
    #define MATH_LIBRARY "ISO C99 with extensions"
#elif defined(__GNU_LIBRARY__)
    #define MATH_LIBRARY "GNU C Library"
#elif defined(_LIBC)
    #define MATH_LIBRARY "Microsoft Visual Studio"
#else
    #define MATH_LIBRARY "Unknown"
#endif

#ifdef ACTIVATE_MESSAGE
    #pragma message("Compiling with " MATH_LIBRARY)
#endif

#define DOUBLE_DIVIDE_PRECISON 8
#define DOUBLE_EXACTLY_PRECISON 8
#define OUT_OF_CALCULATOR_RANGE 1024

/*
* 我们定义CalculatePrint来打印计算结果，其他的printf都是用来打印提示信息的或其他版本
*/
#define CalculatePrint(arg...) printf(arg)
#ifdef DEBUG_MODE
    #define DEBUG_PRINT(arg...) printf(arg)
#endif

const float STEINS_GATE_WORLD_LINE = 1.048596f; // Magic Number only to verify calculator version
const float PI = 3.1415926f;
const float EXP = 2.7182818f;
const float SQRT2 = 1.4142136f;
const float LOG2 = 0.69314718f;
const float LOG10 = 2.3025851f;


enum NumberType{
    NOT_A_NUMBER=-1,
    DOUBLE_CAN_HANDLE =1,
    BIG_INTEAGER = 2,
    BIG_DECIMAL = 3,
    EXP_NUMBER = 4
};

enum NumberStatus{
    A_IS_BIGGER = 1,
    B_IS_BIGGER = 2,
    A_EQUAL_B = 0
};

// 定义一个简单的链表结构来跟踪内存分配
// https://blog.csdn.net/weixin_43308899/article/details/135122404
typedef struct Allocation {
    void* address;
    size_t size;
    struct Allocation* next;
} Allocation;
 
Allocation* allocations = NULL;

/////////////////////////////// Human-Computer Interaction Functions /////////////////////////

void PrintHelp() {
    printf("\n");
    printf("\033[1;32mHBM 5100 \033[1;31m-- a simple command-line calculator.\033[0m\n\n");
    printf("Usage: ./haibinCalculator \033[0;36m[operand1]\033[0m \033[1;34m[operation]\033[0m \033[0;36m[operand2]\033[0m\n");
    printf("Operations:\n");
    printf("  +     : Addition\n");
    printf("  -     : Subtraction\n");
    printf("  x     : Multiplication\n");
    printf("  /     : Division\n");
}

void PrintHelpInLine() {
    printf("\n");
    printf("\033[1;32mHBM 5100 \033[1;31m-- a simple command-line calculator.\033[0m\n\n");
    printf("Usage: \033[0;36m[operand1]\033[0m \033[1;34m[operation]\033[0m \033[0;36m[operand2]\033[0m\n");
    printf("Operations:\n");
    printf("  +     : Addition\n");
    printf("  -     : Subtraction\n");
    printf("  *     : Multiplication\n");
    printf("  /     : Division\n");
}

void PrintVersion(){
    printf("\n");
    printf("HBM 5100\n");
    printf("El Psy Kongroo\n");
    printf("HaibinCalculator - a simple command-line calculator.\n");
    printf("Version:1.1.2\n");
    printf("Steins Gate World Line: ");
    printf("%f\n",STEINS_GATE_WORLD_LINE);
    printf("This program comes with ABSOLUTELY NO WARRANTY;\n");
    printf("Program build time: %s\n", __TIMESTAMP__);
}

void ErrorInput(){
    printf("\033[0;31mPlease input the right element to the calculator.\033[0m\n");
    printf("\033[0;33mFind help using ./haibinCalculator -h in command line mode or using h / help in other mode. \033[0m\n");
}

void ErrorOutOfRange(){
    printf("\033[0;31mError: The input is out of the calculator's range.\033[0m\n");
    printf("\033[0;33mFind help using ./haibinCalculator -h in command line mode or using h / help in other mode. \033[0m\n");
}

void ErrorDivideByZero(){
    printf("\033[0;33mError: Division by zero\033[0m\n");
    printf("A number cannot be divied by zero.\n");
}

void PrintMathLibrary(){
    printf("Math Library: %s\n", MATH_LIBRARY);
    if (CRYPTO_AVAILABLE) {
        printf("Crypto Library: Sodium\n");
    }
    else {
        printf("Crypto Library: Not available\n");
        printf("Please install Sodium to enable cryptography features.\n");
    }
}

int CHECK_ENABLE_CRYPTO_LIB(){
    // 如果密码学库可用，则执行密码学相关的代码
#if CRYPTO_AVAILABLE
    printf("Using Sodium cryptography library.\n");
    // 在这里编写使用Sodium的代码
#else
    printf("Sodium cryptography library is not available.\n");
    // 在这里编写备用方案的代码
#endif
}

//////////////////////////// SYSTEM FUNCTIONS //////////////////////////////

//函数来添加内存分配记录
//  https://blog.csdn.net/weixin_43308899/article/details/135122404
void add_allocation(void* p, size_t size){
    Allocation* newAlloc = (Allocation*)malloc(sizeof(Allocation));
    newAlloc->address = p;
    newAlloc->size = size;
    newAlloc->next = allocations;
    allocations = newAlloc;
}

// 函数来移除内存分配记录
// https://blog.csdn.net/weixin_43308899/article/details/135122404
void remove_allocation(void* p) {
    Allocation **ptr = &allocations;

    // 栈移到下一个指针
    while (*ptr) {
        Allocation* entry = *ptr;
        if (entry->address == p) {
            *ptr = entry->next;
            free(entry);
            return;
        }
        ptr = &entry->next;
    }
}

void* Calculator_malloc(size_t size){
    void* p = malloc(size);
    add_allocation(p, size);
    return p;
}

// 自定义的 free 函数
void Calculator_free(void* p) {
    remove_allocation(p);
    free(p);
}

// made by haibin Lai
void clear_allocations() {
    Allocation* current = allocations;
    while (current) {
        Allocation* temp = current;
        current = current->next;
        free(temp->address);
        free(temp);
    }
    allocations = NULL;
}


// 函数来检查和报告内存泄漏
// https://blog.csdn.net/weixin_43308899/article/details/135122404
void check_for_leaks() {
    Allocation* current = allocations;
    if (current == NULL) {
        printf("No memory leaks detected.\n");
    } else {
        printf("Memory leaks detected:\n");
        while (current) {
            printf("Leaked memory at address %p, size %zu\n", current->address, current->size);
            current = current->next;
        }
    }
}


////////////////////////////////// Math Functions ////////////////////////////////////

int Max(int a, int b){
    if(a>=b){return a;}
    else return b;
}

int Min(int a, int b){
    if(a>=b){return b;}
    else return a;
}

/**
 * A Tribute to the legend
 * 致敬传奇平方根倒数算法
 */
float Q_rsqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
//	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}

// int KaratsubaMultiply(int x, int y) {
//     if (x < 10 || y < 10) {
//         return x * y;
//     }

//     // 计算 n 和 n2
//     int n = fmax(log10(x) + 1, log10(y) + 1);
//     int n2 = n / 2;

//     // 拆分 x 和 y
//     int a = x / pow(10, n2);
//     int b = x % (int)pow(10, n2);
//     int c = y / pow(10, n2);
//     int d = y % (int)pow(10, n2);

//     // 递归计算乘积
//     int ac = KaratsubaMultiply(a, c);
//     int bd = KaratsubaMultiply(b, d);
//     int ad_plus_bc = KaratsubaMultiply(a + b, c + d) - ac - bd;

//     // 计算结果
//     int result = ac * pow(10, 2 * n2) + ad_plus_bc * pow(10, n2) + bd;
//     return result;
// }

// double karatsuba_multiply(double x, double y) {
//     if (x < 10 || y < 10) {
//         return x * y;
//     }

//     // 计算 n 和 n2
//     int n = fmax(log10(x) + 1, log10(y) + 1);
//     int n2 = n / 2;

//     // 拆分 x 和 y
//     int a = x / pow(10, n2);
//     int b = x - a * pow(10, n2);
//     int c = y / pow(10, n2);
//     int d = y - c * pow(10, n2);

//     // 递归计算乘积
//     double ac = karatsuba_multiply(a, c);
//     double bd = karatsuba_multiply(b, d);
//     double ad_plus_bc = karatsuba_multiply(a + b, c + d) - ac - bd;

//     // 计算结果
//     double result = ac * pow(10, 2 * n2) + ad_plus_bc * pow(10, n2) + bd;
//     return result;
// }

////////////// Crytography //////////////
#ifdef USE_CRYPTO
void homomorphic_addition(unsigned char *ciphertext1, unsigned char *ciphertext2) {

    // 假设 ciphertext1 和 ciphertext2 分别是两个密文
    // 执行同态加法
    unsigned char result[crypto_secretbox_MACBYTES + Max(strlen(ciphertext1), strlen(ciphertext2))];
    for (size_t i = 0; i < crypto_secretbox_MACBYTES; i++) {
        result[i] = ciphertext1[i] ^ ciphertext2[i];
    }

    // 输出结果
    printf("Homomorphically added result: ");
    for (size_t i = 0; i < crypto_secretbox_MACBYTES; i++) {
        printf("%02x", result[i]);
    }
    printf("\n");
}
#endif

char* Add_carry(char* number, int carry, char* NewNumber){
    int len = strlen(number);
    NewNumber[0] = carry + '0';
    for(int i = 1; i < len+1; i++){
        NewNumber[i] = number[i-1];
    }
    NewNumber[len+2] = '\0';
    printf("NewNumber:%s\n",NewNumber);
    return NewNumber;
}

////////////////////////////////// Auxiliary Functions ////////////////////////////////////


int* stringToIntArray(char* str, int length, int* arr) {
    for (int i = 0; i < length; i++) {
        arr[i] = str[i] - '0';
    }
    return arr;
}

void exchange(char* number1, char* number2){
    char* temp;

    temp = number1;
    number1 = number2;
    number2 = temp;
}

// warning: 原str会被翻转，如果后面要转回来，还要进行一次操作
char* reverseString(char* str) {
    int length = strlen(str);
    for (int i = 0; i < length / 2; i++) {
        char temp = str[i];
        str[i] = str[length - i - 1];
        str[length - i - 1] = temp;
    }
    return str;
}

/**
 * @brief Decide if numStr is double, big data, or expontienl
 * 
 * @param numStr 
 * @return int num
 * num   type              example
 *  1 | inteager       |     45
 *  2 | big inteager   | 23132214231
 *  3 | double         |  0.1231232
 *  4 | scientific num |    1e400
 */
int NumType(char *numStr){

    if(numStr == NULL){return NOT_A_NUMBER;}

    int length_of_num = strlen(numStr);

    int InteagerPartCount = 0;
    int DecimalPartCount = 0;
    

    int ExpPartCount = 0;
    char ExpLevel[20] = "";

    // they are only to detect the "." and "e" sign
    int DecimalSignCount = 0;
    int ExpSignCount = 0;
   
   
    for(int i = 0; i < length_of_num; i++){
        // if you contain other symbol, game is over
        if(numStr[i] != '1' && numStr[i] != '2' && numStr[i] != '3' && numStr[i] != '4' 
        && numStr[i] != '5' && numStr[i] != '6' && numStr[i] != '7' && numStr[i] != '8'
        && numStr[i] != '9' && numStr[i] != '0' && numStr[i] != 'e' && numStr[i] != 'E' 
        && numStr[i] != '+' && numStr[i] != '-' && numStr[i] != '.' && numStr[i] != ' '){
            return NOT_A_NUMBER;
        }
        if(!DecimalSignCount){
            InteagerPartCount++;
        }
        if(DecimalSignCount){
            DecimalPartCount++;
        }
        if(numStr[i] == '.'){
            DecimalSignCount++;
        }

        if(ExpSignCount==1){
            ExpLevel[ExpPartCount] = numStr[i];
            ExpPartCount++;
        }
        if(numStr[i] == 'e' || numStr[i] == 'E'){
            ExpSignCount++;
        }

    }

    int ExpNum = atof(ExpLevel);

    // ..14 1.1.1 not a number
    // 1e1e1 not a number
    if(DecimalSignCount>1 || DecimalSignCount <0 || ExpSignCount <0 || ExpSignCount >1){
        return NOT_A_NUMBER;
    }

    // we may finish our calculator in double version
    // less than 8 digits (7位小数或8位整数)
    if(length_of_num > 0 && length_of_num <= DOUBLE_EXACTLY_PRECISON 
                         && ExpNum < DOUBLE_EXACTLY_PRECISON){
        return DOUBLE_CAN_HANDLE;
    }
    // more than 8 digits (>8)
    else if(length_of_num > DOUBLE_EXACTLY_PRECISON && DecimalSignCount == 0){
        return BIG_INTEAGER;
    }
    else if(length_of_num > DOUBLE_EXACTLY_PRECISON && DecimalSignCount == 1){
        return BIG_DECIMAL;
    }
    // scientific number
    else if(ExpSignCount == 1){
        return EXP_NUMBER;
    }

    // for other cases we just can't recognize or can't handle
    else{
        return OUT_OF_CALCULATOR_RANGE;
    }
    return NOT_A_NUMBER;
}

/*
*@brief Count the decimal digits of a double number
*/
int CountDecimalDigits(double num) {
    int int_count = 0;
    int isDecimal = 0;

    if (num < 0) {
        num *= -1;
    }

    int max_count = 0;

    char result_in_char_format[64] = "";
    sprintf(result_in_char_format,"%.16f", num);

    //解法：三指针
    // check num in deciaml strlen(result_in_char_format)
    for (int i = 0; i < 18; i++) {
        if(isDecimal){
            if(result_in_char_format[i] != '0'){
                max_count = i;
            }
        }
        if(result_in_char_format[i] == '.'){
            isDecimal = 1;
            int_count = i;
            max_count = i;
        }
    }

    return max_count-int_count;
}

/**
 * @brief Split the string by separator
 * 
 * @param src 
 * @param separator 
 * @param dest 
 * @param num 
 */
void Split(char *src,const char *separator,char **dest,int num) {
    char *pNext;
    int count = 0;
    if (src == NULL || strlen(src) == 0)
        return;
    if (separator == NULL || strlen(separator) == 0)
        return;    
        pNext = strtok(src,separator);
    while(pNext != NULL) {
        *dest++ = pNext;
        ++count;
        pNext = strtok(NULL,separator);  
    }  
    num = count;
 }     

 char* RemoveNextLinrArgs(char* argum){
    char *tmp = NULL;
    if ((tmp = strstr(argum, "\n")))
    {
        *tmp = '\0';
    }
    return argum;
 }

// decide which Inteager number is bigger. if 1 is bigger than 2, return 1, else return 2.
// if both of them are the same, then return 0
 int whichPositiveInteagerIsBigger(char* number1, char* number2){
    int len1 = strlen(number1);
    int len2 = strlen(number2);
    if(len1 > len2){
        return A_IS_BIGGER;
    }
    else if(len1 < len2){
        return B_IS_BIGGER;
    }
    else{
        for(int i=0;i<len1;i++){
            if(number1[i] > number2[i]){
                return A_IS_BIGGER;
            }
            else if(number1[i] < number2[i]){
                return B_IS_BIGGER;
            }
        }
        // if all the digits are the same
        return A_EQUAL_B;
    }
 }


//////////////////////////////// BASIC FUNCTION ///////////////////////////////

/*
    OpenMP version of Add_abs
    我们将add中的number1[i]+number2[i]的for循环并行化
*/
void multiply(char *num1, char *num2, char *result) {
    int len1 = 0, len2 = 0;
    while (num1[len1]) len1++;
    while (num2[len2]) len2++;

    int len_result = len1 + len2;
    int carry = 0;

    #pragma omp parallel for shared(result, carry)
    for (int i = 0; i < len_result; i++) {
        int temp = carry;
        #pragma omp parallel for reduction(+:temp)
        for (int j = 0; j <= i; j++) {
            if (j < len1 && i - j < len2) {
                temp += (num1[len1 - j - 1] - '0') * (num2[len2 - (i - j) - 1] - '0');
            }
        }
        result[len_result - i - 1] = temp % 10 + '0';
        carry = temp / 10;
    }

    result[len_result] = '\0';

    // Remove leading zeros
    int start = 0;
    while (result[start] == '0' && result[start] != '\0') {
        start++;
    }
    if (start > 0) {
        for (int i = start; i <= len_result; i++) {
            result[i - start] = result[i];
        }
    }
}

/*
    我们的数组加法的核心底层，Add_abs
*/
char* Add_abs(char* number1, char* number2, char* result) {
    int len1 = strlen(number1);
    int len2 = strlen(number2);
    int maxlen = Max(len1,len2);
    int carry = 0;

    int ptr1 = len1 - 1;
    int ptr2 = len2 - 1;

    int i;
    for(i= maxlen-1; i>=0; i--){
        int digit1 = 0;
        if(ptr1 >= 0){
            digit1 = number1[ptr1] - '0';
            ptr1--;
        }
        int digit2 = 0;
        if(ptr2 >= 0){
            digit2 = number2[ptr2] - '0';
            ptr2--;
        }
        int sum = digit1 + digit2 + carry;
        result[i] = (sum % 10) + '0'; // 将个位数添加到结果字符串中
        carry = sum / 10; // 计算进位
    }
    
    /*
    * 血泪的教训：内存分配与泄露，上了C的第一课了属于是
    */
    if (carry > 0) {
        for(int i=maxlen; i>=0; i--){
            result[i+1] = result[i];
        }
        result[0] = carry + '0'; // 如果最后还有进位，加入结果中
        
        return result;
    }
    else{
        result[maxlen + 1] = '\0'; // 结束字符串
        return result;
    }
}



// only work for number1 > number2
char* Sub_abs(char* number1, char* number2, char* result){
    int len1 = strlen(number1);
    int len2 = strlen(number2);
    int maxlen = Max(len1,len2);
    int carry = 0;

    int ptr1 = len1 - 1;
    int ptr2 = len2 - 1;

    int i;
    for(i= maxlen-1; i>=0; i--){
        int digit1 = 0;
        if(ptr1 >= 0){
            digit1 = number1[ptr1] - '0';
            ptr1--;
        }
        int digit2 = 0;
        if(ptr2 >= 0){
            digit2 = number2[ptr2] - '0';
            ptr2--;
        }
        int sum = digit1 - digit2 - carry;
        if(sum < 0){
            sum += 10;
            carry = 1;
        }
        else{
            carry = 0;
        }
        result[i] = sum + '0'; // 将个位数添加到结果字符串中
    }
    result[maxlen + 1] = '\0'; // 结束字符串
    return result;

}

/*
实现了内存分配与泄露的检查
*/
void Add_using_array(char* number1, char* number2){
    short OperandAIsNegative = 0;
    short OperandBIsNegative = 0;

    if(number1[0] == '-'){
        OperandAIsNegative = 1;
        number1++; // 天才！这样我们的操作数就直接变成了正数
    } 
    else if(number1[0] == '+'){
        number1++;
    }

    if(number2[0] == '-'){
        OperandBIsNegative = 1;
        number2++; // 天才！指针是这个世界上最天才的东西
    }
    else if(number2[0] == '+'){
        number2++;
    }

    int len1 = strlen(number1);
    int len2 = strlen(number2);
    int maxlen = Max(len1,len2);

    char* result = (char*)Calculator_malloc((maxlen + 2) * sizeof(char)); // 为结果分配足够的内存空间


     // both positive
    if(!OperandAIsNegative && !OperandBIsNegative){
        
        result = Add_abs(number1,number2,result);

        CalculatePrint("%s\n",result);
        Calculator_free(result);
        return;
    }else{
        int whoIsBig = whichPositiveInteagerIsBigger(number1,number2);
        // A is negative
        if(OperandAIsNegative && !OperandBIsNegative){
            if(whoIsBig == A_IS_BIGGER){
                result = Sub_abs(number1,number2,result); // -(a-b)
                CalculatePrint("-%s\n",result);
                Calculator_free(result);
                return;
            }
            else if(whoIsBig == B_IS_BIGGER){
                char* result = Sub_abs(number2,number1,result); // b-a
                CalculatePrint("%s\n",result);
                Calculator_free(result);
                return;
            }
            else{ // a == b
                CalculatePrint("0\n");
                Calculator_free(result);
                return;
            }
        }
        // B is negative
        else if(!OperandAIsNegative && OperandBIsNegative){
            
            if(whoIsBig == A_IS_BIGGER){
                result = Sub_abs(number1,number2,result); // a-b
                CalculatePrint("%s\n",result);
                Calculator_free(result);
                return;
            }
            else if(whoIsBig == B_IS_BIGGER){
                result = Sub_abs(number2,number1,result); // -(b-a)
                CalculatePrint("-%s\n",result);
                Calculator_free(result);
                return;
            }
            else{ // a == b
                CalculatePrint("0\n");
                Calculator_free(result);
                return;
            }
        }
        // both negative
        else{
            int carry = 0;
            result = Add_abs(number1,number2,result); // -(a+b)
            
            CalculatePrint("-%s\n",result);
            Calculator_free(result);
            return;
        }
    }
}

void Sub_using_array(char* number1, char* number2){
    short OperandAIsNegative = 0;
    short OperandBIsNegative = 0;

    if(number1[0] == '-'){
        OperandAIsNegative = 1;
        number1++; // 天才！这样我们的操作数就直接变成了正数
    }
    else if(number1[0] == '+'){
        number1++; // 跳过正号
    }
    if(number2[0] == '-'){
        OperandBIsNegative = 1;
        number2++; // 天才！指针是这个世界上最天才的东西
    }
    else if(number2[0] == '+'){
        number2++; // 跳过正号
    }

    int len1 = strlen(number1);
    int len2 = strlen(number2);
    int maxlen = Max(len1,len2);
    
    char* result = (char*)Calculator_malloc((maxlen + 2) * sizeof(char)); // 为结果分配足够的内存空间

    // both positive
    if(!OperandAIsNegative && !OperandBIsNegative){
        int whoIsBig = whichPositiveInteagerIsBigger(number1,number2);
        if(whoIsBig == A_IS_BIGGER){
            result = Sub_abs(number1,number2,result); // a-b
            CalculatePrint("%s\n",result);
            Calculator_free(result);
            return;
        }
        else if(whoIsBig == B_IS_BIGGER){
            result = Sub_abs(number2,number1,result); // -(b-a)
            CalculatePrint("-%s\n",result);
            Calculator_free(result);
            return;
        }
        else{ // a == b
            CalculatePrint("0\n");
            Calculator_free(result);
            return;
        }
    }else{
        // A is negative
        if(OperandAIsNegative && !OperandBIsNegative){
            int carry = 0;
            result = Add_abs(number1,number2,result); // -(a+b)
            
            CalculatePrint("-%s\n",result);
            Calculator_free(result);
            return;
        }
        // B is negative
        else if(!OperandAIsNegative && OperandBIsNegative){
            int carry = 0;
            result = Add_abs(number1,number2,result); // (a+b)

            /*
            曾经的弯路，我将进位单独在加法之外进行处理，单纯是为了节省那应该byte的内存
            然而这样的内存分配反而会增加，因为我们需要在加法之外再次分配内存
            */
            // if(result[maxlen+1] == 'C'){
            // 复制一份result，然后free。我的期望是free
            // }
            
            CalculatePrint("%s\n",result);
            Calculator_free(result);
            return;
        }
        // A and B are both negative
        else{
            int whoIsBig = whichPositiveInteagerIsBigger(number1,number2);
            if(whoIsBig == A_IS_BIGGER){
                result = Sub_abs(number1,number2,result); // -(a-b)
                CalculatePrint("-%s\n",result);
                Calculator_free(result);
                return;
            }
            else if(whoIsBig == B_IS_BIGGER){
                result = Sub_abs(number2,number1,result); // b-a
                CalculatePrint("%s\n",result);
                Calculator_free(result);
                return;
            }
            else{ // a == b
                CalculatePrint("0\n");
                Calculator_free(result);
                return;
            }
        }

    }
}

/*
* Multiply an array of char and a number
* EX: "987654321" x "9" = "8888888889"
*/
char* Multiply_array_and_num(char* number, char num, char* number_copy){
    if(num == '0'){return "0";}
    if(num == '1'){return number;}

    int num_of_num = num - '0';
    
    for(int i = 0; i < num_of_num; i++){
        number_copy = Add_abs(number_copy,number,number_copy);
    }
    return number_copy;
}



char* Shifting_num_for_n_digit(char* number, int num, char* result,int number_length){
    if(num==0){
        return number;
    }
    int len = number_length;

    for(int i = 0; i < len; i++){
        result[i] = number[i];
    }
    // 全部填0
    for(int i = len; i < len + num; i++){
        result[i] = '0';
    }
    result[len + num] = '\0';
    return result;
}

/*
使用char处理乘法问题
这里最困难的问题是，如何完美的处理内存分配的问题
*/
void Multiply_using_array(char* number1, char* number2){    

    short OperandAIsNegative = 0;
    short OperandBIsNegative = 0;

    if(number1[0] == '-'){
        OperandAIsNegative = 1;
        number1++; 
    }
    if(number2[0] == '-'){
        OperandBIsNegative = 1;
        number2++; 
    }

    if(number1[0] == '+'){
        number1++; 
    }
    if(number2[0] == '+'){
        number2++; 
    }

    if(number1 == "0" || number2 == "0"){
        CalculatePrint("0\n");
        return;
    }

    // let number1 be the bigger one
    // so that we can have fewer iterations
    int whom = whichPositiveInteagerIsBigger(number1,number2);
    if(whom == B_IS_BIGGER){
        exchange(number1,number2);
    }

    int len1 = strlen(number1);
    int len2 = strlen(number2);
    int maxlen = Max(len1,len2);

    /*
        快速反应判断
    */ 
    if(*number2 == '0' && len2 == 1){
        CalculatePrint("0\n");
        return;
    }
    if(number2[0] == '1' && len2 == 1){
        if(OperandAIsNegative == OperandBIsNegative){
            CalculatePrint("%s\n",number1);
            return;
        }else{
            CalculatePrint("-%s\n",number1);
            return;
        }
            
    }

    // DEBUG_Printf("Are we make it?\n"); //yes,we are in 2024/3/7
    char* result = Calculator_malloc( (len1+len2+5) *sizeof(char));
        
    // 计算核心
    {
        /*
            必须要分配一个内存！不然很容易segement fail。
            这个错误是因为Linux的内核出现了问题，而这个问题的本质就是
            我们的temp再次新加入新的东西时，内存不够，然后内核就自动报错
        */ 
        // char* temp  = Calculator_malloc( (len1+2) * sizeof(char));
        // char* temp2 = Calculator_malloc( (len1+len2+2) * sizeof(char));
            
        for(int i = 0; i < len2; i++){
            
            char* temp  = Calculator_malloc( (len1+2) * sizeof(char));
            char* temp2 = Calculator_malloc( (len1+len2+2) * sizeof(char));

            /*
                使用一个array x num的方法，这样可以有效控制我们的乘法
            */
            temp = Multiply_array_and_num(number1,number2[i],temp); // correct

            /*  
                为什么要新定义一个temp的长度？因为此时经过乘法后的temp大小是随我们的乘法而变化的
                此时我们是并不能知道temp的具体的\0是存在在哪一个地址，此时我们要重新定义一个
                否则，我们在接下来的移位中将会出现很多的问题
            */ 
            int temp_now_length = strlen(temp);
            
            /*
             从这里开始我们进行移位操作
             如果想在这里测试，使用：// printf("temp: %s in %d loop \n",temp,i);
             复制number的所有位数到result
            */
            for(int j = 0; j < temp_now_length; j++){
                temp2[j] = temp[j];
            }

            // 你可以在这里使用degbug：printf("temp2: %s in %d loop \n",temp2,i);
            // 在result后面填充num个'0'
            for (int j = temp_now_length; j < (temp_now_length+len2 - i - 1); j++) {
                temp2[j] = '0';
            }

            // 添加字符串结束符
            temp2[temp_now_length+len2 - i - 1] = '\0';

            result = Add_abs(result,temp2,result); // -(a+b)
            
            // Calculator_free(temp2);
            // Calculator_free(temp);
        }
        
    }
        /*
            到这里，我们的temp2就已经是计算好的了，我们就可以直接加在我们的结果上。
            因此，我们或许可以free我们的temp了
            如果你想测试，请在这里使用：printf("\n temp2 length: %ld\n",strlen(temp2));
        */ 

       // 但是不知道为什么，这次内存free后，在下一次使用乘法的时候，会出现错误
       // 由于我的学习知识有限，我实在没有办法解决这个问题
    if(OperandAIsNegative == OperandBIsNegative){
        printf("%s\n",result);
        // free(result);
        // free(temp);
        // free(temp2);
        return;
    }else if(OperandAIsNegative != OperandBIsNegative){
        CalculatePrint("-%s\n",result);
        // Calculator_free(result);
        return;
    }

    return;
}

#ifdef USE_GMP
void Multiply_using_GMP(char* number1, char* number2){
    // 初始化大整数变量
    mpz_t a, b, result;
    mpz_inits(a, b, result, NULL);

    // 将字符串形式的大整数赋值给 GMP 的变量
    mpz_set_str(a, number1, 10); // 基数为10
    mpz_set_str(b, number2, 10); // 基数为10

    // 进行乘法运算
    mpz_mul(result, a, b);

    
    // 打印结果
    gmp_printf("%Zd\n", result);

    // 释放内存
    mpz_clears(a, b, result, NULL);

    return;
}
#endif
void Divide_using_array(char* number1, char* number2){
    short OperandAIsNegative = 0;
    short OperandBIsNegative = 0;

    if(number1[0] == '-'){
        OperandAIsNegative = 1;
        number1++; 
    }
    if(number2[0] == '-'){
        OperandBIsNegative = 1;
        number2++; 
    }

    if(number1[0] == '+'){
        number1++; 
    }
    if(number2[0] == '+'){
        number2++; 
    }

    if(number2 == "0"){
        ErrorDivideByZero();
        return;
    }
    if(number1 == "0"){
        CalculatePrint("0\n");
        return;
    }

    int Greater_than_zero = whichPositiveInteagerIsBigger(number1,0);
    int len1 = strlen(number1);
    int len2 = strlen(number2);
    int maxlen = Max(len1,len2);

    char* mod = Calculator_malloc( (len1+len2+2) * sizeof(char));
    char* result = Calculator_malloc( (len1+len2+2) * sizeof(char));

    

    while (A_IS_BIGGER)
    {
        mod = Sub_abs(number1,number2,mod);
        result = Add_abs(result,"1",result); //result +=1
        Greater_than_zero = whichPositiveInteagerIsBigger(number1,0);
    }

    CalculatePrint("%s",result);

    // Calculator_free(mod);
    // Calculator_free(result);
    return; 
    
    
}


void multiply_exponential(char *a, char *b, char *result) {
    // Extracting exponents from input strings
    int exp_a = atoi(strchr(a, 'e') + 1);
    int exp_b = atoi(strchr(b, 'e') + 1);

    // Computing result exponent
    int result_exp = exp_a + exp_b;

    // Copying base part of the first number
    char *base_a = strtok(a, "e");

    // Copying base part of the second number
    char *base_b = strtok(b, "e");

    // Computing result base
    double base_result = atof(base_a) * atof(base_b);

    // Formatting result
    sprintf(result, "%.2lf", base_result);
    strcat(result, "e");
    sprintf(result + strlen(result), "%d", result_exp);
}

void add_exponential(char *a, char *b, char *result) {
    // Extracting exponents from input strings
    int exp_a = atoi(strchr(a, 'e') + 1);
    int exp_b = atoi(strchr(b, 'e') + 1);

    // If exponents are not equal, adjust bases accordingly
    if (exp_a != exp_b) {
        int max_exp = (exp_a > exp_b) ? exp_a : exp_b;
        int min_exp = (exp_a < exp_b) ? exp_a : exp_b;

        double base_a = atof(strtok(a, "e"));
        double base_b = atof(strtok(b, "e"));

        base_a *= pow(10, max_exp - min_exp);
        base_b *= pow(10, max_exp - min_exp);

        // Computing result base
        double base_result = base_a + base_b;

        // Adjusting result exponent
        // Formatting result
        sprintf(result, "%.2lf", base_result);
        strcat(result, "e");
        sprintf(result + strlen(result), "%d", max_exp);
    } else {
        // If exponents are equal, simply add the bases
        double base_result = atof(strtok(a, "e")) + atof(strtok(b, "e"));

        // Formatting result
        sprintf(result, "%.2lf", base_result);
        strcat(result, "e");
        sprintf(result + strlen(result), "%d", exp_a);
    }
}

//////////////////////////// CALCULATE //////////////////////////////
/*
Mode 2 use char[] to save the result
*/
void CalculateBigInterger(char *operand1, char *operand2, char *operator){
    
    printf("%s %s %s = ",operand1,operator,operand2);
    
    switch (*operator)
    {
    case '+':
        Add_using_array(operand1,operand2);
        break;

    case '-':
        Sub_using_array(operand1,operand2);
        break;
    case 'x':
        #ifdef USE_GMP
        Multiply_using_GMP(operand1,operand2);
        #else
        Multiply_using_array(operand1,operand2);
        #endif
        break;
    case '/':
        break;    
    default:
        break;
    }
    return;
}
/*
    We use this to Calculate Big Decimal
*/
void CalculateBigDecimal(char *operand1, char *operand2, char *operator){}


/*
    We use this to Calculate Exp Number
*/
void CalculateExpNumber(char *operand1, char *operand2, char *operator){
    printf("%s %s %s = ",operand1,operator,operand2);
    switch (*operator)
    {
    case '+':
        char* result = Calculator_malloc( (strlen(operand1)+strlen(operand2)+5) * sizeof(char));
        add_exponential(operand1,operand2,result);
        printf("%s\n",result);
        Calculator_free(result);
        break;

    case '-':
        // Sub_using_array(operand1,operand2);
        break;
    case 'x':
        char* results = Calculator_malloc( (strlen(operand1)+strlen(operand2)+5) * sizeof(char));
        multiply_exponential(operand1,operand2,results);
        printf("%s\n",results);
        Calculator_free(results);
        break;
    case '/':
        break;    
    default:
        break;
    }
    return;
}

/*
Mode 1: 使用double实现的计算器加减乘除
范围：-1e15 ~ 1e15 (15位有效数字)
*/
void CalculateUsingDouble(char *operand1, char *operand2, char *operator){
        double double_operand1 = atof(operand1);
        double double_operand2 = atof(operand2);
        double result;
        int effect_num_of_result;

        int effect_num_of_operand1 = CountDecimalDigits(double_operand1);
        int effect_num_of_operand2 = CountDecimalDigits(double_operand2);

        int max_effect_num_on_operand = Max(effect_num_of_operand1,effect_num_of_operand2);

        switch (*operator)
        {
        case '+':
            result = double_operand1 + double_operand2;
            effect_num_of_result = CountDecimalDigits(result);
            CalculatePrint("%s %s %s = %.*f\n",operand1,operator,operand2,effect_num_of_result,result);
            break;

        case '-':
            result = double_operand1 - double_operand2;
            effect_num_of_result = CountDecimalDigits(result);
            CalculatePrint("%s %s %s = %.*f\n",operand1,operator,operand2,effect_num_of_result,result);
            break;

        case 'x':
            result = (double_operand1 * double_operand2);
            effect_num_of_result = CountDecimalDigits(result);
            CalculatePrint("%s %s %s = %.*f\n",operand1,operator,operand2,effect_num_of_result,result);
            break;
        
        case '/':
        // dog shit!
        // 有效精度：8.8818e-16 (double 有效精度)
            
            if(double_operand2 == 0){
                ErrorDivideByZero();
                return;
            }

            result = double_operand1 / double_operand2;

            int effect_num_of_result = CountDecimalDigits(result);            
            int effect_num = Min(DOUBLE_DIVIDE_PRECISON,effect_num_of_result);

            CalculatePrint("%s / %s = %.*f\n",operand1,operand2,effect_num,result);
            
            break;
        
        case '@':
            printf("Only for testing.\n");
            break;

        default:
            printf("Can not recognize operator symbol. PLease input +, -, x, or / as binary operator!");
            break;
        }
        
        return;
}


////////////////////////////////// Mode ////////////////////////////////////
/*
* Kernel for our calculation
*/
void CalculatorKernel(char *operand1, char *operand2, char *operator){

    // For mode 2 to press enter for many times 
    if(operand1 == "" || operand2 == "" || operator == NULL){
        return;
    }
    
#ifdef USE_CRYPTO
    // for homomorphic addition
    if (*operator=='@'){
        homomorphic_addition(operand1,operand2);
        return;
    }
#endif

    int numtype_operand1 = NumType(operand1);
    int numtype_operand2 = NumType(operand2);

    int n = 5;

    if(numtype_operand1==NOT_A_NUMBER || numtype_operand2==NOT_A_NUMBER){
        ErrorInput();
        return;
    }
    if(numtype_operand1==OUT_OF_CALCULATOR_RANGE || numtype_operand2==OUT_OF_CALCULATOR_RANGE){
        ErrorOutOfRange();
        return;
    }

    // both small number
    // 数据极限： 15位有效数字
    if(numtype_operand1==DOUBLE_CAN_HANDLE && numtype_operand2==DOUBLE_CAN_HANDLE){
        CalculateUsingDouble(operand1,operand2,operator);
        return;
    }

    // for big inteager
    else if(numtype_operand1==BIG_INTEAGER || numtype_operand2==BIG_INTEAGER){
        CalculateBigInterger(operand1,operand2,operator);
        return;
    }
    // for big decimal
    else if(numtype_operand1==BIG_DECIMAL || numtype_operand2==BIG_DECIMAL){
        CalculateBigDecimal(operand1,operand2,operator);
        return;
    }
    // for scientific number
    else if(numtype_operand1==EXP_NUMBER || numtype_operand2==EXP_NUMBER){
        CalculateExpNumber(operand1,operand2,operator);
        return;
    }
    // for other cases
    else{
        ErrorInput();
        return;
    }



}

/**
 * @brief Mode2 for different input
 */
void Mode2(){

    #ifdef USE_LONG_INPUT
    char *expression = NULL;  // 初始时将表达式指针设为 NULL
    size_t expression_size = 0;
    #else
    int max_size = 2048;
    // 声明一个字符串
    char expression[max_size];
    #endif

    while (1) {

        #ifdef USE_LONG_INPUT
        // 使用 getline 动态获取用户输入的表达式
        size_t chars_read = getline(&expression, &expression_size, stdin);
        #else
        fgets(expression,max_size,stdin);
        #endif

        // 检查用户是否输入了 "exit" 以结束循环
        if (strcmp(expression, "exit\n") == 0 || strcmp(expression, "quit\n") == 0) {
            break;
        }
        if(strcmp(expression, "help\n") == 0 || strcmp(expression, "h\n") == 0){
            PrintHelpInLine();
            continue;
        }
        if(strcmp(expression,"l\n") == 0 || strcmp(expression, "mathlib\n") == 0){
            PrintMathLibrary();
            continue;
        }
        if(strcmp(expression, "version\n") == 0 || strcmp(expression, "v\n") == 0){
            PrintVersion();
            continue;
        }
        if(strcmp(expression, "clear\n") == 0 || strcmp(expression, "cls\n") == 0){
            system("clear");
            printf("Please input your expression:\n");
            continue;
        }
        if(strcmp(expression, "check\n") == 0){
            check_for_leaks();
            continue;
        }
        if(strcmp(expression, "cmatrix\n") == 0){
            system("cmatrix");
            continue;
        }

        // 处理用户输入的表达式
        if (strlen(expression) > 0) {
            char* argv[5] = {" ","","","",""};
            Split(expression, " ",argv,4);

            char* operator1 = argv[0];
            char* operand = argv[1];
            char* operator2 = argv[2];

            // since the argv2 may be the last input, the operator will contain
            // a "\n" in the char[]. So we neeed to remove the next line args
            RemoveNextLinrArgs(operator2);

            CalculatorKernel(operator1,operator2,operand);
            
        } else {
            ErrorInput();
            break;
        }


    }

    // 释放动态分配的内存
    // 请注意：千万不要二次释放内存，在rust和c++中会出现问题，因为当你释放内存后，
    // 新的变量可能会被分配到原来的内存地址，继续释放内存会导致你的变量被篡改，这样是不安全的
    clear_allocations();

    #ifdef USE_LONG_INPUT
    free(expression);
    #endif
    
    printf("Exiting the calculator.\n");
}

/////////////////////////////////// MAIN //////////////////////////////////////
/*
* Main 
*   Deal with input
*/
int main(int argc, char **argv) {

    switch (argc)
    {
    // Mode 1: input like 3 + 5   
    case 4:
        char *operand1 = argv[1];
        char *operator = argv[2];
        char *operand2 = argv[3];
        CalculatorKernel(operand1, operand2, operator);
        // free(operand1);
        break;

    // Mode 2: wait for input
    case 1:
        Mode2();
        break;

    // special argument like -h
    case 2:
        // argument is "-h"，then help print the helps
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            PrintHelp();
        }
        else if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0){
            PrintVersion();
        }
        else if (strcmp(argv[1], "-l") == 0 || strcmp(argv[1], "--mathlib") == 0){
            PrintMathLibrary();
        }
        else{    
            ErrorInput();
        }
        break;
    
    // error input
    default:
        ErrorInput();
        break;
    }

    clear_allocations();

    return EXIT_SUCCESS;
}
