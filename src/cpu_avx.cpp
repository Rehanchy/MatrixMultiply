#include <iostream>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>
using namespace std;

int N = 0;
inline void gemm_verify(float *A, float *B, float *C); // you can use inline function
inline void gemm_avx(float *A, float *B, float *C);
inline void Matrix_transpose(float *B,float *B_trans, int N);
clock_t start_time, end_time;
float X = 11.4514;



int main(int argc, char *argv[])
{
    int n = 0;
    srand(time(0));
    n = atoi(argv[1]);
    cout << "specified n as "<< n << endl;
    N = (1 << n);
    // initialize A, B, C
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *B_trans = new float[N * N];
    float *C = new float[N * N];
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            A[i * N + j] = (float)rand() / (float)(RAND_MAX) * X;
            B[i * N + j] = (float)rand() / (float)(RAND_MAX) * X;
        }
    }
    Matrix_transpose(B, B_trans, N);
    gemm_avx(A, B_trans, C);
    printf("time cost: %.12f\n",(double)(end_time - start_time)/CLOCKS_PER_SEC );
    if(n<=10)
    {
        cout << "Matrix Verification" <<endl;
        gemm_verify(A, B, C);
        cout << "Verification success" << endl;
    }
    else
    {
        cout << "Matrix Verification skipped" <<endl;
    }
    delete[] A;
    delete[] B;
    delete[] B_trans;   
    delete[] C;
}

inline void gemm_verify(float *A, float *B, float *C) 
{
    int i, j, k;
    float sum;
    float a, b;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            sum = 0;
            for (k = 0; k < N; k++)
            {
                a = A[i * N + k];
                b = B[k * N + j];
                sum += a * b;
            }
            if(abs(sum - C[i * N + j])>0.99)
            {
                printf("%.12f, %.12f", sum, C[i * N + j]);
                cout << "error" << endl;
                exit(0);
            }
        }
    }
}

inline void gemm_avx(float *A, float *B, float *C)
{   
    start_time = clock();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            __m256 sum = _mm256_setzero_ps(); // 256 bits
            for (int k = 0; k < N; k += 8)
            {
                __m256 a = _mm256_loadu_ps(&A[i * N + k]);
                __m256 b = _mm256_loadu_ps(&B[j * N + k]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
            }
            sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 1));
            sum = _mm256_hadd_ps(sum, sum);
            C[i * N + j] = _mm256_cvtss_f32(_mm256_hadd_ps(sum, sum));
        }
    }
    end_time = clock();
}

inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
{
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(row0, row1);
    __t1 = _mm256_unpackhi_ps(row0, row1);
    __t2 = _mm256_unpacklo_ps(row2, row3);
    __t3 = _mm256_unpackhi_ps(row2, row3);
    __t4 = _mm256_unpacklo_ps(row4, row5);
    __t5 = _mm256_unpackhi_ps(row4, row5);
    __t6 = _mm256_unpacklo_ps(row6, row7);
    __t7 = _mm256_unpackhi_ps(row6, row7);
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}
inline void Matrix_transpose(float *B, float *B_trans, int N)
{
    __m256 row0, row1, row2, row3, row4, row5, row6, row7;
    for (int i = 0; i < N; i += 8)
    {
        for(int j = 0; j < N; j += 8)
        {
            row0 = _mm256_loadu_ps(B + i * N + j);
            row1 = _mm256_loadu_ps(B + (i+1) * N + j);
            row2 = _mm256_loadu_ps(B + (i+2) * N + j);
            row3 = _mm256_loadu_ps(B + (i+3) * N + j);
            row4 = _mm256_loadu_ps(B + (i+4) * N + j);
            row5 = _mm256_loadu_ps(B + (i+5) * N + j);
            row6 = _mm256_loadu_ps(B + (i+6) * N + j);
            row7 = _mm256_loadu_ps(B + (i+7) * N + j);
            transpose8_ps(row0, row1, row2, row3, row4, row5, row6, row7);
            _mm256_storeu_ps(B_trans + j * N + i, row0);
            _mm256_storeu_ps(B_trans + (j + 1) * N + i, row1);
            _mm256_storeu_ps(B_trans + (j + 2) * N + i, row2);
            _mm256_storeu_ps(B_trans + (j + 3) * N + i, row3);
            _mm256_storeu_ps(B_trans + (j + 4) * N + i, row4);
            _mm256_storeu_ps(B_trans + (j + 5) * N + i, row5);
            _mm256_storeu_ps(B_trans + (j + 6) * N + i, row6);
            _mm256_storeu_ps(B_trans + (j + 7) * N + i, row7);
        }
    }
}