#include <iostream>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>
#include "mkl.h"

using namespace std;
#define min(x,y) (((x) < (y)) ? (x) : (y))

int N = 0;
inline void gemm_verify(float *C1, float *C2); // you can use inline function
inline void gemm_Blas(float *A, float *B, float *C);
clock_t start_time, end_time;
float X = 11.4514;

int main()
{
    int num = 0;
    
    srand(time(0));
    cout << "please specify n" << endl;
    cin >> num;
    N = (1 << num);
    // initialize A, B, C
    float *A_o = new float[N * N];
    float *B_o = new float[N * N];
    float *C_o = new float[N * N];
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            A_o[i * N + j] = (float)rand() / (float)(RAND_MAX) * X;
            B_o[i * N + j] = (float)rand() / (float)(RAND_MAX) * X;
        }
    }
    float *A, *B, *C;
    int m, n, k, i, j;
    m = N, k = N, n = N;
    double alpha, beta;

    printf ("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
            " Intel(R) MKL function sgemm, where A, B, and  C are matrices and \n"
            " alpha and beta are double precision scalars\n\n");

    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
    alpha = 1.0; beta = 0.0;

    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n");
    A = (float *)mkl_malloc( m*k*sizeof( float ), 64 );
    B = (float *)mkl_malloc( k*n*sizeof( float ), 64 );
    C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }

    printf (" Intializing matrix data \n\n");
    for (i = 0; i < (m*k); i++) {
        A[i] = A_o[i];
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = B_o[i];
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }
    start_time = clock();
    printf (" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
    printf ("\n Computations completed.\n\n");
    end_time = clock();
    printf("time cost for Intel MKL: %.12f\n",(double)(end_time - start_time)/CLOCKS_PER_SEC );
    gemm_Blas(A_o, B_o, C_o);
    printf("time cost for AVX blocking: %.12f\n",(double)(end_time - start_time)/CLOCKS_PER_SEC );
    if(N<=10)
    {
        cout << "Matrix Verification" << endl;
        gemm_verify(C_o, C);
        cout << "Verification success" << endl;
    }
    else
    {
        cout << "Matrix Verification skipped" << endl;
    }
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    delete[] A_o;
    delete[] B_o;
    delete[] C_o;
    printf (" Example completed. \n\n");
    return 0;
}

inline void gemm_verify(float* C1, float* C2)
{
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{

            if(abs(C1[i * N + j] - C2[i * N + j])>0.1)
            {
                printf("%.12f, %.12f", C1[i * N + j], C2[i * N + j]);
                cout << "error" << endl;
                exit(0);
            }
		}
	}
}

void addDot8x8Pack(float *A, float *B, float *C) 
{
    int p;
	float *Packed_ptr = B;
	__m256 Vec1 = _mm256_setzero_ps();
	__m256 Vec2 = _mm256_setzero_ps();
	__m256 Vec3 = _mm256_setzero_ps();
	__m256 Vec4 = _mm256_setzero_ps();
	__m256 Vec5 = _mm256_setzero_ps();
	__m256 Vec6 = _mm256_setzero_ps();
	__m256 Vec7 = _mm256_setzero_ps();
	__m256 Vec8 = _mm256_setzero_ps();
    
	for (p = 0; p < N; p++) 
    {
		__m256 a0 = _mm256_set1_ps(*(A + p));
		__m256 a1 = _mm256_set1_ps(*(A + N + p));
		__m256 a2 = _mm256_set1_ps(*(A + 2 * N + p));
		__m256 a3 = _mm256_set1_ps(*(A + 3 * N + p));
		__m256 a4 = _mm256_set1_ps(*(A + 4 * N + p));
		__m256 a5 = _mm256_set1_ps(*(A + 5 * N + p));
		__m256 a6 = _mm256_set1_ps(*(A + 6 * N + p));
		__m256 a7 = _mm256_set1_ps(*(A + 7 * N + p)); // each time get one col of A(8 float)

		__m256 bp = _mm256_loadu_ps(Packed_ptr); // load 8 float from B(packed)
                                                // calculate
		Vec1 = _mm256_fmadd_ps(a0, bp, Vec1);
		Vec2 = _mm256_fmadd_ps(a1, bp, Vec2);
		Vec3 = _mm256_fmadd_ps(a2, bp, Vec3);
		Vec4 = _mm256_fmadd_ps(a3, bp, Vec4);
		Vec5 = _mm256_fmadd_ps(a4, bp, Vec5);
		Vec6 = _mm256_fmadd_ps(a5, bp, Vec6);
		Vec7 = _mm256_fmadd_ps(a6, bp, Vec7);
		Vec8 = _mm256_fmadd_ps(a7, bp, Vec8);

		Packed_ptr += 8;
	}
    // 8rows of A * 8cols of B
	_mm256_storeu_ps(C, Vec1);
	_mm256_storeu_ps(C + N, Vec2);
	_mm256_storeu_ps(C + 2 * N, Vec3);
	_mm256_storeu_ps(C + 3 * N, Vec4);
	_mm256_storeu_ps(C + 4 * N, Vec5);
	_mm256_storeu_ps(C + 5 * N, Vec6);
	_mm256_storeu_ps(C + 6 * N, Vec7);
	_mm256_storeu_ps(C + 7 * N, Vec8);
}

void PackedMatrix_B(int j, float *input, float *output) 
{
	for (int i = 0; i < N; i++) // get 8 cols of B each time, improve cache performance
		memcpy(output + i * 8, input + i * N + j, sizeof(float) * 8); // MAKE IT A LONG LINE
}

void PackedMatrix_A(float *input, float *output) 
{
	for (int i = 0; i < 8; i++) // get 8 cols of B each time, improve cache performance
    {
        for(int j = 0; j < N ; j+=8)
        {
            memcpy(output + i * N + j, input + i * N + j, sizeof(float) * 8); // MAKE IT A LONG LINE
        }
    }
}

// impressed by openBLAS
inline void gemm_Blas(float *A, float *B, float *C)
{
    start_time = clock();
    float *PackedA = new float[N*N];
    float *PackedB = new float[8*N];
    for(int j = 0; j < N; j+=8)
    {
        PackedMatrix_B(j, B, PackedB);
        
        for(int i = 0; i < N; i+=8)
        {
            if(j == 0)
                PackedMatrix_A(A + i * N, PackedA+i*N);
            addDot8x8Pack(PackedA+i*N, PackedB, C + i * N + j);
        }
    }
    end_time = clock();
}