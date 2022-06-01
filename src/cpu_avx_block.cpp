#include <iostream>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>
using namespace std;

int N = 0;
inline void gemm_verify(float *A, float *B, float *C); // you can use inline function
inline void gemm_Blas(float *A, float *B, float *C);
//inline void Matrix_transpose(float *B,float *B_trans, int N);
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
    float *C = new float[N * N];
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            A[i * N + j] = (float)rand() / (float)(RAND_MAX) * X;
            B[i * N + j] = (float)rand() / (float)(RAND_MAX) * X;
        }
    }
    gemm_Blas(A, B, C);
    printf("time cost: %.12f\n",(double)(end_time - start_time)/CLOCKS_PER_SEC);
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
    float *PackedA = new float[N*N];
    float *PackedB = new float[8*N];
    start_time = clock();
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