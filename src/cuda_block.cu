#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ctime>
#include <immintrin.h>
 
using namespace std;

int N = 0;
inline void gemm_verify(float *C_cpu, float *C_gpu); // you can use inline function
float X = 11.4514;
__global__ void gemm_block_shared(float* A, float* B, float* C, int N);
inline void gemm_Blas(float *A, float *B, float *C);
// <2d grid, 1d block>
#define get_tid() ((blockIA_gpu.y*gridDim.x + blockIA_gpu.x)*blockDim.x + threadIA_gpu.x)
#define get_bid() (blockIA_gpu.y*gridDim.x + blockIA_gpu.x)
#define local_width 32
int main(int argc, char* argv[])
{
	int num = 0;
    int n_grid = 0;
    float* A_gpu = NULL, *A_cpu = NULL;
	float* B_gpu = NULL, *B_cpu = NULL;
	float* C_gpu = NULL, *C_cpu = NULL, *C_cpu2 = NULL;
    srand(time(0));
    num = atoi(argv[1]);
    cout << "Specified N as " << num << endl;
    N = (1 << num);
    n_grid = int(N / local_width);
    // initialize A, B, C
    A_cpu = new float[N * N];
    B_cpu = new float[N * N];
    C_cpu = new float[N * N];
    C_cpu2 = new float[N * N];
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            A_cpu[i * N + j] = (float)rand() / (float)(RAND_MAX) * X;
            B_cpu[i * N + j] = (float)rand() / (float)(RAND_MAX) * X;
        }
    }
	int nbytes = N * N * sizeof(float);
	dim3 dimGrid(n_grid, n_grid); // 1x1 grid
	dim3 dimBlock(local_width, local_width); // NxN block
	cudaError_t cudaStatus = cudaSetDevice(0); // one GPU
 
 
	/* allocate gpu memory */
	cudaMalloc((void**)&A_gpu, nbytes);
 
	cudaMalloc((void**)&B_gpu, nbytes);
 
	cudaMalloc((void**)&C_gpu, nbytes);

	/* copy data to gpu*/
	cudaMemcpy(A_gpu, A_cpu, nbytes, cudaMemcpyHostToDevice);
 
	cudaMemcpy(B_gpu, B_cpu, nbytes, cudaMemcpyHostToDevice);

 
	// call for gpu
	cudaThreadSynchronize();
	gemm_block_shared << <dimGrid, dimBlock >> > (A_gpu, B_gpu, C_gpu, N);
 
	cudaThreadSynchronize();
 
	// call for cpu
    cudaMemcpy(C_cpu2, C_gpu, nbytes, cudaMemcpyDeviceToHost);
    if(N <= 10)
    {
        cout << "Small Matrix, apply verification" << endl;
        gemm_Blas(A_cpu, B_cpu, C_cpu);
        gemm_verify(C_cpu, C_cpu2);
        cout << "Verify Success" << endl;
    }
    else
    {
        cout << "Large Matrix, abort verification" << endl;
    }
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	delete [] A_cpu;
	delete [] B_cpu;
	delete [] C_cpu;
    delete [] C_cpu2;
	return 0;
}

inline void gemm_verify(float* C_cpu, float* C_gpu)
{
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
            if(abs(C_gpu[i * N + j]- C_cpu[i * N + j])>1e-5)
            {
                printf("%.12f, %.12f", C_gpu[i * N + j], C_cpu[i * N + j]);
                cout << "error" << endl;
                exit(0);
            }
		}
	}
}

// device code
__global__ void gemm_block_shared(float* A, float* B, float* C, int N)
{
    __shared__ float local_A[local_width][local_width];
    __shared__ float local_B[local_width][local_width];
    int th_x = threadIdx.x;
    int th_y = threadIdx.y;
    int i = 0;
    int row = blockIdx.y * local_width + th_y;
    int col = blockIdx.x * local_width + th_x;
    if(row < N && col < N)
    {
        float sum = 0.0, a, b;
        for (i = 0; i < N/local_width; ++i)
        {
            local_A[th_y][th_x] = A[row * N + i*local_width + th_x];
            local_B[th_y][th_x] = B[col + N * (i*local_width + th_y)];
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < local_width; ++j)
            {
                a = local_A[th_y][j];
                b = local_B[j][th_x];
                sum += a * b;   
            }
            __syncthreads();
        }
        C[row * N + col] = sum;
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

void PackedMatrix(int j, float *input, float *output) 
{
	for (int i = 0; i < N; i++) // get 8 cols of B each time, improve cache performance
		memcpy(output + i * 8, input + i * N + j, sizeof(float) * 8); // MAKE IT A LONG LINE
}

// impressed by openBLAS
inline void gemm_Blas(float *A, float *B, float *C)
{
    float *PackedB = new float[8*N];
    for(int j = 0; j < N; j+=8)
    {
        PackedMatrix(j, B, PackedB);
        
        for(int i = 0; i < N; i+=8)
        {
            addDot8x8Pack(A + i * N, PackedB, C + i * N + j);
        }
    }
}