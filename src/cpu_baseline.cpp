#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int N = 0;

inline void gemm_baseline(float *A, float *B, float *C); // you can use inline function
clock_t start_time, end_time;
float X = 11.4514;

int main(int argc, char *argv[])
{
    int n = 0;
    n = atoi(argv[1]);
    srand(time(0));
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
    
    gemm_baseline(A, B, C);
    printf("time cost: %.12f\n",(double)(end_time - start_time)/CLOCKS_PER_SEC );
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}

inline void gemm_baseline(float *A, float *B, float *C) 
{
    int i, j, k;
    float sum, a, b;
    start_time = clock();
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
            C[i * N + j] = sum;
        }
    }
    end_time = clock();
}