EXEC= baseline avx avx_block CUDA_baseline CUDA_block

all: $(EXEC)

clean:
	rm -f $(EXEC)

baseline:cpu_baseline.cpp
	g++ cpu_baseline.cpp -march=native -o baseline
avx:cpu_avx.cpp
	g++ cpu_avx.cpp -march=native -o avx
avx_block:cpu_avx_block.cpp
	g++ cpu_avx_block.cpp -march=native -o avx_block
CUDA_baseline:cuda_baseline.cu
	nvcc cuda_baseline.cu -o CUDA_baseline
CUDA_block:cuda_block.cu
	nvcc cuda_block.cu -o CUDA_block