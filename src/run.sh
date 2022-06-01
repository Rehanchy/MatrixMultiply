echo " "
".\baseline" 8
".\avx" 8
".\avx_block" 8
echo " "
".\baseline" 9
".\avx" 9
".\avx_block" 9
echo " "
".\baseline" 10
".\avx" 10
".\avx_block" 10
echo " "
".\baseline" 11
".\avx" 11
".\avx_block" 11
echo " "
echo " test CUDA program"
echo " "
nvprof --print-gpu-trace ".\CUDA_baseline" 12 4
nvprof --print-gpu-trace ".\CUDA_baseline" 12 8
nvprof --print-gpu-trace ".\CUDA_baseline" 12 16
nvprof --print-gpu-trace ".\CUDA_baseline" 12 32
echo " "
nvprof --print-gpu-trace ".\CUDA_block" 12
nvprof --print-gpu-trace ".\CUDA_block" 13 
read -p "press enter end"
