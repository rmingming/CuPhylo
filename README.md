# CuPhylo
CuPhylo: A CUDA-based Application Program Interface and Library for Phylogenetic Analysis

## File description: 
- CuLibKernel-xxx.cu contains the kernel implementation for the likelihood computation, postfix rooted means for rooted tree, rooted means for unrooted tree
- CuLibKernel-baseline-xxx.cu contains the kernel for state number is not 61
- CuLibKernel-codemlAndMrBayes-xxx.cu contains the kernel for state number is 61

## General compile and link steps 
- First compile all files like CuLibxxx.cu：nvcc -arch=sm_35 -c CuLibxxx.cu
- Then compile and link testCuLibImpl.cu or timeTest.cu


## testCuLibImpl.cu 
- This used for the timing of each kernel function in CuPhylo
- Compile: 
```
nvcc -arch=sm_35 -o testCuLibImpl testCuLibImpl.cu *.o
```
## timeTest.cu 
- This used for the experiments, you need first install beagle
- Compile: 
```
nvcc -arch=sm_35 -L$BEAGLE_DIR$/lib -lhmsbeagle -I$BEAGLE_DIR$/include/libhmsbeagle-1 -o timeTest timeTest.cu *.o
```
