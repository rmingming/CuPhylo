#ifndef CULIB_KERNEL_CODEML_MRBAYES_ROOTED_H
#define CULIB_KERNEL_CODEML_MRBAYES_ROOTED_H

#include "CuLibCommon.h"
#include "CuLibCommon-cuda.h"

// For calculation of PMat:
#define N_ELEMENT_PER_THREAD_PMAT_CODEML 16
#define N_THREAD_PER_BLOCK_PMAT_CODEML 64

// For calculation of condlike:
#define MAX_SHARED_MEMORY 9000

#define N_THREAD_PER_BLOCK_CONDLIKE_MRBAYES 128

#define N_ELEMENT_PER_THREAD_CONDLIKE_CASE1_CODEML 16
#define N_ELEMENT_PER_THREAD_CONDLIKE_CASE2_CODEML 16
#define N_ELEMENT_PER_THREAD_CONDLIKE_CASE3_CODEML 16
#define N_PATTERN_PER_THREAD_BLOCK_CONDLIKE_CASE23_CODEML 32

// 注意：case 1和case 2需要和case 3保持一致，case 3的block大小为8 * 8;
#define BLOCK_DIMENSION_X_CONDLIKE_CODEML 8
#define BLOCK_DIMENSION_Y_CONDLIKE_CODEML 8
//#define BLOCK_DIMENSION_X_CONDLIKE_CODEML_VERSION2 4
//#define BLOCK_DIMENSION_Y_CONDLIKE_CODEML_VERSION2 32

#define TILE_SIZE_CONDLIKE_CODEML 8
#define TILE_SIZE_CONDLIKE_CODEML_VERSION2 4		// for version 2 of case 3's first/not first child;

// For calculation of likelihood:
#define LIKE_EPSILON        1.0e-300

#define N_THREAD_PER_BLOCK_SITE_LNL_MRBAYES 128
#define N_THREAD_PER_BLOCK_REDUCTION_LNL_MRBAYES 256

#define N_THREAD_PER_BLOCK_SITE_LNL_CODEML 128
#define N_THREAD_PER_BLOCK_REDUCTION_LNL_CODEML 256
#define N_SITE_PER_THREAD_REDUCTION_LNL_CODEML 1


void callKernelPMat_codeml(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CUFlt *exptRootAll, CuLPMatOffset *offset, const int nMatrix, const int nState, const int nPaddedState, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t stream);

void transposeMatrix(CUFlt *pMatrix, int nMatrix, int nRow, int nCol);

void callKernelCondlike_rq(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nSitePattern, const int nPaddedState, const int nState, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream);

void callKernelCondlike_codeml(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nSitePattern, const int nPaddedState, const int nState, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream);

int callKernelLikelihood_rq(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *rootCondlike, CUFlt *rateCatWeight, CUFlt *stateFreq, CUFlt *sitePatternWeight, int nNodeScaler, CUFlt *scaleFactor, const int nPaddedState, const int nState, const int nEigenDecomp, const int nRateCategory, const int nSitePattern, const int nPaddedSitePattern, dim3 nBlockPerGrid_siteLnL, dim3 nThreadPerBlock_siteLnL, int nBlockPerGrid_reduce, int nThreadPerBlock_reduce, cudaStream_t &stream);
int callKernelLikelihoodFromSiteLnL_rq(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *sitePatternWeight, const int nSitePattern, int nBlockPerGrid, int nThreadPerBlock, cudaStream_t &stream);

int callKernelLikelihood_codeml(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *rootCondlike, CUFlt *rateCatWeight, CUFlt *stateFreq, CUFlt *sitePatternWeight, int nNodeScaler, CUFlt *scaleFactor, const int nPaddedState, const int nState, const int nEigenDecomp, const int nRateCategory, const int nSitePattern, const int nPaddedSitePattern, dim3 nBlockPerGird_siteLnL, dim3 nThreadPerBlock_siteLnL, int nBlockPerGrid_reduce, int nThreadPerBlock_reduce, cudaStream_t &stream);
int callKernelLikelihoodFromSiteLnL_codeml(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *sitePatternWeight, const int nSitePattern, int nBlockPerGrid, int nThreadPerBlock, cudaStream_t &stream);

#endif