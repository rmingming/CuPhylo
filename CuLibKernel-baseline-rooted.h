#ifndef CULIB_KERNEL_BASELINE_ROOTED_H
#define CULIB_KERNEL_BASELINE_ROOTED_H

#include "CuLibCommon.h"
#include "CuLibCommon-cuda.h"

// For state != 4 / 20 / 61, the state is padded to a multiply of 8;
#define PAD_SIZE 8


// For calculation of transition matrices:
// 对于nPaddedState == 4的情况，for baseline，假设一个block负责移到多个(BLOCK_DIM_Y_PMAT_4STATE_SMALL/LARGE_MATRIX_COUNT)PMat的计算;
#define BLOCK_DIMENSION_X_PMAT_4STATE_BASELINE 16
#define BLOCK_DIMENSION_Y_PMAT_4STATE_BASELINE 8

// 对于nPaddedState == 20的情况，for baseline, 假设一个block负责一到多个(BLOCK_DIM_Y_PMAT_20STATE)PMat的计算;
#define BLOCK_DIMENSION_X_PMAT_20STATE_BASELINE 64
#define BLOCK_DIMENSION_Y_PMAT_20STATE_BASELINE 1
#define TILE_SIZE_20STATE_PMAT_BASELINE 4

// 对于nPaddedState == 64的情况，for baseline，假设一个block负责一到多个(BLOCK_DIM_Y_PMAT_64STATE个)PMat的计算;
#define BLOCK_DIMENSION_X_PMAT_64STATE_BASELINE 256
#define BLOCK_DIMENSION_Y_PMAT_64STATE_BASELINE 1

// 对于nPaddedState != 4 / 20 / 64的情况，都是threadIdx.y相同的k个thread负责一个matrix，具体分三种：< 16时，block dimension为(4, 16); < 32时，block dimension为(16, 8); >= 32时，block dimension为(64, 4)
#define BOUNDARY_STATE_IN_XSTATE_BASELINE 10

#define BLOCK_DIMENSION_X_PMAT_XSTATE_SMALL_BASELINE 4
#define BLOCK_DIMENSION_Y_PMAT_XSTATE_SMALL_BASELINE 16

#define BLOCK_DIMENSION_X_PMAT_XSTATE_MEDIUM_BASELINE 16
#define BLOCK_DIMENSION_Y_PMAT_XSTATE_MEDIUM_BASELINE 8

#define BLOCK_DIMENSION_X_PMAT_XSTATE_LARGE_BASELINE 64
#define BLOCK_DIMENSION_Y_PMAT_XSTATE_LARGE_BASELINE 4


// For the padded version of X state:
#define TILE_SIZE_PMAT_XSTATE_BASELINE 4

// For nPaddedState == 8 / 16, block dimension is: (16, 4), each thread block is responsible for 4 PMat matrix;
#define BLOCK_DIMENSION_X_PMAT_XSTATE_8_BASELINE 16
#define BLOCK_DIMENSION_Y_PMAT_XSTATE_8_BASELINE 4

#define BLOCK_DIMENSION_X_PMAT_XSTATE_16_BASELINE 16
#define BLOCK_DIMENSION_Y_PMAT_XSTATE_16_BASELINE 4

// For nPaddedState == 24, block dimension is: (32, 4), each thread block is responsible for 4 PMat matrix;
#define BLOCK_DIMENSION_X_PMAT_XSTATE_24_BASELINE 32
#define BLOCK_DIMENSION_Y_PMAT_XSTATE_24_BASELINE 4

// For nPaddedState == 32 / 40, block dimension is: (64, 2), each thread block is responsible for 2 PMat matrix;
#define BLOCK_DIMENSION_X_PMAT_XSTATE_32_BASELINE 64
#define BLOCK_DIMENSION_Y_PMAT_XSTATE_32_BASELINE 2

#define BLOCK_DIMENSION_X_PMAT_XSTATE_40_BASELINE 64
#define BLOCK_DIMENSION_Y_PMAT_XSTATE_40_BASELINE 2

// For nPaddedState == 48, block dimension is: (128, 1), each thread block is responsible for 1 PMat matrix;
#define BLOCK_DIMENSION_X_PMAT_XSTATE_48_BASELINE 128
#define BLOCK_DIMENSION_Y_PMAT_XSTATE_48_BASELINE 1

// For nPaddedState == 56, block dimension is: (32 * 7, 1), each thread block is responsible for 1 PMat matrix;
#define BLOCK_DIMENSION_X_PMAT_XSTATE_56_BASELINE 224
#define BLOCK_DIMENSION_Y_PMAT_XSTATE_56_BASELINE 1

// For nPaddedState > 64, block dimension is: (8, 8), every 4 thread block is responsible for a PMat matrix;
#define N_ELEMENT_PER_THREAD_PMAT_XSTATE_LARGE_STATE 8
#define BLOCK_DIMENSION_Y_PMAT_XSTATE_LARGE_STATE_BASELINE 8



// For calculation of conditional likelihood:

// For the old schedule scheme:
#define N_SITE_PER_THREAD_CONDLIKE_4STATE_BASELINE 4
#define N_THREAD_PER_BLOCK_CONDLIKE_4STATE_BASELINE 128
#define PATTERN_THRESHOLD_4STATE_CASE1_BASELINE	17000
#define PATTERN_THRESHOLD_4STATE_CASE2_BASELINE	60000
#define PATTERN_THRESHOLD_4STATE_CASE3_BASELINE	30000
#define PATTERN_THRESHOLD_4STATE_UNROOTED_BASELINE	29000

// For the new schedule scheme:
#define N_SITE_PER_THREAD_CONDLIKE_4STATE_VERSION2 1
#define N_THREAD_PER_BLOCK_CONDLIKE_4STATE_VERSION2 64
#define N_SITE_PER_THREAD_CONDLIKE_4STATE_VERSION3_SMALL 1
#define N_SITE_PER_THREAD_CONDLIKE_4STATE_VERSION3_MEDIUM 2
#define N_SITE_PER_THREAD_CONDLIKE_4STATE_VERSION3_LARGE 4
#define N_THREAD_PER_BLOCK_CONDLIKE_4STATE_VERSION3 128
#define TOTAL_PATTERN_THRESHOLD_4STATE_SMALL 5000		// nOp * nSitePattern
#define TOTAL_PATTERN_THRESHOLD_4STATE_MEDIUM 40000		// nOp * nSitePattern 
#define PATTERN_THRESHOLD_4STATE_LARGE 12000			// nSitePattern

#define N_STATE_PER_THREAD_CONDLIKE_20STATE_BASELINE 5
#define N_SITE_PER_BLOCK_CONDLIKE_20STATE_BASELINE 32
#define TILE_SIZE_20STATE_CONDLIKE_BASELINE	4

#define N_STATE_PER_THREAD_CONDLIKE_64STATE_BASELINE 4
#define N_SITE_PER_BLOCK_CONDLIKE_64STATE_BASELINE 32

#define XSTATE_THRESHOLD 32
#define N_STATE_PER_THREAD_CONDLIKE_XSTATE_SMALL_BASELINE 16		// For state != 4 / 20 / 61 && state < 32
#define N_THREAD_PER_BLOCK_CONDLIKE_XSTATE_SMALL_BASELINE 128

#define N_STATE_PER_THREAD_CONDLIKE_XSTATE_LARGE_BASELINE 8		// For state != 4 / 20 / 61 && state >= 32
#define N_THREAD_PER_BLOCK_CONDLIKE_XSTATE_LARGE_BASELINE 256

#define CONDLIKE_XSTATE_USE_PAD_THRESHOLD 8000			// The threshold for whether to use the (padded + shared memory + register) version or the (padded + no shared memory) version

// For the padded version of xState's condlike:
#define BLOCK_DIMENSION_X_CONDLIKE_XSTATE_8_BASELINE 2
#define BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE 32

#define TILE_SIZE_CONDLINE_XSTATE_16_BASELINE 4
#define BLOCK_DIMENSION_X_CONDLIKE_XSTATE_16_BASELINE 2
#define BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE 32

#define TILE_SIZE_CONDLINE_XSTATE_24_BASELINE 4
#define BLOCK_DIMENSION_X_CONDLIKE_XSTATE_24_BASELINE 2
#define BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE 32

// For nPaddedState == 32, tile size & nElemPerThread is the same with other state;
#define BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_32_BASELINE 16

#define TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE 4
#define N_STATE_PER_THREAD_CONDLIKE_XSTATE_OTHER_BASELINE 8
#define BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_OTHER_BASELINE 32



// For scaling of condlike:
#define N_THREAD_PER_BLOCK_SCALE_BASELINE 128


// For calculation of site likelihood / reduction of site likelihood:
#define N_THREAD_PER_BLOCK_SITE_LNL_4STATE_BASELINE 128

#define CAT_THRESHOLD_BASELINE 12
#define N_THREAD_PER_BLOCK_SITE_LNL_20STATE_SMALL_BASELINE 128
#define N_ELEMENT_PER_THREAD_SITE_LNL_20STATE_LARGE_BASELINE 5
#define N_THREAD_PER_BLOCK_SITE_LNL_20STATE_LARGE_BASELINE 64

#define N_ELEMENT_PER_THREAD_SITE_LNL_64STATE_SMALL_BASELINE 16
#define N_ELEMENT_PER_THREAD_SITE_LNL_64STATE_LARGE_BASELINE 4
#define N_THREAD_PER_BLOCK_SITE_LNL_64STATE_SMALL_BASELINE 64
#define N_THREAD_PER_BLOCK_SITE_LNL_64STATE_LARGE_BASELINE 16

// version 2 of site likelihood:
// For nPaddedState == 4:
#define BLOCK_DIMENSION_X_SITE_LNL_4STATE_BASELINE 2
#define BLOCK_DIMENSION_Y_SITE_LNL_4STATE_BASELINE 32

// For nPaddedState == 20:
#define BLOCK_DIMENSION_X_SITE_LNL_20STATE_BASELINE 8
#define BLOCK_DIMENSION_Y_SITE_LNL_20STATE_BASELINE 16

// For nPaddedState == 64:
#define BLOCK_DIMENSION_X_SITE_LNL_64STATE_BASELINE 16
#define BLOCK_DIMENSION_Y_SITE_LNL_64STATE_BASELINE 8

// For nPaddedState == x:
#define N_THREAD_PER_BLOCK_SITE_LNL_XSTATE_BASELINE_VERSION1 128
#define BLOCK_DIMENSION_X_SITE_LNL_XSTATE_BASELINE_SMALL 4
#define BLOCK_DIMENSION_Y_SITE_LNL_XSTATE_BASELINE_SMALL 32
#define BLOCK_DIMENSION_X_SITE_LNL_XSTATE_BASELINE_LARGE 8
#define BLOCK_DIMENSION_Y_SITE_LNL_XSTATE_BASELINE_LARGE 16

#define N_THREAD_PER_BLOCK_REDUCE_BASELINE 128
#define SITE_PATTERN_THRESHOLD_REDUCE 10000
#define N_ELEMENT_PER_THREAD_REDUCE_BASELINE_SMALL 2
#define N_ELEMENT_PER_THREAD_REDUCE_BASELINE_LARGE 4

/*
typedef struct
{
	int UV_offset;
	int R_offset;
	int brLen_offset;
	int P_offset;
} CuLPMatOffset;		// Offset for calculation of transition matrix


typedef struct
{
	int nChild;
	int whichCase;		// which case is the current node, range from 1 to 6
	int isChildTip[3];
	int child_P_offset[3];			// 注意：PMat的offset直接对应的是node在tree中的id；
	int child_condlike_offset[3];
	int father_condlike_offset;
} CuLCondlikeOp;
*/


void callKernelPMat_baseline(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix, const int nState, const int nPaddedState, dim3 nBlockPerGird, dim3 nThreadPerBlock, cudaStream_t &stream);

void callKernelCondlike_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const bool usePadVersion, const int nSitePattern, const int nPaddedState, const int nState, const int nThreadPerArray, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream);

void callKernelNodeScale_baseline(CUFlt *nodeScaleFactor, CUFlt *intCondlike, int *blkIndToCondlikeOffset, int *blkIndToScaleBufferOffset, int *startBlkInd, int nCategory, int nSitePattern, int nPaddedSitePattern, int nState, int nPaddedState, int nBlockPerGrid, int nThreadPerBlock, cudaStream_t &stream);

int callKernelLikelihood_baseline(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *rootCondlike, CUFlt *rateCatWeight, CUFlt *stateFreq, CUFlt *sitePatternWeight, int nNodeScaler, CUFlt *scaleFactor, const int nPaddedState, const int nState, const int nEigenDecomp, const int nRateCategory, const int nSitePattern, const int nPaddedSitePattern, dim3 nBlockPerGrid_siteLnL, dim3 nThreadPerBlock_siteLnL, int nBlockPerGrid_reduce, int nThreadPerBlock_reduce, cudaStream_t &stream);

int callKernelLikelihoodFromSiteLnL_baseline(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *sitePatternWeight, const int nSitePattern, int nBlockPerGrid, int nThreadPerBlock, cudaStream_t &stream);


#endif