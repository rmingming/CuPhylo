#ifndef CULIB_KERNEL_BASELINE_UNROOTED_H
#define CULIB_KERNEL_BASELINE_UNROOTED_H

#include "CuLibCommon.h"
#include "CuLibCommon-cuda.h"
#include "CuLibKernel-baseline-rooted.h"

void callKernelCondlike_baseline_unrooted(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const bool usePadVersion, const int nSitePattern, const int nPaddedState, const int nState, const int nThreadPerArray, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream);

#endif