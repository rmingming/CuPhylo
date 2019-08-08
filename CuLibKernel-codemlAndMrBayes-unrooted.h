#ifndef CULIB_KERNEL_CODEML_MRBAYES_UNROOTED_H
#define CULIB_KERNEL_CODEML_MRBAYES_UNROOTED_H

#include "CuLibCommon.h"
#include "CuLibCommon-cuda.h"
#include "CuLibKernel-codemlAndMrBayes-rooted.h"


void callKernelCondlike_rq_unrooted(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nSitePattern, const int nPaddedState, const int nState, int blockSize, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream);

void callKernelCondlike_codeml_unrooted(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nSitePattern, const int nPaddedSitePattern, const int nPaddedState, const int nState, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream);

#endif