#include "CuLibKernel-baseline-unrooted.h"


// Calculation of conditional likelihoods:
// case 1: the current child is tip state;
// transpose version: row * col;
// version 2 of case 1 (first & transpose): 
__device__
void deviceCondlike_4State_case1_first_transpose_baseline_version2(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread, int nElement)
{
	int offset_S, curPattern, curState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

#ifdef USING_LDG
		offset_S = (__ldg(&tipState_S[curPattern]) << 2);
#else
		offset_S = (tipState_S[curPattern] << 2);
#endif
		
		condlike_F[ind] = PMat_S[curState + offset_S];
	}
}



// version 3 of case 1 (first & transpose): modification of version 1, 使用寄存器保存中间结果;
__device__
void deviceCondlike_4State_case1_first_transpose_baseline_version1(int *tipState_S, CUFlt *PMat_S, CUFlt *buf, int ind, int nThread, int nSitePattern)
{
	CUFlt *pPMat_S;
	int iState, cnt = 0;

	for(; ind < nSitePattern; ind += nThread){
#ifdef USING_LDG
		pPMat_S = PMat_S + (__ldg(&tipState_S[ind]) << 2);
#else
		pPMat_S = PMat_S + (tipState_S[ind] << 2);
#endif
		//pCondlike_F = condlike_F + (ind << 2);

		for(iState = 0; iState < 4; iState ++, cnt ++){
			buf[cnt] = pPMat_S[iState];
		}
	}
}


// version 2 of case 1 (not first & transpose): 
__device__
void deviceCondlike_4State_case1_notFirst_transpose_baseline_version2(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread, int nElement)
{
	int offset_S, curPattern, curState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

#ifdef USING_LDG
		offset_S = (__ldg(&tipState_S[curPattern]) << 2);
#else
		offset_S = (tipState_S[curPattern] << 2);
#endif
		
		condlike_F[ind] *= PMat_S[curState + offset_S];
	}
}


// version 3 of case 1 (not first & transpose): modification of version 1, 使用寄存器保存中间结果;
__device__
void deviceCondlike_4State_case1_notFirst_transpose_baseline_version1(int *tipState_S, CUFlt *PMat_S, CUFlt *buf, int ind, int nThread, int nSitePattern)
{
	CUFlt *pPMat_S;
	int iState, cnt = 0;

	for(; ind < nSitePattern; ind += nThread){
#ifdef USING_LDG
		pPMat_S = PMat_S + (__ldg(&tipState_S[ind]) << 2);
#else
		pPMat_S = PMat_S + (tipState_S[ind] << 2);
#endif
		//pCondlike_F = condlike_F + (ind << 2);

		for(iState = 0; iState < 4; iState ++, cnt ++){
			buf[cnt] *= pPMat_S[iState];
		}
	}
}


// non-transpose version: row * row;
// version 1 of case 1 (first & non-transpose):
__device__
void deviceCondlike_4State_case1_first_noTranspose_baseline_version1(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread, int nSitePattern)
{
	CUFlt *pCondlike_F, *pPMat_S;
	int iState;

	for(; ind < nSitePattern; ind += nThread){
		pPMat_S = PMat_S + tipState_S[ind];
		pCondlike_F = condlike_F + (ind << 2);

		for(iState = 0; iState < 4; iState ++, pPMat_S += 4)
			pCondlike_F[iState] = pPMat_S[0];
	}
}



// version 2 of case 1 (first & non-transpose):
__device__
void deviceCondlike_4State_case1_first_noTranspose_baseline_version2(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread, int nElement)
{
	CUFlt *pPMat_S;
	int curPattern, curState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		pPMat_S = PMat_S + tipState_S[curPattern] + (curState << 2);
		condlike_F[ind] = pPMat_S[0];
	}
}


// version 1 of case 1 (not first & non-transpose):
__device__
void deviceCondlike_4State_case1_notFirst_noTranspose_baseline_version1(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread, int nSitePattern)
{
	CUFlt *pCondlike_F, *pPMat_S;
	int iState;

	for(; ind < nSitePattern; ind += nThread){
		pPMat_S = PMat_S + tipState_S[ind];
		pCondlike_F = condlike_F + (ind << 2);

		for(iState = 0; iState < 4; iState ++, pPMat_S += 4)
			pCondlike_F[iState] *= pPMat_S[0];
	}
}



// version 2 of case 1 (not first & non-transpose):
__device__
void deviceCondlike_4State_case1_notFirst_noTranspose_baseline_version2(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread, int nElement)
{
	CUFlt *pPMat_S;
	int curPattern, curState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		pPMat_S = PMat_S + tipState_S[curPattern] + (curState << 2);
		condlike_F[ind] *= pPMat_S[0];
	}
}


// case 2: the current child is tip/int condlike:
// transpose version: row * col;
// version 3 of case 2 (first & transpose): roll the iteration of version 2;
__device__
void deviceCondlike_4State_case2_first_transpose_baseline_version2(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThread, const int nElement)
{
	CUFlt *pCondlike_S, *pPMat_S, curSum;
	int curPattern, curState, jState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		pCondlike_S = condlike_S + (curPattern << 2);
		pPMat_S = PMat_S + curState;

		curSum = 0.0f;
		for(jState = 0; jState < 4; jState ++, pPMat_S += 4){
#ifdef USING_LDG
			curSum += __ldg(&pCondlike_S[jState]) * pPMat_S[0];
#else
			curSum += pCondlike_S[jState] * pPMat_S[0];
#endif
		}
		condlike_F[ind] = curSum;
	}
}

// version 7 of case 2 (first & transpose): modification of version 5, 使用寄存器保存中间结果;
__device__
void deviceCondlike_4State_case2_first_transpose_baseline_version1(CUFlt *condlike_S, CUFlt *PMat_S, CUFlt *res_buf, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_S, *pPMat_S, buf[4], curSum;
	int iState, jState, cnt = 0;

	for(; ind < nSitePattern; ind += nThread){
		pCondlike_S = condlike_S + (ind << 2);

#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_S[0]);
		buf[1] = __ldg(&pCondlike_S[1]);
		buf[2] = __ldg(&pCondlike_S[2]);
		buf[3] = __ldg(&pCondlike_S[3]);
#else
		buf[0] = pCondlike_S[0];
		buf[1] = pCondlike_S[1];
		buf[2] = pCondlike_S[2];
		buf[3] = pCondlike_S[3];
#endif
		
		for(iState = 0; iState < 4; iState ++, cnt ++){
			pPMat_S = PMat_S + iState;
			curSum = 0.0f;

			for(jState = 0; jState < 4; jState ++, pPMat_S += 4)
				curSum += buf[jState] * pPMat_S[0];

			res_buf[cnt] = curSum;
		}
	}
}



// version 3 of case 2 (not first & transpose): roll the iteration of version 2;
__device__
void deviceCondlike_4State_case2_notFirst_transpose_baseline_version2(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThread, const int nElement)
{
	CUFlt *pCondlike_S, *pPMat_S, curSum;
	int curPattern, curState, jState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		pCondlike_S = condlike_S + (curPattern << 2);
		pPMat_S = PMat_S + curState;

		curSum = 0.0f;
		for(jState = 0; jState < 4; jState ++, pPMat_S += 4){
#ifdef USING_LDG
			curSum += __ldg(&pCondlike_S[jState]) * pPMat_S[0];
#else
			curSum += pCondlike_S[jState] * pPMat_S[0];
#endif
		}
		condlike_F[ind] *= curSum;
	}
}


// version 7 of case 2 (not first & transpose): modification of version 5, 使用寄存器保存中间结果;
__device__
void deviceCondlike_4State_case2_notFirst_transpose_baseline_version1(CUFlt *condlike_S, CUFlt *PMat_S, CUFlt *res_buf, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_S, *pPMat_S, buf[4], curSum;
	int iState, jState, cnt = 0;

	for(; ind < nSitePattern; ind += nThread){
		pCondlike_S = condlike_S + (ind << 2);

#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_S[0]);
		buf[1] = __ldg(&pCondlike_S[1]);
		buf[2] = __ldg(&pCondlike_S[2]);
		buf[3] = __ldg(&pCondlike_S[3]);
#else
		buf[0] = pCondlike_S[0];
		buf[1] = pCondlike_S[1];
		buf[2] = pCondlike_S[2];
		buf[3] = pCondlike_S[3];
#endif
		
		for(iState = 0; iState < 4; iState ++, cnt ++){
			pPMat_S = PMat_S + iState;
			curSum = 0.0f;

			for(jState = 0; jState < 4; jState ++, pPMat_S += 4)
				curSum += buf[jState] * pPMat_S[0];

			res_buf[cnt] *= curSum;
		}
	}
}



// Non-transpose version: row * row;
// version 2 of case 2 (first & non-transpose):
__device__
void deviceCondlike_4State_case2_first_noTranspose_baseline_version2(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThread, const int nElement)
{
	CUFlt *pCondlike_S, *pPMat_S;
	int curPattern, curState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		pCondlike_S = condlike_S + (curPattern << 2);
		pPMat_S = PMat_S + (curState << 2);
		
		condlike_F[ind] = pCondlike_S[0] * pPMat_S[0] + pCondlike_S[1] * pPMat_S[1] + pCondlike_S[2] * pPMat_S[2] + pCondlike_S[3] * pPMat_S[3];
	}
}


// version 5 of case 2 (first & non-transpose): roll the iteration of version 4;
__device__
void deviceCondlike_4State_case2_first_noTranspose_baseline_version1(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_S, *pPMat_S, buf[4], curSum;
	int iState, jState;

	for(; ind < nSitePattern; ind += nThread){
		pCondlike_S = condlike_S + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

		buf[0] = pCondlike_S[0];
		buf[1] = pCondlike_S[1];
		buf[2] = pCondlike_S[2];
		buf[3] = pCondlike_S[3];
		
		for(iState = 0; iState < 4; iState ++){
			pPMat_S = PMat_S + (iState << 2);
			curSum = 0.0f;

			for(jState = 0; jState < 4; jState ++)
				curSum += buf[jState] * pPMat_S[jState];

			pCondlike_F[iState] = curSum;
		}
	}
}


// version 2 of case 2 (not first & non-transpose):
__device__
void deviceCondlike_4State_case2_notFirst_noTranspose_baseline_version2(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThread, const int nElement)
{
	CUFlt *pCondlike_S, *pPMat_S;
	int curPattern, curState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		pCondlike_S = condlike_S + (curPattern << 2);
		pPMat_S = PMat_S + (curState << 2);
		
		condlike_F[ind] *= pCondlike_S[0] * pPMat_S[0] + pCondlike_S[1] * pPMat_S[1] + pCondlike_S[2] * pPMat_S[2] + pCondlike_S[3] * pPMat_S[3];
	}
}


// version 5 of case 2 (not first & non-transpose): roll the iteration of version 4;
__device__
void deviceCondlike_4State_case2_notFirst_noTranspose_baseline_version1(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_S, *pPMat_S, buf[4], curSum;
	int iState, jState;

	for(; ind < nSitePattern; ind += nThread){
		pCondlike_S = condlike_S + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

		buf[0] = pCondlike_S[0];
		buf[1] = pCondlike_S[1];
		buf[2] = pCondlike_S[2];
		buf[3] = pCondlike_S[3];
		
		for(iState = 0; iState < 4; iState ++){
			pPMat_S = PMat_S + (iState << 2);
			curSum = 0.0f;

			for(jState = 0; jState < 4; jState ++)
				curSum += buf[jState] * pPMat_S[jState];

			pCondlike_F[iState] *= curSum;
		}
	}
}


// 注意：由于case 1的version 1和version 2的任务分配方式不同，case 2的version 1/4/5与case 3的version 2/3的任务分配方式不同，因此，case 1的version 1只能与case 2的version 1/4/5搭配，case 1的version 2只能与case 2的version 2/3搭配，也即一共只有5种可能的组合;
// Transpose version:
// version 1: 用寄存器保存中间结果, case 1 use version 3, case 2 use version 7;
__global__
void kernelCondlike_4State_nChild_transpose_baseline_version1(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_S[16];

	CUFlt buf[N_SITE_PER_THREAD_CONDLIKE_4STATE_BASELINE * 4];
	
	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();

	int startBlkInd = opStartBlkInd[opInd];
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	int iChild;
	void *pCondlike_S;
	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset + (ind << 2);

	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){
		if(tx < 16){
#ifdef USING_LDG
			sh_PMat_S[tx] = __ldg(&PMat[sh_condlikeOp.child_P_offset[iChild] + tx]);
#else
			sh_PMat_S[tx] = PMat[sh_condlikeOp.child_P_offset[iChild] + tx];
#endif
		}
		__syncthreads();

		if(1 == sh_condlikeOp.child_case[iChild]){
			if(0 == iChild){
				deviceCondlike_4State_case1_first_transpose_baseline_version1(tipState + sh_condlikeOp.child_condlike_offset[iChild], 
																	sh_PMat_S, 
																	buf,
																	ind, 
																	nThreadPerArray, 
																	nSitePattern);
			}
			else{
				deviceCondlike_4State_case1_notFirst_transpose_baseline_version1(tipState + sh_condlikeOp.child_condlike_offset[iChild], 
																	sh_PMat_S, 
																	buf,
																	ind, 
																	nThreadPerArray, 
																	nSitePattern);
			}
		}
		else{
			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			if(0 == iChild){
				deviceCondlike_4State_case2_first_transpose_baseline_version1((CUFlt *)pCondlike_S, 
																	sh_PMat_S, 
																	buf,
																	ind, 
																	nThreadPerArray, 
																	nSitePattern);
			}
			else{
				deviceCondlike_4State_case2_notFirst_transpose_baseline_version1((CUFlt *)pCondlike_S, 
																	sh_PMat_S, 
																	buf,
																	ind, 
																	nThreadPerArray, 
																	nSitePattern);
			}
		}
		__syncthreads();		// TODO: 是否可以去掉同步??? 应该不行，因为用到了shared memory;
	}

	int cnt = 0, offset = (nThreadPerArray << 2), iState;
	for(; ind < nSitePattern; ind += nThreadPerArray, pCondlike_F += offset){
		for(iState = 0; iState < 4; iState ++, cnt ++)
			pCondlike_F[iState] = buf[cnt];
	}
}



// version 2: case 1 use version 2, case 2 use version 3;
__global__
void kernelCondlike_4State_nChild_transpose_baseline_version2(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_S[16];
	
	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();

	int startBlkInd = opStartBlkInd[opInd];
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	int iChild;
	void *pCondlike_S;
	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset;

	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){
		if(tx < 16){
#ifdef USING_LDG
			sh_PMat_S[tx] = __ldg(&PMat[sh_condlikeOp.child_P_offset[iChild] + tx]);
#else
			sh_PMat_S[tx] = PMat[sh_condlikeOp.child_P_offset[iChild] + tx];
#endif
		}
		__syncthreads();

		if(1 == sh_condlikeOp.child_case[iChild]){
			if(0 == iChild){
				deviceCondlike_4State_case1_first_transpose_baseline_version2(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild], 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	(nSitePattern << 2));
			}
			else{
				deviceCondlike_4State_case1_notFirst_transpose_baseline_version2(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild], 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	(nSitePattern << 2));
			}
		}
		else{
			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			if(0 == iChild){
				deviceCondlike_4State_case2_first_transpose_baseline_version2(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	(nSitePattern << 2));
			}
			else{
				deviceCondlike_4State_case2_notFirst_transpose_baseline_version2(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	(nSitePattern << 2));
			}
		}
		__syncthreads();		// TODO: 是否可以去掉同步??? 应该不行，因为用到了shared memory;
	}
}



// 一共5种组合：case 1的version 1 + case 2的version 1/4/5，case 1的version 2 + case 2的version 2/3；
// Non-transpose version:
// version 1: case 1 use version 1, case 2 use version 5;
__global__
void kernelCondlike_4State_nChild_noTranspose_baseline_version1(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_S[16];
	
	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();

	int startBlkInd = opStartBlkInd[opInd];
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	int iChild;
	void *pCondlike_S;
	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset;

	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){
		if(tx < 16)
			sh_PMat_S[tx] = PMat[sh_condlikeOp.child_P_offset[iChild] + tx];
		__syncthreads();

		if(1 == sh_condlikeOp.child_case[iChild]){
			if(0 == iChild){
				deviceCondlike_4State_case1_first_noTranspose_baseline_version1(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild], 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern);
			}
			else{
				deviceCondlike_4State_case1_notFirst_noTranspose_baseline_version1(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild], 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern);
			}
		}
		else{
			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			if(0 == iChild){
				deviceCondlike_4State_case2_first_noTranspose_baseline_version1(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern);
			}
			else{
				deviceCondlike_4State_case2_notFirst_noTranspose_baseline_version1(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern);
			}
		}
		__syncthreads();		// TODO: 是否可以去掉同步??? 应该不行，因为用到了shared memory;
	}
}



// version 2: case 1 use version 2, case 2 use version 2;
__global__
void kernelCondlike_4State_nChild_noTranspose_baseline_version2(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_S[16];
	
	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();

	int startBlkInd = opStartBlkInd[opInd];
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	int iChild;
	void *pCondlike_S;
	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset;

	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){
		if(tx < 16)
			sh_PMat_S[tx] = PMat[sh_condlikeOp.child_P_offset[iChild] + tx];
		__syncthreads();

		if(1 == sh_condlikeOp.child_case[iChild]){
			if(0 == iChild){
				deviceCondlike_4State_case1_first_noTranspose_baseline_version2(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild], 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	(nSitePattern << 2));
			}
			else{
				deviceCondlike_4State_case1_notFirst_noTranspose_baseline_version2(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild], 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	(nSitePattern << 2));
			}
		}
		else{
			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			if(0 == iChild){
				deviceCondlike_4State_case2_first_noTranspose_baseline_version2(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	(nSitePattern << 2));
			}
			else{
				deviceCondlike_4State_case2_notFirst_noTranspose_baseline_version2(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	sh_PMat_S, 
																	ind, 
																	nThreadPerArray, 
																	(nSitePattern << 2));
			}
		}
		__syncthreads();		// TODO: 是否可以去掉同步??? 应该不行，因为用到了shared memory;
	}
}



// For nState = 20:
// Transpose version:
// version 3: modification of version 2, 用寄存器保存中间结果，而不是直接写condlike_F[]
__device__
void deviceCondlike_20State_case1_first_transpose_baseline(int *tipState_S, CUFlt *PMat_S, CUFlt *buf, const int blockDim_x, const int tx, const int ty)
{
	int state_S = tipState_S[ty];

	PMat_S += state_S * 20 + tx;

#ifdef USING_LDG
	buf[0] = __ldg(&PMat_S[0]);
	buf[1] = __ldg(&PMat_S[blockDim_x]);
	buf[2] = __ldg(&PMat_S[(blockDim_x << 1)]);
	buf[3] = __ldg(&PMat_S[blockDim_x * 3]);
	buf[4] = __ldg(&PMat_S[(blockDim_x << 2)]);
#else
	buf[0] = PMat_S[0];
	buf[1] = PMat_S[blockDim_x];
	buf[2] = PMat_S[(blockDim_x << 1)];
	buf[3] = PMat_S[blockDim_x * 3];
	buf[4] = PMat_S[(blockDim_x << 2)];
#endif
}


__device__
void deviceCondlike_20State_case1_notFirst_transpose_baseline(int *tipState_S, CUFlt *PMat_S, CUFlt *buf, const int blockDim_x, const int tx, const int ty)
{
	int state_S = tipState_S[ty];

	PMat_S += state_S * 20 + tx;

#ifdef USING_LDG
	buf[0] *= __ldg(&PMat_S[0]);
	buf[1] *= __ldg(&PMat_S[blockDim_x]);
	buf[2] *= __ldg(&PMat_S[(blockDim_x << 1)]);
	buf[3] *= __ldg(&PMat_S[blockDim_x * 3]);
	buf[4] *= __ldg(&PMat_S[(blockDim_x << 2)]);
#else
	buf[0] *= PMat_S[0];
	buf[1] *= PMat_S[blockDim_x];
	buf[2] *= PMat_S[(blockDim_x << 1)];
	buf[3] *= PMat_S[blockDim_x * 3];
	buf[4] *= PMat_S[(blockDim_x << 2)];
#endif
}

// Non-transpose version: row * row;
__device__
void deviceCondlike_20State_case1_first_noTranspose_baseline(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread, int nSitePattern)
{
	int nElement = nSitePattern * 20, offset_S, curSite, curState;

	for(; ind < nElement; ind += nThread){
		curSite = ind / 20;
		curState = ind % 20;

		offset_S = tipState_S[curSite];

		condlike_F[ind] = PMat_S[curState * 20 + offset_S];
	}
}

__device__
void deviceCondlike_20State_case1_notFirst_noTranspose_baseline(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread, int nSitePattern)
{
	int nElement = nSitePattern * 20, offset_S, curSite, curState;

	for(; ind < nElement; ind += nThread){
		curSite = ind / 20;
		curState = ind % 20;

		offset_S = tipState_S[curSite];

		condlike_F[ind] *= PMat_S[curState * 20 + offset_S];
	}
}


// Case 2 of nPaddedState = 20:
// Transpose version:
// version 3 of case 2: modification of version 2, 用寄存器保存中间结果，避免对condlike_F[]的多次读写;
__device__
void deviceCondlike_20State_case2_first_transpose_baseline(CUFlt *condlike_S, CUFlt *PMat_S, CUFlt sh_condlike_S[][TILE_SIZE_20STATE_CONDLIKE_BASELINE], CUFlt sh_PMat_S[][20], CUFlt *buf, const int tx, const int ty)
{
	int ind1 = ty * 20 + tx, ind2 = ty * blockDim.x + tx, nIteration = 20 / TILE_SIZE_20STATE_CONDLIKE_BASELINE, PMat_offset = TILE_SIZE_20STATE_CONDLIKE_BASELINE * 20, itr, iElem;
	int offset1 = tx + TILE_SIZE_20STATE_CONDLIKE_BASELINE, offset2 = offset1 + TILE_SIZE_20STATE_CONDLIKE_BASELINE , offset3 = offset1 + 2 * TILE_SIZE_20STATE_CONDLIKE_BASELINE, offset4 = offset1 + 3 * TILE_SIZE_20STATE_CONDLIKE_BASELINE;
	int nElement = TILE_SIZE_20STATE_CONDLIKE_BASELINE * 20, indx = ind2 / 20, indy = ind2 % 20;
	
	for(itr = 0; itr < nIteration; itr ++, condlike_S += TILE_SIZE_20STATE_CONDLIKE_BASELINE, PMat_S += PMat_offset){
#ifdef USING_LDG
		sh_condlike_S[ty][tx] = __ldg(&condlike_S[ind1]);
#else
		sh_condlike_S[ty][tx] = condlike_S[ind1];
#endif
		if(ind2 < nElement){
#ifdef USING_LDG
			sh_PMat_S[indx][indy] = __ldg(&PMat_S[ind2]);
#else
			sh_PMat_S[indx][indy] = PMat_S[ind2];
#endif
		}
		__syncthreads();

		for(iElem = 0; iElem < TILE_SIZE_20STATE_CONDLIKE_BASELINE; iElem ++){
			buf[0] += sh_condlike_S[ty][iElem] * sh_PMat_S[iElem][tx];
			buf[1] += sh_condlike_S[ty][iElem] * sh_PMat_S[iElem][offset1];
			buf[2] += sh_condlike_S[ty][iElem] * sh_PMat_S[iElem][offset2];
			buf[3] += sh_condlike_S[ty][iElem] * sh_PMat_S[iElem][offset3];
			buf[4] += sh_condlike_S[ty][iElem] * sh_PMat_S[iElem][offset4];
		}
		__syncthreads();
	}

	buf[5] = buf[0];
	buf[6] = buf[1];
	buf[7] = buf[2];
	buf[8] = buf[3];
	buf[9] = buf[4];
}

// not first child:
__device__
void deviceCondlike_20State_case2_notFirst_transpose_baseline(CUFlt *condlike_S, CUFlt *PMat_S, CUFlt sh_condlike_S[][TILE_SIZE_20STATE_CONDLIKE_BASELINE], CUFlt sh_PMat_S[][20], CUFlt *buf, const int tx, const int ty)
{
	int ind1 = ty * 20 + tx, ind2 = ty * blockDim.x + tx, nIteration = 20 / TILE_SIZE_20STATE_CONDLIKE_BASELINE, PMat_offset = TILE_SIZE_20STATE_CONDLIKE_BASELINE * 20, itr, iElem;
	int offset1 = tx + TILE_SIZE_20STATE_CONDLIKE_BASELINE, offset2 = offset1 + TILE_SIZE_20STATE_CONDLIKE_BASELINE , offset3 = offset1 + 2 * TILE_SIZE_20STATE_CONDLIKE_BASELINE, offset4 = offset1 + 3 * TILE_SIZE_20STATE_CONDLIKE_BASELINE;
	int nElement = TILE_SIZE_20STATE_CONDLIKE_BASELINE * 20, indx = ind2 / 20, indy = ind2 % 20;
	
	for(itr = 0; itr < nIteration; itr ++, condlike_S += TILE_SIZE_20STATE_CONDLIKE_BASELINE, PMat_S += PMat_offset){
#ifdef USING_LDG
		sh_condlike_S[ty][tx] = __ldg(&condlike_S[ind1]);
#else
		sh_condlike_S[ty][tx] = condlike_S[ind1];
#endif
		if(ind2 < nElement){
#ifdef USING_LDG
			sh_PMat_S[indx][indy] = __ldg(&PMat_S[ind2]);
#else
			sh_PMat_S[indx][indy] = PMat_S[ind2];
#endif
		}
		__syncthreads();

		for(iElem = 0; iElem < TILE_SIZE_20STATE_CONDLIKE_BASELINE; iElem ++){
			buf[0] += sh_condlike_S[ty][iElem] * sh_PMat_S[iElem][tx];
			buf[1] += sh_condlike_S[ty][iElem] * sh_PMat_S[iElem][offset1];
			buf[2] += sh_condlike_S[ty][iElem] * sh_PMat_S[iElem][offset2];
			buf[3] += sh_condlike_S[ty][iElem] * sh_PMat_S[iElem][offset3];
			buf[4] += sh_condlike_S[ty][iElem] * sh_PMat_S[iElem][offset4];
		}
		__syncthreads();
	}

	buf[5] *= buf[0];
	buf[6] *= buf[1];
	buf[7] *= buf[2];
	buf[8] *= buf[3];
	buf[9] *= buf[4];
}


// Non-transpose version: row * row;
__device__
void deviceCondlike_20State_case2_first_noTranspose_baseline(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThread, const int nSitePattern, const int nState)
{
	int nElement = nSitePattern * 20, curSite, curState, iState;
	CUFlt *pCondlike_S, *pPMat_S, sum_S;

	for(; ind < nElement; ind += nThread){
		curSite = ind / 20;
		curState = ind % 20;

		pCondlike_S = condlike_S + curSite * 20;
		pPMat_S = PMat_S + curState * 20;
		
		sum_S = 0.0f;

		for(iState = 0; iState < nState; iState ++)
			sum_S += pCondlike_S[iState] * pPMat_S[iState];

		condlike_F[ind] = sum_S;
	}
}


__device__
void deviceCondlike_20State_case2_notFirst_noTranspose_baseline(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThread, const int nSitePattern, const int nState)
{
	int nElement = nSitePattern * 20, curSite, curState, iState;
	CUFlt *pCondlike_S, *pPMat_S, sum_S;

	for(; ind < nElement; ind += nThread){
		curSite = ind / 20;
		curState = ind % 20;

		pCondlike_S = condlike_S + curSite * 20;
		pPMat_S = PMat_S + curState * 20;
		
		sum_S = 0.0f;

		for(iState = 0; iState < nState; iState ++)
			sum_S += pCondlike_S[iState] * pPMat_S[iState];

		condlike_F[ind] *= sum_S;
	}
}



// For nState = 20, each thread is responsible for k states' condlike;
// block dimension is: (m, l), m * k = 20, 每个block负责l个site pattern，nSitePattern / l个block负责一个condlike;
// nThreadPerArray: 负责一个condlike array(一个node的一个eigen decomposition的一个rate category的condlike)的thread数目
// Transpose version:
// version 3: case 1 use version 3, case 2 use version 3;
__global__
void kernelCondlike_20State_nChild_transpose_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern, const int nState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int blockDim_x = blockDim.x;
	int opInd = blkIndToOpInd[bx];

	__shared__ CuLCondlikeOp sh_condlikeOp;

	__shared__ CUFlt sh_condlike_S[N_SITE_PER_BLOCK_CONDLIKE_20STATE_BASELINE][TILE_SIZE_20STATE_CONDLIKE_BASELINE];
	__shared__ CUFlt sh_PMat_S[TILE_SIZE_20STATE_CONDLIKE_BASELINE][20];
	
	CUFlt buf[2 * N_STATE_PER_THREAD_CONDLIKE_20STATE_BASELINE];

	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int blkOffset = bx - startBlkInd;
	int iChild, tipStateOffset = blkOffset * N_SITE_PER_BLOCK_CONDLIKE_20STATE_BASELINE, condlikeOffset = tipStateOffset * 20;

	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset + condlikeOffset + ty * 20 + tx;
	void *pCondlike_S;
	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){
		if(1 == sh_condlikeOp.child_case[iChild]){
			if(0 == iChild){
				deviceCondlike_20State_case1_first_transpose_baseline(tipState + sh_condlikeOp.child_condlike_offset[iChild] + tipStateOffset,
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	buf + N_STATE_PER_THREAD_CONDLIKE_20STATE_BASELINE,
																	blockDim_x,
																	tx,
																	ty);
			}
			else{
				deviceCondlike_20State_case1_notFirst_transpose_baseline(tipState + sh_condlikeOp.child_condlike_offset[iChild] + tipStateOffset,
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	buf + N_STATE_PER_THREAD_CONDLIKE_20STATE_BASELINE,
																	blockDim_x,
																	tx,
																	ty);
			}
		}
		else{
			// assert N_STATE_PER_THREAD_CONDLIKE_20STATE_BASELINE = 0;
			buf[0] = 0; 
			buf[1] = 0;
			buf[2] = 0;
			buf[3] = 0;
			buf[4] = 0;

			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			if(0 == iChild){
				deviceCondlike_20State_case2_first_transpose_baseline((CUFlt *)pCondlike_S + condlikeOffset, 
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	sh_condlike_S,
																	sh_PMat_S,
																	buf,
																	tx,
																	ty);
			}
			else{
				deviceCondlike_20State_case2_notFirst_transpose_baseline((CUFlt *)pCondlike_S + condlikeOffset, 
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	sh_condlike_S,
																	sh_PMat_S,
																	buf,
																	tx,
																	ty);
			}
		}
	}

	pCondlike_F[0] = buf[5];
	pCondlike_F[blockDim_x] = buf[6];
	pCondlike_F[(blockDim_x << 1)] = buf[7];
	pCondlike_F[blockDim_x * 3] = buf[8];
	pCondlike_F[(blockDim_x << 2)] = buf[9];
}

// Non-transpose version:
__global__
void kernelCondlike_20State_nChild_noTranspose_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern, const int nState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	__shared__ CuLCondlikeOp sh_condlikeOp;
	
	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int ind = (bx - startBlkInd) * nThreadPerBlock + threadIdx.y * blockDim.x + tx;
	int iChild;

	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset;
	void *pCondlike_S;
	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){
		if(1 == sh_condlikeOp.child_case[iChild]){
			if(0 == iChild){
				deviceCondlike_20State_case1_first_noTranspose_baseline(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild],
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern);
			}
			else{
				deviceCondlike_20State_case1_notFirst_noTranspose_baseline(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild],
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern);
			}
		}
		else{
			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			if(0 == iChild){
				deviceCondlike_20State_case2_first_noTranspose_baseline(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern, 
																	nState);
			}
			else{
				deviceCondlike_20State_case2_notFirst_noTranspose_baseline(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern, 
																	nState);
			}
		}
		__syncthreads();				// TODO: 是否可以去掉同步??? 应该可以，因为每个thread负责的不同的child的site pattern的state没有变化，且未使用shared memory;
	}
}




// For nPaddedState == 8:
// version 1 of case 1:
__device__
void deviceCondlike_xState_case1_first_transpose_baseline_8State(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE << 3);
	int curPattern, curState;
	
	condlike_F += ind;

	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = (ind >> 3);
		curState = (ind & 0x7);

#ifdef USING_LDG
		condlike_F[0] = PMat_S[(__ldg(&tipState_S[curPattern]) << 3) + curState];
#else
		condlike_F[0] = PMat_S[(tipState_S[curPattern] << 3) + curState];
#endif
	}
}


__device__
void deviceCondlike_xState_case1_notFirst_transpose_baseline_8State(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE << 3);
	int curPattern, curState;
	
	condlike_F += ind;

	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = (ind >> 3);
		curState = (ind & 0x7);

#ifdef USING_LDG
		condlike_F[0] *= PMat_S[(__ldg(&tipState_S[curPattern]) << 3) + curState];
#else
		condlike_F[0] *= PMat_S[(tipState_S[curPattern] << 3) + curState];
#endif
	}
}



// version 1 of case 2:
__device__
void deviceCondlike_xState_case2_first_transpose_baseline_8State(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, int nThread, int nState)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE << 3), curPattern, curState, iState;
	CUFlt *pCondlike_S, *pPMat_S, curSum;

	condlike_F += ind;
	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = (ind >> 3);
		curState = (ind & 0x7);

		pPMat_S = PMat_S + curState;
		pCondlike_S = condlike_S + (curPattern << 3);
		curSum = 0.0f;

		for(iState = 0; iState < nState; iState ++, pCondlike_S ++, pPMat_S += 8){
			curSum += pPMat_S[0] * pCondlike_S[0];
		}


		condlike_F[0] = curSum;
	}
}


__device__
void deviceCondlike_xState_case2_notFirst_transpose_baseline_8State(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, int nThread, int nState)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE << 3), curPattern, curState, iState;
	CUFlt *pCondlike_S, *pPMat_S, curSum;

	condlike_F += ind;
	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = (ind >> 3);
		curState = (ind & 0x7);

		pPMat_S = PMat_S + curState;
		pCondlike_S = condlike_S + (curPattern << 3);
		curSum = 0.0f;

		for(iState = 0; iState < nState; iState ++, pCondlike_S ++, pPMat_S += 8){
			curSum += pPMat_S[0] * pCondlike_S[0];
		}


		condlike_F[0] *= curSum;
	}
}


// For nPaddedState == 8:
// 一共两种version，这两种version所需的shared memory大小均为: 64 + 32 * 8，区别在于寄存器的用量;
// version 1: 每个孩子的condlike * PMat的结果直接写condlike_F，需要写condlike_F 2次，读1次;
// version 2: 使用一组寄存器(8 / 2 = 4个)保存各个孩子的condlike * PMat的中间结果，最后写condlike_F，需要写condlike_F 1次，读0次;
// 实验结果为：version 1好于version 2，因此用version 1;
__global__
void kernelCondlike_xState_nChild_transpose_baseline_8State(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];
	int ind = tx + threadIdx.y * BLOCK_DIMENSION_X_CONDLIKE_XSTATE_8_BASELINE;
	int nThread = BLOCK_DIMENSION_X_CONDLIKE_XSTATE_8_BASELINE * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE;

	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_S[64];
	__shared__ CUFlt sh_condlike_S[BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE << 3];
	
	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int blkOffset = bx - startBlkInd;
	int tipState_offset = blkOffset * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE;
	int condlike_offset = (tipState_offset << 3);
	int iChild, curInd;
	int nElemPerBlock = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE << 3);

	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, *pPMat_S;
	CUFlt *pCondlike_S;
	
	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){

		pPMat_S = PMat + sh_condlikeOp.child_P_offset[iChild];
		for(curInd = ind; curInd < 64; curInd += nThread){
#ifdef USING_LDG
			sh_PMat_S[curInd] = __ldg(&pPMat_S[curInd]);
#else
			sh_PMat_S[curInd] = pPMat_S[curInd];
#endif
		}

		__syncthreads();

		if(1 == sh_condlikeOp.child_case[iChild]){
			if(0 == iChild){
				deviceCondlike_xState_case1_first_transpose_baseline_8State(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild] + tipState_offset,
																	sh_PMat_S, 
																	ind, 
																	nThread);
			}
			else{
				deviceCondlike_xState_case1_notFirst_transpose_baseline_8State(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild] + tipState_offset,
																	sh_PMat_S, 
																	ind, 
																	nThread);
			}
		}
		else{
			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			pCondlike_S += condlike_offset;
			for(curInd = ind; curInd < nElemPerBlock; curInd += nThread){
#ifdef USING_LDG
				sh_condlike_S[curInd] = __ldg(&pCondlike_S[curInd]);
#else
				sh_condlike_S[curInd] = pCondlike_S[curInd];
#endif
			}

			__syncthreads();

			if(0 == iChild){
				deviceCondlike_xState_case2_first_transpose_baseline_8State(pCondlike_F, 
																	sh_condlike_S, 
																	sh_PMat_S, 
																	ind, 
																	nThread, 
																	nState);
			}
			else{
				deviceCondlike_xState_case2_notFirst_transpose_baseline_8State(pCondlike_F, 
																	sh_condlike_S, 
																	sh_PMat_S, 
																	ind, 
																	nThread, 
																	nState);
			}
		}
		__syncthreads();				// TODO: 是否可以去掉同步??? 应该可以，因为每个thread负责的不同的child的site pattern的state没有变化，且未使用shared memory;
	}
}


// For nPaddedState == 16:
// version 1 of case 1:
__device__
void deviceCondlike_xState_case1_first_transpose_baseline_16State(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE << 4);
	int curPattern, curState;
	
	condlike_F += ind;

	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = (ind >> 4);
		curState = (ind & 0xf);

#ifdef USING_LDG
		condlike_F[0] = PMat_S[(__ldg(&tipState_S[curPattern]) << 4) + curState];
#else
		condlike_F[0] = PMat_S[(tipState_S[curPattern] << 4) + curState];
#endif
	}
}


__device__
void deviceCondlike_xState_case1_notFirst_transpose_baseline_16State(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE << 4);
	int curPattern, curState;
	
	condlike_F += ind;

	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = (ind >> 4);
		curState = (ind & 0xf);

#ifdef USING_LDG
		condlike_F[0] *= PMat_S[(__ldg(&tipState_S[curPattern]) << 4) + curState];
#else
		condlike_F[0] *= PMat_S[(tipState_S[curPattern] << 4) + curState];
#endif
	}
}


// version 1 of case 2:
__device__
void deviceCondlike_xState_case2_first_transpose_baseline_16State(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE << 4);
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_16_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	const int nElemPerTile_P = (TILE_SIZE_CONDLINE_XSTATE_16_BASELINE << 4);
	const int nElemPerTile_Cl = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE * TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	CUFlt *pSh_condlike, *pCurPMat, *pCurCondlike;
	
	pSh_condlike = sh_buf + (TILE_SIZE_CONDLINE_XSTATE_16_BASELINE << 4);
	
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_S[curInd]);
#else
			sh_buf[curInd] = PMat_S[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_S[(curPattern << 4) + curState]);
#else
			pSh_condlike[curInd] = condlike_S[(curPattern << 4) + curState];
#endif
		}
		__syncthreads();


		for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, iElem ++){
			curState = (curInd & 0xf);
			curPattern = (curInd >> 4);

			pCurPMat = sh_buf + curState;
			pCurCondlike = pSh_condlike + curPattern * TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;

			for(iState = 0; iState < TILE_SIZE_CONDLINE_XSTATE_16_BASELINE; iState ++, pCurPMat += 16)
				temp_buf[iElem] += pCurPMat[0] * pCurCondlike[iState];
		}
		__syncthreads();

		PMat_S += nElemPerTile_P;
		condlike_S += TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] = temp_buf[iElem];
	}
}


__device__
void deviceCondlike_xState_case2_notFirst_transpose_baseline_16State(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE << 4);
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_16_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	const int nElemPerTile_P = (TILE_SIZE_CONDLINE_XSTATE_16_BASELINE << 4);
	const int nElemPerTile_Cl = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE * TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	CUFlt *pSh_condlike, *pCurPMat, *pCurCondlike;
	
	pSh_condlike = sh_buf + (TILE_SIZE_CONDLINE_XSTATE_16_BASELINE << 4);
	
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_S[curInd]);
#else
			sh_buf[curInd] = PMat_S[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_S[(curPattern << 4) + curState]);
#else
			pSh_condlike[curInd] = condlike_S[(curPattern << 4) + curState];
#endif
		}
		__syncthreads();


		for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, iElem ++){
			curState = (curInd & 0xf);
			curPattern = (curInd >> 4);

			pCurPMat = sh_buf + curState;
			pCurCondlike = pSh_condlike + curPattern * TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;

			for(iState = 0; iState < TILE_SIZE_CONDLINE_XSTATE_16_BASELINE; iState ++, pCurPMat += 16)
				temp_buf[iElem] += pCurPMat[0] * pCurCondlike[iState];
		}
		__syncthreads();

		PMat_S += nElemPerTile_P;
		condlike_S += TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] *= temp_buf[iElem];
	}
}



// For nPaddedState == 16:
// 一共两种version，所需的shared memory大小均为: 16 * 16 = 256;
// version 1: 使用一组寄存器，用于保存分块乘法的中间结果，需要写condlike_F k次，读k-1次(k为孩子数目);
// version 2: 使用2组寄存器，一组用于保存当前孩子的condlike * PMat的结果，一组用于保存分块乘法的中间结果，需要写condlike_F 1次，读0次;
// 实验结果为：version 1结果好于version 2，因此用version 1;
__global__
void kernelCondlike_xState_nChild_transpose_baseline_16State(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];
	int ind = tx + threadIdx.y * BLOCK_DIMENSION_X_CONDLIKE_XSTATE_16_BASELINE;
	int nThread = BLOCK_DIMENSION_X_CONDLIKE_XSTATE_16_BASELINE * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE;

	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_buf[16 * 16];
	
	CUFlt temp_buf[16 / BLOCK_DIMENSION_X_CONDLIKE_XSTATE_16_BASELINE];

	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int blkOffset = bx - startBlkInd;
	int tipState_offset = blkOffset * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE;
	int condlike_offset = (tipState_offset << 4);
	int iChild, iElem;
	int nElemPerThread = 16 / BLOCK_DIMENSION_X_CONDLIKE_XSTATE_16_BASELINE;

	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, *pPMat_S;
	CUFlt *pCondlike_S;
	
	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){

		pPMat_S = PMat + sh_condlikeOp.child_P_offset[iChild];

		if(1 == sh_condlikeOp.child_case[iChild]){

			for(iElem = ind; iElem < 256; iElem += nThread){
#ifdef USING_LDG
				sh_buf[iElem] = __ldg(&pPMat_S[iElem]);
#else
				sh_buf[iElem] = pPMat_S[iElem];
#endif
			}

			__syncthreads();

			if(0 == iChild){
				deviceCondlike_xState_case1_first_transpose_baseline_16State(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild] + tipState_offset,
																	sh_buf, 
																	ind, 
																	nThread);
			}
			else{
				deviceCondlike_xState_case1_notFirst_transpose_baseline_16State(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild] + tipState_offset,
																	sh_buf, 
																	ind, 
																	nThread);
			}
		}
		else{
			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			pCondlike_S += condlike_offset;
			
			for(iElem = 0; iElem < nElemPerThread; iElem ++)
				temp_buf[iElem] = 0.0f;

			if(0 == iChild){
				deviceCondlike_xState_case2_first_transpose_baseline_16State(pCondlike_F, 
																	pCondlike_S,
																	pPMat_S,
																	sh_buf,
																	temp_buf,
																	ind, 
																	nThread, 
																	nState);
			}
			else{
				deviceCondlike_xState_case2_notFirst_transpose_baseline_16State(pCondlike_F, 
																	pCondlike_S,
																	pPMat_S,
																	sh_buf,
																	temp_buf,
																	ind, 
																	nThread, 
																	nState);
			}
		}
		__syncthreads();
	}
}


// For nPaddedState == 24:
// version 1 of case 1:
__device__
void deviceCondlike_xState_case1_first_transpose_baseline_24State(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread)
{
	int nElemToCalc = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE * 24;
	int curPattern, curState;
	
	condlike_F += ind;

	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = ind / 24;
		curState = ind % 24;

#ifdef USING_LDG
		condlike_F[0] = __ldg(&PMat_S[(__ldg(&tipState_S[curPattern]) * 24) + curState]);
#else
		condlike_F[0] = PMat_S[(tipState_S[curPattern] * 24) + curState];
#endif
	}
}


__device__
void deviceCondlike_xState_case1_notFirst_transpose_baseline_24State(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread)
{
	int nElemToCalc = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE * 24;
	int curPattern, curState;
	
	condlike_F += ind;

	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = ind / 24;
		curState = ind % 24;

#ifdef USING_LDG
		condlike_F[0] *= __ldg(&PMat_S[(__ldg(&tipState_S[curPattern]) * 24) + curState]);
#else
		condlike_F[0] *= PMat_S[(tipState_S[curPattern] * 24) + curState];
#endif
	}
}



// version 1 of case 2:
__device__
void deviceCondlike_xState_case2_first_transpose_baseline_24State(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState)
{
	int nElemToCalc = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE * 24;
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_24_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	const int nElemPerTile_P = TILE_SIZE_CONDLINE_XSTATE_24_BASELINE * 24;
	const int nElemPerTile_Cl = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE * TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	CUFlt *pSh_condlike, *pCurPMat, *pCurCondlike;
	
	pSh_condlike = sh_buf + TILE_SIZE_CONDLINE_XSTATE_24_BASELINE * 24;
	
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_S[curInd]);
#else
			sh_buf[curInd] = PMat_S[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_S[curPattern * 24 + curState]);
#else
			pSh_condlike[curInd] = condlike_S[curPattern * 24 + curState];
#endif
		}
		__syncthreads();


		for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, iElem ++){
			curState = curInd % 24;
			curPattern = curInd / 24;

			pCurPMat = sh_buf + curState;
			pCurCondlike = pSh_condlike + curPattern * TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;

			for(iState = 0; iState < TILE_SIZE_CONDLINE_XSTATE_24_BASELINE; iState ++, pCurPMat += 24)
				temp_buf[iElem] += pCurPMat[0] * pCurCondlike[iState];
		}
		__syncthreads();

		PMat_S += nElemPerTile_P;
		condlike_S += TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] = temp_buf[iElem];
	}
}


__device__
void deviceCondlike_xState_case2_notFirst_transpose_baseline_24State(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState)
{
	int nElemToCalc = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE * 24;
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_24_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	const int nElemPerTile_P = TILE_SIZE_CONDLINE_XSTATE_24_BASELINE * 24;
	const int nElemPerTile_Cl = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE * TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	CUFlt *pSh_condlike, *pCurPMat, *pCurCondlike;
	
	pSh_condlike = sh_buf + TILE_SIZE_CONDLINE_XSTATE_24_BASELINE * 24;
	
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_S[curInd]);
#else
			sh_buf[curInd] = PMat_S[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_S[curPattern * 24 + curState]);
#else
			pSh_condlike[curInd] = condlike_S[curPattern * 24 + curState];
#endif
		}
		__syncthreads();


		for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, iElem ++){
			curState = curInd % 24;
			curPattern = curInd / 24;

			pCurPMat = sh_buf + curState;
			pCurCondlike = pSh_condlike + curPattern * TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;

			for(iState = 0; iState < TILE_SIZE_CONDLINE_XSTATE_24_BASELINE; iState ++, pCurPMat += 24)
				temp_buf[iElem] += pCurPMat[0] * pCurCondlike[iState];
		}
		__syncthreads();

		PMat_S += nElemPerTile_P;
		condlike_S += TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] *= temp_buf[iElem];
	}
}



// For nPaddedState == 24:
// 一共两种version，所需的shared memory大小均为: (32 + 24) * 4 = 226;
// version 1: 使用一组寄存器，用于保存分块乘法的中间结果，需要写condlike_F k次，读k-1次(k为孩子数目);
// version 2: 使用2组寄存器，一组用于保存当前孩子的condlike * PMat的结果，一组用于保存分块乘法的中间结果，需要写condlike_F 1次，读0次;
// version 1结果略好于version 2，因此用version 1;
__global__
void kernelCondlike_xState_nChild_transpose_baseline_24State(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];
	int ind = tx + threadIdx.y * BLOCK_DIMENSION_X_CONDLIKE_XSTATE_24_BASELINE;
	int nThread = BLOCK_DIMENSION_X_CONDLIKE_XSTATE_24_BASELINE * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE;

	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_buf[(32 + 24) * TILE_SIZE_CONDLINE_XSTATE_24_BASELINE];

	CUFlt temp_buf[24 / BLOCK_DIMENSION_X_CONDLIKE_XSTATE_24_BASELINE];

	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int blkOffset = bx - startBlkInd;
	int tipState_offset = blkOffset * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE;
	int condlike_offset = tipState_offset * 24;
	int iChild, iElem;
	int nElemPerThread = 24 / BLOCK_DIMENSION_X_CONDLIKE_XSTATE_24_BASELINE;

	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, *pPMat_S;
	CUFlt *pCondlike_S;
	
	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){

		pPMat_S = PMat + sh_condlikeOp.child_P_offset[iChild];

		if(1 == sh_condlikeOp.child_case[iChild]){

			if(0 == iChild){
				deviceCondlike_xState_case1_first_transpose_baseline_24State(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild] + tipState_offset,
																	pPMat_S,
																	ind, 
																	nThread);
			}
			else{
				deviceCondlike_xState_case1_notFirst_transpose_baseline_24State(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild] + tipState_offset,
																	pPMat_S, 
																	ind, 
																	nThread);
			}
		}
		else{
			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			pCondlike_S += condlike_offset;
			
			for(iElem = 0; iElem < nElemPerThread; iElem ++)
				temp_buf[iElem] = 0.0f;

			if(0 == iChild){
				deviceCondlike_xState_case2_first_transpose_baseline_24State(pCondlike_F, 
																	pCondlike_S,
																	pPMat_S,
																	sh_buf,
																	temp_buf,
																	ind, 
																	nThread, 
																	nState);
			}
			else{
				deviceCondlike_xState_case2_notFirst_transpose_baseline_24State(pCondlike_F, 
																	pCondlike_S,
																	pPMat_S,
																	sh_buf,
																	temp_buf,
																	ind, 
																	nThread, 
																	nState);
			}
		}
		__syncthreads();
	}
}


// For nPaddedState > 24 && != 64:
// version 1 of case 1:
__device__
void deviceCondlike_xState_case1_first_transpose_baseline_largeState(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread, int nPaddedState, int nElemToCalc)
{
	int curPattern, curState;
	
	condlike_F += ind;

	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = ind / nPaddedState;
		curState = ind % nPaddedState;

#ifdef USING_LDG
		condlike_F[0] = __ldg(&PMat_S[(__ldg(&tipState_S[curPattern]) * nPaddedState) + curState]);
#else
		condlike_F[0] = PMat_S[(tipState_S[curPattern] * nPaddedState) + curState];
#endif
	}
}


__device__
void deviceCondlike_xState_case1_notFirst_transpose_baseline_largeState(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThread, int nPaddedState, int nElemToCalc)
{
	int curPattern, curState;
	
	condlike_F += ind;

	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = ind / nPaddedState;
		curState = ind % nPaddedState;

#ifdef USING_LDG
		condlike_F[0] *= __ldg(&PMat_S[(__ldg(&tipState_S[curPattern]) * nPaddedState) + curState]);
#else
		condlike_F[0] *= PMat_S[(tipState_S[curPattern] * nPaddedState) + curState];
#endif
	}
}


// version 1 of case 2:
__device__
void deviceCondlike_xState_case2_first_transpose_baseline_largeState(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState, int nPaddedState, int blockDim_y)
{
	int nElemToCalc = blockDim_y * nPaddedState;
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	const int nElemPerTile_P = TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE * nPaddedState;
	const int nElemPerTile_Cl = blockDim_y * TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	CUFlt *pSh_condlike, *pCurPMat, *pCurCondlike;
	
	pSh_condlike = sh_buf + TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE * nPaddedState;
	
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_S[curInd]);
#else
			sh_buf[curInd] = PMat_S[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_S[curPattern * nPaddedState + curState]);
#else
			pSh_condlike[curInd] = condlike_S[curPattern * nPaddedState + curState];
#endif
		}
		__syncthreads();


		for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, iElem ++){
			curState = curInd % nPaddedState;
			curPattern = curInd / nPaddedState;

			pCurPMat = sh_buf + curState;
			pCurCondlike = pSh_condlike + curPattern * TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;

			for(iState = 0; iState < TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE; iState ++, pCurPMat += nPaddedState)
				temp_buf[iElem] += pCurPMat[0] * pCurCondlike[iState];
		}
		__syncthreads();

		PMat_S += nElemPerTile_P;
		condlike_S += TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] = temp_buf[iElem];
	}
}


__device__
void deviceCondlike_xState_case2_notFirst_transpose_baseline_largeState(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState, int nPaddedState, int blockDim_y)
{
	int nElemToCalc = blockDim_y * nPaddedState;
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	const int nElemPerTile_P = TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE * nPaddedState;
	const int nElemPerTile_Cl = blockDim_y * TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	CUFlt *pSh_condlike, *pCurPMat, *pCurCondlike;
	
	pSh_condlike = sh_buf + TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE * nPaddedState;
	
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_S[curInd]);
#else
			sh_buf[curInd] = PMat_S[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_S[curPattern * nPaddedState + curState]);
#else
			pSh_condlike[curInd] = condlike_S[curPattern * nPaddedState + curState];
#endif
		}
		__syncthreads();


		for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, iElem ++){
			curState = curInd % nPaddedState;
			curPattern = curInd / nPaddedState;

			pCurPMat = sh_buf + curState;
			pCurCondlike = pSh_condlike + curPattern * TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;

			for(iState = 0; iState < TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE; iState ++, pCurPMat += nPaddedState)
				temp_buf[iElem] += pCurPMat[0] * pCurCondlike[iState];
		}
		__syncthreads();

		PMat_S += nElemPerTile_P;
		condlike_S += TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] *= temp_buf[iElem];
	}
}


// For nPaddedState > 24 && != 64:
// 一共两种version，所需的shared memory大小均为: (32 + nPaddedState) * 4;
// version 1: 使用一组寄存器，用于保存分块乘法的中间结果，需要写condlike_F k次，读k-1次(k为孩子数目);
// version 2: 使用2组寄存器，一组用于保存当前孩子的condlike * PMat的结果，一组用于保存分块乘法的中间结果，需要写condlike_F 1次，读0次;
// 实验结果为：version 1略好于version 2，因此用version 1;
__global__
void kernelCondlike_xState_nChild_transpose_baseline_largeState(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState, const int nPaddedState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int blockDim_x = blockDim.x;
	int blockDim_y = blockDim.y;
	int opInd = blkIndToOpInd[bx];
	int ind = tx + threadIdx.y * blockDim_x;
	int nThread = blockDim_x * blockDim_y;

	__shared__ CuLCondlikeOp sh_condlikeOp;
	extern __shared__ CUFlt sh_buf[];

	CUFlt temp_buf[N_STATE_PER_THREAD_CONDLIKE_XSTATE_OTHER_BASELINE];

	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int blkOffset = bx - startBlkInd;
	int tipState_offset = blkOffset * blockDim_y;
	int condlike_offset = tipState_offset * nPaddedState;
	int iChild, iElem;
	int nElemPerThread = N_STATE_PER_THREAD_CONDLIKE_XSTATE_OTHER_BASELINE;
	int nElemPerBlock = blockDim_y * nPaddedState;

	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, *pPMat_S;
	CUFlt *pCondlike_S;
	
	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){

		pPMat_S = PMat + sh_condlikeOp.child_P_offset[iChild];

		if(1 == sh_condlikeOp.child_case[iChild]){

			if(0 == iChild){
				deviceCondlike_xState_case1_first_transpose_baseline_largeState(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild] + tipState_offset,
																	pPMat_S,
																	ind, 
																	nThread,
																	nPaddedState,
																	nElemPerBlock);
			}
			else{
				deviceCondlike_xState_case1_notFirst_transpose_baseline_largeState(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild] + tipState_offset,
																	pPMat_S, 
																	ind, 
																	nThread,
																	nPaddedState,
																	nElemPerBlock);
			}
		}
		else{
			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			pCondlike_S += condlike_offset;
			
			for(iElem = 0; iElem < nElemPerThread; iElem ++)
				temp_buf[iElem] = 0.0f;

			if(0 == iChild){
				deviceCondlike_xState_case2_first_transpose_baseline_largeState(pCondlike_F, 
																	pCondlike_S,
																	pPMat_S,
																	sh_buf,
																	temp_buf,
																	ind, 
																	nThread, 
																	nState,
																	nPaddedState,
																	blockDim_y);
			}
			else{
				deviceCondlike_xState_case2_notFirst_transpose_baseline_largeState(pCondlike_F, 
																	pCondlike_S,
																	pPMat_S,
																	sh_buf,
																	temp_buf,
																	ind, 
																	nThread, 
																	nState,
																	nPaddedState,
																	blockDim_y);
			}
		}
		__syncthreads();
	}
}



// For nState != 4 / 20 / 61:
// nThreadPerArray: 一共有多少个thread负责该condlike
// Transpose version:
__device__
void deviceCondlike_xState_case1_first_transpose_baseline(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThreadPerArray, int nSitePattern, const int nPaddedState)
{
	int nElement = nSitePattern * nPaddedState,offset_S, curSite, curState;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		offset_S = tipState_S[curSite] * nPaddedState;

		condlike_F[ind] = PMat_S[curState + offset_S];
	}
}


__device__
void deviceCondlike_xState_case1_notFirst_transpose_baseline(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThreadPerArray, int nSitePattern, int nPaddedState)
{
	int nElement = nSitePattern * nPaddedState,offset_S, curSite, curState;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		offset_S = tipState_S[curSite] * nPaddedState;

		condlike_F[ind] *= PMat_S[curState + offset_S];
	}
}


// Non-transpose version:
__device__
void deviceCondlike_xState_case1_first_noTranspose_baseline(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThreadPerArray, int nSitePattern, int nPaddedState)
{
	int nElement = nSitePattern * nPaddedState,offset_S, curSite, curState;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		offset_S = tipState_S[curSite];

		condlike_F[ind] = PMat_S[curState * nPaddedState + offset_S];
	}
}


__device__
void deviceCondlike_xState_case1_notFirst_noTranspose_baseline(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThreadPerArray, int nSitePattern, int nPaddedState)
{
	int nElement = nSitePattern * nPaddedState,offset_S, curSite, curState;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		offset_S = tipState_S[curSite];

		condlike_F[ind] *= PMat_S[curState * nPaddedState + offset_S];
	}
}



// case 2: the current child is tip/int condlike:
// Transpose version:
__device__
void deviceCondlike_xState_case2_first_transpose_baseline(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThreadPerArray, const int nSitePattern, const int nState, const int nPaddedState)
{
	CUFlt *pCondlike_S, sum_S;
	int nElement = nSitePattern * nPaddedState,curSite, curState;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		pCondlike_S = condlike_S + curSite * nPaddedState;
		
		sum_S = 0.0f;
		for(int iState = 0; iState < nState; iState ++){
			sum_S += pCondlike_S[iState] * PMat_S[curState + iState * nPaddedState];
		}

		condlike_F[ind] = sum_S;
	}
}


__device__
void deviceCondlike_xState_case2_notFirst_transpose_baseline(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThreadPerArray, const int nSitePattern, const int nState, const int nPaddedState)
{
	int nElement = nSitePattern * nPaddedState,curSite, curState;
	CUFlt *pCondlike_S, sum_S;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		pCondlike_S = condlike_S + curSite * nPaddedState;
		
		sum_S = 0.0f;
		for(int iState = 0; iState < nState; iState ++){
			sum_S += pCondlike_S[iState] * PMat_S[curState + iState * nPaddedState];
		}

		condlike_F[ind] *= sum_S;
	}
}


// Non-transpose version:
__device__
void deviceCondlike_xState_case2_first_noTranspose_baseline(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThreadPerArray, const int nSitePattern, const int nState, const int nPaddedState)
{
	int nElement = nSitePattern * nPaddedState, curSite, curState;
	CUFlt *pCondlike_S, sum_S;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		pCondlike_S = condlike_S + curSite * nPaddedState;
		
		curState *= nPaddedState;
		sum_S = 0.0f;
		for(int iState = 0; iState < nState; iState ++){
			sum_S += pCondlike_S[iState] * PMat_S[curState + iState];
		}

		condlike_F[ind] = sum_S;
	}
}


__device__
void deviceCondlike_xState_case2_notFirst_noTranspose_baseline(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, int ind, const int nThreadPerArray, const int nSitePattern, const int nState, const int nPaddedState)
{
	CUFlt *pCondlike_S, sum_S;
	int nElement = nSitePattern * nPaddedState, curSite, curState;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		pCondlike_S = condlike_S + curSite * nPaddedState;
		
		curState *= nPaddedState;
		sum_S = 0.0f;
		for(int iState = 0; iState < nState; iState ++){
			sum_S += pCondlike_S[iState] * PMat_S[curState + iState];
		}

		condlike_F[ind] *= sum_S;
	}
}



// Transpose version:
__global__
void kernelCondlike_xState_nChild_transpose_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern, const int nState, const int nPaddedState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	__shared__ CuLCondlikeOp sh_condlikeOp;
	
	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int ind = (bx - startBlkInd) * nThreadPerBlock + threadIdx.y * blockDim.x + tx;
	int iChild;

	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset;
	void *pCondlike_S;
	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){
		if(1 == sh_condlikeOp.child_case[iChild]){
			if(0 == iChild){
				deviceCondlike_xState_case1_first_transpose_baseline(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild],
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern,
																	nPaddedState);
			}
			else{
				deviceCondlike_xState_case1_notFirst_transpose_baseline(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild],
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern,
																	nPaddedState);
			}
		}
		else{
			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			if(0 == iChild){
				deviceCondlike_xState_case2_first_transpose_baseline(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern, 
																	nState,
																	nPaddedState);
			}
			else{
				deviceCondlike_xState_case2_notFirst_transpose_baseline(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern, 
																	nState,
																	nPaddedState);
			}
		}
		__syncthreads();				// TODO: 是否可以去掉同步??? 应该可以，因为每个thread负责的不同的child的site pattern的state没有变化，且未使用shared memory;
	}
}


// Non-transpose version:
__global__
void kernelCondlike_xState_nChild_noTranspose_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern, const int nState, const int nPaddedState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	__shared__ CuLCondlikeOp sh_condlikeOp;
	
	
	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int ind = (bx - startBlkInd) * nThreadPerBlock + threadIdx.y * blockDim.x + tx;
	int iChild;

	CUFlt *pCondlike_F = intCondlike + sh_condlikeOp.father_condlike_offset;
	void *pCondlike_S;
	for(iChild = 0; iChild < sh_condlikeOp.nChild; iChild ++){
		if(1 == sh_condlikeOp.child_case[iChild]){
			if(0 == iChild){
				deviceCondlike_xState_case1_first_noTranspose_baseline(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild],
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern,
																	nPaddedState);
			}
			else{
				deviceCondlike_xState_case1_notFirst_noTranspose_baseline(pCondlike_F, 
																	tipState + sh_condlikeOp.child_condlike_offset[iChild],
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern,
																	nPaddedState);
			}
		}
		else{
			if(2 == sh_condlikeOp.child_case[iChild])
				pCondlike_S = tipCondlike + sh_condlikeOp.child_condlike_offset[iChild];
			else
				pCondlike_S = intCondlike + sh_condlikeOp.child_condlike_offset[iChild];

			if(0 == iChild){
				deviceCondlike_xState_case2_first_noTranspose_baseline(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern, 
																	nState,
																	nPaddedState);
			}
			else{
				deviceCondlike_xState_case2_notFirst_noTranspose_baseline(pCondlike_F, 
																	(CUFlt *)pCondlike_S, 
																	PMat + sh_condlikeOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray, 
																	nSitePattern, 
																	nState,
																	nPaddedState);
			}
		}
		__syncthreads();				// TODO: 是否可以去掉同步??? 应该可以，因为每个thread负责的不同的child的site pattern的state没有变化，且未使用shared memory;
	}
}



void callKernelCondlike_baseline_unrooted(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const bool usePadVersion, const int nSitePattern, const int nPaddedState, const int nState, const int nThreadPerArray, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream)
{
	const int blockSize = nThreadPerBlock.x * nThreadPerBlock.y * nThreadPerBlock.z;
	if(4 == nPaddedState){
		//printf("\n=======\nGoing to call kernel for 4 state of condlike...\n==========\n");
		const int nTotalPattern = nOp * nSitePattern;
#ifdef TRANSPOSE_PMAT
		if(nTotalPattern < PATTERN_THRESHOLD_4STATE_UNROOTED_BASELINE)
			kernelCondlike_4State_nChild_transpose_baseline_version1<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		else
			kernelCondlike_4State_nChild_transpose_baseline_version2<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		cutilCheckMsg("kernel kernelCondlike_4State_nChild_transpose_baseline() failed");
#else
		if(nTotalPattern < PATTERN_THRESHOLD_4STATE_UNROOTED_BASELINE)
			kernelCondlike_4State_nChild_noTranspose_baseline_version1<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		else
			kernelCondlike_4State_nChild_noTranspose_baseline_version2<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		cutilCheckMsg("kernel kernelCondlike_4State_nChild_noTranspose_baseline() failed");
#endif
	}
	else if(20 == nPaddedState){
		//printf("\n=======\nGoing to call kernel for 20 state of condlike...\n==========\n");
#ifdef TRANSPOSE_PMAT
		kernelCondlike_20State_nChild_transpose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern, nState);
		cutilCheckMsg("kernel kernelCondlike_20State_nChild_transpose_baseline() failed");
#else
		kernelCondlike_20State_nChild_noTranspose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern, nState);
		cutilCheckMsg("kernel kernelCondlike_20State_nChild_noTranspose_baseline() failed");
#endif
	}
	else{
		// For nPaddedState != 4 / 20 / 64:
#ifdef TRANSPOSE_PMAT
		//printf("\n===========\nusePadVersion = %d, nSitePattern = %d\n===========\n", usePadVersion == true, nSitePattern);
		if(usePadVersion){
			if(8 == nPaddedState){
				//printf("\n=======\nGoing to call kernel for 8 state of condlike...\n==========\n");
				kernelCondlike_xState_nChild_transpose_baseline_8State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nState);
			}
			else if(16 == nPaddedState){
				//printf("\n=======\nGoing to call kernel for 16 state of condlike...\n==========\n");
				kernelCondlike_xState_nChild_transpose_baseline_16State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nState);
			}
			else if(24 == nPaddedState){
				//printf("\n=======\nGoing to call kernel for 24 state of condlike...\n==========\n");
				kernelCondlike_xState_nChild_transpose_baseline_24State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nState);
			}
			else{
				//printf("\n=======\nGoing to call kernel for large state of condlike...\n==========\n");

				const int sharedMem_needed = TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE * (nPaddedState + BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_OTHER_BASELINE) * sizeof(CUFlt);
				kernelCondlike_xState_nChild_transpose_baseline_largeState<<<nBlockPerGrid, nThreadPerBlock, sharedMem_needed, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nState, nPaddedState);
			}
		}
		else{
			// Use the traditional version, that is do not use shared memory;
			//printf("\n=======\nGoing to call no-pad version kernel for X state of condlike...\n==========\n");
			kernelCondlike_xState_nChild_transpose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern, nState, nPaddedState);
		}
		cutilCheckMsg("kernel kernelCondlike_xState_nChild_transpose_baseline() failed");
#else
		kernelCondlike_xState_nChild_noTranspose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern, nState, nPaddedState);
		cutilCheckMsg("kernel kernelCondlike_xState_nChild_noTranspose_baseline() failed");
#endif
	}
}