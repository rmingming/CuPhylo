#include "CuLibKernel-codemlAndMrBayes-unrooted.h"

// Codeml version:
// codeml's kernel for calculation of condlike:
// Case 1: tip state;
// 注意：原本若PMat没有转置，计算时应该是condlike的一行乘以PMat的一列，由于codeml计算PMat时进行了转置，因此这里在计算时是condlike的一行乘以PMat的一列；
// TODO: 对于case 1，使用shared memory存储PMat???
// transpose version:
// version 1.2 of case 1's first child: 每个thread block负责一个32 * 32的sub-matrix，其中各个thread负责4个2 * 2的sub-matrix，任务划分方式同case 2的version 1完全一样，故不需要同步即可同时使用;
// 该version要求block dimension为: (BLOCK_DIMENSION_X_CONDLIKE_CODEML, BLOCK_DIMENSION_Y_CONDLIKE_CODEML), 也即(8, 8);
// 另外，该version将condlike * PMat的结果写入寄存器，最后才写condlike_F;
__device__
void deviceCondlike_64State_case1_first_transpose_codeml(int *tipState_S, CUFlt *PMat_S, CUFlt *temp_buf, int tx, int ty)
{
	int ttx = (tx << 1), tty = (ty << 1), curState;
	CUFlt *pPMat_S;

	tipState_S += tty;
	PMat_S += ttx;

	// The first 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState = __ldg(&tipState_S[0]);
#else
	curState = tipState_S[0];
#endif
	pPMat_S = PMat_S + (curState << 6);
	
#ifdef USING_LDG
	temp_buf[0] = __ldg(&pPMat_S[0]);
	temp_buf[1] = __ldg(&pPMat_S[1]);
	temp_buf[2] = __ldg(&pPMat_S[16]);
	temp_buf[3] = __ldg(&pPMat_S[17]);
#else
	temp_buf[0] = pPMat_S[0];
	temp_buf[1] = pPMat_S[1];
	temp_buf[2] = pPMat_S[16];
	temp_buf[3] = pPMat_S[17];
#endif

	// The second 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState = __ldg(&tipState_S[1]);
#else
	curState = tipState_S[1];
#endif
	pPMat_S = PMat_S + (curState << 6);

#ifdef USING_LDG
	temp_buf[4] = __ldg(&pPMat_S[0]);
	temp_buf[5] = __ldg(&pPMat_S[1]);
	temp_buf[6] = __ldg(&pPMat_S[16]);
	temp_buf[7] = __ldg(&pPMat_S[17]);
#else
	temp_buf[4] = pPMat_S[0];
	temp_buf[5] = pPMat_S[1];
	temp_buf[6] = pPMat_S[16];
	temp_buf[7] = pPMat_S[17];
#endif

	// The third 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState = __ldg(&tipState_S[16]);
#else
	curState = tipState_S[16];
#endif
	pPMat_S = PMat_S + (curState << 6);

#ifdef USING_LDG
	temp_buf[8] = __ldg(&pPMat_S[0]);
	temp_buf[9] = __ldg(&pPMat_S[1]);
	temp_buf[10] = __ldg(&pPMat_S[16]);
	temp_buf[11] = __ldg(&pPMat_S[17]);
#else
	temp_buf[8] = pPMat_S[0];
	temp_buf[9] = pPMat_S[1];
	temp_buf[10] = pPMat_S[16];
	temp_buf[11] = pPMat_S[17];
#endif

	// The fourth 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState = __ldg(&tipState_S[17]);
#else
	curState = tipState_S[17];
#endif
	pPMat_S = PMat_S + (curState << 6);

#ifdef USING_LDG
	temp_buf[12] = __ldg(&pPMat_S[0]);
	temp_buf[13] = __ldg(&pPMat_S[1]);
	temp_buf[14] = __ldg(&pPMat_S[16]);
	temp_buf[15] = __ldg(&pPMat_S[17]);
#else
	temp_buf[12] = pPMat_S[0];
	temp_buf[13] = pPMat_S[1];
	temp_buf[14] = pPMat_S[16];
	temp_buf[15] = pPMat_S[17];
#endif
}



// version 1.2 of case 1's not first child:
// 该version要求block dimension为: (BLOCK_DIMENSION_X_CONDLIKE_CODEML, BLOCK_DIMENSION_Y_CONDLIKE_CODEML), 也即(8, 8);
__device__
void deviceCondlike_64State_case1_notFirst_transpose_codeml(int *tipState_S, CUFlt *PMat_S, CUFlt *temp_buf, int tx, int ty)
{
	int ttx = (tx << 1), tty = (ty << 1), curState;
	CUFlt *pPMat_S;

	tipState_S += tty;
	PMat_S += ttx;

	// The first 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState = __ldg(&tipState_S[0]);
#else
	curState = tipState_S[0];
#endif
	pPMat_S = PMat_S + (curState << 6);
	
#ifdef USING_LDG
	temp_buf[0] *= __ldg(&pPMat_S[0]);
	temp_buf[1] *= __ldg(&pPMat_S[1]);
	temp_buf[2] *= __ldg(&pPMat_S[16]);
	temp_buf[3] *= __ldg(&pPMat_S[17]);
#else
	temp_buf[0] *= pPMat_S[0];
	temp_buf[1] *= pPMat_S[1];
	temp_buf[2] *= pPMat_S[16];
	temp_buf[3] *= pPMat_S[17];
#endif

	// The second 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState = __ldg(&tipState_S[1]);
#else
	curState = tipState_S[1];
#endif
	pPMat_S = PMat_S + (curState << 6);

#ifdef USING_LDG
	temp_buf[4] *= __ldg(&pPMat_S[0]);
	temp_buf[5] *= __ldg(&pPMat_S[1]);
	temp_buf[6] *= __ldg(&pPMat_S[16]);
	temp_buf[7] *= __ldg(&pPMat_S[17]);
#else
	temp_buf[4] *= pPMat_S[0];
	temp_buf[5] *= pPMat_S[1];
	temp_buf[6] *= pPMat_S[16];
	temp_buf[7] *= pPMat_S[17];
#endif

	// The third 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState = __ldg(&tipState_S[16]);
#else
	curState = tipState_S[16];
#endif
	pPMat_S = PMat_S + (curState << 6);

#ifdef USING_LDG
	temp_buf[8] *= __ldg(&pPMat_S[0]);
	temp_buf[9] *= __ldg(&pPMat_S[1]);
	temp_buf[10] *= __ldg(&pPMat_S[16]);
	temp_buf[11] *= __ldg(&pPMat_S[17]);
#else
	temp_buf[8] *= pPMat_S[0];
	temp_buf[9] *= pPMat_S[1];
	temp_buf[10] *= pPMat_S[16];
	temp_buf[11] *= pPMat_S[17];
#endif

	// The fourth 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState = __ldg(&tipState_S[17]);
#else
	curState = tipState_S[17];
#endif
	pPMat_S = PMat_S + (curState << 6);

#ifdef USING_LDG
	temp_buf[12] *= __ldg(&pPMat_S[0]);
	temp_buf[13] *= __ldg(&pPMat_S[1]);
	temp_buf[14] *= __ldg(&pPMat_S[16]);
	temp_buf[15] *= __ldg(&pPMat_S[17]);
#else
	temp_buf[12] *= pPMat_S[0];
	temp_buf[13] *= pPMat_S[1];
	temp_buf[14] *= pPMat_S[16];
	temp_buf[15] *= pPMat_S[17];
#endif
}



// Non-transpose version:
__device__
void deviceCondlike_64State_case1_first_noTranspose_codeml(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThreadPerArray, int nElementPerArray)
{
	int curPattern, curState, itr;

	for(itr = ind; itr < nElementPerArray; itr += nThreadPerArray){
		curPattern = (itr >> 6);
		curState = (itr & 0x3f);

		// row * row
#ifdef USING_LDG
		condlike_F[itr] = __ldg(&PMat_S[(curState << 6) + __ldg(&tipState_S[curPattern])]);
#else
		condlike_F[itr] = PMat_S[(curState << 6) + tipState_S[curPattern]];
#endif
	}
}

__device__
void deviceCondlike_64State_case1_notFirst_noTranspose_codeml(CUFlt *condlike_F, int *tipState_S, CUFlt *PMat_S, int ind, int nThreadPerArray, int nElementPerArray)
{
	int curPattern, curState, itr;

	for(itr = ind; itr < nElementPerArray; itr += nThreadPerArray){
		curPattern = (itr >> 6);
		curState = (itr & 0x3f);

		// row * row
#ifdef USING_LDG
		condlike_F[itr] *= __ldg(&PMat_S[(curState << 6) + __ldg(&tipState_S[curPattern])]);
#else
		condlike_F[itr] *= PMat_S[(curState << 6) + tipState_S[curPattern]];
#endif
	}
}


// case 2: for condlike:
// Transpose version, row * col:
// version 1.2 of case 2's first child: 每个thread负责4个2 * 2的sub-matrix的计算，一个thread block负责一个32 * 32的sub-matrix:
// 该version每个thread block计算一个32 * 32的sub-matrix，要求thread block dimension为: (8, 8);
// 该version将当前孩子的condlike * PMat的计算结果写入寄存器，最后才写condlike_F;
__device__
void deviceCondlike_64State_case2_first_transpose_codeml(CUFlt *condlike_S, CUFlt *PMat_S, CUFlt sh_condlike[][TILE_SIZE_CONDLIKE_CODEML], CUFlt sh_PMat[][32], CUFlt *temp_buf, const int tx, const int ty)
{
	const int tty = (ty << 1);
	const int ttx = (tx << 1);

	condlike_S += (tty << 6) + tx;
	PMat_S += (ty << 6) + ttx;

	int itr, i, nIteration = 64 / TILE_SIZE_CONDLIKE_CODEML;
	for(itr = 0; itr < nIteration; itr ++){
#ifdef USING_LDG
		sh_condlike[tty][tx] = __ldg(&condlike_S[0]);
		sh_condlike[tty + 1][tx] = __ldg(&condlike_S[64]);
		sh_condlike[tty + 16][tx] = __ldg(&condlike_S[16 << 6]);
		sh_condlike[tty + 17][tx] = __ldg(&condlike_S[17 << 6]);

		sh_PMat[ty][ttx] = __ldg(&PMat_S[0]);
		sh_PMat[ty][ttx + 1] = __ldg(&PMat_S[1]);
		sh_PMat[ty][ttx + 16] = __ldg(&PMat_S[16]);
		sh_PMat[ty][ttx + 17] = __ldg(&PMat_S[17]);
#else
		sh_condlike[tty][tx] = condlike_S[0];
		sh_condlike[tty + 1][tx] = condlike_S[64];
		sh_condlike[tty + 16][tx] = condlike_S[16 << 6];
		sh_condlike[tty + 17][tx] = condlike_S[17 << 6];

		sh_PMat[ty][ttx] = PMat_S[0];
		sh_PMat[ty][ttx + 1] = PMat_S[1];
		sh_PMat[ty][ttx + 16] = PMat_S[16];
		sh_PMat[ty][ttx + 17] = PMat_S[17];
#endif
        __syncthreads();

	    for(i = 0; i < TILE_SIZE_CONDLIKE_CODEML; i ++){
	      temp_buf[0] += sh_condlike[tty][i] * sh_PMat[i][ttx]; 
	      temp_buf[1] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 1];
	      temp_buf[2] += sh_condlike[tty][i] * sh_PMat[i][ttx+1];
	      temp_buf[3] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx];

	      temp_buf[4] += sh_condlike[tty][i] * sh_PMat[i][ttx + 16];
	      temp_buf[5] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 17];
	      temp_buf[6] += sh_condlike[tty][i] * sh_PMat[i][ttx + 17];
	      temp_buf[7] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 16];
	   
	      temp_buf[8] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx];
	      temp_buf[9] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 1];
	      temp_buf[10] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 1];
	      temp_buf[11] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx];

	      temp_buf[12] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 16];
	      temp_buf[13] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 17];
	      temp_buf[14] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 17];
	      temp_buf[15] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 16];
	  }
      __syncthreads();

      condlike_S += TILE_SIZE_CONDLIKE_CODEML;
      PMat_S += (TILE_SIZE_CONDLIKE_CODEML << 6);	
	}
  
	
	temp_buf[16] = temp_buf[0];
	temp_buf[17] = temp_buf[2];
	temp_buf[18] = temp_buf[4];
	temp_buf[19] = temp_buf[6];

	temp_buf[20] = temp_buf[3];
	temp_buf[21] = temp_buf[1];
	temp_buf[22] = temp_buf[7];
	temp_buf[23] = temp_buf[5];

	temp_buf[24] = temp_buf[8];
	temp_buf[25] = temp_buf[10];
	temp_buf[26] = temp_buf[12];
	temp_buf[27] = temp_buf[14];

	temp_buf[28] = temp_buf[11];
	temp_buf[29] = temp_buf[9];
	temp_buf[30] = temp_buf[15];
	temp_buf[31] = temp_buf[13];
}


// version 1.2 of case 2's not first child:
__device__
void deviceCondlike_64State_case2_notFirst_transpose_codeml(CUFlt *condlike_S, CUFlt *PMat_S, CUFlt sh_condlike[][TILE_SIZE_CONDLIKE_CODEML], CUFlt sh_PMat[][32], CUFlt *temp_buf, const int tx, const int ty)
{
	const int tty = (ty << 1);
	const int ttx = (tx << 1);

	condlike_S += (tty << 6) + tx;
	PMat_S += (ty << 6) + ttx;

	int itr, i, nIteration = 64 / TILE_SIZE_CONDLIKE_CODEML;
	for(itr = 0; itr < nIteration; itr ++){
#ifdef USING_LDG
		sh_condlike[tty][tx] = __ldg(&condlike_S[0]);
		sh_condlike[tty + 1][tx] = __ldg(&condlike_S[64]);
		sh_condlike[tty + 16][tx] = __ldg(&condlike_S[16 << 6]);
		sh_condlike[tty + 17][tx] = __ldg(&condlike_S[17 << 6]);

		sh_PMat[ty][ttx] = __ldg(&PMat_S[0]);
		sh_PMat[ty][ttx + 1] = __ldg(&PMat_S[1]);
		sh_PMat[ty][ttx + 16] = __ldg(&PMat_S[16]);
		sh_PMat[ty][ttx + 17] = __ldg(&PMat_S[17]);
#else
		sh_condlike[tty][tx] = condlike_S[0];
		sh_condlike[tty + 1][tx] = condlike_S[64];
		sh_condlike[tty + 16][tx] = condlike_S[16 << 6];
		sh_condlike[tty + 17][tx] = condlike_S[17 << 6];

		sh_PMat[ty][ttx] = PMat_S[0];
		sh_PMat[ty][ttx + 1] = PMat_S[1];
		sh_PMat[ty][ttx + 16] = PMat_S[16];
		sh_PMat[ty][ttx + 17] = PMat_S[17];
#endif
        __syncthreads();

	    for(i = 0; i < TILE_SIZE_CONDLIKE_CODEML; i ++){
	      temp_buf[0] += sh_condlike[tty][i] * sh_PMat[i][ttx]; 
	      temp_buf[1] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 1];
	      temp_buf[2] += sh_condlike[tty][i] * sh_PMat[i][ttx+1];
	      temp_buf[3] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx];

	      temp_buf[4] += sh_condlike[tty][i] * sh_PMat[i][ttx + 16];
	      temp_buf[5] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 17];
	      temp_buf[6] += sh_condlike[tty][i] * sh_PMat[i][ttx + 17];
	      temp_buf[7] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 16];
	   
	      temp_buf[8] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx];
	      temp_buf[9] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 1];
	      temp_buf[10] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 1];
	      temp_buf[11] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx];

	      temp_buf[12] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 16];
	      temp_buf[13] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 17];
	      temp_buf[14] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 17];
	      temp_buf[15] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 16];
	  }
      __syncthreads();

      condlike_S += TILE_SIZE_CONDLIKE_CODEML;
      PMat_S += (TILE_SIZE_CONDLIKE_CODEML << 6);	
	}
  
	
	temp_buf[16] *= temp_buf[0];
	temp_buf[17] *= temp_buf[2];
	temp_buf[18] *= temp_buf[4];
	temp_buf[19] *= temp_buf[6];

	temp_buf[20] *= temp_buf[3];
	temp_buf[21] *= temp_buf[1];
	temp_buf[22] *= temp_buf[7];
	temp_buf[23] *= temp_buf[5];

	temp_buf[24] *= temp_buf[8];
	temp_buf[25] *= temp_buf[10];
	temp_buf[26] *= temp_buf[12];
	temp_buf[27] *= temp_buf[14];

	temp_buf[28] *= temp_buf[11];
	temp_buf[29] *= temp_buf[9];
	temp_buf[30] *= temp_buf[15];
	temp_buf[31] *= temp_buf[13];
}


// Non-transpose version, row * row:
__device__
void deviceCondlike_64State_case2_first_noTranspose_codeml(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, CUFlt sh_condlike[][TILE_SIZE_CONDLIKE_CODEML], CUFlt sh_PMat[][32], CUFlt *temp_buf, const int bx, const int by, const int tx, const int ty)
{
	const int tty = (ty << 1);
	const int ttx = (tx << 1);

	condlike_S += (tty << 6) + tx;
	PMat_S += (tty << 6) + (bx << 11) + tx;

	int itr, i, nIteration = 64 / TILE_SIZE_CONDLIKE_CODEML;
	for(itr = 0; itr < nIteration; itr ++){
#ifdef USING_LDG
		sh_condlike[tty][tx] = __ldg(&condlike_S[0]);
		sh_condlike[tty + 1][tx] = __ldg(&condlike_S[64]);
		sh_condlike[tty + 16][tx] = __ldg(&condlike_S[16 << 6]);
		sh_condlike[tty + 17][tx] = __ldg(&condlike_S[17 << 6]);

		sh_PMat[tx][tty] = __ldg(&PMat_S[0]);
		sh_PMat[tx][tty + 1] = __ldg(&PMat_S[64]);
		sh_PMat[tx][tty + 16] = __ldg(&PMat_S[16 << 6]);
		sh_PMat[tx][tty + 17] = __ldg(&PMat_S[17 << 6]);

#else
		sh_condlike[tty][tx] = condlike_S[0];
		sh_condlike[tty + 1][tx] = condlike_S[64];
		sh_condlike[tty + 16][tx] = condlike_S[16 << 6];
		sh_condlike[tty + 17][tx] = condlike_S[17 << 6];

		sh_PMat[tx][tty] = PMat_S[0];
		sh_PMat[tx][tty + 1] = PMat_S[64];
		sh_PMat[tx][tty + 16] = PMat_S[16 << 6];
		sh_PMat[tx][tty + 17] = PMat_S[17 << 6];
#endif

        __syncthreads();

	    for(i = 0; i < TILE_SIZE_CONDLIKE_CODEML; i ++){
	      temp_buf[0] += sh_condlike[tty][i] * sh_PMat[i][ttx]; 
	      temp_buf[1] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 1];
	      temp_buf[2] += sh_condlike[tty][i] * sh_PMat[i][ttx+1];
	      temp_buf[3] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx];

	      temp_buf[4] += sh_condlike[tty][i] * sh_PMat[i][ttx + 16];
	      temp_buf[5] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 17];
	      temp_buf[6] += sh_condlike[tty][i] * sh_PMat[i][ttx + 17];
	      temp_buf[7] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 16];
	   
	      temp_buf[8] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx];
	      temp_buf[9] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 1];
	      temp_buf[10] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 1];
	      temp_buf[11] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx];

	      temp_buf[12] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 16];
	      temp_buf[13] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 17];
	      temp_buf[14] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 17];
	      temp_buf[15] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 16];
	  }
      __syncthreads();

      condlike_S += TILE_SIZE_CONDLIKE_CODEML;
      PMat_S += TILE_SIZE_CONDLIKE_CODEML;
  }
	
  condlike_F += (tty << 6) + (bx << 5) + ttx;
  
  int curOffset;
  condlike_F[0] = temp_buf[0];
  condlike_F[1] = temp_buf[2];
  condlike_F[16] = temp_buf[4];
  condlike_F[17] = temp_buf[6];

  condlike_F[64] = temp_buf[3];
  condlike_F[64 + 1] = temp_buf[1];
  condlike_F[64 + 16] = temp_buf[7];
  condlike_F[64 + 17] = temp_buf[5];

  curOffset = (16 << 6);
  condlike_F[curOffset] = temp_buf[8];
  condlike_F[curOffset + 1] = temp_buf[10];
  condlike_F[curOffset + 16] = temp_buf[12];
  condlike_F[curOffset + 17] = temp_buf[14];

  curOffset += 64;
  condlike_F[curOffset] = temp_buf[11];
  condlike_F[curOffset + 1] = temp_buf[9];
  condlike_F[curOffset + 16] = temp_buf[15];
  condlike_F[curOffset + 17] = temp_buf[13];
}


__device__
void deviceCondlike_64State_case2_notFirst_noTranspose_codeml(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, CUFlt sh_condlike[][TILE_SIZE_CONDLIKE_CODEML], CUFlt sh_PMat[][32], CUFlt *temp_buf, const int bx, const int by, const int tx, const int ty)
{
	const int tty = (ty << 1);
	const int ttx = (tx << 1);

	condlike_S += (tty << 6) + tx;
	PMat_S += (tty << 6) + (bx << 11) + tx;	
   
	int itr, i, nIteration = 64 / TILE_SIZE_CONDLIKE_CODEML;
	for(itr = 0; itr < nIteration; itr ++){
#ifdef USING_LDG
		sh_condlike[tty][tx] = __ldg(&condlike_S[0]);
		sh_condlike[tty + 1][tx] = __ldg(&condlike_S[64]);
		sh_condlike[tty + 16][tx] = __ldg(&condlike_S[16 << 6]);
		sh_condlike[tty + 17][tx] = __ldg(&condlike_S[17 << 6]);

		sh_PMat[tx][tty] = __ldg(&PMat_S[0]);
		sh_PMat[tx][tty + 1] = __ldg(&PMat_S[64]);
		sh_PMat[tx][tty + 16] = __ldg(&PMat_S[16 << 6]);
		sh_PMat[tx][tty + 17] = __ldg(&PMat_S[17 << 6]);
#else
		sh_condlike[tty][tx] = condlike_S[0];
		sh_condlike[tty + 1][tx] = condlike_S[64];
		sh_condlike[tty + 16][tx] = condlike_S[16 << 6];
		sh_condlike[tty + 17][tx] = condlike_S[17 << 6];

		sh_PMat[tx][tty] = PMat_S[0];
		sh_PMat[tx][tty + 1] = PMat_S[64];
		sh_PMat[tx][tty + 16] = PMat_S[16 << 6];
		sh_PMat[tx][tty + 17] = PMat_S[17 << 6];
#endif

        __syncthreads();

	    for(i = 0; i < TILE_SIZE_CONDLIKE_CODEML; i ++){
	      temp_buf[0] += sh_condlike[tty][i] * sh_PMat[i][ttx]; 
	      temp_buf[1] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 1];
	      temp_buf[2] += sh_condlike[tty][i] * sh_PMat[i][ttx+1];
	      temp_buf[3] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx];

	      temp_buf[4] += sh_condlike[tty][i] * sh_PMat[i][ttx + 16];
	      temp_buf[5] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 17];
	      temp_buf[6] += sh_condlike[tty][i] * sh_PMat[i][ttx + 17];
	      temp_buf[7] += sh_condlike[tty + 1][i] * sh_PMat[i][ttx + 16];
	   
	      temp_buf[8] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx];
	      temp_buf[9] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 1];
	      temp_buf[10] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 1];
	      temp_buf[11] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx];

	      temp_buf[12] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 16];
	      temp_buf[13] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 17];
	      temp_buf[14] += sh_condlike[tty + 16][i] * sh_PMat[i][ttx + 17];
	      temp_buf[15] += sh_condlike[tty + 17][i] * sh_PMat[i][ttx + 16];
	  }
      __syncthreads();

      condlike_S += TILE_SIZE_CONDLIKE_CODEML;
      PMat_S += TILE_SIZE_CONDLIKE_CODEML;
  }
	
  condlike_F += (tty << 6) + (bx << 5) + ttx;
  
  int curOffset;
  condlike_F[0] *= temp_buf[0];
  condlike_F[1] *= temp_buf[2];
  condlike_F[16] *= temp_buf[4];
  condlike_F[17] *= temp_buf[6];

  condlike_F[64] *= temp_buf[3];
  condlike_F[64 + 1] *= temp_buf[1];
  condlike_F[64 + 16] *= temp_buf[7];
  condlike_F[64 + 17] *= temp_buf[5];

  curOffset = (16 << 6);
  condlike_F[curOffset] *= temp_buf[8];
  condlike_F[curOffset + 1] *= temp_buf[10];
  condlike_F[curOffset + 16] *= temp_buf[12];
  condlike_F[curOffset + 17] *= temp_buf[14];

  curOffset += 64;
  condlike_F[curOffset] *= temp_buf[11];
  condlike_F[curOffset + 1] *= temp_buf[9];
  condlike_F[curOffset + 16] *= temp_buf[15];
  condlike_F[curOffset + 17] *= temp_buf[13];
}



// case 1的6种version中只有前5种可以用于unrooted version，case 2的5种version均可用于unrooted version;
// case 1的version 1/2/3的任务分配方式为每个thread block负责一个32 * 32的sub-matrix，可与case 2的version 1搭配使用;
// case 2的version 4/5的任务分配方式为每个thread block负责k个site pattern(也即k * 64的sub-matrix)，可与case 2的version 2/3/5搭配使用，但由于version 4用寄存器而不是shared memory保存condlike的分块，因此搭配方式为：4 & 4; 5 & 2; 5 & 3; 5 & 5;
// 因此一共有7种搭配方式: 1 & 1, 2 & 1, 3 & 1, 4 & 4, 5 & 2, 5 & 3, 5 & 5；
// 其中1 & 1, 4 & 4和5 & 5由于每个thread负责的state完全一样，又可以有两种方式：(1) 每个孩子的condlike * PMat的结果直接写condlike_F；(2) 每个孩子的condlike * PMat的结果写寄存器，最后才写condlike_F;
// version 1.2: case 1为version 1.2, case 2为version 1.2
// 每个孩子的condlike * PMat写寄存器，最后才写condlike_F;
// 实验结果为：version 1.2效果最好，因此用version 1.2;
__global__ void kernelCondlike_nChild_transpose_codeml(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState)
{
	const int bx = blockIdx.x;
	const int ty = threadIdx.y;
    const int tx = threadIdx.x;

	const int opInd = blkIndToOpInd[bx];
	CuLCondlikeOp curOp = condlikeOp[opInd];

	int iChild;
	CUFlt *pCondlike_F = intCondlike + curOp.father_condlike_offset;

	const int blkOffset = bx - opStartBlkInd[opInd];
	const int tile_y = (blkOffset >> 1);
	const int tile_x = (blkOffset & 0x1);
	const int tipState_offset = (tile_y << 5);
	const int condlike_offset = (tile_y << 11);
	const int tile_offset = (tile_x << 5);

	CUFlt temp_buf[32];

	if(1 == curOp.whichCase){		// All children are tip state, do not use shared memory;

		for(iChild = 0; iChild < curOp.nChild; iChild ++){
			if(0 == iChild){
				deviceCondlike_64State_case1_first_transpose_codeml(tipState + curOp.child_condlike_offset[iChild] + tipState_offset, 
																	PMat + curOp.child_P_offset[iChild] + tile_offset,
																	temp_buf + 16,
																	tx,
																	ty);
			}
			else{
				deviceCondlike_64State_case1_notFirst_transpose_codeml(tipState + curOp.child_condlike_offset[iChild] + tipState_offset, 
																	PMat + curOp.child_P_offset[iChild] + tile_offset,
																	temp_buf + 16,
																	tx,
																	ty);
			}
		}
	}		// At least one child is condlike, use shared memory;
	else{
		CUFlt *pCondlike_S;

		int iElem;

		// Use shared memory to store a block of PMat and condlike;
		__shared__ CUFlt sh_condlike[32][TILE_SIZE_CONDLIKE_CODEML];
		__shared__ CUFlt sh_PMat[TILE_SIZE_CONDLIKE_CODEML][32];

		for(iChild = 0; iChild < curOp.nChild; iChild ++){
			if(1 == curOp.child_case[iChild]){			// Current child is tip state;
				if(0 == iChild){
					deviceCondlike_64State_case1_first_transpose_codeml(tipState + curOp.child_condlike_offset[iChild] + tipState_offset, 
																	PMat + curOp.child_P_offset[iChild] + tile_offset,
																	temp_buf + 16,
																	tx,
																	ty);
				}
				else{
					deviceCondlike_64State_case1_notFirst_transpose_codeml(tipState + curOp.child_condlike_offset[iChild] + tipState_offset, 
																	PMat + curOp.child_P_offset[iChild] + tile_offset,
																	temp_buf + 16,
																	tx,
																	ty);
				}
			}
			else{				// Current child is tip condlike / int condlike;
				for(iElem = 0; iElem < 16; iElem ++)
					temp_buf[iElem] = 0;

				if(2 == curOp.child_case[iChild])
					pCondlike_S = tipCondlike + curOp.child_condlike_offset[iChild] + condlike_offset;
				else
					pCondlike_S = intCondlike + curOp.child_condlike_offset[iChild] + condlike_offset;

				if(0 == iChild){
					deviceCondlike_64State_case2_first_transpose_codeml(pCondlike_S, 
																			PMat + curOp.child_P_offset[iChild] + tile_offset, 
																			sh_condlike,
																			sh_PMat,
																			temp_buf,
																			tx,
																			ty);
				}
				else{
					deviceCondlike_64State_case2_notFirst_transpose_codeml(pCondlike_S, 
																			PMat + curOp.child_P_offset[iChild] + tile_offset, 
																			sh_condlike,
																			sh_PMat,
																			temp_buf,
																			tx,
																			ty);
				}
			}
			//__syncthreads();		// TODO: 是否可以去掉该同步指令???
		}
	}

	pCondlike_F += condlike_offset + tile_offset + (ty << 7) + (tx << 1);
	// the first 2 * 2 sub-matrix:
	pCondlike_F[0] = temp_buf[16];
	pCondlike_F[1] = temp_buf[17];
	pCondlike_F[16] = temp_buf[18];
	pCondlike_F[17] = temp_buf[19];

	// the second 2 * 2 sub-matrix:
	pCondlike_F += 64;
	pCondlike_F[0] = temp_buf[20];
	pCondlike_F[1] = temp_buf[21];
	pCondlike_F[16] = temp_buf[22];
	pCondlike_F[17] = temp_buf[23];

	// the third 2 * 2 sub-matrix:
	pCondlike_F += (15 << 6);
	pCondlike_F[0] = temp_buf[24];
	pCondlike_F[1] = temp_buf[25];
	pCondlike_F[16] = temp_buf[26];
	pCondlike_F[17] = temp_buf[27];

	// the fourth 2 * 2 sub-matrix:
	pCondlike_F += 64;
	pCondlike_F[0] = temp_buf[28];
	pCondlike_F[1] = temp_buf[29];
	pCondlike_F[16] = temp_buf[30];
	pCondlike_F[17] = temp_buf[31];
}


// Non-transpose version:
__global__ void kernelCondlike_nChild_noTranspose_codeml(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState, const int nThreadPerBlock, const int nThreadPerArray_case1, const int nThreadPerArray_case2, const int nElementPerArray)
{
/*
  Kernel of calculating the conditional probability of the node whose current son(ison) is an internal node.
*/
	const int bx = blockIdx.x;
	const int ty = threadIdx.y;
    const int tx = threadIdx.x;

	const int thdInd = ty * blockDim.x + tx;
	const int opInd = blkIndToOpInd[bx];
	CuLCondlikeOp curOp = condlikeOp[opInd];

	int iChild;
	CUFlt *pCondlike_F = intCondlike + curOp.father_condlike_offset;
	int ind = (bx - opStartBlkInd[opInd]) * nThreadPerBlock + thdInd;

	if(1 == curOp.whichCase){		// All children are tip state, do not use shared memory;

		for(iChild = 0; iChild < curOp.nChild; iChild ++){
			if(0 == iChild){
				deviceCondlike_64State_case1_first_noTranspose_codeml(pCondlike_F, 
																	tipState + curOp.child_condlike_offset[iChild], 
																	PMat + curOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray_case1, 
																	nElementPerArray);
			}
			else{
				deviceCondlike_64State_case1_notFirst_noTranspose_codeml(pCondlike_F, 
																	tipState + curOp.child_condlike_offset[iChild], 
																	PMat + curOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray_case1, 
																	nElementPerArray);
			}
		}
	}		// At least one child is condlike, use shared memory;
	else{
		CUFlt *pCondlike_S;

		const int blkOffset = bx - opStartBlkInd[opInd];
		const int tile_y = (blkOffset >> 1);
		const int tile_x = (blkOffset & 0x1);
		const int condlike_offset = (tile_y << 11);
		int iElem;

		// Use shared memory to store a block of PMat and condlike;
		__shared__ CUFlt sh_condlike[32][TILE_SIZE_CONDLIKE_CODEML];
		__shared__ CUFlt sh_PMat[TILE_SIZE_CONDLIKE_CODEML][32];

		CUFlt temp_buf[16];

		for(iChild = 0; iChild < curOp.nChild; iChild ++){
			if(1 == curOp.child_case[iChild]){			// Current child is tip state;
				if(0 == iChild){
					deviceCondlike_64State_case1_first_noTranspose_codeml(pCondlike_F, 
																	tipState + curOp.child_condlike_offset[iChild], 
																	PMat + curOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray_case2, 
																	nElementPerArray);
				}
				else{
					deviceCondlike_64State_case1_notFirst_noTranspose_codeml(pCondlike_F, 
																	tipState + curOp.child_condlike_offset[iChild], 
																	PMat + curOp.child_P_offset[iChild], 
																	ind, 
																	nThreadPerArray_case2, 
																	nElementPerArray);
				}
			}
			else{				// Current child is tip condlike / int condlike;
				for(iElem = 0; iElem < 16; iElem ++)
					temp_buf[iElem] = 0;

				if(2 == curOp.child_case[iChild])
					pCondlike_S = tipCondlike + curOp.child_condlike_offset[iChild] + condlike_offset;
				else
					pCondlike_S = intCondlike + curOp.child_condlike_offset[iChild] + condlike_offset;

				if(0 == iChild){
					deviceCondlike_64State_case2_first_noTranspose_codeml(pCondlike_F + condlike_offset, 
																			pCondlike_S, 
																			PMat + curOp.child_P_offset[iChild], 
																			sh_condlike,
																			sh_PMat,
																			temp_buf,
																			tile_x,
																			tile_y,
																			tx,
																			ty);
				}
				else{
					deviceCondlike_64State_case2_notFirst_noTranspose_codeml(pCondlike_F + condlike_offset, 
																			pCondlike_S, 
																			PMat + curOp.child_P_offset[iChild], 
																			sh_condlike,
																			sh_PMat,
																			temp_buf,
																			tile_x,
																			tile_y,
																			tx,
																			ty);
				}
			}
			__syncthreads();		// TODO: 是否可以去掉该同步指令???
		}
	}
}



void callKernelCondlike_codeml_unrooted(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nSitePattern, const int nPaddedSitePattern, const int nPaddedState, const int nState, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream)
{	
	//printf("\n=======\nGoing to call codeml's kernel for 64 state of condlike...\n==========\n");
#ifdef TRANSPOSE_PMAT
	kernelCondlike_nChild_transpose_codeml<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, nOp, condlikeOp, blkIndToOpInd, opStartBlkInd, nState);
	cutilCheckMsg("kernel kernelCondlike_nChild_transpose_codeml() failed");
#else
	int nElementPerArray, blockSize, nElemPerBlock_case1, nThreadPerArray_case1, nThreadPerArray_case2;
	
	nElementPerArray = nSitePattern * nPaddedState;
	blockSize = BLOCK_DIMENSION_X_CONDLIKE_CODEML * BLOCK_DIMENSION_Y_CONDLIKE_CODEML;
	
	nElemPerBlock_case1 = blockSize * N_ELEMENT_PER_THREAD_CONDLIKE_CASE1_CODEML;
	
	nThreadPerArray_case1 = (nElementPerArray + nElemPerBlock_case1 - 1) / nElemPerBlock_case1 * blockSize;
	nThreadPerArray_case2 = nPaddedSitePattern / 32 * 2 * blockSize;

	//printf("Going to call the non-transpose version...\n");
	kernelCondlike_nChild_noTranspose_codeml<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, nOp, condlikeOp, blkIndToOpInd, opStartBlkInd, nState, blockSize, nThreadPerArray_case1, nThreadPerArray_case2, nElementPerArray);
	cutilCheckMsg("kernel kernelCondlike_nChild_noTranspose_codeml() failed");
#endif
}

