#include "CuLibKernel-codemlAndMrBayes-rooted.h"



// PMat的计算: 
// CuCodeML的PMat计算函数，得到的为转置后的PMat;
// 只适用于nPaddedState = 64的情况:
__device__
inline void saxpy(CUFlt a, const CUFlt *b, CUFlt *c, int n)
{
/*
  Calculate c[] += a * b[], later used in calculating transition matrix.
*/
#pragma unroll
    for (int i = 0; i < n; i++) {
        c[i] += a * b[i];
    }
}


// TODO: offset的加载是否有必要用shared memory???
__global__ void kernelPMatExptRoot_codeml(CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, CUFlt *exptRootAll)
{
/*
  Kernel of calculating exptRootAll[]=exp(t*R[]);
*/
    const int inode  = blockIdx.x;

	int tx = threadIdx.x;
	offset += inode;

	int idx = (inode << 6) + tx;
    R += offset->R_offset;
	CUFlt t = brLen[offset->brLen_offset];

    exptRootAll[idx] = exp(t * R[tx]);
}


// 4个block计算一个PMat:
__global__ void kernelPMatUVRoot_codeml(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *brLen, CuLPMatOffset *offset, const CUFlt *exptRootAll)
{
/*
  Kernel of calculating transition matrix.
*/

	const int inode = blockIdx.y;
    const int  n = 64, m = N_ELEMENT_PER_THREAD_PMAT_CODEML;

	offset += inode;
    const int id = threadIdx.x;
    const int ibx = blockIdx.x * m;

	const int UV_offset = offset->UV_offset;
	P += offset->P_offset + id * n + ibx;
    U += UV_offset + ibx * n;
    V += UV_offset + id;

    double exptRoot = 0;
    exptRoot = exptRootAll[(inode << 6) + id];

    __shared__ double bs[n][m + 1];
#pragma unroll
    for (int i = 0; i < m; i++) {
        bs[id][i] = U[i * n + id] * exptRoot;
    }
    __syncthreads();

    double c[m] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double v0, v1 = V[0];
#pragma unroll
    for (int i = 0; i < n; i++) {
        v0 = v1;
        V += n;
        v1 = V[0];
        saxpy(v0, bs[i], c, m);
    }

#pragma unroll
    for (int i = 0; i < m; i++) {
        if (c[i] < 0) c[i] = 0;
        P[i] = c[i];
    }
}


void callKernelPMat_codeml(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CUFlt *exptRootAll, CuLPMatOffset *offset, const int nMatrix, const int nState, const int nPaddedState, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t stream)
{
	//printf("\n=======\nGoing to call codeml's kernel for 64 state of PMat...\n==========\n");
	// For nPaddedState == 64, codeml version is used and the result PMat matrix is transposed:
	kernelPMatExptRoot_codeml<<<nMatrix, nPaddedState, 0, stream>>>(R, brLen, offset, exptRootAll);
	cutilCheckMsg("kernel kernelPMatExptRoot() failed");

	kernelPMatUVRoot_codeml<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, brLen, offset, exptRootAll);
	cutilCheckMsg("kernel kernelPMatUVRoot() failed");
}



// Transpose the transition matrix:
void transposeMatrix(CUFlt *pMatrix, int nMatrix, int nRow, int nCol)
{
	const int matrixSize = nRow * nCol;
	for(int iMatrix = 0; iMatrix < nMatrix; iMatrix ++, pMatrix += matrixSize){
		for(int iRow = 0; iRow < nRow; iRow ++){
			for(int iCol = 0; iCol < iRow; iCol ++){
				swap(pMatrix[iRow * nCol + iCol], pMatrix[iCol * nCol + iRow]);
			}
		}
	}
}



// calculation of conditional likelihood;
// codeml's kernel for calculation of condlike:
// 注意：原本若PMat没有转置，计算时应该是condlike的一行乘以PMat的一列，由于codeml计算PMat时进行了转置，因此这里在计算时是condlike的一行乘以PMat的一列；
// TODO: 对于case 1 / case 2，使用shared memory存储PMat???

// case 1:
// version 1 of case 1: 每个thread block负责一个32 * 32的sub-matrix，其中各个thread负责4个2 * 2的sub-matrix;
__device__
void deviceCondlike_64State_case1_transpose_codeml(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int tx, int ty)
{
	int ttx = (tx << 1), tty = (ty << 1), curState_L, curState_R;
	CUFlt *pPMat_L, *pPMat_R;

	tipState_L += tty;
	tipState_R += tty;
	PMat_L += ttx;
	PMat_R += ttx;
	condlike_F += (tty << 6) + ttx;

	// The first 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState_L = __ldg(&tipState_L[0]);
	curState_R = __ldg(&tipState_R[0]);
#else
	curState_L = tipState_L[0];
	curState_R = tipState_R[0];
#endif
	pPMat_L = PMat_L + (curState_L << 6);
	pPMat_R = PMat_R + (curState_R << 6);
	
#ifdef USING_LDG
	condlike_F[0] = __ldg(&pPMat_L[0]) * __ldg(&pPMat_R[0]);
	condlike_F[1] = __ldg(&pPMat_L[1]) * __ldg(&pPMat_R[1]);
	condlike_F[16] = __ldg(&pPMat_L[16]) * __ldg(&pPMat_R[16]);
	condlike_F[17] = __ldg(&pPMat_L[17]) * __ldg(&pPMat_R[17]);
#else
	condlike_F[0] = pPMat_L[0] * pPMat_R[0];
	condlike_F[1] = pPMat_L[1] * pPMat_R[1];
	condlike_F[16] = pPMat_L[16] * pPMat_R[16];
	condlike_F[17] = pPMat_L[17] * pPMat_R[17];
#endif

	// The second 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState_L = __ldg(&tipState_L[1]);
	curState_R = __ldg(&tipState_R[1]);
#else
	curState_L = tipState_L[1];
	curState_R = tipState_R[1];
#endif
	pPMat_L = PMat_L + (curState_L << 6);
	pPMat_R = PMat_R + (curState_R << 6);
	condlike_F += 64;

#ifdef USING_LDG
	condlike_F[0] = __ldg(&pPMat_L[0]) * __ldg(&pPMat_R[0]);
	condlike_F[1] = __ldg(&pPMat_L[1]) * __ldg(&pPMat_R[1]);
	condlike_F[16] = __ldg(&pPMat_L[16]) * __ldg(&pPMat_R[16]);
	condlike_F[17] = __ldg(&pPMat_L[17]) * __ldg(&pPMat_R[17]);
#else
	condlike_F[0] = pPMat_L[0] * pPMat_R[0];
	condlike_F[1] = pPMat_L[1] * pPMat_R[1];
	condlike_F[16] = pPMat_L[16] * pPMat_R[16];
	condlike_F[17] = pPMat_L[17] * pPMat_R[17];
#endif

	// The third 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState_L = __ldg(&tipState_L[16]);
	curState_R = __ldg(&tipState_R[16]);
#else
	curState_L = tipState_L[16];
	curState_R = tipState_R[16];
#endif
	pPMat_L = PMat_L + (curState_L << 6);
	pPMat_R = PMat_R + (curState_R << 6);
	condlike_F += (15 << 6);

#ifdef USING_LDG
	condlike_F[0] = __ldg(&pPMat_L[0]) * __ldg(&pPMat_R[0]);
	condlike_F[1] = __ldg(&pPMat_L[1]) * __ldg(&pPMat_R[1]);
	condlike_F[16] = __ldg(&pPMat_L[16]) * __ldg(&pPMat_R[16]);
	condlike_F[17] = __ldg(&pPMat_L[17]) * __ldg(&pPMat_R[17]);
#else
	condlike_F[0] = pPMat_L[0] * pPMat_R[0];
	condlike_F[1] = pPMat_L[1] * pPMat_R[1];
	condlike_F[16] = pPMat_L[16] * pPMat_R[16];
	condlike_F[17] = pPMat_L[17] * pPMat_R[17];
#endif

	// The fourth 2 * 2 sub-matrix:
#ifdef USING_LDG
	curState_L = __ldg(&tipState_L[17]);
	curState_R = __ldg(&tipState_R[17]);
#else
	curState_L = tipState_L[17];
	curState_R = tipState_R[17];
#endif
	pPMat_L = PMat_L + (curState_L << 6);
	pPMat_R = PMat_R + (curState_R << 6);
	condlike_F += 64;

#ifdef USING_LDG
	condlike_F[0] = __ldg(&pPMat_L[0]) * __ldg(&pPMat_R[0]);
	condlike_F[1] = __ldg(&pPMat_L[1]) * __ldg(&pPMat_R[1]);
	condlike_F[16] = __ldg(&pPMat_L[16]) * __ldg(&pPMat_R[16]);
	condlike_F[17] = __ldg(&pPMat_L[17]) * __ldg(&pPMat_R[17]);
#else
	condlike_F[0] = pPMat_L[0] * pPMat_R[0];
	condlike_F[1] = pPMat_L[1] * pPMat_R[1];
	condlike_F[16] = pPMat_L[16] * pPMat_R[16];
	condlike_F[17] = pPMat_L[17] * pPMat_R[17];
#endif
}



// case 2: for condlike:
// Transpose version, row * col:
// version 1 of case 2: 每个thread负责4个2 * 2的sub-matrix的计算，一个thread block负责一个32 * 32的sub-matrix:
__device__
void deviceCondlike_64State_case2_transpose_codeml(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt sh_condlike[][TILE_SIZE_CONDLIKE_CODEML], CUFlt sh_PMat[][32], CUFlt *temp_buf, const int tx, const int ty)
{
	const int tty = (ty << 1);
	const int ttx = (tx << 1);

	condlike_R += (tty << 6) + tx;
	tipState_L += tty;

	PMat_L += ttx;
	PMat_R += (ty << 6) + ttx;

	int itr, i, nIteration = 64 / TILE_SIZE_CONDLIKE_CODEML;
	for(itr = 0; itr < nIteration; itr ++){
#ifdef USING_LDG
		sh_condlike[tty][tx] = __ldg(&condlike_R[0]);
		sh_condlike[tty + 1][tx] = __ldg(&condlike_R[64]);
		sh_condlike[tty + 16][tx] = __ldg(&condlike_R[16 << 6]);
		sh_condlike[tty + 17][tx] = __ldg(&condlike_R[17 << 6]);

		sh_PMat[ty][ttx] = __ldg(&PMat_R[0]);
		sh_PMat[ty][ttx + 1] = __ldg(&PMat_R[1]);
		sh_PMat[ty][ttx + 16] = __ldg(&PMat_R[16]);
		sh_PMat[ty][ttx + 17] = __ldg(&PMat_R[17]);
#else
		sh_condlike[tty][tx] = condlike_R[0];
		sh_condlike[tty + 1][tx] = condlike_R[64];
		sh_condlike[tty + 16][tx] = condlike_R[16 << 6];
		sh_condlike[tty + 17][tx] = condlike_R[17 << 6];

		sh_PMat[ty][ttx] = PMat_R[0];
		sh_PMat[ty][ttx + 1] = PMat_R[1];
		sh_PMat[ty][ttx + 16] = PMat_R[16];
		sh_PMat[ty][ttx + 17] = PMat_R[17];
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

      condlike_R += TILE_SIZE_CONDLIKE_CODEML;
      PMat_R += (TILE_SIZE_CONDLIKE_CODEML << 6);	
	}
	
	condlike_F += (tty << 6) + ttx;
  
	int curOffset, curState;
	CUFlt *pPMat_L;

#ifdef USING_LDG
	curState = __ldg(&tipState_L[0]);
	pPMat_L = PMat_L + (curState << 6);
	condlike_F[0] = temp_buf[0] * __ldg(&pPMat_L[0]);
	condlike_F[1] = temp_buf[2] * __ldg(&pPMat_L[1]);
	condlike_F[16] = temp_buf[4] * __ldg(&pPMat_L[16]);
	condlike_F[17] = temp_buf[6] * __ldg(&pPMat_L[17]);

	curState = __ldg(&tipState_L[1]);
	pPMat_L = PMat_L + (curState << 6);
	condlike_F[64] = temp_buf[3] * __ldg(&pPMat_L[0]);
	condlike_F[64 + 1] = temp_buf[1]* __ldg(&pPMat_L[1]);
	condlike_F[64 + 16] = temp_buf[7]* __ldg(&pPMat_L[16]);
	condlike_F[64 + 17] = temp_buf[5]* __ldg(&pPMat_L[17]);

	curOffset = (16 << 6);
	curState = __ldg(&tipState_L[16]);
	pPMat_L = PMat_L + (curState << 6);
	condlike_F[curOffset] = temp_buf[8] * __ldg(&pPMat_L[0]);
	condlike_F[curOffset + 1] = temp_buf[10] * __ldg(&pPMat_L[1]);
	condlike_F[curOffset + 16] = temp_buf[12] * __ldg(&pPMat_L[16]);
	condlike_F[curOffset + 17] = temp_buf[14] * __ldg(&pPMat_L[17]);

	curOffset += 64;
	curState = __ldg(&tipState_L[17]);
	pPMat_L = PMat_L + (curState << 6);
	condlike_F[curOffset] = temp_buf[11] * __ldg(&pPMat_L[0]);
	condlike_F[curOffset + 1] = temp_buf[9] * __ldg(&pPMat_L[1]);
	condlike_F[curOffset + 16] = temp_buf[15] * __ldg(&pPMat_L[16]);
	condlike_F[curOffset + 17] = temp_buf[13] * __ldg(&pPMat_L[17]);
#else
	curState = tipState_L[0];
	pPMat_L = PMat_L + (curState << 6);
	condlike_F[0] = temp_buf[0] * pPMat_L[0];
	condlike_F[1] = temp_buf[2] * pPMat_L[1];
	condlike_F[16] = temp_buf[4] * pPMat_L[16];
	condlike_F[17] = temp_buf[6] * pPMat_L[17];

	curState = tipState_L[1];
	pPMat_L = PMat_L + (curState << 6);
	condlike_F[64] = temp_buf[3] * pPMat_L[0];
	condlike_F[64 + 1] = temp_buf[1]* pPMat_L[1];
	condlike_F[64 + 16] = temp_buf[7]* pPMat_L[16];
	condlike_F[64 + 17] = temp_buf[5]* pPMat_L[17];

	curOffset = (16 << 6);
	curState = tipState_L[16];
	pPMat_L = PMat_L + (curState << 6);
	condlike_F[curOffset] = temp_buf[8] * pPMat_L[0];
	condlike_F[curOffset + 1] = temp_buf[10] * pPMat_L[1];
	condlike_F[curOffset + 16] = temp_buf[12] * pPMat_L[16];
	condlike_F[curOffset + 17] = temp_buf[14] * pPMat_L[17];

	curOffset += 64;
	curState = tipState_L[17];
	pPMat_L = PMat_L + (curState << 6);
	condlike_F[curOffset] = temp_buf[11] * pPMat_L[0];
	condlike_F[curOffset + 1] = temp_buf[9] * pPMat_L[1];
	condlike_F[curOffset + 16] = temp_buf[15] * pPMat_L[16];
	condlike_F[curOffset + 17] = temp_buf[13] * pPMat_L[17];
#endif
}



// version 1.2 of case 3: 任务分配方式以及计算模式完全同version 1.1，但寄存器使用量为version 1的一半，代价为：增加了对condlike_F的读写次数;
// 该version假设temp_buf[]的数目为16，其余同version 1;
__device__
void deviceCondlike_64State_case3_transpose_codeml(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt sh_condlike[][TILE_SIZE_CONDLIKE_CODEML], CUFlt sh_PMat[][32], CUFlt *temp_buf, const int tx, const int ty)
{
	const int tty = (ty << 1);
	const int ttx = (tx << 1);

	int itr, i, nIteration = 64 / TILE_SIZE_CONDLIKE_CODEML;

	// For the first child:
	condlike_L += (tty << 6) + tx;
	PMat_L += (ty << 6) + ttx;

	for(itr = 0; itr < nIteration; itr ++){
#ifdef USING_LDG
		sh_condlike[tty][tx] = __ldg(&condlike_L[0]);
		sh_condlike[tty + 1][tx] = __ldg(&condlike_L[64]);
		sh_condlike[tty + 16][tx] = __ldg(&condlike_L[16 << 6]);
		sh_condlike[tty + 17][tx] = __ldg(&condlike_L[17 << 6]);

		sh_PMat[ty][ttx] = __ldg(&PMat_L[0]);
		sh_PMat[ty][ttx + 1] = __ldg(&PMat_L[1]);
		sh_PMat[ty][ttx + 16] = __ldg(&PMat_L[16]);
		sh_PMat[ty][ttx + 17] = __ldg(&PMat_L[17]);
#else
		sh_condlike[tty][tx] = condlike_L[0];
		sh_condlike[tty + 1][tx] = condlike_L[64];
		sh_condlike[tty + 16][tx] = condlike_L[16 << 6];
		sh_condlike[tty + 17][tx] = condlike_L[17 << 6];

		sh_PMat[ty][ttx] = PMat_L[0];
		sh_PMat[ty][ttx + 1] = PMat_L[1];
		sh_PMat[ty][ttx + 16] = PMat_L[16];
		sh_PMat[ty][ttx + 17] = PMat_L[17];
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

      condlike_L += TILE_SIZE_CONDLIKE_CODEML;
      PMat_L += (TILE_SIZE_CONDLIKE_CODEML << 6);	
	}
	
	condlike_F += (tty << 6) + ttx;
  
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


	// For the second child:
	condlike_R += (tty << 6) + tx;
	PMat_R += (ty << 6) + ttx;

	for(itr = 0; itr < 16; itr ++)
		temp_buf[itr] = 0;

	for(itr = 0; itr < nIteration; itr ++){
#ifdef USING_LDG
		sh_condlike[tty][tx] = __ldg(&condlike_R[0]);
		sh_condlike[tty + 1][tx] = __ldg(&condlike_R[64]);
		sh_condlike[tty + 16][tx] = __ldg(&condlike_R[16 << 6]);
		sh_condlike[tty + 17][tx] = __ldg(&condlike_R[17 << 6]);

		sh_PMat[ty][ttx] = __ldg(&PMat_R[0]);
		sh_PMat[ty][ttx + 1] = __ldg(&PMat_R[1]);
		sh_PMat[ty][ttx + 16] = __ldg(&PMat_R[16]);
		sh_PMat[ty][ttx + 17] = __ldg(&PMat_R[17]);
#else
		sh_condlike[tty][tx] = condlike_R[0];
		sh_condlike[tty + 1][tx] = condlike_R[64];
		sh_condlike[tty + 16][tx] = condlike_R[16 << 6];
		sh_condlike[tty + 17][tx] = condlike_R[17 << 6];

		sh_PMat[ty][ttx] = PMat_R[0];
		sh_PMat[ty][ttx + 1] = PMat_R[1];
		sh_PMat[ty][ttx + 16] = PMat_R[16];
		sh_PMat[ty][ttx + 17] = PMat_R[17];
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

      condlike_R += TILE_SIZE_CONDLIKE_CODEML;
      PMat_R += (TILE_SIZE_CONDLIKE_CODEML << 6);	
	}

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



// Non-transpose version:
__device__
void deviceCondlike_64State_case1_noTranspose_codeml(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThreadPerArray, int nElementPerArray)
{
	int curPattern, curState, itr;

	for(itr = ind; itr < nElementPerArray; itr += nThreadPerArray){
		curState = (itr & 0x3f);
		curPattern = (itr >> 6);
		curState <<= 6;

		// row * row
#ifdef USING_LDG
		condlike_F[itr] = __ldg(&PMat_L[curState + __ldg(&tipState_L[curPattern])]) * __ldg(&PMat_R[curState + __ldg(&tipState_R[curPattern])]);
#else
		condlike_F[itr] = PMat_L[curState + tipState_L[curPattern]] * PMat_R[curState + tipState_R[curPattern]];
#endif
	}
}

__device__
void deviceCondlike_64State_case2_noTranspose_codeml(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThreadPerArray, int nElementPerArray, int nState)
{
	int  curPattern, curState, itr, iState;
	CUFlt sum_L, sum_R, *pCondlike_R, *pPMat_R;

	for(itr = ind; itr < nElementPerArray; itr += nThreadPerArray){
		curPattern = (itr >> 6);
		curState = (itr & 0x3f);

		// row * row
		pPMat_R = PMat_R + (curState << 6);
		pCondlike_R = condlike_R + itr - curState;
#ifdef USING_LDG
		sum_L = __ldg(&PMat_L[(curState << 6) + __ldg(&tipState_L[curPattern])]);
#else
		sum_L = PMat_L[(curState << 6) + tipState_L[curPattern]];
#endif
		sum_R = 0.0f;

		for(iState = 0; iState < nState; iState ++){
#ifdef USING_LDG
			sum_R += __ldg(&pPMat_R[iState]) * __ldg(&pCondlike_R[iState]);
#else
			sum_R += pPMat_R[iState] * pCondlike_R[iState];
#endif
		}

		condlike_F[itr] = sum_L * sum_R;
	}
}


// case 3 of no-transpose version:
__device__
void deviceCondlike_64State_case3_first_noTranspose_codeml(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, CUFlt sh_condlike[][TILE_SIZE_CONDLIKE_CODEML], CUFlt sh_PMat[][32], CUFlt *temp_buf, const int bx, const int by, const int tx, const int ty)
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

// Not first child of non-transpose version of case 3:
__device__
void deviceCondlike_64State_case3_notFirst_noTranspose_codeml(CUFlt *condlike_F, CUFlt *condlike_S, CUFlt *PMat_S, CUFlt sh_condlike[][TILE_SIZE_CONDLIKE_CODEML], CUFlt sh_PMat[][32], CUFlt *temp_buf, const int bx, const int by, const int tx, const int ty)
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




// TODO: 目前一个block大小为64, 导致block数目过多，不利于cuda stream之间的并行，考虑将block尺寸扩大???
// TODO: condlike operation可以加载到shared memory中;
// version 1.2: 该version与1.1的唯一区别在于：该version使用的temp_buf[]的寄存器量为16;
// 其中，case 1可以调用的version包括: 1 / 2 / 3，case 2可以调用的version包括: 1，case 3可以调用的version为: 1.2;
__global__ void kernelCondlike_transpose_codeml(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState)
{
	const int bx = blockIdx.x;
	const int ty = threadIdx.y;
    const int tx = threadIdx.x;

	//const int thdInd = ty * blockDim.x + tx;
	const int opInd = blkIndToOpInd[bx];
	CuLCondlikeOp curOp = condlikeOp[opInd];

	int blkOffset = bx - opStartBlkInd[opInd];
	int tile_y = (blkOffset >> 1);	
	int tile_x = (blkOffset & 0x1);	

	int tipState_offset = (tile_y << 5), condlike_offset = (tipState_offset << 6);
	int tileOffset = (tile_x << 5);

	__shared__ CUFlt sh_condlike[32][TILE_SIZE_CONDLIKE_CODEML];
	__shared__ CUFlt sh_PMat[TILE_SIZE_CONDLIKE_CODEML][32];
	
	CUFlt temp_buf[16] = {0};

	if(1 == curOp.whichCase){
		
		deviceCondlike_64State_case1_transpose_codeml(intCondlike + curOp.father_condlike_offset + condlike_offset + tileOffset,
											tipState + curOp.child_condlike_offset[0] + tipState_offset, 
											tipState + curOp.child_condlike_offset[1] + tipState_offset,
											PMat + curOp.child_P_offset[0] + tileOffset, 
											PMat + curOp.child_P_offset[1] + tileOffset,
											tx,
											ty);
	}
	else if(2 == curOp.whichCase){
		
		CUFlt *pCondlike_R;
		if(1 == curOp.isChildTip[1])
			pCondlike_R = tipCondlike + curOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + curOp.child_condlike_offset[1];

		deviceCondlike_64State_case2_transpose_codeml(intCondlike + curOp.father_condlike_offset + condlike_offset + tileOffset,
											tipState + curOp.child_condlike_offset[0] + tipState_offset, 
											pCondlike_R + condlike_offset,
											PMat + curOp.child_P_offset[0] + tileOffset, 
											PMat + curOp.child_P_offset[1] + tileOffset,
											sh_condlike,
											sh_PMat,
											temp_buf,
											tx,
											ty);
	}
	else if(3 == curOp.whichCase){

		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == curOp.isChildTip[0])
			pCondlike_L = tipCondlike + curOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + curOp.child_condlike_offset[0];

		if(1 == curOp.isChildTip[1])
			pCondlike_R = tipCondlike + curOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + curOp.child_condlike_offset[1];
		
		deviceCondlike_64State_case3_transpose_codeml(intCondlike + curOp.father_condlike_offset + condlike_offset + tileOffset,
											pCondlike_L + condlike_offset,
											pCondlike_R + condlike_offset,
											PMat + curOp.child_P_offset[0] + tileOffset, 
											PMat + curOp.child_P_offset[1] + tileOffset,
											sh_condlike,
											sh_PMat,
											temp_buf,
											tx,
											ty);
	}
	else{
		// Error case;
		return;
	}
}


// Non-transpose version:
__global__ void kernelCondlike_noTranspose_codeml(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState, const int nThreadPerBlock, const int nThreadPerArray_case1, const int nThreadPerArray_case2, const int nElementPerArray)
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


	if(1 == curOp.whichCase){
		int ind = (bx - opStartBlkInd[opInd]) * nThreadPerBlock + thdInd;
		deviceCondlike_64State_case1_noTranspose_codeml(intCondlike + curOp.father_condlike_offset,
											tipState + curOp.child_condlike_offset[0], 
											tipState + curOp.child_condlike_offset[1],
											PMat + curOp.child_P_offset[0], 
											PMat + curOp.child_P_offset[1],
											ind, 
											nThreadPerArray_case1, 
											nElementPerArray);
	}
	else if(2 == curOp.whichCase){
		int ind = (bx - opStartBlkInd[opInd]) * nThreadPerBlock + thdInd;

		CUFlt *pCondlike_R;
		if(1 == curOp.isChildTip[1])
			pCondlike_R = tipCondlike + curOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + curOp.child_condlike_offset[1];

		deviceCondlike_64State_case2_noTranspose_codeml(intCondlike + curOp.father_condlike_offset,
											tipState + curOp.child_condlike_offset[0], 
											pCondlike_R,
											PMat + curOp.child_P_offset[0], 
											PMat + curOp.child_P_offset[1],
											ind, 
											nThreadPerArray_case2, 
											nElementPerArray,
											nState);
	}
	else if(3 == curOp.whichCase){
		const int blkOffset = bx - opStartBlkInd[opInd];
		const int tile_x = (blkOffset & 0x3);
		const int tile_y = (blkOffset >> 1);

		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == curOp.isChildTip[0])
			pCondlike_L = tipCondlike + curOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + curOp.child_condlike_offset[0];

		if(1 == curOp.isChildTip[1])
			pCondlike_R = tipCondlike + curOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + curOp.child_condlike_offset[1];

		const int condlike_offset = (tile_y << 11);

		// Use shared memory to store a block of PMat and condlike;
		__shared__ CUFlt sh_condlike[32][TILE_SIZE_CONDLIKE_CODEML];
		__shared__ CUFlt sh_PMat[TILE_SIZE_CONDLIKE_CODEML][32];

		CUFlt temp_buf[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


		// For the first child:
		deviceCondlike_64State_case3_first_noTranspose_codeml(intCondlike + curOp.father_condlike_offset + condlike_offset, 
										pCondlike_L + condlike_offset, 
										PMat + curOp.child_P_offset[0], 
										sh_condlike,
										sh_PMat,
										temp_buf,
										tile_x,
										tile_y,
										tx,
										ty);

		// For the second child:
		for(int i = 0; i < 16; i ++)
			temp_buf[i] = 0;

		deviceCondlike_64State_case3_notFirst_noTranspose_codeml(intCondlike + curOp.father_condlike_offset + condlike_offset, 
										pCondlike_R + condlike_offset,  
										PMat + curOp.child_P_offset[1],
										sh_condlike,
										sh_PMat,
										temp_buf,
										tile_x,
										tile_y,
										tx,
										ty);
	}
	else{
		// Error case;
		return;
	}
}



void callKernelCondlike_codeml(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nSitePattern, const int nPaddedState, const int nState, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream)
{
	printf("\n=======\nGoing to call codeml's kernel for 64 state of condlike...\n==========\n");
#ifdef TRANSPOSE_PMAT
	kernelCondlike_transpose_codeml<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, nOp, condlikeOp, blkIndToOpInd, opStartBlkInd, nState);
	cutilCheckMsg("kernel kernelCondlike_transpose_codeml() failed");
#else
	int nElementPerArray, blockSize, nElemPerBlock_case1, nElemPerBlock_case2, nThreadPerArray_case1, nThreadPerArray_case2;
	
	nElementPerArray = nSitePattern * nPaddedState;
	blockSize = BLOCK_DIMENSION_X_CONDLIKE_CODEML * BLOCK_DIMENSION_Y_CONDLIKE_CODEML;
	
	nElemPerBlock_case1 = blockSize * N_ELEMENT_PER_THREAD_CONDLIKE_CASE1_CODEML;
	nElemPerBlock_case2 = blockSize * N_ELEMENT_PER_THREAD_CONDLIKE_CASE2_CODEML;

	nThreadPerArray_case1 = (nElementPerArray + nElemPerBlock_case1 - 1) / nElemPerBlock_case1 * blockSize;
	nThreadPerArray_case2 = (nElementPerArray + nElemPerBlock_case2 - 1) / nElemPerBlock_case2 * blockSize;

	kernelCondlike_noTranspose_codeml<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, nOp, condlikeOp, blkIndToOpInd, opStartBlkInd, nState, blockSize, nThreadPerArray_case1, nThreadPerArray_case2, nElementPerArray);
	cutilCheckMsg("kernel kernelCondlike_noTranspose_codeml() failed");
#endif
}



// For the calculation of site likelihood:
// CuCodeML-version:
__global__
void kernelSiteLnL_codeml(CUFlt *siteLnL, CUFlt *rootCondlike, CUFlt *catWeight, CUFlt *stateFreq, int nNodeScaler, CUFlt *scaleFactor, const int nCategory, const int nSitePattern, const int nPaddedSitePattern, const int nState, const int nPaddedState)
{
/*
  Kernel of intergrating the conditional probability of the root.
*/
	const int tx = threadIdx.x;
    const int ind = blockIdx.x * blockDim.x + tx;
	
	extern __shared__ CUFlt sh_catWeightAndStateFreq[];

	if(tx < nCategory)
		sh_catWeightAndStateFreq[tx] = catWeight[tx];
	else if(tx >= 32 && tx < 32 + nState)
		sh_catWeightAndStateFreq[nCategory + tx - 32] = stateFreq[tx - 32];

	__syncthreads();

    if (ind >= nSitePattern) return ;

    CUFlt *pRootCondlike = rootCondlike + ind * nPaddedState;
	CUFlt *pStateFreq = sh_catWeightAndStateFreq + nCategory;
    CUFlt sum = 0.0f, sumState;
    int i, j;
	const int offset = nPaddedSitePattern * nPaddedState;

	for(i = 0; i < nCategory; i ++){
		sumState = 0.0f;

		for(j = 0; j < nState; j ++){
			sumState += pStateFreq[j] * pRootCondlike[j];
		}

		sum += sumState * sh_catWeightAndStateFreq[i];
		pRootCondlike += offset;
	}
    
	/*
    if (fh <= 0) {
        fh = 1e-100;
    }
	*/
	if(sum <= 0)
		sum = log(CUFLT_MIN);
	else
		sum = log(sum);
	
	int iNode;
	for(iNode = 0; iNode < nNodeScaler; iNode ++)
		sum += scaleFactor[ind + iNode * nPaddedSitePattern];

    siteLnL[ind] = sum;
}

__global__
void kernelReductionOfSiteLnL_codeml(CUFlt *reduceLnL, const CUFlt *siteLnL, const CUFlt *sitePatternWeight, const int nSitePattern, const int nThread){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int ind = bx * blockDim.x + tx;
	
	CUFlt sumLnL = 0.0f;
	
	for(; ind < nSitePattern; ind += nThread)
		sumLnL += siteLnL[ind] * sitePatternWeight[ind];

	__shared__ CUFlt sh_lnL[N_THREAD_PER_BLOCK_REDUCTION_LNL_CODEML];
	sh_lnL[tx] = sumLnL;

	__syncthreads();

	int reduceSize = (N_THREAD_PER_BLOCK_REDUCTION_LNL_CODEML >> 1);
	for( ; reduceSize > 0; (reduceSize >>= 1)){
		if(tx < reduceSize)
			sh_lnL[tx] += sh_lnL[tx + reduceSize];
		__syncthreads();
	}

	if(tx == 0)
		reduceLnL[bx] = sh_lnL[0];
}

int callKernelLikelihood_codeml(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *rootCondlike, CUFlt *rateCatWeight, CUFlt *stateFreq, CUFlt *sitePatternWeight, int nNodeScaler, CUFlt *scaleFactor, const int nPaddedState, const int nState, const int nEigenDecomp, const int nRateCategory, const int nSitePattern, const int nPaddedSitePattern, dim3 nBlockPerGrid_siteLnL, dim3 nThreadPerBlock_siteLnL, int nBlockPerGrid_reduce, int nThreadPerBlock_reduce, cudaStream_t &stream)
{
	const int nCategory = nEigenDecomp * nRateCategory;
	
	// Calculation of site likelihood values:
	kernelSiteLnL_codeml<<<nBlockPerGrid_siteLnL, nThreadPerBlock_siteLnL, (nCategory + nState) * sizeof(CUFlt), stream>>>(siteLnL, rootCondlike, rateCatWeight, stateFreq, nNodeScaler, scaleFactor, nCategory, nSitePattern, nPaddedSitePattern, nState, nPaddedState);
	cutilCheckMsg("kernel kernelSiteLnL_codeml() failed");


	// Reduction of site likelihood values:
	const int nThread = nBlockPerGrid_reduce * nThreadPerBlock_reduce;

	kernelReductionOfSiteLnL_codeml<<<nBlockPerGrid_reduce, nThreadPerBlock_reduce, 0, stream>>>(reduceLnL, siteLnL, sitePatternWeight, nSitePattern, nThread);
	cutilCheckMsg("cudaKernel kernelReductionOfSiteLnL_codeml failed");

	return nBlockPerGrid_reduce;
}


// Calculate likelihood from site likelihood values:
int callKernelLikelihoodFromSiteLnL_codeml(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *sitePatternWeight, const int nSitePattern, int nBlockPerGrid, int nThreadPerBlock, cudaStream_t &stream)
{
	const int nThread = nBlockPerGrid * nThreadPerBlock;

	kernelReductionOfSiteLnL_codeml<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(reduceLnL, siteLnL, sitePatternWeight, nSitePattern, nThread);
	cutilCheckMsg("cudaKernel kernelReductionOfSiteLnL_codeml failed");

	return nBlockPerGrid;
}

