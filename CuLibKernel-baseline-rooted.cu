#include "CuLibKernel-baseline-rooted.h"

//#include <queue>
#include <cuda_runtime.h>
#include <sys/time.h>

using namespace std;


// Calculation of transition matrices:
// ����nState = 4��Ŀǰ����task�ķ��䷽��Ϊÿ��threadһ��element��֮����Կ����������䷽����������task������̬�������䷽��������task���϶࣬��ÿ��thread��������task�����磺4��element��
// ���⣬����block dimensionΪ��(16, k)��Ҳ��threadIdx.y��ͬ��16��thread����һ��PMat�ļ��㣬ÿ��block����k��PMat�ļ��㣬k���ݺ궨��õ������Ե�����
// ���⣬offset����ÿ��PMat��Ӧһ����������ÿ��thread block��Ӧһ������Ҫע��;
// ���⣬ע������ͬһ��block������PMat��ͬ���Ƿ������⣻
__global__
void kernelPMat_4State_noTranspose_baseline(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;
	
	if(curMatrix < nMatrix){
		
		CuLPMatOffset reg_offset = offset[curMatrix];

		CUFlt curBrLen = brLen[reg_offset.brLen_offset];
		int row = tx / 4, col = tx % 4;

		CUFlt *pU, *pV, *pR, *pP;
		
		pU = U + reg_offset.UV_offset +  row * 4;
		pV = V + reg_offset.UV_offset + col;
		pR = R + reg_offset.R_offset;
		pP = P + reg_offset.P_offset + tx;
		

		CUFlt curSum = pU[0] * exp(pR[0] * curBrLen) * pV[0] +  pU[1] * exp(pR[1] * curBrLen) * pV[4] + pU[2] * exp(pR[2] * curBrLen) * pV[8] + pU[3] * exp(pR[3] * curBrLen) * pV[12];

		pP[0] = curSum;
	}
}


// �����������汾(�ֶ�չ���ڲ��ѭ���Լ�#pragma unroll)�ıȽϽ��Ϊ����չ��ѭ��(Ҳ���ӱ���ָ��)Ч�����(�������������ֻ��һ����);
// ���⣬�벻��shared memory�İ汾�Ƚ�: ʹ��shared memory��Ч�����ã�Լ�����5%��10%;
__global__
void kernelPMat_4State_transpose_baseline(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;
	//int ind = ty * blockDim.x + tx;

	CuLPMatOffset reg_offset;

	__shared__ CUFlt sh_U[BLOCK_DIMENSION_Y_PMAT_4STATE_BASELINE][16];
	__shared__ CUFlt sh_V[BLOCK_DIMENSION_Y_PMAT_4STATE_BASELINE][16];
	__shared__ CUFlt sh_R[BLOCK_DIMENSION_Y_PMAT_4STATE_BASELINE][4];

	CUFlt *pU, *pV, *pR, *pP;

	if(curMatrix < nMatrix){
		reg_offset = offset[curMatrix];

		pU = U + reg_offset.UV_offset;
		pV = V + reg_offset.UV_offset;
		pR = R + reg_offset.R_offset;

		sh_U[ty][tx] = pU[tx];
		sh_V[ty][tx] = pV[tx];
		if(tx < 4)
			sh_R[ty][tx] = pR[tx];
	}
	__syncthreads();

	if(curMatrix < nMatrix){
		int row = tx >> 2, col = tx & 0x3, iState;
		CUFlt curBrLen = brLen[reg_offset.brLen_offset], curSum = 0.0f;
		
		pU = sh_U[ty] +  (row << 2);
		pV = sh_V[ty] + col;
		pR = sh_R[ty];
		pP = P + reg_offset.P_offset + (col << 2) + row;

		for(iState = 0; iState < 4; iState ++)
			curSum += pU[iState] * exp(pR[iState] * curBrLen) * pV[(iState<<2)];

		pP[0] = curSum;
	}
}



// For nPaddedState == 20:
// ����nState = 20��һ��thread����k��element(kĿǰΪ5)��Ҳ��80��thread����һ��PMat�����ڵ����⣺80��Ϊ32����������
// Ŀǰ���õ�block dimensionΪ��(80, m), mѡ2/4/6��Щż���Ų����˷�warp��Ŀǰѡ4��Ҳ��һ��block����4��PMat��
// ����һ��block��Ӧm��PMat����ˣ���Ҫ������m��PMat��Ӧ��offset��shared memory�У�
// Non-transpose version:
__global__
void kernelPMat_20State_noTranspose_baseline(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;

	if(curMatrix < nMatrix){
		
		CuLPMatOffset reg_offset = offset[curMatrix];
		CUFlt curBrLen = brLen[reg_offset.brLen_offset];
		
		const int blockDim_x = blockDim.x;
		
		CUFlt *pU, *pV, *pR, *pP;
		
		pU = U + reg_offset.UV_offset;
		pV = V + reg_offset.UV_offset;
		pR = R + reg_offset.R_offset;
		pP = P + reg_offset.P_offset;
		

		int curElem, row, col;
		for(curElem = tx; curElem < 400; curElem += blockDim_x){
			row = curElem / 20;
			col = curElem % 20;

			CUFlt curSum = 0.0f;
			for(int iState = 0; iState < 20; iState ++){
				curSum += pU[row * 20 + iState] * exp(pR[iState] * curBrLen) * pV[col + iState * 20];
			}

			pP[curElem] = curSum;
		}
	}
}


// Transpose version of nPaddedState = 20:
// ��U, V, R��������shared memory��;
// ���Թ���ʹ��shared memory����ֻ��U/V֮һ����shared memory�У�Ч��������ֿ��ҽ�U, V, R������shared memory�У�������Գ��Ե����ֿ�Ĵ�С;
// �ð汾��Ч����õ�;
__global__
void kernelPMat_20State_transpose_baseline(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;
	int blockDim_x = blockDim.x;

	CuLPMatOffset reg_offset;
	CUFlt *pU, *pV, *pR, *pP, curBrLen;
	CUFlt buf[(400 + BLOCK_DIMENSION_X_PMAT_20STATE_BASELINE - 1) / BLOCK_DIMENSION_X_PMAT_20STATE_BASELINE] = {0, 0};

	__shared__ CUFlt sh_U[BLOCK_DIMENSION_Y_PMAT_20STATE_BASELINE][20][TILE_SIZE_20STATE_PMAT_BASELINE];
	__shared__ CUFlt sh_V[BLOCK_DIMENSION_Y_PMAT_20STATE_BASELINE][TILE_SIZE_20STATE_PMAT_BASELINE][20];
	__shared__ CUFlt sh_R[BLOCK_DIMENSION_Y_PMAT_20STATE_BASELINE][TILE_SIZE_20STATE_PMAT_BASELINE];

	if(curMatrix < nMatrix){
		reg_offset = offset[curMatrix];

		curBrLen = brLen[reg_offset.brLen_offset];

		pU = U + reg_offset.UV_offset;
		pV = V + reg_offset.UV_offset;
		pR = R + reg_offset.R_offset;
		//pP = P + reg_offset.P_offset;
	}

	int itr, ind, row, col, iElem, k, nIteration = 20 / TILE_SIZE_20STATE_PMAT_BASELINE;
	int nElemToLoad = 20 * TILE_SIZE_20STATE_PMAT_BASELINE;

	for(itr = 0; itr < nIteration; itr ++){
		if(curMatrix < nMatrix){
			for(ind = tx; ind < nElemToLoad; ind += blockDim_x){
				row = ind / TILE_SIZE_20STATE_PMAT_BASELINE;
				col = ind % TILE_SIZE_20STATE_PMAT_BASELINE;
				sh_V[ty][ind / 20][ind % 20] = pV[ind];
				sh_U[ty][row][col]  = pU[row * 20 + col];
			}
			if(tx < TILE_SIZE_20STATE_PMAT_BASELINE)
				sh_R[ty][tx] = pR[tx];

			pU += TILE_SIZE_20STATE_PMAT_BASELINE;
			pV += nElemToLoad;
			pR += TILE_SIZE_20STATE_PMAT_BASELINE;
		}
		__syncthreads();

		if(curMatrix < nMatrix){
			for(ind = tx, iElem = 0; ind < 400; ind += blockDim_x, iElem ++){
				row = ind / 20;
				col = ind % 20;

				for(k = 0; k < TILE_SIZE_20STATE_PMAT_BASELINE; k ++)
					buf[iElem] += sh_U[ty][row][k] * exp(sh_R[ty][k] * curBrLen) * sh_V[ty][k][col];
			}
		}
		__syncthreads();
	}
	
	if(curMatrix < nMatrix){
		pP = P + reg_offset.P_offset;
		for(ind = tx, iElem = 0; ind < 400; ind += blockDim_x, iElem ++){
			row = ind / 20;
			col = ind % 20;

			pP[col * 20 + row] = buf[iElem];
		}
	}
}


// For nPaddedState != 4 / 20 / 64:
// ����nState != 4 / 20 / 61�����������blockIdx.y��ͬ��thread����һ��matrix����nState < 16ʱ��block dimensionΪ: (4, 16); ��nState < 32ʱ��block dimensionΪ:(16, 8)����nState >= 32ʱ��block dimensionΪ: (64, 4);
// Non-transpose version:
__global__
void kernelPMat_xState_noTranspose_baseline(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix, const int nState, const int nPaddedState)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;

	if(curMatrix < nMatrix){
		
		CuLPMatOffset reg_offset = offset[curMatrix];
		CUFlt curBrLen = brLen[reg_offset.brLen_offset];
		
		const int blockDim_x = blockDim.x;
		
		CUFlt *pU, *pV, *pR, *pP;
		
		pU = U + reg_offset.UV_offset;
		pV = V + reg_offset.UV_offset;
		pR = R + reg_offset.R_offset;
		pP = P + reg_offset.P_offset;
		

		int nElement = nPaddedState * nPaddedState, curElem, row, col;
		for(curElem = tx; curElem < nElement; curElem += blockDim_x){
			row = curElem / nPaddedState;
			col = curElem % nPaddedState;

			CUFlt curSum = 0.0f;
			for(int iState = 0; iState < nState; iState ++){
				curSum += pU[row * nPaddedState + iState] * exp(pR[iState] * curBrLen) * pV[col + iState * nPaddedState];
			}

			pP[curElem] = curSum;
		}
	}
}

// Transpose version of nPaddedState != 4 / 20 / 64, pad the state to a multiply of 8:
// For padded version of state = X:
// For nPaddedState == 8, use shared memory to store the entire U/V/R matrix, the total shared memory needed for each thread block is: (8 * 8 * 2 + 8) * BLOCK_DIMENSION_Y_PMAT_XSTATE_8_BASELINE;
__global__
void kernelPMat_xState_transpose_baseline_8State(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix, const int nState)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;
	
	CuLPMatOffset reg_offset;
	CUFlt *pU, *pV, *pR, *pP, curBrLen;
	
	__shared__ CUFlt sh_U[BLOCK_DIMENSION_Y_PMAT_XSTATE_8_BASELINE][8][8];
	__shared__ CUFlt sh_V[BLOCK_DIMENSION_Y_PMAT_XSTATE_8_BASELINE][8][8];
	__shared__ CUFlt sh_R[BLOCK_DIMENSION_Y_PMAT_XSTATE_8_BASELINE][8];

	if(curMatrix < nMatrix){
		reg_offset = offset[curMatrix];

		curBrLen = brLen[reg_offset.brLen_offset];

		pU = U + reg_offset.UV_offset;
		pV = V + reg_offset.UV_offset;
		pR = R + reg_offset.R_offset;
		pP = P + reg_offset.P_offset;
	}

	int ind, row, col, iElem;
	CUFlt curSum;
		
	// Transpose U/V matrix when loading into shared memory:
	if(curMatrix < nMatrix){
		for(ind = tx; ind < 64; ind += BLOCK_DIMENSION_X_PMAT_XSTATE_8_BASELINE){
			row = (ind & 0x7);
			col = (ind >> 3);
			sh_U[ty][row][col] = pU[ind];
			sh_V[ty][row][col] = pV[ind];
		}
		
		if(tx < 8)
			sh_R[ty][tx] = pR[tx];
	}

	__syncthreads();

	if(curMatrix < nMatrix){
		for(ind = tx; ind < 64; ind += BLOCK_DIMENSION_X_PMAT_XSTATE_8_BASELINE){
			row = (ind >> 3);
			col = (ind & 0x7);

			curSum = 0.0f;
			for(iElem = 0; iElem < nState; iElem ++)
				curSum += sh_V[ty][row][iElem] * exp(sh_R[ty][iElem] * curBrLen) * sh_U[ty][iElem][col];

			pP[ind] = curSum;
		}
	}
}


// For nPaddedState == 16, use shared memory to store a tile of the U/V/R matrix;
// The total size of shared memory needed is: (16 * TILE_SIZE_PMAT_XSTATE_BASELINE * 2 + TILE_SIZE_PMAT_XSTATE_BASELINE) * BLOCK_DIMENSION_Y_PMAT_XSTATE_16_BASELINE = (16 * 4 * 2 + 4) * 4 = 528;
__global__
void kernelPMat_xState_transpose_baseline_16State(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix, const int nState)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;
	int blockDim_x = blockDim.x;

	CuLPMatOffset reg_offset;
	CUFlt *pU, *pV, *pR, *pP, curBrLen;
	CUFlt buf[256 / BLOCK_DIMENSION_X_PMAT_XSTATE_16_BASELINE] = {0, 0};

	__shared__ CUFlt sh_U[BLOCK_DIMENSION_Y_PMAT_XSTATE_16_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE][16];
	__shared__ CUFlt sh_V[BLOCK_DIMENSION_Y_PMAT_XSTATE_16_BASELINE][16][TILE_SIZE_PMAT_XSTATE_BASELINE];
	__shared__ CUFlt sh_R[BLOCK_DIMENSION_Y_PMAT_XSTATE_16_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE];

	if(curMatrix < nMatrix){
		reg_offset = offset[curMatrix];

		curBrLen = brLen[reg_offset.brLen_offset];

		pU = U + reg_offset.UV_offset;
		pV = V + reg_offset.UV_offset;
		pR = R + reg_offset.R_offset;
	}

	int itr, ind, row, col, iElem, k, nIteration = (nState + TILE_SIZE_PMAT_XSTATE_BASELINE - 1) / TILE_SIZE_PMAT_XSTATE_BASELINE;
	const int nElement = (TILE_SIZE_PMAT_XSTATE_BASELINE << 4);	

	for(itr = 0; itr < nIteration; itr ++){
		if(curMatrix < nMatrix){
			for(ind = tx; ind < nElement; ind += blockDim_x){
				row = ind / TILE_SIZE_PMAT_XSTATE_BASELINE;
				col = ind % TILE_SIZE_PMAT_XSTATE_BASELINE;
				sh_V[ty][(ind & 0xf)][(ind >> 4)] = pV[ind];
				sh_U[ty][col][row]  = pU[(row << 4) + col];
			}
			if(tx < TILE_SIZE_PMAT_XSTATE_BASELINE)
				sh_R[ty][tx] = pR[tx];

			pU += TILE_SIZE_PMAT_XSTATE_BASELINE;
			pV += nElement;
			pR += TILE_SIZE_PMAT_XSTATE_BASELINE;
		}

		__syncthreads();

		if(curMatrix < nMatrix){
			for(ind = tx, iElem = 0; ind < 256; ind += blockDim_x, iElem ++){
				row = (ind >> 4);
				col = (ind & 0xf);

				for(k = 0; k < TILE_SIZE_PMAT_XSTATE_BASELINE; k ++)
					buf[iElem] += sh_V[ty][row][k] * exp(sh_R[ty][k] * curBrLen) * sh_U[ty][k][col];
			}
		}

		__syncthreads();
	}
	
	// Write the results to global memory:
	if(curMatrix < nMatrix){
		pP = P + reg_offset.P_offset;

		for(ind = tx, iElem = 0; ind < 256; ind += blockDim_x, iElem ++){
			pP[ind] = buf[iElem];
		}
	}
}



// For nPaddedState == 24, use shared memory to store a tile of the U/V/R matrix;
// The total size of shared memory needed is: (24 * TILE_SIZE_PMAT_XSTATE_BASELINE * 2 + TILE_SIZE_PMAT_XSTATE_BASELINE) * BLOCK_DIMENSION_Y_PMAT_XSTATE_24_BASELINE = (24 * 4 * 2 + 4) * 4 = 784;
__global__
void kernelPMat_xState_transpose_baseline_24State(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix, const int nState)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;
	int blockDim_x = blockDim.x;

	CuLPMatOffset reg_offset;
	CUFlt *pU, *pV, *pR, *pP, curBrLen;
	CUFlt buf[576 / BLOCK_DIMENSION_X_PMAT_XSTATE_24_BASELINE] = {0, 0};

	__shared__ CUFlt sh_U[BLOCK_DIMENSION_Y_PMAT_XSTATE_24_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE][24];
	__shared__ CUFlt sh_V[BLOCK_DIMENSION_Y_PMAT_XSTATE_24_BASELINE][24][TILE_SIZE_PMAT_XSTATE_BASELINE];
	__shared__ CUFlt sh_R[BLOCK_DIMENSION_Y_PMAT_XSTATE_24_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE];

	if(curMatrix < nMatrix){
		reg_offset = offset[curMatrix];

		curBrLen = brLen[reg_offset.brLen_offset];

		pU = U + reg_offset.UV_offset;
		pV = V + reg_offset.UV_offset;
		pR = R + reg_offset.R_offset;
	}

	int itr, ind, row, col, iElem, k, nIteration = (nState + TILE_SIZE_PMAT_XSTATE_BASELINE - 1) / TILE_SIZE_PMAT_XSTATE_BASELINE;
	const int nElement = 24 * TILE_SIZE_PMAT_XSTATE_BASELINE;

	for(itr = 0; itr < nIteration; itr ++){
		if(curMatrix < nMatrix){
			for(ind = tx; ind < nElement; ind += blockDim_x){
				row = ind / TILE_SIZE_PMAT_XSTATE_BASELINE;
				col = ind % TILE_SIZE_PMAT_XSTATE_BASELINE;
				sh_V[ty][ind % 24][ind / 24] = pV[ind];
				sh_U[ty][col][row]  = pU[row * 24 + col];
			}
			if(tx < TILE_SIZE_PMAT_XSTATE_BASELINE)
				sh_R[ty][tx] = pR[tx];

			pU += TILE_SIZE_PMAT_XSTATE_BASELINE;
			pV += nElement;
			pR += TILE_SIZE_PMAT_XSTATE_BASELINE;
		}

		__syncthreads();

		if(curMatrix < nMatrix){
			for(ind = tx, iElem = 0; ind < 576; ind += blockDim_x, iElem ++){
				row = ind / 24;
				col = ind % 24;

				for(k = 0; k < TILE_SIZE_PMAT_XSTATE_BASELINE; k ++)
					buf[iElem] += sh_V[ty][row][k] * exp(sh_R[ty][k] * curBrLen) * sh_U[ty][k][col];
			}
		}

		__syncthreads();
	}
	
	// Write the results to global memory:
	if(curMatrix < nMatrix){
		pP = P + reg_offset.P_offset;

		for(ind = tx, iElem = 0; ind < 576; ind += blockDim_x, iElem ++){
			pP[ind] = buf[iElem];
		}
	}
}



// For nPaddedState == 32, use shared memory to store a tile of the U/V/R matrix;
// The total size of shared memory needed is: (32 * TILE_SIZE_PMAT_XSTATE_BASELINE * 2 + TILE_SIZE_PMAT_XSTATE_BASELINE) * BLOCK_DIMENSION_Y_PMAT_XSTATE_32_BASELINE = (32 * 4 * 2 + 4) * 2 =520 ;
__global__
void kernelPMat_xState_transpose_baseline_32State(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix, const int nState)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;
	int blockDim_x = blockDim.x;

	CuLPMatOffset reg_offset;
	CUFlt *pU, *pV, *pR, *pP, curBrLen;
	CUFlt buf[1024 / BLOCK_DIMENSION_X_PMAT_XSTATE_32_BASELINE] = {0, 0};

	__shared__ CUFlt sh_U[BLOCK_DIMENSION_Y_PMAT_XSTATE_32_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE][32];
	__shared__ CUFlt sh_V[BLOCK_DIMENSION_Y_PMAT_XSTATE_32_BASELINE][32][TILE_SIZE_PMAT_XSTATE_BASELINE];
	__shared__ CUFlt sh_R[BLOCK_DIMENSION_Y_PMAT_XSTATE_32_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE];

	if(curMatrix < nMatrix){
		reg_offset = offset[curMatrix];

		curBrLen = brLen[reg_offset.brLen_offset];

		pU = U + reg_offset.UV_offset;
		pV = V + reg_offset.UV_offset;
		pR = R + reg_offset.R_offset;
	}

	int itr, ind, row, col, iElem, k, nIteration = (nState + TILE_SIZE_PMAT_XSTATE_BASELINE - 1) / TILE_SIZE_PMAT_XSTATE_BASELINE;
	const int nElement = (TILE_SIZE_PMAT_XSTATE_BASELINE << 5);	

	for(itr = 0; itr < nIteration; itr ++){
		if(curMatrix < nMatrix){
			for(ind = tx; ind < nElement; ind += blockDim_x){
				row = ind / TILE_SIZE_PMAT_XSTATE_BASELINE;
				col = ind % TILE_SIZE_PMAT_XSTATE_BASELINE;
				sh_V[ty][(ind & 0x1f)][(ind >> 5)] = pV[ind];
				sh_U[ty][col][row]  = pU[(row << 5) + col];
			}
			if(tx < TILE_SIZE_PMAT_XSTATE_BASELINE)
				sh_R[ty][tx] = pR[tx];

			pU += TILE_SIZE_PMAT_XSTATE_BASELINE;
			pV += nElement;
			pR += TILE_SIZE_PMAT_XSTATE_BASELINE;
		}

		__syncthreads();

		if(curMatrix < nMatrix){
			for(ind = tx, iElem = 0; ind < 1024; ind += blockDim_x, iElem ++){
				row = (ind >> 5);
				col = (ind & 0x1f);

				for(k = 0; k < TILE_SIZE_PMAT_XSTATE_BASELINE; k ++)
					buf[iElem] += sh_V[ty][row][k] * exp(sh_R[ty][k] * curBrLen) * sh_U[ty][k][col];
			}
		}

		__syncthreads();
	}
	
	// Write the results to global memory:
	if(curMatrix < nMatrix){
		pP = P + reg_offset.P_offset;

		for(ind = tx, iElem = 0; ind < 1024; ind += blockDim_x, iElem ++){
			pP[ind] = buf[iElem];
		}
	}
}



// For nPaddedState == 40, use shared memory to store a tile of the U/V/R matrix;
// The total size of shared memory needed is: (40 * TILE_SIZE_PMAT_XSTATE_BASELINE * 2 + TILE_SIZE_PMAT_XSTATE_BASELINE) * BLOCK_DIMENSION_Y_PMAT_XSTATE_40_BASELINE = (40 * 4 * 2 + 4) * 2 = 648 ;
__global__
void kernelPMat_xState_transpose_baseline_40State(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix, const int nState)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;
	int blockDim_x = blockDim.x;

	CuLPMatOffset reg_offset;
	CUFlt *pU, *pV, *pR, *pP, curBrLen;
	CUFlt buf[1600 / BLOCK_DIMENSION_X_PMAT_XSTATE_40_BASELINE] = {0, 0};

	__shared__ CUFlt sh_U[BLOCK_DIMENSION_Y_PMAT_XSTATE_40_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE][40];
	__shared__ CUFlt sh_V[BLOCK_DIMENSION_Y_PMAT_XSTATE_40_BASELINE][40][TILE_SIZE_PMAT_XSTATE_BASELINE];
	__shared__ CUFlt sh_R[BLOCK_DIMENSION_Y_PMAT_XSTATE_40_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE];

	if(curMatrix < nMatrix){
		reg_offset = offset[curMatrix];

		curBrLen = brLen[reg_offset.brLen_offset];

		pU = U + reg_offset.UV_offset;
		pV = V + reg_offset.UV_offset;
		pR = R + reg_offset.R_offset;
	}

	int itr, ind, row, col, iElem, k, nIteration = (nState + TILE_SIZE_PMAT_XSTATE_BASELINE - 1) / TILE_SIZE_PMAT_XSTATE_BASELINE;
	const int nElement = 40 * TILE_SIZE_PMAT_XSTATE_BASELINE;	

	for(itr = 0; itr < nIteration; itr ++){
		if(curMatrix < nMatrix){
			for(ind = tx; ind < nElement; ind += blockDim_x){
				row = ind / TILE_SIZE_PMAT_XSTATE_BASELINE;
				col = ind % TILE_SIZE_PMAT_XSTATE_BASELINE;
				sh_V[ty][ind % 40][ind / 40] = pV[ind];
				sh_U[ty][col][row]  = pU[row * 40 + col];
			}
			if(tx < TILE_SIZE_PMAT_XSTATE_BASELINE)
				sh_R[ty][tx] = pR[tx];

			pU += TILE_SIZE_PMAT_XSTATE_BASELINE;
			pV += nElement;
			pR += TILE_SIZE_PMAT_XSTATE_BASELINE;
		}

		__syncthreads();

		if(curMatrix < nMatrix){
			for(ind = tx, iElem = 0; ind < 1600; ind += blockDim_x, iElem ++){
				row = ind / 40;
				col = ind % 40;

				for(k = 0; k < TILE_SIZE_PMAT_XSTATE_BASELINE; k ++)
					buf[iElem] += sh_V[ty][row][k] * exp(sh_R[ty][k] * curBrLen) * sh_U[ty][k][col];
			}
		}

		__syncthreads();
	}
	
	// Write the results to global memory:
	if(curMatrix < nMatrix){
		pP = P + reg_offset.P_offset;

		for(ind = tx, iElem = 0; ind < 1600; ind += blockDim_x, iElem ++){
			pP[ind] = buf[iElem];
		}
	}
}



// For nPaddedState == 48, use shared memory to store a tile of the U/V/R matrix;
// The total size of shared memory needed is: (48 * TILE_SIZE_PMAT_XSTATE_BASELINE * 2 + TILE_SIZE_PMAT_XSTATE_BASELINE) * BLOCK_DIMENSION_Y_PMAT_XSTATE_48_BASELINE = (48 * 4 * 2 + 4) * 1 = 388;
__global__
void kernelPMat_xState_transpose_baseline_48State(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix, const int nState)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;
	int blockDim_x = blockDim.x;

	CuLPMatOffset reg_offset;
	CUFlt *pU, *pV, *pR, *pP, curBrLen;
	CUFlt buf[2304 / BLOCK_DIMENSION_X_PMAT_XSTATE_48_BASELINE] = {0, 0};

	__shared__ CUFlt sh_U[BLOCK_DIMENSION_Y_PMAT_XSTATE_48_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE][48];
	__shared__ CUFlt sh_V[BLOCK_DIMENSION_Y_PMAT_XSTATE_48_BASELINE][48][TILE_SIZE_PMAT_XSTATE_BASELINE];
	__shared__ CUFlt sh_R[BLOCK_DIMENSION_Y_PMAT_XSTATE_48_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE];

	if(curMatrix < nMatrix){
		reg_offset = offset[curMatrix];

		curBrLen = brLen[reg_offset.brLen_offset];

		pU = U + reg_offset.UV_offset;
		pV = V + reg_offset.UV_offset;
		pR = R + reg_offset.R_offset;
	}

	int itr, ind, row, col, iElem, k, nIteration = (nState + TILE_SIZE_PMAT_XSTATE_BASELINE - 1) / TILE_SIZE_PMAT_XSTATE_BASELINE;
	const int nElement = 48 * TILE_SIZE_PMAT_XSTATE_BASELINE;	

	for(itr = 0; itr < nIteration; itr ++){
		if(curMatrix < nMatrix){
			for(ind = tx; ind < nElement; ind += blockDim_x){
				row = ind / TILE_SIZE_PMAT_XSTATE_BASELINE;
				col = ind % TILE_SIZE_PMAT_XSTATE_BASELINE;
				sh_V[ty][ind % 48][ind / 48] = pV[ind];
				sh_U[ty][col][row]  = pU[row * 48 + col];
			}
			if(tx < TILE_SIZE_PMAT_XSTATE_BASELINE)
				sh_R[ty][tx] = pR[tx];

			pU += TILE_SIZE_PMAT_XSTATE_BASELINE;
			pV += nElement;
			pR += TILE_SIZE_PMAT_XSTATE_BASELINE;
		}

		__syncthreads();

		if(curMatrix < nMatrix){
			for(ind = tx, iElem = 0; ind < 2304; ind += blockDim_x, iElem ++){
				row = ind / 48;
				col = ind % 48;

				for(k = 0; k < TILE_SIZE_PMAT_XSTATE_BASELINE; k ++)
					buf[iElem] += sh_V[ty][row][k] * exp(sh_R[ty][k] * curBrLen) * sh_U[ty][k][col];
			}
		}

		__syncthreads();
	}
	
	// Write the results to global memory:
	if(curMatrix < nMatrix){
		pP = P + reg_offset.P_offset;

		for(ind = tx, iElem = 0; ind < 2304; ind += blockDim_x, iElem ++){
			pP[ind] = buf[iElem];
		}
	}
}



// For nPaddedState == 56, use shared memory to store a tile of the U/V/R matrix;
// The total size of shared memory needed is: (56 * TILE_SIZE_PMAT_XSTATE_BASELINE * 2 + TILE_SIZE_PMAT_XSTATE_BASELINE) * BLOCK_DIMENSION_Y_PMAT_XSTATE_56_BASELINE = (56 * 4 * 2 + 4) * 1 = 452 ;
__global__
void kernelPMat_xState_transpose_baseline_56State(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix, const int nState)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int curMatrix = blockIdx.x * blockDim.y + ty;
	int blockDim_x = blockDim.x;

	CuLPMatOffset reg_offset;
	CUFlt *pU, *pV, *pR, *pP, curBrLen;
	CUFlt buf[3136 / BLOCK_DIMENSION_X_PMAT_XSTATE_56_BASELINE] = {0, 0};

	__shared__ CUFlt sh_U[BLOCK_DIMENSION_Y_PMAT_XSTATE_56_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE][56];
	__shared__ CUFlt sh_V[BLOCK_DIMENSION_Y_PMAT_XSTATE_56_BASELINE][56][TILE_SIZE_PMAT_XSTATE_BASELINE];
	__shared__ CUFlt sh_R[BLOCK_DIMENSION_Y_PMAT_XSTATE_56_BASELINE][TILE_SIZE_PMAT_XSTATE_BASELINE];

	if(curMatrix < nMatrix){
		reg_offset = offset[curMatrix];

		curBrLen = brLen[reg_offset.brLen_offset];

		pU = U + reg_offset.UV_offset;
		pV = V + reg_offset.UV_offset;
		pR = R + reg_offset.R_offset;
	}

	int itr, ind, row, col, iElem, k, nIteration = (nState + TILE_SIZE_PMAT_XSTATE_BASELINE - 1) / TILE_SIZE_PMAT_XSTATE_BASELINE;
	const int nElement = 56 * TILE_SIZE_PMAT_XSTATE_BASELINE;

	for(itr = 0; itr < nIteration; itr ++){
		if(curMatrix < nMatrix){
			for(ind = tx; ind < nElement; ind += blockDim_x){
				row = ind / TILE_SIZE_PMAT_XSTATE_BASELINE;
				col = ind % TILE_SIZE_PMAT_XSTATE_BASELINE;
				sh_V[ty][ind % 56][ind / 56] = pV[ind];
				sh_U[ty][col][row]  = pU[row * 56 + col];
			}
			if(tx < TILE_SIZE_PMAT_XSTATE_BASELINE)
				sh_R[ty][tx] = pR[tx];

			pU += TILE_SIZE_PMAT_XSTATE_BASELINE;
			pV += nElement;
			pR += TILE_SIZE_PMAT_XSTATE_BASELINE;
		}

		__syncthreads();

		if(curMatrix < nMatrix){
			for(ind = tx, iElem = 0; ind < 3136; ind += blockDim_x, iElem ++){
				row = ind / 56;
				col = ind % 56;

				for(k = 0; k < TILE_SIZE_PMAT_XSTATE_BASELINE; k ++)
					buf[iElem] += sh_V[ty][row][k] * exp(sh_R[ty][k] * curBrLen) * sh_U[ty][k][col];
			}
		}

		__syncthreads();
	}
	
	// Write the results to global memory:
	if(curMatrix < nMatrix){
		pP = P + reg_offset.P_offset;

		for(ind = tx, iElem = 0; ind < 3136; ind += blockDim_x, iElem ++){
			pP[ind] = buf[iElem];
		}
	}
}



// For xState > 64, each thread is responsible for 8 elements, and the block dimension is: (nPaddedState / 8, 8)��Ҳ��threadIdx.y��ͬ��nPaddedState / 8��thread����ͬһ�У�ÿ��block����8��;
// ÿnPaddedState / 8��thread block����һ��PMat matrix;
// grid dimension is: (nPaddedState / 8, cntMatrix), no need for if(curMatrix < nMatrix) statement;

__global__
void kernelPMat_xState_transpose_baseline_largeState(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix, const int nState, const int nPaddedState)
{
	int blockDim_x = blockDim.x;
	int ind = threadIdx.y * blockDim_x + threadIdx.x;
	int curMatrix = blockIdx.y;
	int curOffset = blockIdx.x * N_ELEMENT_PER_THREAD_PMAT_XSTATE_LARGE_STATE * nPaddedState;

	const int nElemPerTile_U = TILE_SIZE_PMAT_XSTATE_BASELINE * BLOCK_DIMENSION_Y_PMAT_XSTATE_LARGE_STATE_BASELINE;
	const int nElemPerTile_V = nPaddedState * TILE_SIZE_PMAT_XSTATE_BASELINE;
	const int nElemToCalc = BLOCK_DIMENSION_Y_PMAT_XSTATE_LARGE_STATE_BASELINE * nPaddedState;
	const int nThread = blockDim_x * BLOCK_DIMENSION_Y_PMAT_XSTATE_LARGE_STATE_BASELINE;

	CuLPMatOffset reg_offset;
	CUFlt *pU, *pV, *pR, *pP, *pSh_V, curBrLen;
	CUFlt buf[N_ELEMENT_PER_THREAD_PMAT_XSTATE_LARGE_STATE] = {0, 0};

	extern __shared__ CUFlt sh_UV[];
	__shared__ CUFlt sh_R[TILE_SIZE_PMAT_XSTATE_BASELINE];

	reg_offset = offset[curMatrix];

	curBrLen = brLen[reg_offset.brLen_offset];

	pU = U + reg_offset.UV_offset + curOffset;
	pV = V + reg_offset.UV_offset;
	pR = R + reg_offset.R_offset;
	
	pSh_V = sh_UV + nElemPerTile_U;

	int itr, curInd, row, col, iElem, k, nIteration = (nState + TILE_SIZE_PMAT_XSTATE_BASELINE - 1) / TILE_SIZE_PMAT_XSTATE_BASELINE;
	CUFlt *pCur_U, *pCur_V;

	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_U; curInd += nThread){
			row = curInd / TILE_SIZE_PMAT_XSTATE_BASELINE;
			col = curInd % TILE_SIZE_PMAT_XSTATE_BASELINE;
			sh_UV[col * BLOCK_DIMENSION_Y_PMAT_XSTATE_LARGE_STATE_BASELINE + row] = pU[row * nPaddedState + col];
		}

		for(curInd = ind; curInd < nElemPerTile_V; curInd += nThread){
			row = curInd / nPaddedState;
			col = curInd % nPaddedState;
			pSh_V[col * TILE_SIZE_PMAT_XSTATE_BASELINE + row] = pV[curInd];
		}

		if(ind < TILE_SIZE_PMAT_XSTATE_BASELINE)
			sh_R[ind] = pR[ind];

		pU += TILE_SIZE_PMAT_XSTATE_BASELINE;
		pV += nElemPerTile_V;
		pR += TILE_SIZE_PMAT_XSTATE_BASELINE;

		__syncthreads();

		for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, iElem ++){
			row = curInd / BLOCK_DIMENSION_Y_PMAT_XSTATE_LARGE_STATE_BASELINE;
			col = curInd % BLOCK_DIMENSION_Y_PMAT_XSTATE_LARGE_STATE_BASELINE;
			
			pCur_U = sh_UV + col;
			pCur_V = pSh_V + row * TILE_SIZE_PMAT_XSTATE_BASELINE;

			for(k = 0; k < TILE_SIZE_PMAT_XSTATE_BASELINE; k ++, pCur_U += BLOCK_DIMENSION_Y_PMAT_XSTATE_LARGE_STATE_BASELINE)
				buf[iElem] += pCur_V[k] * exp(sh_R[k] * curBrLen) * pCur_U[0];
		}

		__syncthreads();
	}
	
	// Write the results to global memory:
	pP = P + reg_offset.P_offset + blockIdx.x * N_ELEMENT_PER_THREAD_PMAT_XSTATE_LARGE_STATE;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, iElem ++){
		row = curInd / N_ELEMENT_PER_THREAD_PMAT_XSTATE_LARGE_STATE;
		col = curInd % N_ELEMENT_PER_THREAD_PMAT_XSTATE_LARGE_STATE;
		pP[row * nPaddedState + col] = buf[iElem];
	}
}



void callKernelPMat_baseline(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CuLPMatOffset *offset, const int nMatrix, const int nState, const int nPaddedState, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream)
{
	// Baseline is used when nPaddedState != 64:
	if(4 == nPaddedState){
		//printf("\n=======\nGoing to call kernel for 4 state of PMat...\n==========\n");
#ifdef TRANSPOSE_PMAT
		kernelPMat_4State_transpose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, R, brLen, offset, nMatrix);
		cutilCheckMsg("kernel kernelPMat_4State_transpose_baseline() failed");
#else
		kernelPMat_4State_noTranspose_baseline<<<nBlockPerGrid, nThreadPerBlock, 4 * BLOCK_DIMENSION_Y_PMAT_4STATE_BASELINE * sizeof(CUFlt), stream>>>(P, U, V, R, brLen, offset, nMatrix);
		cutilCheckMsg("kernel kernelPMat_4State_noTranspose_baseline() failed");
#endif
	}
	else if(20 == nPaddedState){
#ifdef TRANSPOSE_PMAT
		//printf("\n=======\nGoing to call kernel for 20 state of PMat...\n==========\n");
		kernelPMat_20State_transpose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, R, brLen, offset, nMatrix);
		cutilCheckMsg("kernel kernelPMat_20State_transpose_baseline() failed");
#else
		kernelPMat_20State_noTranspose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, R, brLen, offset, nMatrix);
		cutilCheckMsg("kernel kernelPMat_20State_noTranspose_baseline() failed");
#endif
	}
	else{
		// For nPaddedState != 4 / 20 / 64:
#ifdef TRANSPOSE_PMAT
		if(8 == nPaddedState){
			//printf("\n=======\nGoing to call kernel for 8 state of PMat...\n==========\n");
			kernelPMat_xState_transpose_baseline_8State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, R, brLen, offset, nMatrix, nState);
		}
		else if(16 == nPaddedState){
			//printf("\n=======\nGoing to call kernel for 16 state of PMat...\n==========\n");
			kernelPMat_xState_transpose_baseline_16State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, R, brLen, offset, nMatrix, nState);
		}
		else if(24 == nPaddedState){
			//printf("\n=======\nGoing to call kernel for 24 state of PMat...\n==========\n");
			kernelPMat_xState_transpose_baseline_24State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, R, brLen, offset, nMatrix, nState);
		}
		else if(32 == nPaddedState){
			//printf("\n=======\nGoing to call kernel for 32 state of PMat...\n==========\n");
			kernelPMat_xState_transpose_baseline_32State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, R, brLen, offset, nMatrix, nState);
		}
		else if(40 == nPaddedState){
			//printf("\n=======\nGoing to call kernel for 40 state of PMat...\n==========\n");
			kernelPMat_xState_transpose_baseline_40State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, R, brLen, offset, nMatrix, nState);
		}
		else if(48 == nPaddedState){
			//printf("\n=======\nGoing to call kernel for 48 state of PMat...\n==========\n");
			kernelPMat_xState_transpose_baseline_48State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, R, brLen, offset, nMatrix, nState);
		}
		else if(56 == nPaddedState){
			//printf("\n=======\nGoing to call kernel for 56 state of PMat...\n==========\n");
			kernelPMat_xState_transpose_baseline_56State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, R, brLen, offset, nMatrix, nState);
		}
		else{
			// For nPaddedState > 64:
			//printf("\n=======\nGoing to call kernel for large state of PMat...\n==========\n");
			const int sharedMem_size = TILE_SIZE_PMAT_XSTATE_BASELINE * (BLOCK_DIMENSION_Y_PMAT_XSTATE_LARGE_STATE_BASELINE + nPaddedState);
			
			kernelPMat_xState_transpose_baseline_largeState<<<nBlockPerGrid, nThreadPerBlock, sharedMem_size * sizeof(CUFlt), stream>>>(P, U, V, R, brLen, offset, nMatrix, nState, nPaddedState);
		}
		cutilCheckMsg("kernel kernelPMat_xState_transpose_baseline() failed");
#else
		kernelPMat_xState_noTranspose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(P, U, V, R, brLen, offset, nMatrix, nState, nPaddedState);
		cutilCheckMsg("kernel kernelPMat_xState_noTranspose_baseline() failed");
#endif
	}
}




// Calculation of conditional likelihoods:
// case 1: both children are tip states;
// transpose version: row * col;
// version 1 of case 1: each thread is responsible for all states of k site patterns;
__device__
void deviceCondlike_4State_case1_transpose_baseline_version1(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread, int nSitePattern)
{
	CUFlt *pCondlike_F;
	int offset_L, offset_R, iState;

	for(; ind < nSitePattern; ind += nThread){
		offset_L = (tipState_L[ind] << 2);
		offset_R = (tipState_R[ind] << 2);

		pCondlike_F = condlike_F + (ind << 2);

		for(iState = 0; iState < 4; iState ++)
			pCondlike_F[iState] = PMat_L[iState + offset_L] * PMat_R[iState + offset_R];
	}
}


// version 2 of case 1: each thread is responsible for 1 state of k site patterns;
__device__
void deviceCondlike_4State_case1_transpose_baseline_version2(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread, int nElement)
{
	int offset_L, offset_R, curPattern, curState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		offset_L = (tipState_L[curPattern] << 2);
		offset_R = (tipState_R[curPattern] << 2);

		condlike_F[ind] = PMat_L[offset_L + curState] * PMat_R[offset_R + curState];
	}
}

// non-transpose version: row * row;
__device__
void deviceCondlike_4State_case1_noTranspose_baseline_version1(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread, int nSitePattern)
{
	CUFlt *pCondlike_F;
	int offset_L, offset_R, iState, offset;

	for(; ind < nSitePattern; ind += nThread){
		offset_L = tipState_L[ind];
		offset_R = tipState_R[ind];

		pCondlike_F = condlike_F + (ind << 2);

		for(iState = 0, offset = 0; iState < 4; iState ++, offset += 4)
			pCondlike_F[iState] = PMat_L[offset + offset_L] * PMat_R[offset + offset_R];
	}
}


__device__
void deviceCondlike_4State_case1_noTranspose_baseline_version2(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread, int nElement)
{
	int offset_L, offset_R, curPattern, curState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		offset_L = tipState_L[curPattern];
		offset_R = tipState_R[curPattern];

		curState <<= 2;

		condlike_F[ind] = PMat_L[offset_L + curState] * PMat_R[offset_R + curState];
	}
}


// case 2: one child is tip state, the other child is tip condlike:
// transpose version: row * col;
__device__
void deviceCondlike_4State_case2_transpose_baseline_version1(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_R;
	int offset_L, iState;

	for(; ind < nSitePattern; ind += nThread){
		pCondlike_F = condlike_F + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
#ifdef USING_LDG
		offset_L = (__ldg(&tipState_L[ind]) << 2);
#else
		offset_L = (tipState_L[ind] << 2);
#endif

		for(iState = 0; iState < 4; iState ++){
#ifdef USING_LDG
			pCondlike_F[iState] = (PMat_L[iState + offset_L]) * (__ldg(&pCondlike_R[0]) * PMat_R[iState] + __ldg(&pCondlike_R[1]) * PMat_R[iState + 4] + __ldg(&pCondlike_R[2]) * PMat_R[iState + 8] + __ldg(&pCondlike_R[3]) * PMat_R[iState + 12]);
#else
			pCondlike_F[iState] = (PMat_L[iState + offset_L]) * (pCondlike_R[0] * PMat_R[iState] + pCondlike_R[1] * PMat_R[iState + 4] + pCondlike_R[2] * PMat_R[iState + 8] + pCondlike_R[3] * PMat_R[iState + 12]);
#endif
		}
	}
}


__device__
void deviceCondlike_4State_case2_transpose_baseline_version2(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nElement)
{
	CUFlt *pPMat_R, *pCondlike_R;
	int offset_L, curPattern, curState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);
		pCondlike_R = condlike_R + (curPattern << 2);
		pPMat_R = PMat_R + curState;
#ifdef USING_LDG
		offset_L = (__ldg(&tipState_L[curPattern]) << 2);
#else
		offset_L = (tipState_L[curPattern] << 2);
#endif
		
#ifdef USING_LDG
		condlike_F[ind] = PMat_L[offset_L + curState] * (__ldg(&pCondlike_R[0]) * pPMat_R[0] + __ldg(&pCondlike_R[1]) * pPMat_R[4] + __ldg(&pCondlike_R[2]) * pPMat_R[8] + __ldg(&pCondlike_R[3]) * pPMat_R[12]);
#else
		condlike_F[ind] = PMat_L[offset_L + curState] * (pCondlike_R[0] * pPMat_R[0] + pCondlike_R[1] * pPMat_R[4] + pCondlike_R[2] * pPMat_R[8] + pCondlike_R[3] * pPMat_R[12]);
#endif
	}
}


// version 3: roll the iteration;
__device__
void deviceCondlike_4State_case2_transpose_baseline_version3(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nElement)
{
	CUFlt *pPMat_R, *pCondlike_R, sum_R;
	int offset_L, curPattern, curState, iState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		sum_R = 0.0f;
		pCondlike_R = condlike_R + (curPattern << 2);
		pPMat_R = PMat_R + curState;
#ifdef USING_LDG
		offset_L = (__ldg(&tipState_L[curPattern]) << 2);
#else
		offset_L = (tipState_L[curPattern] << 2);
#endif
		
		for(iState = 0; iState < 4; iState ++, pPMat_R += 4){
#ifdef USING_LDG
			sum_R += __ldg(&pCondlike_R[iState]) * pPMat_R[0];
#else
			sum_R += pCondlike_R[iState] * pPMat_R[0];
#endif
		}

		condlike_F[ind] = PMat_L[offset_L + curState] * sum_R;
	}
}


__device__
void deviceCondlike_4State_case2_transpose_baseline_version4(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_R, buf[4];
	int offset_L, iState;

	for(; ind < nSitePattern; ind += nThread){
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_R[0]);
		buf[1] = __ldg(&pCondlike_R[1]);
		buf[2] = __ldg(&pCondlike_R[2]);
		buf[3] = __ldg(&pCondlike_R[3]);
#else
		buf[0] = pCondlike_R[0];
		buf[1] = pCondlike_R[1];
		buf[2] = pCondlike_R[2];
		buf[3] = pCondlike_R[3];
#endif

#ifdef USING_LDG
		offset_L = (__ldg(&tipState_L[ind]) << 2);
#else
		offset_L = (tipState_L[ind] << 2);
#endif

		for(iState = 0; iState < 4; iState ++){
			pCondlike_F[iState] = (PMat_L[iState + offset_L]) * (buf[0] * PMat_R[iState] + buf[1] * PMat_R[iState + 4] + buf[2] * PMat_R[iState + 8] + buf[3] * PMat_R[iState + 12]);
		}
	}
}



__device__
void deviceCondlike_4State_case2_transpose_baseline_version5(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_R, *pPMat_R, sum_R, buf[4];
	int offset_L, iState, jState;

	for(; ind < nSitePattern; ind += nThread){
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_R[0]);
		buf[1] = __ldg(&pCondlike_R[1]);
		buf[2] = __ldg(&pCondlike_R[2]);
		buf[3] = __ldg(&pCondlike_R[3]);
#else
		buf[0] = pCondlike_R[0];
		buf[1] = pCondlike_R[1];
		buf[2] = pCondlike_R[2];
		buf[3] = pCondlike_R[3];
#endif

#ifdef USING_LDG
		offset_L = (__ldg(&tipState_L[ind]) << 2);
#else
		offset_L = (tipState_L[ind] << 2);
#endif

		for(iState = 0; iState < 4; iState ++){
			pPMat_R = PMat_R + iState;
			sum_R = 0.0f;

			for(jState = 0; jState < 4; jState ++, pPMat_R += 4)
				sum_R += buf[jState] * pPMat_R[0];
			
			pCondlike_F[iState] = (PMat_L[iState + offset_L]) * sum_R;
		}
	}
}

/*
// version 8: use shared memory for condlike;
__device__
void deviceCondlike_4State_case2_transpose_baseline_version8(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThreadPerArray, const int nElement)
{
	CUFlt *pPMat_R, *pCondlike_R;
	int offset_L, curPattern, curState;

	__shared__ CUFlt sh_condlike[N_THREAD_PER_BLOCK_CONDLIKE_4STATE_BASELINE * 4];

	for(; ind < nElement; ind += nThreadPerArray){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);
		pCondlike_R = condlike_R + (curPattern << 2);
		pPMat_R = PMat_R + curState;
		offset_L = (tipState_L[curPattern] << 2);
		
		condlike_F[ind] = PMat_L[offset_L + curState] * (pCondlike_R[0] * pPMat_R[0] + pCondlike_R[1] * pPMat_R[4] + pCondlike_R[2] * pPMat_R[8] + pCondlike_R[3] * pPMat_R[12]);
	}
}
*/



// Non-transpose version: row * row;
__device__
void deviceCondlike_4State_case2_noTranspose_baseline_version1(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_R, *pPMat_R;
	int offset_L, iState, offset;

	for(; ind < nSitePattern; ind += nThread){
		pCondlike_F = condlike_F + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
#ifdef USING_LDG
		offset_L = __ldg(&tipState_L[ind]);
#else
		offset_L = tipState_L[ind];
#endif

		for(iState = 0, offset = 0; iState < 4; iState ++, offset += 4){
			pPMat_R = PMat_R + offset;
#ifdef USING_LDG
			pCondlike_F[iState] = (PMat_L[offset + offset_L]) * (__ldg(&pCondlike_R[0]) * pPMat_R[0] + __ldg(&pCondlike_R[1]) * pPMat_R[1] + __ldg(&pCondlike_R[2]) * pPMat_R[2] + __ldg(&pCondlike_R[3]) * pPMat_R[3]);
#else
			pCondlike_F[iState] = (PMat_L[offset + offset_L]) * (pCondlike_R[0] * pPMat_R[0] + pCondlike_R[1] * pPMat_R[1] + pCondlike_R[2] * pPMat_R[2] + pCondlike_R[3] * pPMat_R[3]);
#endif
		}
	}
}


__device__
void deviceCondlike_4State_case2_noTranspose_baseline_version2(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nElement)
{
	CUFlt *pCondlike_R, *pPMat_R;
	int offset_L, curPattern, curState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		pCondlike_R = condlike_R + (curPattern << 2);
		curState <<= 2;
#ifdef USING_LDG
		offset_L = __ldg(&tipState_L[curPattern]);
#else
		offset_L = tipState_L[curPattern];
#endif
		
		pPMat_R = PMat_R + curState;

#ifdef USING_LDG
		condlike_F[ind] = PMat_L[offset_L + curState] * (__ldg(&pCondlike_R[0]) * pPMat_R[0] + __ldg(&pCondlike_R[1]) * pPMat_R[1] + __ldg(&pCondlike_R[2]) * pPMat_R[2] + __ldg(&pCondlike_R[3]) * pPMat_R[3]);
#else
		condlike_F[ind] = PMat_L[offset_L + curState] * (pCondlike_R[0] * pPMat_R[0] + pCondlike_R[1] * pPMat_R[1] + pCondlike_R[2] * pPMat_R[2] + pCondlike_R[3] * pPMat_R[3]);
#endif
	}
}


__device__
void deviceCondlike_4State_case2_noTranspose_baseline_version3(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nElement)
{
	CUFlt *pCondlike_R, *pPMat_R, sum_R;
	int offset_L, curPattern, curState, iState;

	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		sum_R = 0.0f;
		pCondlike_R = condlike_R + (curPattern << 2);
		curState <<= 2;
#ifdef USING_LDG
		offset_L = __ldg(&tipState_L[curPattern]);
#else
		offset_L = tipState_L[curPattern];
#endif
		
		pPMat_R = PMat_R + curState;

		for(iState = 0; iState < 4; iState ++){
#ifdef USING_LDG
			sum_R += __ldg(&pCondlike_R[iState]) * pPMat_R[iState];
#else
			sum_R += pCondlike_R[iState] * pPMat_R[iState];
#endif
		}

		condlike_F[ind] = PMat_L[offset_L + curState] * sum_R;
	}
}


__device__
void deviceCondlike_4State_case2_noTranspose_baseline_version4(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_R, *pPMat_R, buf[4];
	int offset_L, iState, offset;

	for(; ind < nSitePattern; ind += nThread){
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);
		
#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_R[0]);
		buf[1] = __ldg(&pCondlike_R[1]);
		buf[2] = __ldg(&pCondlike_R[2]);
		buf[3] = __ldg(&pCondlike_R[3]);
#else
		buf[0] = pCondlike_R[0];
		buf[1] = pCondlike_R[1];
		buf[2] = pCondlike_R[2];
		buf[3] = pCondlike_R[3];
#endif

#ifdef USING_LDG
		offset_L = __ldg(&tipState_L[ind]);
#else
		offset_L = tipState_L[ind];
#endif

		pPMat_R = PMat_R;
		for(iState = 0, offset = 0; iState < 4; iState ++, pPMat_R += 4, offset += 4){
			pCondlike_F[iState] = (PMat_L[offset + offset_L]) * (buf[0] * pPMat_R[0] + buf[1] * pPMat_R[1] + buf[2] * pPMat_R[2] + buf[3] * pPMat_R[3]);
		}
	}
}


__device__
void deviceCondlike_4State_case2_noTranspose_baseline_version5(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_R, *pPMat_R, sum_R, buf[4];
	int offset_L, iState, jState, offset;

	for(; ind < nSitePattern; ind += nThread){
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_R[0]);
		buf[1] = __ldg(&pCondlike_R[1]);
		buf[2] = __ldg(&pCondlike_R[2]);
		buf[3] = __ldg(&pCondlike_R[3]);
#else
		buf[0] = pCondlike_R[0];
		buf[1] = pCondlike_R[1];
		buf[2] = pCondlike_R[2];
		buf[3] = pCondlike_R[3];
#endif

#ifdef USING_LDG
		offset_L = __ldg(&tipState_L[ind]);
#else
		offset_L = tipState_L[ind];
#endif

		pPMat_R = PMat_R;
		for(iState = 0, offset = 0; iState < 4; iState ++, pPMat_R += 4, offset += 4){
			sum_R = 0;
			for(jState = 0; jState < 4; jState ++)
				sum_R += buf[jState] * pPMat_R[jState];

			pCondlike_F[iState] = (PMat_L[offset + offset_L]) * sum_R;
		}
	}
}



// case 3: both children are condlike:
// transpose version: row * col;
__device__
void deviceCondlike_4State_case3_transpose_baseline_version1(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_L, *pCondlike_R;

	int iState;
	for(; ind < nSitePattern; ind += nThread){
		pCondlike_F = condlike_F + (ind << 2);
		pCondlike_L = condlike_L + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
		
		for(iState = 0; iState < 4; iState ++){
#ifdef USING_LDG
			pCondlike_F[iState] = (__ldg(&pCondlike_L[0]) * PMat_L[iState] + __ldg(&pCondlike_L[1]) * PMat_L[iState + 4] + __ldg(&pCondlike_L[2]) * PMat_L[iState + 8] + __ldg(&pCondlike_L[3]) * PMat_L[iState + 12]) * (__ldg(&pCondlike_R[0]) * PMat_R[iState] + __ldg(&pCondlike_R[1]) * PMat_R[iState + 4] + __ldg(&pCondlike_R[2]) * PMat_R[iState + 8] + __ldg(&pCondlike_R[3]) * PMat_R[iState + 12]);
#else
			pCondlike_F[iState] = (pCondlike_L[0] * PMat_L[iState] + pCondlike_L[1] * PMat_L[iState + 4] + pCondlike_L[2] * PMat_L[iState + 8] + pCondlike_L[3] * PMat_L[iState + 12]) * (pCondlike_R[0] * PMat_R[iState] + pCondlike_R[1] * PMat_R[iState + 4] + pCondlike_R[2] * PMat_R[iState + 8] + pCondlike_R[3] * PMat_R[iState + 12]);
#endif
		}
	}
}


__device__
void deviceCondlike_4State_case3_transpose_baseline_version2(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nElement)
{
	CUFlt *pCondlike_L, *pCondlike_R, *pPMat_L, *pPMat_R;

	int curPattern, curState;
	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);
		pCondlike_L = condlike_L + (curPattern << 2);
		pCondlike_R = condlike_R + (curPattern << 2);
		pPMat_L = PMat_L + curState;
		pPMat_R = PMat_R + curState;
		
#ifdef USING_LDG
		condlike_F[ind] = (__ldg(&pCondlike_L[0]) * pPMat_L[0] + __ldg(&pCondlike_L[1]) * pPMat_L[4] + __ldg(&pCondlike_L[2]) * pPMat_L[8] + __ldg(&pCondlike_L[3]) * pPMat_L[12]) * (__ldg(&pCondlike_R[0]) * pPMat_R[0] + __ldg(&pCondlike_R[1]) * pPMat_R[4] + __ldg(&pCondlike_R[2]) * pPMat_R[8] + __ldg(&pCondlike_R[3]) * pPMat_R[12]);
#else
		condlike_F[ind] = (pCondlike_L[0] * pPMat_L[0] + pCondlike_L[1] * pPMat_L[4] + pCondlike_L[2] * pPMat_L[8] + pCondlike_L[3] * pPMat_L[12]) * (pCondlike_R[0] * pPMat_R[0] + pCondlike_R[1] * pPMat_R[4] + pCondlike_R[2] * pPMat_R[8] + pCondlike_R[3] * pPMat_R[12]);
#endif
	}
}



__device__
void deviceCondlike_4State_case3_transpose_baseline_version3(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nElement)
{
	CUFlt *pCondlike_L, *pCondlike_R, *pPMat_L, *pPMat_R, sum_L, sum_R;

	int curPattern, curState, iState;
	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		sum_L = 0.0f;
		sum_R = 0.0f;

		pCondlike_L = condlike_L + (curPattern << 2);
		pCondlike_R = condlike_R + (curPattern << 2);
		pPMat_L = PMat_L + curState;
		pPMat_R = PMat_R + curState;
		
		for(iState = 0; iState < 4; iState ++, pPMat_L += 4, pPMat_R += 4){
#ifdef USING_LDG
			sum_L += __ldg(&pCondlike_L[iState]) * pPMat_L[0];
			sum_R += __ldg(&pCondlike_R[iState]) * pPMat_R[0];
#else
			sum_L += pCondlike_L[iState] * pPMat_L[0];
			sum_R += pCondlike_R[iState] * pPMat_R[0];
#endif
		}

		condlike_F[ind] = sum_L * sum_R;
	}
}


__device__
void deviceCondlike_4State_case3_transpose_baseline_version4(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_L, *pCondlike_R, buf_L[4], buf_R[4];

	int iState;
	for(; ind < nSitePattern; ind += nThread){
		pCondlike_L = condlike_L + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

#ifdef USING_LDG
		buf_L[0] = __ldg(&pCondlike_L[0]);
		buf_L[1] = __ldg(&pCondlike_L[1]);
		buf_L[2] = __ldg(&pCondlike_L[2]);
		buf_L[3] = __ldg(&pCondlike_L[3]);

		buf_R[0] = __ldg(&pCondlike_R[0]);
		buf_R[1] = __ldg(&pCondlike_R[1]);
		buf_R[2] = __ldg(&pCondlike_R[2]);
		buf_R[3] = __ldg(&pCondlike_R[3]);
#else
		buf_L[0] = pCondlike_L[0];
		buf_L[1] = pCondlike_L[1];
		buf_L[2] = pCondlike_L[2];
		buf_L[3] = pCondlike_L[3];

		buf_R[0] = pCondlike_R[0];
		buf_R[1] = pCondlike_R[1];
		buf_R[2] = pCondlike_R[2];
		buf_R[3] = pCondlike_R[3];
#endif

		for(iState = 0; iState < 4; iState ++){
			pCondlike_F[iState] = (buf_L[0] * PMat_L[iState] + buf_L[1] * PMat_L[iState + 4] + buf_L[2] * PMat_L[iState + 8] + buf_L[3] * PMat_L[iState + 12]) * (buf_R[0] * PMat_R[iState] + buf_R[1] * PMat_R[iState + 4] + buf_R[2] * PMat_R[iState + 8] + buf_R[3] * PMat_R[iState + 12]);
		}
	}
}


__device__
void deviceCondlike_4State_case3_transpose_baseline_version5(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_L, *pCondlike_R, buf_L[4], buf_R[4], sum_L, sum_R;

	int iState, jState, offset;
	for(; ind < nSitePattern; ind += nThread){
		pCondlike_L = condlike_L + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

#ifdef USING_LDG
		buf_L[0] = __ldg(&pCondlike_L[0]);
		buf_L[1] = __ldg(&pCondlike_L[1]);
		buf_L[2] = __ldg(&pCondlike_L[2]);
		buf_L[3] = __ldg(&pCondlike_L[3]);

		buf_R[0] = __ldg(&pCondlike_R[0]);
		buf_R[1] = __ldg(&pCondlike_R[1]);
		buf_R[2] = __ldg(&pCondlike_R[2]);
		buf_R[3] = __ldg(&pCondlike_R[3]);
#else
		buf_L[0] = pCondlike_L[0];
		buf_L[1] = pCondlike_L[1];
		buf_L[2] = pCondlike_L[2];
		buf_L[3] = pCondlike_L[3];

		buf_R[0] = pCondlike_R[0];
		buf_R[1] = pCondlike_R[1];
		buf_R[2] = pCondlike_R[2];
		buf_R[3] = pCondlike_R[3];
#endif

		for(iState = 0; iState < 4; iState ++){
			sum_L = 0.0f;
			sum_R = 0.0f;

			for(jState = 0, offset = iState; jState < 4; jState ++, offset += 4){
				sum_L += buf_L[jState] * PMat_L[offset];
				sum_R += buf_R[jState] * PMat_R[offset];
			}

			pCondlike_F[iState] = sum_L * sum_R;
		}
	}
}


__device__
void deviceCondlike_4State_case3_transpose_baseline_version6(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_L, *pCondlike_R, buf[4];

	int iState;
	for(; ind < nSitePattern; ind += nThread){
		pCondlike_L = condlike_L + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_L[0]);
		buf[1] = __ldg(&pCondlike_L[1]);
		buf[2] = __ldg(&pCondlike_L[2]);
		buf[3] = __ldg(&pCondlike_L[3]);
#else
		buf[0] = pCondlike_L[0];
		buf[1] = pCondlike_L[1];
		buf[2] = pCondlike_L[2];
		buf[3] = pCondlike_L[3];
#endif

		for(iState = 0; iState < 4; iState ++)
			pCondlike_F[iState] = buf[0] * PMat_L[iState] + buf[1] * PMat_L[iState + 4] + buf[2] * PMat_L[iState + 8] + buf[3] * PMat_L[iState + 12];

#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_R[0]);
		buf[1] = __ldg(&pCondlike_R[1]);
		buf[2] = __ldg(&pCondlike_R[2]);
		buf[3] = __ldg(&pCondlike_R[3]);
#else
		buf[0] = pCondlike_R[0];
		buf[1] = pCondlike_R[1];
		buf[2] = pCondlike_R[2];
		buf[3] = pCondlike_R[3];
#endif

		for(iState = 0; iState < 4; iState ++)
			pCondlike_F[iState] *= buf[0] * PMat_R[iState] + buf[1] * PMat_R[iState + 4] + buf[2] * PMat_R[iState + 8] + buf[3] * PMat_R[iState + 12];
	}
}


__device__
void deviceCondlike_4State_case3_transpose_baseline_version7(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_L, *pCondlike_R, buf[4], sum;

	int iState, jState, offset;
	for(; ind < nSitePattern; ind += nThread){
		pCondlike_L = condlike_L + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_L[0]);
		buf[1] = __ldg(&pCondlike_L[1]);
		buf[2] = __ldg(&pCondlike_L[2]);
		buf[3] = __ldg(&pCondlike_L[3]);
#else
		buf[0] = pCondlike_L[0];
		buf[1] = pCondlike_L[1];
		buf[2] = pCondlike_L[2];
		buf[3] = pCondlike_L[3];
#endif

		for(iState = 0; iState < 4; iState ++){
			sum = 0.0f;

			for(jState = 0, offset = iState; jState < 4; jState ++, offset += 4){
				sum += buf[jState] * PMat_L[offset];
			}

			pCondlike_F[iState] = sum;
		}


#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_R[0]);
		buf[1] = __ldg(&pCondlike_R[1]);
		buf[2] = __ldg(&pCondlike_R[2]);
		buf[3] = __ldg(&pCondlike_R[3]);
#else
		buf[0] = pCondlike_R[0];
		buf[1] = pCondlike_R[1];
		buf[2] = pCondlike_R[2];
		buf[3] = pCondlike_R[3];
#endif

		for(iState = 0; iState < 4; iState ++){
			sum = 0.0f;

			for(jState = 0, offset = iState; jState < 4; jState ++, offset += 4){
				sum += buf[jState] * PMat_R[offset];
			}

			pCondlike_F[iState] *= sum;
		}
	}
}




// Non-transpose version: row * row;
__device__
void deviceCondlike_4State_case3_noTranspose_baseline_version1(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_L, *pCondlike_R, *pPMat_L, *pPMat_R;

	int iState, offset;
	for(; ind < nSitePattern; ind += nThread){
		pCondlike_F = condlike_F + (ind << 2);
		pCondlike_L = condlike_L + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
		
		for(iState = 0, offset = 0; iState < 4; iState ++, offset += 4){
			pPMat_L = PMat_L + offset;
			pPMat_R = PMat_R + offset;

#ifdef USING_LDG
			pCondlike_F[iState] = (__ldg(&pCondlike_L[0]) * pPMat_L[0] + __ldg(&pCondlike_L[1]) * pPMat_L[1] + __ldg(&pCondlike_L[2]) * pPMat_L[2] + __ldg(&pCondlike_L[3]) * pPMat_L[3]) * (__ldg(&pCondlike_R[0]) * pPMat_R[0] + __ldg(&pCondlike_R[1]) * pPMat_R[1] + __ldg(&pCondlike_R[2]) * pPMat_R[2] + __ldg(&pCondlike_R[3]) * pPMat_R[3]);
#else
			pCondlike_F[iState] = (pCondlike_L[0] * pPMat_L[0] + pCondlike_L[1] * pPMat_L[1] + pCondlike_L[2] * pPMat_L[2] + pCondlike_L[3] * pPMat_L[3]) * (pCondlike_R[0] * pPMat_R[0] + pCondlike_R[1] * pPMat_R[1] + pCondlike_R[2] * pPMat_R[2] + pCondlike_R[3] * pPMat_R[3]);
#endif
		}
	}
}



__device__
void deviceCondlike_4State_case3_noTranspose_baseline_version2(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nElement)
{
	CUFlt *pCondlike_L, *pCondlike_R, *pPMat_L, *pPMat_R;

	int curPattern, curState;
	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		pCondlike_L = condlike_L + (curPattern << 2);
		curState <<= 2;
		pCondlike_R = condlike_R + (curPattern << 2);
		
		pPMat_L = PMat_L + curState;
		pPMat_R = PMat_R + curState;

#ifdef USING_LDG
		condlike_F[ind] = (__ldg(&pCondlike_L[0]) * pPMat_L[0] + __ldg(&pCondlike_L[1]) * pPMat_L[1] + __ldg(&pCondlike_L[2]) * pPMat_L[2] + __ldg(&pCondlike_L[3]) * pPMat_L[3]) * (__ldg(&pCondlike_R[0]) * pPMat_R[0] + __ldg(&pCondlike_R[1]) * pPMat_R[1] + __ldg(&pCondlike_R[2]) * pPMat_R[2] + __ldg(&pCondlike_R[3]) * pPMat_R[3]);
#else
		condlike_F[ind] = (pCondlike_L[0] * pPMat_L[0] + pCondlike_L[1] * pPMat_L[1] + pCondlike_L[2] * pPMat_L[2] + pCondlike_L[3] * pPMat_L[3]) * (pCondlike_R[0] * pPMat_R[0] + pCondlike_R[1] * pPMat_R[1] + pCondlike_R[2] * pPMat_R[2] + pCondlike_R[3] * pPMat_R[3]);
#endif
	}
}


__device__
void deviceCondlike_4State_case3_noTranspose_baseline_version3(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nElement)
{
	CUFlt *pCondlike_L, *pCondlike_R, *pPMat_L, *pPMat_R, sum_L, sum_R;

	int curPattern, curState, iState;
	for(; ind < nElement; ind += nThread){
		curPattern = (ind >> 2);
		curState = (ind & 0x3);

		sum_L = 0.0f;
		sum_R = 0.0f;

		pCondlike_L = condlike_L + (curPattern << 2);
		curState <<= 2;
		pCondlike_R = condlike_R + (curPattern << 2);
		
		pPMat_L = PMat_L + curState;
		pPMat_R = PMat_R + curState;

		for(iState = 0; iState < 4; iState ++){
#ifdef USING_LDG
			sum_L += __ldg(&pCondlike_L[iState]) * pPMat_L[iState];
			sum_R += __ldg(&pCondlike_R[iState]) * pPMat_R[iState];
#else
			sum_L += pCondlike_L[iState] * pPMat_L[iState];
			sum_R += pCondlike_R[iState] * pPMat_R[iState];
#endif
		}

		condlike_F[ind] = sum_L * sum_R;
	}
}


__device__
void deviceCondlike_4State_case3_noTranspose_baseline_version4(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_L, *pCondlike_R, *pPMat_L, *pPMat_R, buf_L[4], buf_R[4];

	int iState, offset;
	for(; ind < nSitePattern; ind += nThread){
		pCondlike_L = condlike_L + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

#ifdef USING_LDG
		buf_L[0] = __ldg(&pCondlike_L[0]);
		buf_L[1] = __ldg(&pCondlike_L[1]);
		buf_L[2] = __ldg(&pCondlike_L[2]);
		buf_L[3] = __ldg(&pCondlike_L[3]);

		buf_R[0] = __ldg(&pCondlike_R[0]);
		buf_R[1] = __ldg(&pCondlike_R[1]);
		buf_R[2] = __ldg(&pCondlike_R[2]);
		buf_R[3] = __ldg(&pCondlike_R[3]);
#else
		buf_L[0] = pCondlike_L[0];
		buf_L[1] = pCondlike_L[1];
		buf_L[2] = pCondlike_L[2];
		buf_L[3] = pCondlike_L[3];

		buf_R[0] = pCondlike_R[0];
		buf_R[1] = pCondlike_R[1];
		buf_R[2] = pCondlike_R[2];
		buf_R[3] = pCondlike_R[3];
#endif
		
		for(iState = 0, offset = 0; iState < 4; iState ++, offset += 4){
			pPMat_L = PMat_L + offset;
			pPMat_R = PMat_R + offset;

			pCondlike_F[iState] = (buf_L[0] * pPMat_L[0] + buf_L[1] * pPMat_L[1] + buf_L[2] * pPMat_L[2] + buf_L[3] * pPMat_L[3]) * (buf_R[0] * pPMat_R[0] + buf_R[1] * pPMat_R[1] + buf_R[2] * pPMat_R[2] + buf_R[3] * pPMat_R[3]);
		}
	}
}


__device__
void deviceCondlike_4State_case3_noTranspose_baseline_version5(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_L, *pCondlike_R, *pPMat_L, *pPMat_R, buf_L[4], buf_R[4], sum_L, sum_R;

	int iState, jState, offset;
	for(; ind < nSitePattern; ind += nThread){
		pCondlike_L = condlike_L + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

#ifdef USING_LDG
		buf_L[0] = __ldg(&pCondlike_L[0]);
		buf_L[1] = __ldg(&pCondlike_L[1]);
		buf_L[2] = __ldg(&pCondlike_L[2]);
		buf_L[3] = __ldg(&pCondlike_L[3]);

		buf_R[0] = __ldg(&pCondlike_R[0]);
		buf_R[1] = __ldg(&pCondlike_R[1]);
		buf_R[2] = __ldg(&pCondlike_R[2]);
		buf_R[3] = __ldg(&pCondlike_R[3]);
#else
		buf_L[0] = pCondlike_L[0];
		buf_L[1] = pCondlike_L[1];
		buf_L[2] = pCondlike_L[2];
		buf_L[3] = pCondlike_L[3];

		buf_R[0] = pCondlike_R[0];
		buf_R[1] = pCondlike_R[1];
		buf_R[2] = pCondlike_R[2];
		buf_R[3] = pCondlike_R[3];
#endif
		
		for(iState = 0, offset = 0; iState < 4; iState ++, offset += 4){
			pPMat_L = PMat_L + offset;
			pPMat_R = PMat_R + offset;
			sum_L = 0.0f;
			sum_R = 0.0f;

			for(jState = 0; jState < 4; jState ++){
				sum_L += buf_L[jState] * pPMat_L[jState];
				sum_R += buf_R[jState] * pPMat_R[jState];
			}

			pCondlike_F[iState] = sum_L * sum_R;
		}
	}
}


__device__
void deviceCondlike_4State_case3_noTranspose_baseline_version6(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_L, *pCondlike_R, *pPMat_L, *pPMat_R, buf[4];

	int iState, offset;
	for(; ind < nSitePattern; ind += nThread){
		pCondlike_L = condlike_L + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_L[0]);
		buf[1] = __ldg(&pCondlike_L[1]);
		buf[2] = __ldg(&pCondlike_L[2]);
		buf[3] = __ldg(&pCondlike_L[3]);
#else
		buf[0] = pCondlike_L[0];
		buf[1] = pCondlike_L[1];
		buf[2] = pCondlike_L[2];
		buf[3] = pCondlike_L[3];
#endif
		
		for(iState = 0, offset = 0; iState < 4; iState ++, offset += 4){
			pPMat_L = PMat_L + offset;
			pPMat_R = PMat_R + offset;

			pCondlike_F[iState] = buf[0] * pPMat_L[0] + buf[1] * pPMat_L[1] + buf[2] * pPMat_L[2] + buf[3] * pPMat_L[3];
		}

#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_R[0]);
		buf[1] = __ldg(&pCondlike_R[1]);
		buf[2] = __ldg(&pCondlike_R[2]);
		buf[3] = __ldg(&pCondlike_R[3]);
#else
		buf[0] = pCondlike_R[0];
		buf[1] = pCondlike_R[1];
		buf[2] = pCondlike_R[2];
		buf[3] = pCondlike_R[3];
#endif

		for(iState = 0, offset = 0; iState < 4; iState ++, offset += 4){
			pPMat_L = PMat_L + offset;
			pPMat_R = PMat_R + offset;

			pCondlike_F[iState] *= buf[0] * pPMat_R[0] + buf[1] * pPMat_R[1] + buf[2] * pPMat_R[2] + buf[3] * pPMat_R[3];
		}
	}
}


__device__
void deviceCondlike_4State_case3_noTranspose_baseline_version7(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern)
{
	CUFlt *pCondlike_F, *pCondlike_L, *pCondlike_R, *pPMat, buf[4], sum;

	int iState, jState, offset;
	for(; ind < nSitePattern; ind += nThread){
		pCondlike_L = condlike_L + (ind << 2);
		pCondlike_R = condlike_R + (ind << 2);
		pCondlike_F = condlike_F + (ind << 2);

#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_L[0]);
		buf[1] = __ldg(&pCondlike_L[1]);
		buf[2] = __ldg(&pCondlike_L[2]);
		buf[3] = __ldg(&pCondlike_L[3]);
#else
		buf[0] = pCondlike_L[0];
		buf[1] = pCondlike_L[1];
		buf[2] = pCondlike_L[2];
		buf[3] = pCondlike_L[3];
#endif
		
		for(iState = 0, offset = 0; iState < 4; iState ++, offset += 4){
			pPMat = PMat_L + offset;
			sum = 0.0f;

			for(jState = 0; jState < 4; jState ++){
				sum += buf[jState] * pPMat[jState];
			}

			pCondlike_F[iState] = sum;
		}


#ifdef USING_LDG
		buf[0] = __ldg(&pCondlike_R[0]);
		buf[1] = __ldg(&pCondlike_R[1]);
		buf[2] = __ldg(&pCondlike_R[2]);
		buf[3] = __ldg(&pCondlike_R[3]);
#else
		buf[0] = pCondlike_R[0];
		buf[1] = pCondlike_R[1];
		buf[2] = pCondlike_R[2];
		buf[3] = pCondlike_R[3];
#endif
		
		for(iState = 0, offset = 0; iState < 4; iState ++, offset += 4){
			pPMat = PMat_R + offset;
			sum = 0.0f;

			for(jState = 0; jState < 4; jState ++){
				sum += buf[jState] * pPMat[jState];
			}

			pCondlike_F[iState] *= sum;
		}
	}
}



// calculate condlike of multiple nodes:
// ע�⣺Ӧ����һ��node��һ��eigen decomposition��һ��rate category��Ӧһ��condlike operation�����һ��nNode * nEigenDecomp * nRateCategory��operation;
// ����nState = 4���ԣ�һ��thread����k��site pattern��block dimensionΪm��nSitePattern / m��block����һ��condlike����ļ���;
// ���⣬ע�⣺Լ���ã�CuLCondlikeOp�ĳ�Ա�е�����child���������Ͳ�ͬʱ��child[0]��Ӧ�����Ǽ򵥵����֣����磺����case 2, child[0]Ϊtip state, child[1]Ϊtip condlike; 
// blkIndToOpInd: block index��operation/condlike array index�Ķ�Ӧ; opStartBlkInd: ����ÿ��operation/condlike array����ʼblock��index;
// ���Թ���condlikeOp����Ĵ����У�Ч��������ķ�ʽ���(����shared memory����ÿ��thread�������һ������)��Ҳ���Թ�����shared memory����ֻ��һ��thread���أ�Ч����������ķ�ʽ;

// Transpose version:
// version 1: case 1 use version 1, case 2 uses version 5, case 3 use version 4;
__global__
void kernelCondlike_4State_transpose_baseline_version1(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	int nElem = (sizeof(CuLCondlikeOp) + sizeof(int) - 1) / sizeof(int);
	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_L[16];
	__shared__ CUFlt sh_PMat_R[16];			// TODO: ����unrooted tree��Ҳ�����������ӵ������ô����??? �ָ�������ڵ�???
	
	if(tx < nElem){
		((int *)&sh_condlikeOp)[tx] = ((int*)&(condlikeOp[opInd]))[tx];
	}
	__syncthreads();

	if(tx < 16){
		sh_PMat_L[tx] = PMat[sh_condlikeOp.child_P_offset[0] + tx];
	}
	else if(tx < 32){
		sh_PMat_R[tx - 16] = PMat[sh_condlikeOp.child_P_offset[1] + tx - 16];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	if(1 == curCase){
		deviceCondlike_4State_case1_transpose_baseline_version1(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case2_transpose_baseline_version5(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case3_transpose_baseline_version4(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R,
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else{		// Error
		return;
	}
}


// version 2: case 1 use version 2, case 2 use version 5, case 3 use version 4;
__global__
void kernelCondlike_4State_transpose_baseline_version2(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	int nElem = (sizeof(CuLCondlikeOp) + sizeof(int) - 1) / sizeof(int);
	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_L[16];
	__shared__ CUFlt sh_PMat_R[16];			// TODO: ����unrooted tree��Ҳ�����������ӵ������ô����??? �ָ�������ڵ�???
	
	if(tx < nElem){
		((int *)&sh_condlikeOp)[tx] = ((int*)&(condlikeOp[opInd]))[tx];
	}
	__syncthreads();

	if(tx < 16){
		sh_PMat_L[tx] = PMat[sh_condlikeOp.child_P_offset[0] + tx];
	}
	else if(tx < 32){
		sh_PMat_R[tx - 16] = PMat[sh_condlikeOp.child_P_offset[1] + tx - 16];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	if(1 == curCase){
		deviceCondlike_4State_case1_transpose_baseline_version2(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case2_transpose_baseline_version5(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case3_transpose_baseline_version4(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R,
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else{		// Error
		return;
	}
}


// version 3: case 1 use version 2, case 2 use version 5, case 3 use version 3;
__global__
void kernelCondlike_4State_transpose_baseline_version3(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	int nElem = (sizeof(CuLCondlikeOp) + sizeof(int) - 1) / sizeof(int);
	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_L[16];
	__shared__ CUFlt sh_PMat_R[16];			// TODO: ����unrooted tree��Ҳ�����������ӵ������ô����??? �ָ�������ڵ�???
	
	if(tx < nElem){
		((int *)&sh_condlikeOp)[tx] = ((int*)&(condlikeOp[opInd]))[tx];
	}
	__syncthreads();

	if(tx < 16){
		sh_PMat_L[tx] = PMat[sh_condlikeOp.child_P_offset[0] + tx];
	}
	else if(tx < 32){
		sh_PMat_R[tx - 16] = PMat[sh_condlikeOp.child_P_offset[1] + tx - 16];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	if(1 == curCase){
		deviceCondlike_4State_case1_transpose_baseline_version2(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case2_transpose_baseline_version5(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case3_transpose_baseline_version3(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R,
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else{		// Error
		return;
	}
}


// version 4: case 1 use version 2, case 2 use version 3, case 3 use version 3;
__global__
void kernelCondlike_4State_transpose_baseline_version4(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	int nElem = (sizeof(CuLCondlikeOp) + sizeof(int) - 1) / sizeof(int);
	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_L[16];
	__shared__ CUFlt sh_PMat_R[16];			// TODO: ����unrooted tree��Ҳ�����������ӵ������ô����??? �ָ�������ڵ�???
	
	if(tx < nElem){
		((int *)&sh_condlikeOp)[tx] = ((int*)&(condlikeOp[opInd]))[tx];
	}
	__syncthreads();

	if(tx < 16){
		sh_PMat_L[tx] = PMat[sh_condlikeOp.child_P_offset[0] + tx];
	}
	else if(tx < 32){
		sh_PMat_R[tx - 16] = PMat[sh_condlikeOp.child_P_offset[1] + tx - 16];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	if(1 == curCase){
		deviceCondlike_4State_case1_transpose_baseline_version2(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case2_transpose_baseline_version3(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case3_transpose_baseline_version3(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R,
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else{		// Error
		return;
	}
}



// Non-transpose version:
// version 1: case 1 use version 1, case 2 use version 5, case 3 use version 4;
__global__
void kernelCondlike_4State_noTranspose_baseline_version1(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	int nElem = (sizeof(CuLCondlikeOp) + sizeof(int) - 1) / sizeof(int);
	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_L[16];
	__shared__ CUFlt sh_PMat_R[16];			// TODO: ����unrooted tree��Ҳ�����������ӵ������ô����??? �ָ�������ڵ�???
	
	if(tx < nElem){
		((int *)&sh_condlikeOp)[tx] = ((int*)&(condlikeOp[opInd]))[tx];
	}
	__syncthreads();

	if(tx < 16){
		sh_PMat_L[tx] = PMat[sh_condlikeOp.child_P_offset[0] + tx];
	}
	else if(tx < 32){
		sh_PMat_R[tx - 16] = PMat[sh_condlikeOp.child_P_offset[1] + tx - 16];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	if(1 == curCase){
		deviceCondlike_4State_case1_noTranspose_baseline_version1(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case2_noTranspose_baseline_version5(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case3_noTranspose_baseline_version4(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R,
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else{		// Error
		return;
	}
}



// version 2: case 1 use version 2, case 2 use version 5, case 3 use version 4;
__global__
void kernelCondlike_4State_noTranspose_baseline_version2(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	int nElem = (sizeof(CuLCondlikeOp) + sizeof(int) - 1) / sizeof(int);
	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_L[16];
	__shared__ CUFlt sh_PMat_R[16];			// TODO: ����unrooted tree��Ҳ�����������ӵ������ô����??? �ָ�������ڵ�???
	
	if(tx < nElem){
		((int *)&sh_condlikeOp)[tx] = ((int*)&(condlikeOp[opInd]))[tx];
	}
	__syncthreads();

	if(tx < 16){
		sh_PMat_L[tx] = PMat[sh_condlikeOp.child_P_offset[0] + tx];
	}
	else if(tx < 32){
		sh_PMat_R[tx - 16] = PMat[sh_condlikeOp.child_P_offset[1] + tx - 16];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	if(1 == curCase){
		deviceCondlike_4State_case1_noTranspose_baseline_version2(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case2_noTranspose_baseline_version5(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case3_noTranspose_baseline_version4(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R,
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else{		// Error
		return;
	}
}



// version 3: case 1 use version 2, case 2 use version 5, case 3 use version 3;
__global__
void kernelCondlike_4State_noTranspose_baseline_version3(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	int nElem = (sizeof(CuLCondlikeOp) + sizeof(int) - 1) / sizeof(int);
	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_L[16];
	__shared__ CUFlt sh_PMat_R[16];			// TODO: ����unrooted tree��Ҳ�����������ӵ������ô����??? �ָ�������ڵ�???
	
	if(tx < nElem){
		((int *)&sh_condlikeOp)[tx] = ((int*)&(condlikeOp[opInd]))[tx];
	}
	__syncthreads();

	if(tx < 16){
		sh_PMat_L[tx] = PMat[sh_condlikeOp.child_P_offset[0] + tx];
	}
	else if(tx < 32){
		sh_PMat_R[tx - 16] = PMat[sh_condlikeOp.child_P_offset[1] + tx - 16];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	if(1 == curCase){
		deviceCondlike_4State_case1_noTranspose_baseline_version2(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case2_noTranspose_baseline_version5(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case3_noTranspose_baseline_version3(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R,
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else{		// Error
		return;
	}
}



// version 4: case 1 use version 2, case 2 use version 3, case 3 use version 3;
__global__
void kernelCondlike_4State_noTranspose_baseline_version4(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	int nElem = (sizeof(CuLCondlikeOp) + sizeof(int) - 1) / sizeof(int);
	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_L[16];
	__shared__ CUFlt sh_PMat_R[16];			// TODO: ����unrooted tree��Ҳ�����������ӵ������ô����??? �ָ�������ڵ�???
	
	if(tx < nElem){
		((int *)&sh_condlikeOp)[tx] = ((int*)&(condlikeOp[opInd]))[tx];
	}
	__syncthreads();

	if(tx < 16){
		sh_PMat_L[tx] = PMat[sh_condlikeOp.child_P_offset[0] + tx];
	}
	else if(tx < 32){
		sh_PMat_R[tx - 16] = PMat[sh_condlikeOp.child_P_offset[1] + tx - 16];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + tx;

	if(1 == curCase){
		deviceCondlike_4State_case1_noTranspose_baseline_version2(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case2_noTranspose_baseline_version3(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_4State_case3_noTranspose_baseline_version3(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R,
									sh_PMat_L, 
									sh_PMat_R,
									ind,
									nThreadPerArray,
									(nSitePattern << 2));
	}
	else{		// Error
		return;
	}
}



// For nState = 20:
// case 1: both children are tip states; case 2: one child is tip state, the other is tip/internal condlike; case 3: both children are tip/internal condlike;
// Transpose version:
// Case 1 & transpose version: row * col;
// nThread: һ���ж��ٸ�thread�����condlike
// version 1 of case 1: ÿ��thread����ͬsite pattern��ĳ��state;
__device__
void deviceCondlike_20State_case1_transpose_baseline(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread, int nSitePattern)
{
	int offset_L, offset_R, curSite, curState, nElement = nSitePattern * 20;

	for(; ind < nElement; ind += nThread){
		curSite = ind / 20;
		curState = ind % 20;

		offset_L = tipState_L[curSite] * 20;
		offset_R = tipState_R[curSite] * 20;

		condlike_F[ind] = PMat_L[curState + offset_L] * PMat_R[curState + offset_R];
	}
}



// version 2 of case 2: �ֿ�˷� + ʹ��shared memory����condlike��PMat�ķֿ�;
__device__
void deviceCondlike_20State_case2_transpose_baseline(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt sh_condlike_R[][TILE_SIZE_20STATE_CONDLIKE_BASELINE], CUFlt sh_PMat_R[][20], CUFlt *buf, const int tx, const int ty)
{
	int ind1 = ty * 20 + tx, ind2 = ty * blockDim.x + tx, nIteration = 20 / TILE_SIZE_20STATE_CONDLIKE_BASELINE, PMat_offset = TILE_SIZE_20STATE_CONDLIKE_BASELINE * 20, itr, iElem;
	int offset1 = tx + TILE_SIZE_20STATE_CONDLIKE_BASELINE, offset2 = offset1 + TILE_SIZE_20STATE_CONDLIKE_BASELINE , offset3 = offset1 + 2 * TILE_SIZE_20STATE_CONDLIKE_BASELINE, offset4 = offset1 + 3 * TILE_SIZE_20STATE_CONDLIKE_BASELINE;
	int nElement = TILE_SIZE_20STATE_CONDLIKE_BASELINE * 20, indx = ind2 / 20, indy = ind2 % 20;
	
	for(itr = 0; itr < nIteration; itr ++, condlike_R += TILE_SIZE_20STATE_CONDLIKE_BASELINE, PMat_R += PMat_offset){
#ifdef USING_LDG
		sh_condlike_R[ty][tx] = __ldg(&condlike_R[ind1]);
#else
		sh_condlike_R[ty][tx] = condlike_R[ind1];
#endif

		if(ind2 < nElement){
#ifdef USING_LDG
			sh_PMat_R[indx][indy] = __ldg(&PMat_R[ind2]);
#else
			sh_PMat_R[indx][indy] = PMat_R[ind2];
#endif
		}
		__syncthreads();

		for(iElem = 0; iElem < TILE_SIZE_20STATE_CONDLIKE_BASELINE; iElem ++){
			buf[0] += sh_condlike_R[ty][iElem] * sh_PMat_R[iElem][tx];
			buf[1] += sh_condlike_R[ty][iElem] * sh_PMat_R[iElem][offset1];
			buf[2] += sh_condlike_R[ty][iElem] * sh_PMat_R[iElem][offset2];
			buf[3] += sh_condlike_R[ty][iElem] * sh_PMat_R[iElem][offset3];
			buf[4] += sh_condlike_R[ty][iElem] * sh_PMat_R[iElem][offset4];
		}
		__syncthreads();
	}

#ifdef USING_LDG
	PMat_L += __ldg(&tipState_L[ty]) * 20;
#else
	PMat_L += tipState_L[ty] * 20;
#endif
	ind2 = ty * 20;

	condlike_F += ind2;
#ifdef USING_LDG
	condlike_F[tx] = buf[0] * __ldg(&PMat_L[tx]);
	condlike_F[offset1] = buf[1] * __ldg(&PMat_L[offset1]);
	condlike_F[offset2] = buf[2] * __ldg(&PMat_L[offset2]);
	condlike_F[offset3] = buf[3] * __ldg(&PMat_L[offset3]);
	condlike_F[offset4] = buf[4] * __ldg(&PMat_L[offset4]);
#else
	condlike_F[tx] = buf[0] * PMat_L[tx];
	condlike_F[offset1] = buf[1] * PMat_L[offset1];
	condlike_F[offset2] = buf[2] * PMat_L[offset2];
	condlike_F[offset3] = buf[3] * PMat_L[offset3];
	condlike_F[offset4] = buf[4] * PMat_L[offset4];
#endif
}


// version 3 of case 3: �������ӹ���һ��shared memory;
__device__
void deviceCondlike_20State_case3_transpose_baseline(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt sh_condlike[][TILE_SIZE_20STATE_CONDLIKE_BASELINE], CUFlt sh_PMat[][20], CUFlt *buf, const int tx, const int ty)
{
	int ind1 = ty * 20 + tx, ind2 = ty * blockDim.x + tx, nIteration = 20 / TILE_SIZE_20STATE_CONDLIKE_BASELINE, PMat_offset = TILE_SIZE_20STATE_CONDLIKE_BASELINE * 20, itr, iElem;
	int offset1 = tx + TILE_SIZE_20STATE_CONDLIKE_BASELINE, offset2 = offset1 + TILE_SIZE_20STATE_CONDLIKE_BASELINE , offset3 = offset1 + 2 * TILE_SIZE_20STATE_CONDLIKE_BASELINE, offset4 = offset1 + 3 * TILE_SIZE_20STATE_CONDLIKE_BASELINE;
	int nElement = TILE_SIZE_20STATE_CONDLIKE_BASELINE * 20, indx = ind2 / 20, indy = ind2 % 20;

	// For left child:
	for(itr = 0; itr < nIteration; itr ++){
#ifdef USING_LDG
		sh_condlike[ty][tx] = __ldg(&condlike_L[ind1]);
#else
		sh_condlike[ty][tx] = condlike_L[ind1];
#endif
		if(ind2 < nElement){
#ifdef USING_LDG
			sh_PMat[indx][indy] = __ldg(&PMat_L[ind2]);
#else
			sh_PMat[indx][indy] = PMat_L[ind2];
#endif
		}
		__syncthreads();

		for(iElem = 0; iElem < TILE_SIZE_20STATE_CONDLIKE_BASELINE; iElem ++){
			buf[0] += sh_condlike[ty][iElem] * sh_PMat[iElem][tx];
			buf[1] += sh_condlike[ty][iElem] * sh_PMat[iElem][offset1];
			buf[2] += sh_condlike[ty][iElem] * sh_PMat[iElem][offset2];
			buf[3] += sh_condlike[ty][iElem] * sh_PMat[iElem][offset3];
			buf[4] += sh_condlike[ty][iElem] * sh_PMat[iElem][offset4];
		}
		__syncthreads();

		condlike_L += TILE_SIZE_20STATE_CONDLIKE_BASELINE;

		PMat_L += PMat_offset;
	}

	// For right child:
	for(itr = 0; itr < nIteration; itr ++){
#ifdef USING_LDG
		sh_condlike[ty][tx] = __ldg(&condlike_R[ind1]);
#else
		sh_condlike[ty][tx] = condlike_R[ind1];
#endif
		if(ind2 < nElement){
#ifdef USING_LDG
			sh_PMat[indx][indy] = __ldg(&PMat_R[ind2]);
#else
			sh_PMat[indx][indy] = PMat_R[ind2];
#endif
		}
		__syncthreads();

		for(iElem = 0; iElem < TILE_SIZE_20STATE_CONDLIKE_BASELINE; iElem ++){
			buf[5] += sh_condlike[ty][iElem] * sh_PMat[iElem][tx];
			buf[6] += sh_condlike[ty][iElem] * sh_PMat[iElem][offset1];
			buf[7] += sh_condlike[ty][iElem] * sh_PMat[iElem][offset2];
			buf[8] += sh_condlike[ty][iElem] * sh_PMat[iElem][offset3];
			buf[9] += sh_condlike[ty][iElem] * sh_PMat[iElem][offset4];
		}
		__syncthreads();

		condlike_R += TILE_SIZE_20STATE_CONDLIKE_BASELINE;

		PMat_R += PMat_offset;
	}

	ind2 = ty * 20;
	condlike_F += ind2;

	condlike_F[tx] = buf[0] * buf[5];
	condlike_F[offset1] = buf[1] * buf[6];
	condlike_F[offset2] = buf[2] * buf[7];
	condlike_F[offset3] = buf[3] * buf[8];
	condlike_F[offset4] = buf[4] * buf[9];
}


// Non-transpose version:
// Case 1 & non-transpose version: row * row;
__device__
void deviceCondlike_20State_case1_noTranspose_baseline(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread, int nSitePattern)
{
	int offset_L, offset_R, curSite, curState, nElement = nSitePattern * 20;

	for(; ind < nElement; ind += nThread){
		curSite = ind / 20;
		curState = ind % 20;

#ifdef USING_LDG
		offset_L = __ldg(&tipState_L[curSite]);
#else
		offset_L = tipState_L[curSite];
#endif
		curState *= 20;

#ifdef USING_LDG
		offset_R = __ldg(&tipState_R[curSite]);
#else
		offset_R = tipState_R[curSite];
#endif

#ifdef USING_LDG
		condlike_F[ind] = __ldg(&PMat_L[curState + offset_L]) * __ldg(&PMat_R[curState + offset_R]);
#else
		condlike_F[ind] = PMat_L[curState + offset_L] * PMat_R[curState + offset_R];
#endif
	}
}


// Case 2 & non-transpose version: row * row;
__device__
void deviceCondlike_20State_case2_noTranspose_baseline(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern, const int nState)
{
	CUFlt *pCondlike_R, *pPMat_R, sum_L, sum_R;
	int offset_L, curSite, curState, iState, nElement = nSitePattern * 20;

	for(; ind < nElement; ind += nThread){
		curSite = ind / 20;
		curState = ind % 20;
#ifdef USING_LDG
		offset_L = __ldg(&tipState_L[curSite]);
#else
		offset_L = tipState_L[curSite];
#endif
		curState *= 20;

		pCondlike_R = condlike_R + curSite * 20;
		pPMat_R = PMat_R + curState;
		
		sum_R = 0.0f;
#ifdef USING_LDG
		sum_L = __ldg(&PMat_L[curState + offset_L]);
#else
		sum_L = PMat_L[curState + offset_L];
#endif

		for(iState = 0; iState < nState; iState ++){
#ifdef USING_LDG
			sum_R += __ldg(&pCondlike_R[iState]) * __ldg(&pPMat_R[iState]);
#else
			sum_R += pCondlike_R[iState] * pPMat_R[iState];
#endif
		}

		condlike_F[ind] = sum_L * sum_R;
	}
}



// Case 3 & non-transpose version: row * row;
__device__
void deviceCondlike_20State_case3_noTranspose_baseline(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern, const int nState)
{
	CUFlt *pCondlike_L, *pCondlike_R, *pPMat_L, *pPMat_R, sum_L, sum_R;
	int curSite, curState, iState, nElement = nSitePattern * 20;

	for(; ind < nElement; ind += nThread){
		curSite = ind / 20;
		curState = ind % 20;

		curSite *= 20;
		curState *= 20;

		pCondlike_L = condlike_L + curSite;
		pCondlike_R = condlike_R + curSite;
		pPMat_L = PMat_L + curState;
		pPMat_R = PMat_R + curState;
		
		sum_L = 0.0f;
		sum_R = 0.0f;

		for(iState = 0; iState < nState; iState ++){
#ifdef USING_LDG
			sum_L += __ldg(&pCondlike_L[iState]) * __ldg(&pPMat_L[iState]);
			sum_R += __ldg(&pCondlike_R[iState]) * __ldg(&pPMat_R[iState]);
#else
			sum_L += pCondlike_L[iState] * pPMat_L[iState];
			sum_R += pCondlike_R[iState] * pPMat_R[iState];
#endif
		}

		condlike_F[ind] = sum_L * sum_R;
	}
}


// For nState = 20, each thread is responsible for k states' condlike;
// block dimension is: (m, l), m * k = 20, ÿ��block����l��site pattern��nSitePattern / l��block����һ��condlike;
// nThreadPerArray: ����һ��condlike array(һ��node��һ��eigen decomposition��һ��rate category��condlike)��thread��Ŀ
// Transpose version:
// ���Թ���condlikeOp����Ĵ���������shared memory��Ч���������shared memory��Ҳ���Թ�ÿ��thread�������condlikeOp��һ��int���ݵ�shared memory�У�����Ч��Ҳ��������ķ�ʽ(��һ���߳�ֱ�Ӽ���ȫ������);
// version 6: �������ӵ�PMat/condlike����ͬһ��shared memory�ռ䣬shared memory������Ϊversion 3��һ�룬�����ڱ����м����ļĴ���(buf)������û�б䣬case 3�������������μ��㣬������ͬʱ���㣬case 1/2ͬversion 3;
__global__
void kernelCondlike_20State_transpose_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern, const int nState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int opInd = blkIndToOpInd[bx];
	int ind = ty * blockDim.x + tx;

	__shared__ CuLCondlikeOp sh_condlikeOp;
	
	__shared__ CUFlt sh_PMat[TILE_SIZE_20STATE_CONDLIKE_BASELINE][20];
	__shared__ CUFlt sh_condlike[N_SITE_PER_BLOCK_CONDLIKE_20STATE_BASELINE][TILE_SIZE_20STATE_CONDLIKE_BASELINE];
	

	CUFlt buf[2 * N_STATE_PER_THREAD_CONDLIKE_20STATE_BASELINE] = {0};

	if(0 == ind){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int blkOffset = bx - startBlkInd;
	ind += blkOffset * nThreadPerBlock;

	int tipStateOffset = (blkOffset * N_SITE_PER_BLOCK_CONDLIKE_20STATE_BASELINE);

	CUFlt *pPMat_L = PMat + sh_condlikeOp.child_P_offset[0], *pPMat_R = PMat + sh_condlikeOp.child_P_offset[1];

	if(1 == curCase){
		deviceCondlike_20State_case1_transpose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									pPMat_L, 
									pPMat_R,
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1] + (tipStateOffset * 20);
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1] + (tipStateOffset * 20);

		deviceCondlike_20State_case2_transpose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset + (tipStateOffset * 20), 
									tipState + sh_condlikeOp.child_condlike_offset[0] + tipStateOffset,
									pCondlike_R, 
									pPMat_L, 
									pPMat_R,
									sh_condlike,
									sh_PMat,
									buf,
									tx,
									ty);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		tipStateOffset *= 20;

		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_20State_case3_transpose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset + tipStateOffset, 
									pCondlike_L + tipStateOffset,
									pCondlike_R + tipStateOffset, 
									pPMat_L, 
									pPMat_R,
									sh_condlike,
									sh_PMat,
									buf,
									tx,
									ty);
	}
	else{		// Error
		return;
	}
}


// Non-transpose version:
__global__
void kernelCondlike_20State_noTranspose_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern, const int nState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];
	int ind = threadIdx.y * blockDim.x + tx;

	__shared__ CuLCondlikeOp sh_condlikeOp;
	
	if(0 == ind){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	ind += (bx - startBlkInd) * nThreadPerBlock;

	if(1 == curCase){
		deviceCondlike_20State_case1_noTranspose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_20State_case2_noTranspose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nState);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_20State_case3_noTranspose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R, 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nState);
	}
	else{		// Error
		return;
	}
}



// For nState = 64:
// case 1: both children are tip states;
// nThread: һ���ж��ٸ�thread�����condlike
// Transpose version:
__device__
void deviceCondlike_64State_case1_transpose_baseline(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread, int nSitePattern)
{
	int offset_L, offset_R, curSite, curState;

	for(; ind < nSitePattern * 64; ind += nThread){
		curSite = ind / 64;
		curState = ind % 64;

		offset_L = tipState_L[curSite] * 64;
		offset_R = tipState_R[curSite] * 64;

		//pCondlike_F = condlike_F + curSite * 64;

		condlike_F[ind] = PMat_L[curState + offset_L] * PMat_R[curState + offset_R];
	}
}

// Non-transpose version:
__device__
void deviceCondlike_64State_case1_noTranspose_baseline(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread, int nSitePattern)
{
	int offset_L, offset_R, curSite, curState;

	for(; ind < nSitePattern * 64; ind += nThread){
		curSite = ind / 64;
		curState = ind % 64;

		offset_L = tipState_L[curSite];
		offset_R = tipState_R[curSite];

		//pCondlike_F = condlike_F + curSite * 64;

		condlike_F[ind] = PMat_L[curState * 64 + offset_L] * PMat_R[curState * 64 + offset_R];
	}
}


// case 2: one child is tip state, the other child is tip condlike:
// TODO: ��������state�Ƿ���Ҫ���㣿����Ҳ��curState > nState��ʱ�򣬶�Ӧ��thread��ֱ�����û�����Ȼ���㣬֮�����site lnLʱ���ø�ֵ???
// Transpose version:
__device__
void deviceCondlike_64State_case2_transpose_baseline(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern, const int nState)
{
	CUFlt *pCondlike_R, sum_L, sum_R;
	int offset_L, curSite, curState;

	for(; ind < nSitePattern * 64; ind += nThread){
		curSite = ind / 64;
		curState = ind % 64;
		offset_L = tipState_L[curSite] * 64;

		//pCondlike_F = condlike_F + curSite * 64;
		pCondlike_R = condlike_R + curSite * 64;
		
		sum_R = 0.0f;
		sum_L = PMat_L[curState + offset_L];

		for(int iState = 0; iState < nState; iState ++)
			sum_R += pCondlike_R[iState] * PMat_R[curState + iState * 64];

		condlike_F[ind] = sum_L * sum_R;
	}
}


// Non-transpose version:
__device__
void deviceCondlike_64State_case2_noTranspose_baseline(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern, const int nState)
{
	CUFlt *pCondlike_R, sum_L, sum_R;
	int offset_L, curSite, curState;

	for(; ind < nSitePattern * 64; ind += nThread){
		curSite = ind / 64;
		curState = ind % 64;
		offset_L = tipState_L[curSite];

		//pCondlike_F = condlike_F + curSite * 64;
		pCondlike_R = condlike_R + curSite * 64;
		
		sum_R = 0.0f;
		sum_L = PMat_L[curState * 64 + offset_L];

		for(int iState = 0; iState < nState; iState ++)
			sum_R += pCondlike_R[iState] * PMat_R[curState * 64 + iState];

		condlike_F[ind] = sum_L * sum_R;
	}
}


// case 3: both children are condlike:
// Transpose version:
__device__
void deviceCondlike_64State_case3_transpose_baseline(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern, const int nState)
{
	CUFlt *pCondlike_L, *pCondlike_R, sum_L, sum_R;
	int curSite, curState;

	for(; ind < nSitePattern * 64; ind += nThread){
		curSite = ind / 64;
		curState = ind % 64;

		//pCondlike_F = condlike_F + curSite * 64;
		pCondlike_L = condlike_L + curSite * 64;
		pCondlike_R = condlike_R + curSite * 64;
		
		sum_L = 0.0f;
		sum_R = 0.0f;
		for(int iState = 0; iState < nState; iState ++){
			sum_L += pCondlike_L[iState] * PMat_L[curState + iState * 64];
			sum_R += pCondlike_R[iState] * PMat_R[curState + iState * 64];
		}

		condlike_F[ind] = sum_L * sum_R;
	}
}


// Non-transpose version:
__device__
void deviceCondlike_64State_case3_noTranspose_baseline(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThread, const int nSitePattern, const int nState)
{
	CUFlt *pCondlike_L, *pCondlike_R, sum_L, sum_R;
	int curSite, curState;

	for(; ind < nSitePattern * 64; ind += nThread){
		curSite = ind / 64;
		curState = ind % 64;

		//pCondlike_F = condlike_F + curSite * 64;
		pCondlike_L = condlike_L + curSite * 64;
		pCondlike_R = condlike_R + curSite * 64;
		
		sum_L = 0.0f;
		sum_R = 0.0f;
		for(int iState = 0; iState < nState; iState ++){
			sum_L += pCondlike_L[iState] * PMat_L[curState * 64 + iState];
			sum_R += pCondlike_R[iState] * PMat_R[curState * 64 + iState];
		}

		condlike_F[ind] = sum_L * sum_R;
	}
}



// For nState = 64, ����ÿ��thread����k��state��condlike�ļ��㣬block dimensionΪ: (m, l)��m * k = 64��Ҳ��threadIdx.y��ͬ��m��thread����һ��site pattern��ÿ��block����l��site pattern��ÿnSitePattern / l��block����һ��condlike array:
// Transpose version:
__global__
void kernelCondlike_64State_transpose_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern, const int nState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	__shared__ CuLCondlikeOp sh_condlikeOp;
	
	/*
	int nElem = (sizeof(CuLCondlikeOp) + sizeof(int) - 1) / sizeof(int);
	if(tx < nElem)
		((int *)sh_condlikeOp)[tx] = ((int *)(&(condlikeOp[opInd])))[tx];
	*/
	if(tx == 0){
		sh_condlikeOp.nChild = condlikeOp[opInd].nChild;
		sh_condlikeOp.whichCase = condlikeOp[opInd].whichCase;
		sh_condlikeOp.father_condlike_offset = condlikeOp[opInd].father_condlike_offset;
		for(int i = 0; i < 2; i ++){
			sh_condlikeOp.isChildTip[i] = condlikeOp[opInd].isChildTip[i];
			sh_condlikeOp.child_P_offset[i] = condlikeOp[opInd].child_P_offset[i];
			sh_condlikeOp.child_condlike_offset[i] = condlikeOp[opInd].child_condlike_offset[i];
		}
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + threadIdx.y * blockDim.x + tx;

	if(1 == curCase){
		deviceCondlike_64State_case1_transpose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_64State_case2_transpose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nState);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_64State_case3_transpose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R, 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nState);
	}
	else{		// Error
		return;
	}
}


// Non-transpose version:
__global__
void kernelCondlike_64State_noTranspose_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern, const int nState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	__shared__ CuLCondlikeOp sh_condlikeOp;
	
	/*
	int nElem = (sizeof(CuLCondlikeOp) + sizeof(int) - 1) / sizeof(int);
	if(tx < nElem)
		((int *)sh_condlikeOp)[tx] = ((int *)(&(condlikeOp[opInd])))[tx];
	*/
	if(tx == 0){
		sh_condlikeOp.nChild = condlikeOp[opInd].nChild;
		sh_condlikeOp.whichCase = condlikeOp[opInd].whichCase;
		sh_condlikeOp.father_condlike_offset = condlikeOp[opInd].father_condlike_offset;
		for(int i = 0; i < 2; i ++){
			sh_condlikeOp.isChildTip[i] = condlikeOp[opInd].isChildTip[i];
			sh_condlikeOp.child_P_offset[i] = condlikeOp[opInd].child_P_offset[i];
			sh_condlikeOp.child_condlike_offset[i] = condlikeOp[opInd].child_condlike_offset[i];
		}
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + threadIdx.y * blockDim.x + tx;

	if(1 == curCase){
		deviceCondlike_64State_case1_noTranspose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_64State_case2_noTranspose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nState);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_64State_case3_noTranspose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R, 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nState);
	}
	else{		// Error
		return;
	}
}



// For nState != 4 / 20 / 61:
// case 1: both children are tip states;
// nThread: һ���ж��ٸ�thread�����condlike
// Transpose version:
__device__
void deviceCondlike_xState_case1_transpose_baseline(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThreadPerArray, int nSitePattern, int nPaddedState)
{
	int nElement = nSitePattern * nPaddedState, offset_L, offset_R, curSite, curState;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		offset_L = tipState_L[curSite] * nPaddedState;
		offset_R = tipState_R[curSite] * nPaddedState;

		condlike_F[ind] = PMat_L[curState + offset_L] * PMat_R[curState + offset_R];
	}
}

// Non-transpose version:
__device__
void deviceCondlike_xState_case1_noTranspose_baseline(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThreadPerArray, int nSitePattern, int nPaddedState)
{
	int nElement = nSitePattern * nPaddedState, offset_L, offset_R, curSite, curState;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		offset_L = tipState_L[curSite];
		offset_R = tipState_R[curSite];

		condlike_F[ind] = PMat_L[curState * nPaddedState + offset_L] * PMat_R[curState * nPaddedState + offset_R];
	}
}


// case 2: one child is tip state, the other child is tip condlike:
// TODO: ��������state�Ƿ���Ҫ���㣿����Ҳ��curState > nState��ʱ�򣬶�Ӧ��thread��ֱ�����û�����Ȼ���㣬֮�����site lnLʱ���ø�ֵ???
// Transpose version:
__device__
void deviceCondlike_xState_case2_transpose_baseline(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThreadPerArray, const int nSitePattern, const int nState, const int nPaddedState)
{
	int nElement = nSitePattern * nPaddedState, offset_L, curSite, curState, iState;
	CUFlt *pCondlike_R, sum_L, sum_R;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;
		offset_L = tipState_L[curSite] * nPaddedState;

		pCondlike_R = condlike_R + curSite * nPaddedState;
		
		sum_R = 0.0f;
		sum_L = PMat_L[curState + offset_L];

		for(iState = 0; iState < nState; iState ++)
			sum_R += pCondlike_R[iState] * PMat_R[curState + iState * nPaddedState];

		condlike_F[ind] = sum_L * sum_R;
	}
}


// Non-transpose version:
__device__
void deviceCondlike_xState_case2_noTranspose_baseline(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThreadPerArray, const int nSitePattern, const int nState, const int nPaddedState)
{
	int nElement = nSitePattern * nPaddedState, offset_L, curSite, curState, iState;
	CUFlt *pCondlike_R, sum_L, sum_R;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;
		offset_L = tipState_L[curSite];

		//pCondlike_F = condlike_F + curSite * 64;
		pCondlike_R = condlike_R + curSite * nPaddedState;
		
		curState *= nPaddedState;
		sum_R = 0.0f;
		sum_L = PMat_L[curState + offset_L];

		for(iState = 0; iState < nState; iState ++)
			sum_R += pCondlike_R[iState] * PMat_R[curState + iState];

		condlike_F[ind] = sum_L * sum_R;
	}
}


// case 3: both children are condlike:
// Transpose version:
__device__
void deviceCondlike_xState_case3_transpose_baseline(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThreadPerArray, const int nSitePattern, const int nState, const int nPaddedState)
{
	int curSite, curState, iState, nElement = nSitePattern * nPaddedState;
	CUFlt *pCondlike_L, *pCondlike_R, sum_L, sum_R;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		//pCondlike_F = condlike_F + curSite * 64;
		pCondlike_L = condlike_L + curSite * nPaddedState;
		pCondlike_R = condlike_R + curSite * nPaddedState;
		
		sum_L = 0.0f;
		sum_R = 0.0f;
		for(iState = 0; iState < nState; iState ++){
			sum_L += pCondlike_L[iState] * PMat_L[curState + iState * nPaddedState];
			sum_R += pCondlike_R[iState] * PMat_R[curState + iState * nPaddedState];
		}

		condlike_F[ind] = sum_L * sum_R;
	}
}


// Non-transpose version:
__device__
void deviceCondlike_xState_case3_noTranspose_baseline(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, const int nThreadPerArray, const int nSitePattern, const int nState, const int nPaddedState)
{
	CUFlt *pCondlike_L, *pCondlike_R, sum_L, sum_R;
	int nElement = nSitePattern * nPaddedState, curSite, curState, iState;

	for(; ind < nElement; ind += nThreadPerArray){
		curSite = ind / nPaddedState;
		curState = ind % nPaddedState;

		//pCondlike_F = condlike_F + curSite * 64;
		pCondlike_L = condlike_L + curSite * nPaddedState;
		pCondlike_R = condlike_R + curSite * nPaddedState;
		
		curState *= nPaddedState;
		sum_L = 0.0f;
		sum_R = 0.0f;
		for(iState = 0; iState < nState; iState ++){
			sum_L += pCondlike_L[iState] * PMat_L[curState + iState];
			sum_R += pCondlike_R[iState] * PMat_R[curState + iState];
		}

		condlike_F[ind] = sum_L * sum_R;
	}
}


// Transpose version:
__global__
void kernelCondlike_xState_transpose_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern, const int nState, const int nPaddedState)
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
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + threadIdx.y * blockDim.x + tx;

	if(1 == curCase){
		deviceCondlike_xState_case1_transpose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nPaddedState);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_xState_case2_transpose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nState,
									nPaddedState);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_xState_case3_transpose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R, 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nState,
									nPaddedState);
	}
	else{		// Error
		return;
	}
}


// Padded version of xState:
// ����kernel����nState��padΪ8�ı�������nSitePattern��padΪN
// Ŀǰ��xState���Ż�Ϊ������nPaddedState = 8 / 16 / 24�ĸ��Զ�Ӧһ��kernel���������������block dimensionΪ: (2, 32)��Ҳ��ÿ��thread block����32��site pattern; (��˽�site pattern padΪ32�ı���);
// ����nPaddedState > 24(Ҳ��nPaddedState >= 32)�������ÿ��thread����һ����Ŀ(16/8)��state��thread block dimensionΪ: (nPaddedState / nElemPerThread, nPatternPerBlock); ����nElemPerThreadΪ8, ����������������nPaddedState <= 56, ��nPatternPerBlockΪ32������Ϊ16
// ��ˣ���nPaddedState <= 64����site pattern��ĿpadΪ32�ı���������Ϊ16�ı���;(֮����ԱȽϣ������ж�padΪ16�ı����뽫���ж�padΪ32�ı�����Ч���ĸ�����);
__device__
void deviceCondlike_xState_case1_transpose_baseline_8State(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE << 3);
	int curPattern, curState;
	
	condlike_F += ind;

	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = (ind >> 3);
		curState = (ind & 0x7);

#ifdef USING_LDG
		condlike_F[0] = PMat_L[(__ldg(&tipState_L[curPattern]) << 3) + curState] * PMat_R[(__ldg(&tipState_R[curPattern]) << 3) + curState];
#else
		condlike_F[0] = PMat_L[(tipState_L[curPattern] << 3) + curState] * PMat_R[(tipState_R[curPattern] << 3) + curState];
#endif
	}
}


__device__
void deviceCondlike_xState_case2_transpose_baseline_8State(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread, int nState)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE << 3), curPattern, curState, iState;
	CUFlt *pCondlike_R, *pPMat_R, sum_R;

	condlike_F += ind;
	for(; ind < nElemToCalc; ind += nThread, condlike_F += nThread){
		curPattern = (ind >> 3);
		curState = (ind & 0x7);

		pPMat_R = PMat_R + curState;
		pCondlike_R = condlike_R + (curPattern << 3);
		sum_R = 0.0f;

		for(iState = 0; iState < nState; iState ++, pCondlike_R ++, pPMat_R += 8){
			sum_R += pPMat_R[0] * pCondlike_R[0];
		}

#ifdef USING_LDG
		condlike_F[0] = PMat_L[(__ldg(&tipState_L[curPattern]) << 3) + curState] * sum_R;
#else
		condlike_F[0] = PMat_L[(tipState_L[curPattern] << 3) + curState] * sum_R;
#endif
	}
}


// version 1 of case 3: �������ӵ�condlike����һ��shared memory����ʹ��register�������ӵ�condlike * PMat�ļ���������Ҫдcondlike_F 2�Σ���1��;
__device__
void deviceCondlike_xState_case3_transpose_baseline_8State(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt *sh_condlike_S, int ind, int nThread, int nState)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE << 3), curPattern, curState, iState, curInd;
	CUFlt *pCondlike_F, *pCondlike_S, *pPMat_S, curSum;

	// For the left child:
	for(curInd = ind; curInd < nElemToCalc; curInd += nThread){
#ifdef USING_LDG
		sh_condlike_S[curInd] = __ldg(&condlike_L[curInd]);
#else
		sh_condlike_S[curInd] = condlike_L[curInd];
#endif
	}

	__syncthreads();

	pCondlike_F = condlike_F + ind;
	for(curInd = ind; curInd < nElemToCalc; curInd += nThread, pCondlike_F += nThread){
		curPattern = (curInd >> 3);
		curState = (curInd & 0x7);

		pPMat_S = PMat_L + curState;
		pCondlike_S = sh_condlike_S + (curPattern << 3);
		curSum = 0.0f;

		for(iState = 0; iState < nState; iState ++, pPMat_S += 8, pCondlike_S ++)
			curSum += pPMat_S[0] * pCondlike_S[0];

		pCondlike_F[0] = curSum;
	}
	__syncthreads();

	// For the right child:
	for(curInd = ind; curInd < nElemToCalc; curInd += nThread){
#ifdef USING_LDG
		sh_condlike_S[curInd] = __ldg(&condlike_R[curInd]);
#else
		sh_condlike_S[curInd] = condlike_R[curInd];
#endif
	}

	__syncthreads();

	pCondlike_F = condlike_F + ind;
	for(curInd = ind; curInd < nElemToCalc; curInd += nThread, pCondlike_F += nThread){
		curPattern = (curInd >> 3);
		curState = (curInd & 0x7);

		pPMat_S = PMat_R + curState;
		pCondlike_S = sh_condlike_S + (curPattern << 3);
		curSum = 0.0f;

		for(iState = 0; iState < nState; iState ++, pPMat_S += 8, pCondlike_S ++)
			curSum += pPMat_S[0] * pCondlike_S[0];

		pCondlike_F[0] *= curSum;
	}
}



// For nPaddedState == 8: use shared memory to store the entire PMat matrix and the k site patterns' condlike the thread block is responsible for;
// ����һ�ַ������������ӹ���һ��PMat��shared memory�Լ�condlike��shared memory, ������Ҫ�üĴ��������м���
// version 1: �������ӵ�condlike����һ��shared memory����ʹ�üĴ����������ӵ�condlike * PMat��ֵ����Ҫ��condlike_F 1�Σ�д2��;
// version 1������version������Ч����õģ����ʹ��version1;
__global__
void kernelCondlike_xState_transpose_baseline_8State(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int ind = tx + threadIdx.y * BLOCK_DIMENSION_X_CONDLIKE_XSTATE_8_BASELINE;
	int opInd = blkIndToOpInd[bx];
	int nThread = BLOCK_DIMENSION_X_CONDLIKE_XSTATE_8_BASELINE * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE;
	int curInd;

	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_PMat_L[64];
	__shared__ CUFlt sh_PMat_R[64];
	__shared__ CUFlt sh_condlike_S[BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE << 3];
	

	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();

	for(curInd = ind; curInd < 64; curInd += nThread){
#ifdef USING_LDG
		sh_PMat_L[curInd] = __ldg(&PMat[sh_condlikeOp.child_P_offset[0] + curInd]);
		sh_PMat_R[curInd] = __ldg(&PMat[sh_condlikeOp.child_P_offset[1] + curInd]);
#else
		sh_PMat_L[curInd] = PMat[sh_condlikeOp.child_P_offset[0] + curInd];
		sh_PMat_R[curInd] = PMat[sh_condlikeOp.child_P_offset[1] + curInd];
#endif
	}
	__syncthreads();

	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int blkOffset = bx - startBlkInd;
	int tipState_offset = blkOffset * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE;
	int condlike_offset = (tipState_offset << 3);
	int nElemPerBlock = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE << 3);

	if(1 == curCase){

		deviceCondlike_xState_case1_transpose_baseline_8State(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0] + tipState_offset,
									tipState + sh_condlikeOp.child_condlike_offset[1] + tipState_offset, 
									sh_PMat_L,
									sh_PMat_R,
									ind,
									nThread);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		pCondlike_R += condlike_offset;

		for(curInd = ind; curInd < nElemPerBlock; curInd += nThread){
#ifdef USING_LDG
			sh_condlike_S[curInd] = __ldg(&pCondlike_R[curInd]);
#else
			sh_condlike_S[curInd] = pCondlike_R[curInd];
#endif
		}

		__syncthreads();

		deviceCondlike_xState_case2_transpose_baseline_8State(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0] + tipState_offset,
									sh_condlike_S, 
									sh_PMat_L,
									sh_PMat_R,
									ind,
									nThread,
									nState);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_xState_case3_transpose_baseline_8State(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									pCondlike_L + condlike_offset,
									pCondlike_R + condlike_offset, 
									sh_PMat_L, 
									sh_PMat_R,
									sh_condlike_S,
									ind,
									nThread,
									nState);
	}
	else{		// Error
		return;
	}
}


// For nPaddedState == 16:
__device__
void deviceCondlike_xState_case1_transpose_baseline_16State(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE << 4);
	int curPattern, curState, curInd, iElem;
	
	// For the left child:
	for(curInd = ind; curInd < 256; curInd += nThread){
#ifdef USING_LDG
		sh_buf[curInd] = __ldg(&PMat_L[curInd]);
#else
		sh_buf[curInd] = PMat_L[curInd];
#endif
	}

	__syncthreads();

	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, iElem ++){
		curPattern = (curInd >> 4);
		curState = (curInd & 0xf);

#ifdef USING_LDG
		temp_buf[iElem] = sh_buf[(__ldg(&tipState_L[curPattern]) << 4) + curState];
#else
		temp_buf[iElem] = sh_buf[(tipState_L[curPattern] << 4) + curState];
#endif
	}

	__syncthreads();


	// For the right child:
	for(curInd = ind; curInd < 256; curInd += nThread){
#ifdef USING_LDG
		sh_buf[curInd] = __ldg(&PMat_R[curInd]);
#else
		sh_buf[curInd] = PMat_R[curInd];
#endif
	}

	__syncthreads();

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		curPattern = (curInd >> 4);
		curState = (curInd & 0xf);

#ifdef USING_LDG
		condlike_F[0] = sh_buf[(__ldg(&tipState_R[curPattern]) << 4) + curState] * temp_buf[iElem];
#else
		condlike_F[0] = sh_buf[(tipState_R[curPattern] << 4) + curState] * temp_buf[iElem];
#endif
	}
}


// version 1 of case 2: ����temp_buf[]����ĿΪ16 / 2 = 8�����ڱ���ֿ�˷����м�������дcondlike_F 2�Σ���1��;
__device__
void deviceCondlike_xState_case2_transpose_baseline_16State(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE << 4);
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_16_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	const int nElemPerTile_P = (TILE_SIZE_CONDLINE_XSTATE_16_BASELINE << 4);
	const int nElemPerTile_Cl = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE * TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	CUFlt *pCondlike_F, *pSh_condlike, *pCurPMat, *pCurCondlike;
	
	// For the left child:
	for(curInd = ind; curInd < 256; curInd += nThread){
#ifdef USING_LDG
		sh_buf[curInd] = __ldg(&PMat_L[curInd]);
#else
		sh_buf[curInd] = PMat_L[curInd];
#endif
	}

	__syncthreads();

	pCondlike_F = condlike_F + ind;
	for(curInd = ind; curInd < nElemToCalc; curInd += nThread, pCondlike_F += nThread){
		curPattern = (curInd >> 4);
		curState = (curInd & 0xf);

#ifdef USING_LDG
		pCondlike_F[0] = sh_buf[(__ldg(&tipState_L[curPattern]) << 4) + curState];
#else
		pCondlike_F[0] = sh_buf[(tipState_L[curPattern] << 4) + curState];
#endif
	}

	__syncthreads();


	// For the right child:
	pSh_condlike = sh_buf + (TILE_SIZE_CONDLINE_XSTATE_16_BASELINE << 4);
	
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_R[curInd]);
#else
			sh_buf[curInd] = PMat_R[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_R[(curPattern << 4) + curState]);
#else
			pSh_condlike[curInd] = condlike_R[(curPattern << 4) + curState];
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

		PMat_R += nElemPerTile_P;
		condlike_R += TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] *= temp_buf[iElem];
	}
}


// version 1 of case 3: ����temp_buf[]����ĿΪ16 / 2 = 8�����ڱ���ֿ�˷����м�������дcondlike_F 2�Σ���1��;
__device__
void deviceCondlike_xState_case3_transpose_baseline_16State(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState)
{
	int nElemToCalc = (BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE << 4);
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_16_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	const int nElemPerTile_P = (TILE_SIZE_CONDLINE_XSTATE_16_BASELINE << 4);
	const int nElemPerTile_Cl = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE * TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	CUFlt *pSh_condlike, *pCurPMat, *pCurCondlike;

	pSh_condlike = sh_buf + (TILE_SIZE_CONDLINE_XSTATE_16_BASELINE << 4);
	
	// For the left child:
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_L[curInd]);
#else
			sh_buf[curInd] = PMat_L[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_L[(curPattern << 4) + curState]);
#else
			pSh_condlike[curInd] = condlike_L[(curPattern << 4) + curState];
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

		PMat_L += nElemPerTile_P;
		condlike_L += TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	}

	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, iElem ++){
		condlike_F[curInd] = temp_buf[iElem];
	}


	// For the right child:
	for(itr = 0; itr < 16 / BLOCK_DIMENSION_X_CONDLIKE_XSTATE_16_BASELINE; itr ++)
		temp_buf[itr] = 0.0f;

	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_R[curInd]);
#else
			sh_buf[curInd] = PMat_R[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_R[(curPattern << 4) + curState]);
#else
			pSh_condlike[curInd] = condlike_R[(curPattern << 4) + curState];
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

		PMat_R += nElemPerTile_P;
		condlike_R += TILE_SIZE_CONDLINE_XSTATE_16_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] *= temp_buf[iElem];
	}
}


// For nPaddedState == 16:
// һ��2��version: 
// version 1: Ҫ���shared memory��СΪ: 16 * 16 = 256������case 1�����ڱ�����/�Һ��ӵ�PMat������case 2�����ڱ������ӵ�PMat/�Һ��ӵ�PMat��condlike�ķֿ�(�ֿ��СΪ4)��ʹ�üĴ���(16 / 2 = 8��)����ֿ���м���������case 2/3��Ҫдcondlike_F 2�Σ���1�Σ�����case 1��Ҫдcondlike_F 1�Σ���0��;
// version 2: ���version 1���ԣ�ʹ������Ĵ���(16 / 2 * 2 = 16��)��һ�����ڱ������ӵ�condlike * PMat�Ľ������һ�����ڱ���ֿ�˷����м�������������case��ֻ��дcondlike_F 1�Σ���0��;
// ʵ����Ϊ: version 1����version 2����ʹ��version 1;
__global__
void kernelCondlike_xState_transpose_baseline_16State(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int ind = tx + threadIdx.y * BLOCK_DIMENSION_X_CONDLIKE_XSTATE_16_BASELINE;
	int opInd = blkIndToOpInd[bx];
	int nThread = BLOCK_DIMENSION_X_CONDLIKE_XSTATE_16_BASELINE * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE;

	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_buf[16 * 16];

	CUFlt temp_buf[16 / BLOCK_DIMENSION_X_CONDLIKE_XSTATE_16_BASELINE] = {0};


	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int blkOffset = bx - startBlkInd;
	int tipState_offset = blkOffset * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE;
	int condlike_offset = (tipState_offset << 4);
	
	if(1 == curCase){
		deviceCondlike_xState_case1_transpose_baseline_16State(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0] + tipState_offset,
									tipState + sh_condlikeOp.child_condlike_offset[1] + tipState_offset,
									PMat + sh_condlikeOp.child_P_offset[0],
									PMat + sh_condlikeOp.child_P_offset[1],
									sh_buf,
									temp_buf,
									ind,
									nThread);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		pCondlike_R += condlike_offset;

		deviceCondlike_xState_case2_transpose_baseline_16State(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0] + tipState_offset,
									pCondlike_R,
									PMat + sh_condlikeOp.child_P_offset[0],
									PMat + sh_condlikeOp.child_P_offset[1],
									sh_buf,
									temp_buf,
									ind,
									nThread,
									nState);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_xState_case3_transpose_baseline_16State(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									pCondlike_L + condlike_offset,
									pCondlike_R + condlike_offset, 
									PMat + sh_condlikeOp.child_P_offset[0],
									PMat + sh_condlikeOp.child_P_offset[1],
									sh_buf,
									temp_buf,
									ind,
									nThread,
									nState);
	}
	else{		// Error
		return;
	}
}


// For nPaddedState == 24:
// version 2 of case 1: ����shared memory��СΪ224 = (24 * 4 + 32 * 4) < 576���������ڱ���һ�����ӵ�������PMat����1/2��Ĵ���(��case 1������);
__device__
void deviceCondlike_xState_case1_transpose_baseline_24State(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread)
{
	int nElemToCalc = 24 * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE;
	int curPattern, curState, curInd, iElem;
	
	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		curPattern = curInd / 24;
		curState = curInd % 24;

#ifdef USING_LDG
		condlike_F[0] = __ldg(&PMat_L[__ldg(&tipState_L[curPattern]) * 24 + curState]) * __ldg(&PMat_R[__ldg(&tipState_R[curPattern]) * 24 + curState]);
#else
		condlike_F[0] = PMat_L[tipState_L[curPattern] * 24 + curState] * PMat_R[tipState_R[curPattern] * 24 + curState];
#endif
	}
}



// version 3 of case 2: ����temp_buf[]����ĿΪ24 / 2 = 12�����ڱ���ֿ�˷����м�������дcondlike_F 2�Σ���1��;
// ����shared memory��СΪ 224 < 24 * 24���������ڱ������ӵ�PMat;
__device__
void deviceCondlike_xState_case2_transpose_baseline_24State(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState)
{
	int nElemToCalc = 24 * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE;
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_24_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	const int nElemPerTile_P = 24 * TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	const int nElemPerTile_Cl = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE * TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	CUFlt *pCondlike_F, *pSh_condlike, *pCurPMat, *pCurCondlike;
	
	// For the left child:
	pCondlike_F = condlike_F + ind;
	for(curInd = ind; curInd < nElemToCalc; curInd += nThread, pCondlike_F += nThread){
		curPattern = curInd / 24;
		curState = curInd % 24;

#ifdef USING_LDG
		pCondlike_F[0] = __ldg(&PMat_L[__ldg(&tipState_L[curPattern]) * 24 + curState]);
#else
		pCondlike_F[0] = PMat_L[tipState_L[curPattern] * 24 + curState];
#endif
	}


	// For the right child:
	pSh_condlike = sh_buf + 24 * TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_R[curInd]);
#else
			sh_buf[curInd] = PMat_R[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_R[curPattern * 24 + curState]);
#else
			pSh_condlike[curInd] = condlike_R[curPattern * 24 + curState];
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

		PMat_R += nElemPerTile_P;
		condlike_R += TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] *= temp_buf[iElem];
	}
}



// version 1 of case 3: ����temp_buf[]����ĿΪ24 / 2 = 12�����ڱ���ֿ�˷����м�������дcondlike_F 2�Σ���1��;
// ����shared memory��СΪ576 > 2 * (24 * 4 + 32 * 4)���ʿ���ͬʱ�������Һ��ӽڵ��condlike��PMat�ķֿ�(����ֿ��СΪ4);
__device__
void deviceCondlike_xState_case3_transpose_baseline_24State(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState)
{
	int nElemToCalc = 24 * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE;
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_24_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	const int nElemPerTile_P = 24 * TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	const int nElemPerTile_Cl = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE * TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	CUFlt *pSh_condlike, *pCurPMat, *pCurCondlike;

	pSh_condlike = sh_buf + nElemPerTile_P;

	// For the left child:
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_L[curInd]);
#else
			sh_buf[curInd] = PMat_L[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_L[curPattern * 24 + curState]);
#else
			pSh_condlike[curInd] = condlike_L[curPattern * 24 + curState];
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

		PMat_L += nElemPerTile_P;
		condlike_L += TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	}

	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, iElem ++){
		condlike_F[curInd] = temp_buf[iElem];
	}


	// For the right child:
	for(itr = 0; itr < 24 / BLOCK_DIMENSION_X_CONDLIKE_XSTATE_24_BASELINE; itr ++)
		temp_buf[itr] = 0.0f;

	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_R[curInd]);
#else
			sh_buf[curInd] = PMat_R[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_R[curPattern * 24 + curState]);
#else
			pSh_condlike[curInd] = condlike_R[curPattern * 24 + curState];
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

		PMat_R += nElemPerTile_P;
		condlike_R += TILE_SIZE_CONDLINE_XSTATE_24_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] *= temp_buf[iElem];
	}
}



// For nPaddedState == 24:
// һ��4��version:
// version 1: Ҫ���shared memory��СΪ: 24 * 24 = 576, �Ĵ���1��(ÿ��24 / 2��)�� ����case 1��shared memory���ڱ�������PMat������case 2��shared memory���ڱ������ӵ�PMat�����Һ��ӵ�condlike��PMat�ķֿ飬����case 3��shared memory���ڱ������Һ��ӵ�condlike��PMat�ķֿ飬��Ҫдcondlike_F 2�Σ���1��;
// version 2: Ҫ���shared memoryͬversion 1�����ǼĴ���2�飬��Ҫдcondlike_F 1�Σ���0��;
// version 3: Ҫ���shared memory��СΪ: (24 * 4 + 32 * 4) = 224(����ֿ��СΪ4)���Ĵ���1��;
// version 4: shared memory��Сͬversion 3, �Ĵ���2��;
// ʵ����Ϊ��version 3����Ч����ã����ʹ��version 3;
// version 3 of nPaddedState = 24:
__global__
void kernelCondlike_xState_transpose_baseline_24State(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int ind = tx + threadIdx.y * BLOCK_DIMENSION_X_CONDLIKE_XSTATE_24_BASELINE;
	int opInd = blkIndToOpInd[bx];
	int nThread = BLOCK_DIMENSION_X_CONDLIKE_XSTATE_24_BASELINE * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE;

	__shared__ CuLCondlikeOp sh_condlikeOp;
	__shared__ CUFlt sh_buf[(32 + 24) * TILE_SIZE_CONDLINE_XSTATE_24_BASELINE];

	CUFlt temp_buf[24 / BLOCK_DIMENSION_X_CONDLIKE_XSTATE_24_BASELINE] = {0};


	if(tx == 0){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int blkOffset = bx - startBlkInd;
	int tipState_offset = blkOffset * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE;
	int condlike_offset = tipState_offset * 24;
	
	if(1 == curCase){
		deviceCondlike_xState_case1_transpose_baseline_24State(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0] + tipState_offset,
									tipState + sh_condlikeOp.child_condlike_offset[1] + tipState_offset,
									PMat + sh_condlikeOp.child_P_offset[0],
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThread);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		pCondlike_R += condlike_offset;

		deviceCondlike_xState_case2_transpose_baseline_24State(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0] + tipState_offset,
									pCondlike_R,
									PMat + sh_condlikeOp.child_P_offset[0],
									PMat + sh_condlikeOp.child_P_offset[1],
									sh_buf,
									temp_buf,
									ind,
									nThread,
									nState);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_xState_case3_transpose_baseline_24State(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									pCondlike_L + condlike_offset,
									pCondlike_R + condlike_offset, 
									PMat + sh_condlikeOp.child_P_offset[0],
									PMat + sh_condlikeOp.child_P_offset[1],
									sh_buf,
									temp_buf,
									ind,
									nThread,
									nState);
	}
	else{		// Error
		return;
	}
}


// For nPaddedState = 32 / 40 / 48 / 56 / 72 / ...
// case 1:
__device__
void deviceCondlike_xState_case1_transpose_baseline_largeState(CUFlt *condlike_F, int *tipState_L, int *tipState_R, CUFlt *PMat_L, CUFlt *PMat_R, int ind, int nThread, int nPaddedState, int nPatternPerBlock)
{
	int nElemToCalc = nPaddedState * nPatternPerBlock;
	int curPattern, curState, curInd, iElem;
	
	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		curPattern = curInd / nPaddedState;
		curState = curInd % nPaddedState;

#ifdef USING_LDG
		condlike_F[0] = __ldg(&PMat_L[__ldg(&tipState_L[curPattern]) * nPaddedState + curState]) * __ldg(&PMat_R[__ldg(&tipState_R[curPattern]) * nPaddedState + curState]);
#else
		condlike_F[0] = PMat_L[tipState_L[curPattern] * nPaddedState + curState] * PMat_R[tipState_R[curPattern] * nPaddedState + curState];
#endif
	}
}


// version 1 of case 2: һ��Ĵ��������ڱ���ֿ�˷����м�������Ҫдcondlike_F 2�Σ���1��;
__device__
void deviceCondlike_xState_case2_transpose_baseline_largeState(CUFlt *condlike_F, int *tipState_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState, int nPaddedState, int nPatternPerBlock)
{
	int nElemToCalc = nPaddedState * nPatternPerBlock;
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	const int nElemPerTile_P = nPaddedState * TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	const int nElemPerTile_Cl = nPatternPerBlock * TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	CUFlt *pCondlike_F, *pSh_condlike, *pCurPMat, *pCurCondlike;
	
	// For the left child:
	pCondlike_F = condlike_F + ind;
	for(curInd = ind; curInd < nElemToCalc; curInd += nThread, pCondlike_F += nThread){
		curPattern = curInd / nPaddedState;
		curState = curInd % nPaddedState;

#ifdef USING_LDG
		pCondlike_F[0] = __ldg(&PMat_L[__ldg(&tipState_L[curPattern]) * nPaddedState + curState]);
#else
		pCondlike_F[0] = PMat_L[tipState_L[curPattern] * nPaddedState + curState];
#endif
	}


	// For the right child:
	pSh_condlike = sh_buf + nPaddedState * TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_R[curInd]);
#else
			sh_buf[curInd] = PMat_R[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_R[curPattern * nPaddedState + curState]);
#else
			pSh_condlike[curInd] = condlike_R[curPattern * nPaddedState + curState];
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

		PMat_R += nElemPerTile_P;
		condlike_R += TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] *= temp_buf[iElem];
	}
}



// version 1 of case 3: һ��Ĵ��������ڱ���ֿ�˷����м�������Ҫдcondlike_F 2�Σ���1��;
__device__
void deviceCondlike_xState_case3_transpose_baseline_largeState(CUFlt *condlike_F, CUFlt *condlike_L, CUFlt *condlike_R, CUFlt *PMat_L, CUFlt *PMat_R, CUFlt *sh_buf, CUFlt *temp_buf, int ind, int nThread, int nState, int nPaddedState, int nPatternPerBlock)
{
	int nElemToCalc = nPaddedState * nPatternPerBlock;
	int curPattern, curState, curInd, iElem, iState, itr, nIteration = (nState + TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE - 1) / TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	const int nElemPerTile_P = nPaddedState * TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	const int nElemPerTile_Cl = nPatternPerBlock * TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	CUFlt *pCondlike_F, *pSh_condlike, *pCurPMat, *pCurCondlike;
	
	// For the left child:
	pSh_condlike = sh_buf + nPaddedState * TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	
	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_L[curInd]);
#else
			sh_buf[curInd] = PMat_L[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_L[curPattern * nPaddedState + curState]);
#else
			pSh_condlike[curInd] = condlike_L[curPattern * nPaddedState + curState];
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

		PMat_L += nElemPerTile_P;
		condlike_L += TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	}

	pCondlike_F = condlike_F + ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, pCondlike_F += nThread, iElem ++){
		pCondlike_F[0] = temp_buf[iElem];
	}


	// For the right child:
	for(itr = 0; itr < N_STATE_PER_THREAD_CONDLIKE_XSTATE_OTHER_BASELINE; itr ++)
		temp_buf[itr] = 0;

	for(itr = 0; itr < nIteration; itr ++){
		for(curInd = ind; curInd < nElemPerTile_P; curInd += nThread){
#ifdef USING_LDG
			sh_buf[curInd] = __ldg(&PMat_R[curInd]);
#else
			sh_buf[curInd] = PMat_R[curInd];
#endif
		}

		for(curInd = ind; curInd < nElemPerTile_Cl; curInd += nThread){
			curPattern = curInd / TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
			curState = curInd % TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;

#ifdef USING_LDG
			pSh_condlike[curInd] = __ldg(&condlike_R[curPattern * nPaddedState + curState]);
#else
			pSh_condlike[curInd] = condlike_R[curPattern * nPaddedState + curState];
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

		PMat_R += nElemPerTile_P;
		condlike_R += TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE;
	}

	condlike_F += ind;
	for(curInd = ind, iElem = 0; curInd < nElemToCalc; curInd += nThread, condlike_F += nThread, iElem ++){
		condlike_F[0] *= temp_buf[iElem];
	}
}



// For nPaddedState = 32 / 40 / 48 / 54 / 72 / ...
// ÿ��thread����N_STATE_PER_THREAD_CONDLIKE_XSTATE_OTHER_BASELINE��element��ÿ��thread block����BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_OTHER_BASELINE��site pattern���ֿ�˷���shared memory����condlike��PMat�ķֿ�;
// ����version: 
// version 1: һ��Ĵ��������ڱ���ֿ�˷����м�������Ҫдcondlike_F 2�Σ���1��;
// version 2: ����Ĵ�����һ�����ڱ������ӵ�condlike * PMat�Ľ������һ�����ڱ���ֿ�˷����м���;
// ʵ����Ϊ��version 1���
__global__
void kernelCondlike_xState_transpose_baseline_largeState(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nState, const int nPaddedState)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int blockDim_x = blockDim.x;
	int blockDim_y = blockDim.y;
	int ind = tx + threadIdx.y * blockDim_x;
	int opInd = blkIndToOpInd[bx];
	int nThread = blockDim_x * blockDim_y;

	__shared__ CuLCondlikeOp sh_condlikeOp;
	extern __shared__ CUFlt sh_buf[];

	CUFlt temp_buf[N_STATE_PER_THREAD_CONDLIKE_XSTATE_OTHER_BASELINE] = {0};


	if(0 == tx){
		sh_condlikeOp = condlikeOp[opInd];
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int blkOffset = bx - startBlkInd;
	int tipState_offset = blkOffset * blockDim_y;
	int condlike_offset = tipState_offset * nPaddedState;
	
	if(1 == curCase){
		deviceCondlike_xState_case1_transpose_baseline_largeState(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0] + tipState_offset,
									tipState + sh_condlikeOp.child_condlike_offset[1] + tipState_offset,
									PMat + sh_condlikeOp.child_P_offset[0],
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThread,
									nPaddedState,
									blockDim_y);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		pCondlike_R += condlike_offset;

		deviceCondlike_xState_case2_transpose_baseline_largeState(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0] + tipState_offset,
									pCondlike_R,
									PMat + sh_condlikeOp.child_P_offset[0],
									PMat + sh_condlikeOp.child_P_offset[1],
									sh_buf,
									temp_buf,
									ind,
									nThread,
									nState,
									nPaddedState,
									blockDim_y);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_xState_case3_transpose_baseline_largeState(intCondlike + sh_condlikeOp.father_condlike_offset + condlike_offset, 
									pCondlike_L + condlike_offset,
									pCondlike_R + condlike_offset, 
									PMat + sh_condlikeOp.child_P_offset[0],
									PMat + sh_condlikeOp.child_P_offset[1],
									sh_buf,
									temp_buf,
									ind,
									nThread,
									nState,
									nPaddedState,
									blockDim_y);
	}
	else{		// Error
		return;
	}
}


// Non-transpose version:
__global__
void kernelCondlike_xState_noTranspose_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const int nThreadPerArray, const int nThreadPerBlock, const int nSitePattern, const int nState, const int nPaddedState)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int opInd = blkIndToOpInd[bx];

	__shared__ CuLCondlikeOp sh_condlikeOp;
	
	if(tx == 0){
		sh_condlikeOp.nChild = condlikeOp[opInd].nChild;
		sh_condlikeOp.whichCase = condlikeOp[opInd].whichCase;
		sh_condlikeOp.father_condlike_offset = condlikeOp[opInd].father_condlike_offset;
		for(int i = 0; i < 2; i ++){
			sh_condlikeOp.isChildTip[i] = condlikeOp[opInd].isChildTip[i];
			sh_condlikeOp.child_P_offset[i] = condlikeOp[opInd].child_P_offset[i];
			sh_condlikeOp.child_condlike_offset[i] = condlikeOp[opInd].child_condlike_offset[i];
		}
	}
	__syncthreads();


	int startBlkInd = opStartBlkInd[opInd];
	int curCase = sh_condlikeOp.whichCase;
	int ind = (bx - startBlkInd) * nThreadPerBlock + threadIdx.y * blockDim.x + tx;

	if(1 == curCase){
		deviceCondlike_xState_case1_noTranspose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									tipState + sh_condlikeOp.child_condlike_offset[1], 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nPaddedState);
	}
	else if(2 == curCase){
		CUFlt *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_xState_case2_noTranspose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									tipState + sh_condlikeOp.child_condlike_offset[0],
									pCondlike_R, 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nState,
									nPaddedState);
	}
	else if(3 == curCase){
		CUFlt *pCondlike_L, *pCondlike_R;
		if(1 == sh_condlikeOp.isChildTip[0])
			pCondlike_L = tipCondlike + sh_condlikeOp.child_condlike_offset[0];
		else
			pCondlike_L = intCondlike + sh_condlikeOp.child_condlike_offset[0];

		if(1 == sh_condlikeOp.isChildTip[1])
			pCondlike_R = tipCondlike + sh_condlikeOp.child_condlike_offset[1];
		else
			pCondlike_R = intCondlike + sh_condlikeOp.child_condlike_offset[1];

		deviceCondlike_xState_case3_noTranspose_baseline(intCondlike + sh_condlikeOp.father_condlike_offset, 
									pCondlike_L,
									pCondlike_R, 
									PMat + sh_condlikeOp.child_P_offset[0], 
									PMat + sh_condlikeOp.child_P_offset[1],
									ind,
									nThreadPerArray,
									nSitePattern,
									nState,
									nPaddedState);
	}
	else{		// Error
		return;
	}
}



void callKernelCondlike_baseline(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const bool usePadVersion, const int nSitePattern, const int nPaddedState, const int nState, const int nThreadPerArray, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream)
{
	const int blockSize = nThreadPerBlock.x * nThreadPerBlock.y * nThreadPerBlock.z;
	// For nPaddedState != 64, use the baseline version for condlike:
	if(4 == nPaddedState){
		//printf("\n=======\nGoing to call kernel for 4 state of condlike...\n==========\n");
		const int nTotalPattern = nOp * nSitePattern;
#ifdef TRANSPOSE_PMAT
#ifdef USE_OLD_VERSION
		// The old schedule scheme:
		if(nTotalPattern < PATTERN_THRESHOLD_4STATE_CASE1_BASELINE)
			kernelCondlike_4State_transpose_baseline_version1<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		else if(nTotalPattern < PATTERN_THRESHOLD_4STATE_CASE3_BASELINE)
			kernelCondlike_4State_transpose_baseline_version2<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		else if(nTotalPattern < PATTERN_THRESHOLD_4STATE_CASE2_BASELINE)
			kernelCondlike_4State_transpose_baseline_version3<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		else
			kernelCondlike_4State_transpose_baseline_version4<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
#else
		// The new schedule scheme:
		if(nTotalPattern < TOTAL_PATTERN_THRESHOLD_4STATE_SMALL)
			kernelCondlike_4State_transpose_baseline_version2<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		else
			kernelCondlike_4State_transpose_baseline_version3<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
#endif
		cutilCheckMsg("kernel kernelCondlike_4State_transpose_baseline() failed");
#else
		if(nTotalPattern < PATTERN_THRESHOLD_4STATE_CASE1_BASELINE)
			kernelCondlike_4State_noTranspose_baseline_version1<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		else if(nTotalPattern < PATTERN_THRESHOLD_4STATE_CASE3_BASELINE)
			kernelCondlike_4State_noTranspose_baseline_version2<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		else if(nTotalPattern < PATTERN_THRESHOLD_4STATE_CASE2_BASELINE)
			kernelCondlike_4State_noTranspose_baseline_version3<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		else
			kernelCondlike_4State_noTranspose_baseline_version4<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern);
		cutilCheckMsg("kernel kernelCondlike_4State_transpose_baseline() failed");

#endif
	}
	else if(20 == nPaddedState){
		//printf("\n=======\nGoing to call kernel for 20 state of condlike...\n==========\n");
#ifdef TRANSPOSE_PMAT
		kernelCondlike_20State_transpose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern, nState);
		cutilCheckMsg("kernel kernelCondlike_20State_transpose_baseline() failed");
#else
		kernelCondlike_20State_noTranspose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern, nState);
		cutilCheckMsg("kernel kernelCondlike_20State_noTranspose_baseline() failed");
#endif
	}
	else{
		// For nPaddedState != 4 / 20 / 64:
#ifdef TRANSPOSE_PMAT
		if(usePadVersion){
			if(8 == nPaddedState){
				//printf("\n=======\nGoing to call kernel for 8 state of condlike...\n==========\n");
				kernelCondlike_xState_transpose_baseline_8State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nState);
			}
			else if(16 == nPaddedState){
				//printf("\n=======\nGoing to call kernel for 16 state of condlike...\n==========\n");
				kernelCondlike_xState_transpose_baseline_16State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nState);
			}
			else if(24 == nPaddedState){
				//printf("\n=======\nGoing to call kernel for 24 state of condlike...\n==========\n");
				kernelCondlike_xState_transpose_baseline_24State<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nState);
			}
			else{
				//printf("\n=======\nGoing to call kernel for large state of condlike...\n==========\n");
				const int sharedMem_needed = TILE_SIZE_CONDLINE_XSTATE_OTHER_BASELINE * (nPaddedState + BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_OTHER_BASELINE) * sizeof(CUFlt);
				kernelCondlike_xState_transpose_baseline_largeState<<<nBlockPerGrid, nThreadPerBlock, sharedMem_needed, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nState, nPaddedState);
			}
		}
		else{
			//printf("\n=======\nGoing to call kernel for X state of condlike...\n==========\n");
			kernelCondlike_xState_transpose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern, nState, nPaddedState);
		}
		cutilCheckMsg("kernel kernelCondlike_xState_transpose_baseline() failed");
#else
		kernelCondlike_xState_noTranspose_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(intCondlike, tipState, tipCondlike, PMat, condlikeOp, blkIndToOpInd, opStartBlkInd, nThreadPerArray, blockSize, nSitePattern, nState, nPaddedState);
		cutilCheckMsg("kernel kernelCondlike_xState_noTranspose_baseline() failed");
#endif
	}
}




// Node scale:
// TODO: �����Ƿ��������shuffle���Ƹ��ƴ�global memory�м������ݽ��бȽϵ�Ч��;
// TODO: ��scale factor��С���ӽ���0��ô��???
// version 1: ÿ��thread����һ��site pattern������eigen decomposition������rate category��state�ıȽ��Լ�����scale����;
// ʵ����Ϊ��version 1��version 2Ч����࣬��version 1����Ҫʹ��shared memory�������version 1;
__global__
void kernelNodeScale_baseline(CUFlt *nodeScaleFactor, CUFlt *intCondlike, int *blkIndToCondlikeOffset, int *blkIndToScaleBufferOffset, int *startBlkInd, int nCategory, int nSitePattern, int nPaddedSitePattern, int nState, int nPaddedState)
{
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int curPattern = (bx - startBlkInd[bx]) * N_THREAD_PER_BLOCK_SCALE_BASELINE + tx;
	
	if(curPattern >= nSitePattern)
		return;

	const int txOffset = blkIndToCondlikeOffset[bx] + tx * nPaddedState;
	const int condlike_size = nPaddedSitePattern * nPaddedState;

	CUFlt *pCondlike = intCondlike + txOffset;

	CUFlt maxValue = 0.0f, curValue;
	int iCategory, ind;
	for(iCategory = 0; iCategory < nCategory; iCategory ++, pCondlike += condlike_size){
		for(ind = 0; ind < nState; ind ++){
			curValue = pCondlike[ind];
			if(curValue > maxValue)
				maxValue = curValue;
		}
	}

	if(maxValue <= 0.0)
		maxValue = 1.0;

	CUFlt factor = 1.0 / maxValue;
	pCondlike = intCondlike + txOffset;
	for(iCategory = 0; iCategory < nCategory; iCategory ++, pCondlike += condlike_size){
		for(ind = 0; ind < nState; ind ++){
			pCondlike[ind] *= factor;
		}
	}

	ind = blkIndToScaleBufferOffset[bx] + tx;
	nodeScaleFactor[ind] = log(maxValue);
}


void callKernelNodeScale_baseline(CUFlt *nodeScaleFactor, CUFlt *intCondlike, int *blkIndToCondlikeOffset, int *blkIndToScaleBufferOffset, int *startBlkInd, int nCategory, int nSitePattern, int nPaddedSitePattern, int nState, int nPaddedState, int nBlockPerGrid, int nThreadPerBlock, cudaStream_t &stream)
{
	kernelNodeScale_baseline<<<nBlockPerGrid, nThreadPerBlock, 0, stream>>>(nodeScaleFactor, intCondlike, blkIndToCondlikeOffset, blkIndToScaleBufferOffset, startBlkInd, nCategory, nSitePattern, nPaddedSitePattern, nState, nPaddedState);
	cutilCheckMsg("kernel kernelNodeScale_baseline() failed");
}



// Calculation of site likelihoods and reduction of site likelihoods:
// Baseline kernel to calculate site likelihood
// for state == 4, a thread is responsible for the calculation of all eigen decomposition and all rate categories of a site pattern, for state == 64, a thread is responsible for the calculation of 16 states of a site pattern;
// TODO: rateCatWeight[]�Ĵ�СΪnRateCategory����nEigenDecomposition * nRateCategory???��Ҳ��eigen weight�Ƿ��Ѿ���������rate category weight����???
// ����MrBayes����beagle�����beagleSetCategoryWeights()�������ƺ�Ӧ����nEigenDecomposition(nCijk)�ף�
// ���ڵ����⣺����condlike����֯��ʽ��ͬһ��site pattern������nState��condlike��һ�����load condlikeʱͬһ��thread�������ڵ��ĸ�condlike��load���ô淽��Ч�ʽϵͣ������޸�condlike����֯��ʽ��Ҳ����Ϊͬһ��state������site pattern����һ�𣿣���
// �����ʵ�ּ���condlike����֯��ʽΪ��ͬһsite pattern������nState��condlike��һ��
// ע��padded state��ʵ�ʵ�state�Լ�padded site pattern��ʵ�ʵ�site pattern������
// Accumulate the scaling factors if needed(when nNodeScaler > 0)
 __global__
void kernelSiteLnL_4State_baseline(CUFlt *siteLnL, CUFlt *rootCondlike, CUFlt *rateCatWeight, CUFlt *stateFreq, int nNodeScaler, CUFlt *scaleFactor, const int nEigenDecomp, const int nRateCategory, const int nSitePattern, const int nPaddedSitePattern)
{
	int step = blockDim.x;
	int tx = threadIdx.x;
	int curSite = threadIdx.y + blockIdx.x * blockDim.y;
	int ind = threadIdx.y * step + tx;
	const int nCategory = nEigenDecomp * nRateCategory;
	
	__shared__ CUFlt sh_stateFreq[4];
	extern __shared__ CUFlt sh_catWeightAndTempLnL[];
	
	if(ind < 4){
		sh_stateFreq[ind] = stateFreq[ind];
	}
	else if(ind >= 4 && ind < 4 + nCategory){
		sh_catWeightAndTempLnL[ind - 4] = rateCatWeight[ind - 4];
	}
	__syncthreads();

	
	CUFlt totalSum = 0.0f;
	CUFlt *pSh_tempLnL = sh_catWeightAndTempLnL + nCategory;
	if(curSite < nSitePattern){
		int curState, totalCatCnt = 0, iEigen, iRateCat;
		int step = blockDim.x;
		CUFlt curSum;
		CUFlt *pCondlike = rootCondlike + (curSite << 2);
		const int condlike_rateCat_offset = (nPaddedSitePattern << 2);

		for(iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			for(iRateCat = 0; iRateCat < nRateCategory; iRateCat ++, pCondlike += condlike_rateCat_offset, totalCatCnt ++){
				curSum = 0.0f;

//#pragma unroll
				
				for(curState = tx; curState < 4; curState += step){
					curSum += pCondlike[curState] * sh_stateFreq[curState];
				}
				
				totalSum += curSum * sh_catWeightAndTempLnL[totalCatCnt];
			}
		}
	}
	
	pSh_tempLnL[ind] = totalSum;
	__syncthreads();
	
	
	for(int reduceSize = (step >> 1); reduceSize > 0; reduceSize >>= 1){
		if(tx < reduceSize)
			pSh_tempLnL[ind] += pSh_tempLnL[ind + reduceSize];
		__syncthreads();
	}

	
	if(curSite < nSitePattern && tx == 0){
		if(pSh_tempLnL[threadIdx.y * step] <= 0)
			totalSum = log(CUFLT_MIN);
		else
			totalSum = log(pSh_tempLnL[threadIdx.y * step]);

		int iNode;
		scaleFactor += curSite;
		for(iNode = 0; iNode < nNodeScaler; iNode ++, scaleFactor += nPaddedSitePattern)
			totalSum += scaleFactor[0];

		siteLnL[curSite] = totalSum;
	}
}


// For nPaddedState == 20:
// ע�⣺����֮����ҪblockDim.x��temp results�Ĺ�Լ����ˣ���Ҫ��֤blockDim.xΪ2�ı�������ÿ��thread����5��stateʱ�������е���4��temp lnL�Ĺ�Լ������������
__global__
void kernelSiteLnL_20State_baseline(CUFlt *siteLnL, CUFlt *rootCondlike, CUFlt *rateCatWeight, CUFlt *stateFreq, int nNodeScaler, CUFlt *scaleFactor, const int nEigenDecomp, const int nRateCategory, const int nSitePattern, const int nPaddedSitePattern)
{
	int step = blockDim.x;
	int tx = threadIdx.x;
	int curSite = blockIdx.x * blockDim.y + threadIdx.y;
	int ind = threadIdx.y * step + tx;
	const int nCategory = nEigenDecomp * nRateCategory;

	__shared__ CUFlt sh_stateFreq[20];
	extern __shared__ CUFlt sh_catWeightAndTempLnL[];
	
	if(ind < 20){
		sh_stateFreq[ind] = stateFreq[ind];
	}
	else if(ind >= 32 && ind < 32 + nCategory){
		sh_catWeightAndTempLnL[ind - 32] = rateCatWeight[ind - 32];
	}
	__syncthreads();


	CUFlt totalSum = 0.0f;
	CUFlt *pSh_tempLnL = sh_catWeightAndTempLnL + nCategory;
	if(curSite < nSitePattern){
		CUFlt curSum;
		CUFlt *pCondlike = rootCondlike + curSite * 20;
		const int condlike_rateCat_offset = nPaddedSitePattern * 20;
		int curState, totalCatCnt = 0, iEigen, iRateCat;

		for(iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			for(iRateCat = 0; iRateCat < nRateCategory; iRateCat ++, pCondlike += condlike_rateCat_offset, totalCatCnt ++){
				curSum = 0.0f;
				for(curState = tx; curState < 20; curState += step){
					curSum += pCondlike[curState] * sh_stateFreq[curState];
				}

				totalSum += curSum * sh_catWeightAndTempLnL[totalCatCnt];
			}
		}
	}

	pSh_tempLnL[ind] = totalSum;
	__syncthreads();

	for(int reduceSize = (step >> 1); reduceSize > 0; reduceSize >>= 1){
		if(tx < reduceSize)
			pSh_tempLnL[ind] += pSh_tempLnL[ind + reduceSize];
		__syncthreads();
	}

	/*
	int reduceSize = 4;
	if(tx + reduceSize < step)
		pSh_tempLnL[ind] += pSh_tempLnL[ind + reduceSize];
	__syncthreads();

	for(reduceSize >>= 1; reduceSize > 0; reduceSize >>= 1){
		if(tx < reduceSize)
			pSh_tempLnL[ind] += pSh_tempLnL[ind + reduceSize];
		__syncthreads();
	}
	*/

	if(curSite < nSitePattern && 0 == tx){
		if(pSh_tempLnL[threadIdx.y * step] <= 0)
			totalSum = log(CUFLT_MIN);
		else
			totalSum = log(pSh_tempLnL[threadIdx.y * step]);

		int iNode;
		scaleFactor += curSite;
		for(iNode = 0; iNode < nNodeScaler; iNode ++, scaleFactor += nPaddedSitePattern)
			totalSum += scaleFactor[0];

		siteLnL[curSite] = totalSum;
	}
}



// For nPaddedState == 64:
/* For 64 state:
 For small rate category(nEigenDecomposition * nRateCategory < 12), each thread is responsible for 16 states' condlike, and the block dimension and grid dimension are : nThreadPerBlock: (4, 64), nBlockPerGrid: (nSitePattern / 64)
 For large rate category(), and the block and grid dimension are : nThreadPerBlock = (16, 16), nBlockPerGrid = nSitePattern / 16;
 TODO: * 64���Ա�Ϊ��λ�������ʽ��������ԱȽ�һ��ѭ��չ���Ƿ������Ч��
 */
 __global__
void kernelSiteLnL_64State_baseline(CUFlt *siteLnL, CUFlt *rootCondlike, CUFlt *rateCatWeight, CUFlt *stateFreq, int nNodeScaler, CUFlt *scaleFactor, const int nEigenDecomp, const int nRateCategory, const int nSitePattern, const int nPaddedSitePattern, const int nState)
{
	int step = blockDim.x;
	int tx = threadIdx.x;
	int curSite = threadIdx.y + blockIdx.x * blockDim.y;
	int ind = threadIdx.y * step + tx;
	const int nCategory = nEigenDecomp * nRateCategory;
	
	__shared__ CUFlt sh_stateFreq[64];
	extern __shared__ CUFlt sh_catWeightAndTempLnL[];
	
	if(ind < 64){
		sh_stateFreq[ind] = stateFreq[ind];
	}
	else if(ind >= 64 && ind < 64 + nCategory){			// assert 64 + nEigenDecomp * nRateCategory < 4 * 64;
		sh_catWeightAndTempLnL[ind - 64] = rateCatWeight[ind - 64];
	}
	__syncthreads();

	
	CUFlt totalSum = 0.0f;
	CUFlt *pSh_tempLnL = sh_catWeightAndTempLnL + nCategory;
	if(curSite < nSitePattern){
		int curState, totalCatCnt = 0, iEigen, iRateCat;
		int step = blockDim.x;
		CUFlt curSum;
		CUFlt *pCondlike = rootCondlike + (curSite << 6);
		const int condlike_rateCat_offset = (nPaddedSitePattern << 6);

		for(iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			for(iRateCat = 0; iRateCat < nRateCategory; iRateCat ++, pCondlike += condlike_rateCat_offset, totalCatCnt ++){
				curSum = 0.0f;

//#pragma unroll
				
				for(curState = tx; curState < nState; curState += step){
					curSum += pCondlike[curState] * sh_stateFreq[curState];
				}
				
				totalSum += curSum * sh_catWeightAndTempLnL[totalCatCnt];
			}
		}
	}
	
	pSh_tempLnL[ind] = totalSum;
	__syncthreads();
	
	
	for(int reduceSize = (step >> 1); reduceSize > 0; reduceSize >>= 1){
		if(tx < reduceSize)
			pSh_tempLnL[ind] += pSh_tempLnL[ind + reduceSize];
		__syncthreads();
	}

	
	if(curSite < nSitePattern && tx == 0){
		if(pSh_tempLnL[threadIdx.y * step] <= 0)
			totalSum = log(CUFLT_MIN);
		else
			totalSum = log(pSh_tempLnL[threadIdx.y * step]);

		int iNode;
		scaleFactor += curSite;
		for(iNode = 0; iNode < nNodeScaler; iNode ++, scaleFactor += nPaddedSitePattern)
			totalSum += scaleFactor[0];

		siteLnL[curSite] = totalSum;
	}
}


// For nPaddedState != 4 / 20 / 64:
__global__
void kernelSiteLnL_xState_baseline(CUFlt *siteLnL, CUFlt *rootCondlike, CUFlt *rateCatWeight, CUFlt *stateFreq, int nNodeScaler, CUFlt *scaleFactor, const int nEigenDecomp, const int nRateCategory, const int nSitePattern, const int nPaddedSitePattern, const int nState, const int nPaddedState)
{
	int step = blockDim.x;
	int tx = threadIdx.x;
	int curSite = threadIdx.y + blockIdx.x * blockDim.y;
	int ind = threadIdx.y * step + tx;
	const int nCategory = nEigenDecomp * nRateCategory;
	
	extern __shared__ CUFlt sh_stateFreqAndCatWeight[];

	if(ind < nState){
		sh_stateFreqAndCatWeight[ind] = stateFreq[ind];
	}
	else if(ind < nState + nEigenDecomp * nRateCategory){			// assert blockDim.x >= 16 + nEigenDecomp * nRateCategory
		sh_stateFreqAndCatWeight[ind] = rateCatWeight[ind - nState];
	}
	__syncthreads();
	
	
	CUFlt totalSum = 0.0f;
	CUFlt *pSh_catWeight = sh_stateFreqAndCatWeight + nState;
	CUFlt *pSh_tempLnL = pSh_catWeight + nCategory;
	if(curSite < nSitePattern){
		int curState, totalCatCnt = 0, iEigen, iRateCat;
		int step = blockDim.x;
		CUFlt curSum;
		CUFlt *pCondlike = rootCondlike + curSite * nPaddedState;
		const int condlike_rateCat_offset = nPaddedSitePattern * nPaddedState;

		for(iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			for(iRateCat = 0; iRateCat < nRateCategory; iRateCat ++, pCondlike += condlike_rateCat_offset, totalCatCnt ++){
				curSum = 0.0f;

//#pragma unroll
				
				for(curState = tx; curState < nState; curState += step){
					curSum += pCondlike[curState] * sh_stateFreqAndCatWeight[curState];
				}
				
				totalSum += curSum * pSh_catWeight[totalCatCnt];
			}
		}
	}
	
	pSh_tempLnL[ind] = totalSum;
	__syncthreads();
	
	// assert step is 2^m (4 / 8 / 16 / ....)
	for(int reduceSize = (step >> 1); reduceSize > 0; reduceSize >>= 1){
		if(tx < reduceSize)
			pSh_tempLnL[ind] += pSh_tempLnL[ind + reduceSize];
		__syncthreads();
	}

	
	if(curSite < nSitePattern && tx == 0){
		if(pSh_tempLnL[threadIdx.y * step] <= 0)
			totalSum = log(CUFLT_MIN);
		else
			totalSum = log(pSh_tempLnL[threadIdx.y * step]);

		int iNode;
		scaleFactor += curSite;
		for(iNode = 0; iNode < nNodeScaler; iNode ++, scaleFactor += nPaddedSitePattern)
			totalSum += scaleFactor[0];

		siteLnL[curSite] = totalSum;
	}
}





 // reduction of site likelihoods:
__global__
void kernelReductionOfSiteLnL_xState_baseline(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *sitePatternWeight, const int nThreadPerBlock, const int nThread, const int nSitePattern)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	extern __shared__ CUFlt sh_lnL[];

	CUFlt curSum = 0.0f;
	int ind;
#pragma unroll
	for(ind = bx * blockDim.x + tx; ind < nSitePattern; ind += nThread){
		curSum += siteLnL[ind] * sitePatternWeight[ind];
	}

	sh_lnL[tx] = curSum;
	__syncthreads();

	
	for(int reduceSize = (nThreadPerBlock >> 1); reduceSize > 0; reduceSize >>= 1){
		if(tx < reduceSize)
			sh_lnL[tx] += sh_lnL[tx + reduceSize];
		__syncthreads();
	}

	if(0 == tx)
		reduceLnL[bx] = sh_lnL[0];
}



// First call kernel to calculate site likelihood, then call kernel for reduction of site likelihoods;
// TODO: ��������state��Ŀ��ô����??? ���⣬��stateΪpad�����Ŀ���Ƿ��������⣿��ϸ���һ��kernel�Ƿ��pad���state��Ч!!!
int callKernelLikelihood_baseline(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *rootCondlike, CUFlt *rateCatWeight, CUFlt *stateFreq, CUFlt *sitePatternWeight, int nNodeScaler, CUFlt *scaleFactor, const int nPaddedState, const int nState, const int nEigenDecomp, const int nRateCategory, const int nSitePattern, const int nPaddedSitePattern, dim3 nBlockPerGrid_siteLnL, dim3 nThreadPerBlock_siteLnL, int nBlockPerGrid_reduce, int nThreadPerBlock_reduce, cudaStream_t &stream)
{
	const int nCategory = nEigenDecomp * nRateCategory;
	int sharedMemSize;

	// Calculation of site likelihood values:
	if(4 == nPaddedState){
		//printf("\n=======\nGoing to call kernel for 4 state of site lnL...\n==========\n");
		sharedMemSize = (nCategory + BLOCK_DIMENSION_X_SITE_LNL_4STATE_BASELINE * BLOCK_DIMENSION_Y_SITE_LNL_4STATE_BASELINE) * sizeof(CUFlt);
		kernelSiteLnL_4State_baseline<<<nBlockPerGrid_siteLnL, nThreadPerBlock_siteLnL, sharedMemSize, stream>>>(siteLnL, rootCondlike, rateCatWeight, stateFreq, nNodeScaler, scaleFactor, nEigenDecomp, nRateCategory, nSitePattern, nPaddedSitePattern);
		cutilCheckMsg("kernelSiteLnL_4State_baseline() failed");
	}
	else if(20 == nPaddedState){
		//printf("\n=======\nGoing to call kernel for 20 state of site lnL...\n==========\n");
		sharedMemSize = (nCategory + BLOCK_DIMENSION_X_SITE_LNL_20STATE_BASELINE * BLOCK_DIMENSION_Y_SITE_LNL_20STATE_BASELINE) * sizeof(CUFlt);
		kernelSiteLnL_20State_baseline<<<nBlockPerGrid_siteLnL, nThreadPerBlock_siteLnL, sharedMemSize, stream>>>(siteLnL, rootCondlike, rateCatWeight, stateFreq, nNodeScaler, scaleFactor, nEigenDecomp, nRateCategory, nSitePattern, nPaddedSitePattern);
		cutilCheckMsg("kernelSiteLnL_20State_baseline() failed");
	}
	else if(64 == nPaddedState){
		//printf("\n=======\nGoing to call kernel for 64 state of site lnL...\n==========\n");
		sharedMemSize = (nCategory + BLOCK_DIMENSION_X_SITE_LNL_64STATE_BASELINE * BLOCK_DIMENSION_Y_SITE_LNL_64STATE_BASELINE) * sizeof(CUFlt);
		kernelSiteLnL_64State_baseline<<<nBlockPerGrid_siteLnL, nThreadPerBlock_siteLnL, sharedMemSize, stream>>>(siteLnL, rootCondlike, rateCatWeight, stateFreq, nNodeScaler, scaleFactor, nEigenDecomp, nRateCategory, nSitePattern, nPaddedSitePattern, nState);
		cutilCheckMsg("kernelSiteLnL_64State_baseline() failed");
	}
	else{ 
		// For nPaddedState != 4 / 20 / 64:
		//printf("\n=======\nGoing to call kernel for X state of site lnL...\n==========\n");
		sharedMemSize = (nState + nCategory +  nThreadPerBlock_siteLnL.x * nThreadPerBlock_siteLnL.y) * sizeof(CUFlt);
		kernelSiteLnL_xState_baseline<<<nBlockPerGrid_siteLnL, nThreadPerBlock_siteLnL, sharedMemSize, stream>>>(siteLnL, rootCondlike, rateCatWeight, stateFreq, nNodeScaler, scaleFactor, nEigenDecomp, nRateCategory, nSitePattern, nPaddedSitePattern, nState, nPaddedState);
		cutilCheckMsg("kernelSiteLnL_xState_baseline() failed");
	}

	// Reduction of site likelihood values:
	int nThread = nBlockPerGrid_reduce * nThreadPerBlock_reduce;
	kernelReductionOfSiteLnL_xState_baseline<<<nBlockPerGrid_reduce, nThreadPerBlock_reduce, nThreadPerBlock_reduce * sizeof(CUFlt), stream>>>(reduceLnL, siteLnL, sitePatternWeight, nThreadPerBlock_reduce, nThread, nSitePattern);
	cutilCheckMsg("kernelReductionOfSiteLnL_xState_baseline() failed");

	return nBlockPerGrid_reduce;
}



int callKernelLikelihoodFromSiteLnL_baseline(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *sitePatternWeight, const int nSitePattern, int nBlockPerGrid, int nThreadPerBlock, cudaStream_t &stream)
{
	const int nThread = nBlockPerGrid * nThreadPerBlock;

	kernelReductionOfSiteLnL_xState_baseline<<<nBlockPerGrid, nThreadPerBlock, nThreadPerBlock * sizeof(CUFlt), stream>>>(reduceLnL, siteLnL, sitePatternWeight, nThreadPerBlock, nThread, nSitePattern);
	cutilCheckMsg("kernelReductionOfSiteLnL_xState_baseline() failed");

	return nBlockPerGrid;
}

