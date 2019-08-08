#include "CuLibPartitionImpl.h"
#include "CuLibKernel-baseline-rooted.h"
#include "CuLibKernel-codemlAndMrBayes-rooted.h"
#include "CuLibKernel-baseline-unrooted.h"
#include "CuLibKernel-codemlAndMrBayes-unrooted.h"
//#include "CuLibImpl-common.h"
#include <queue>
#include <set>
#include <sys/time.h>


#ifdef DEBUG_TIME
const int nIteration = 1000;
long long PMat_time, condlike_time, lnL_time, scale_time;
long long multiple = 1000000ll;
//static int conp_cnt=0;
struct timeval tBegin;
void timeBegin(){
	gettimeofday(&tBegin, NULL);
}

long long timeEnd(){
	 struct timeval tEnd;
     gettimeofday(&tEnd, NULL);
	
	 long long usec = tEnd.tv_sec * multiple + tEnd.tv_usec - (tBegin.tv_sec * multiple + tBegin.tv_usec);
	 return usec;
}
#endif


/*
// TODO: 创建和销毁cuda stream的工作之后应该交给更上层的接口完成;
CuLErrorCode PartitionInstance::createCUDAStream(int nStream,
												int streamId)
{
	int streamCnt = nStream;
	if(nStream > 32)
		streamCnt = 32;
	for(int iStream = 0; iStream < streamCnt; iStream ++){
		cutilSafeCall(cudaStreamCreate(&_stream[iStream]));
		cutilCheckMsg("cudaStreamCreate() failed");
	}

	_streamId = streamId;

	return CUL_SUCCESS;
}

CuLErrorCode PartitionInstance::destroyCUDAStream(int nStream)
{
	int streamCnt = nStream;
	if(nStream > 32)
		streamCnt = 32;
	for(int iStream = 0; iStream < streamCnt; iStream ++){
		cutilSafeCall(cudaStreamDestroy(_stream[iStream]));
		cutilCheckMsg("cudaStreamDestroy() failed");
	}

	return CUL_SUCCESS;
}
*/



/*
// 构造函数：
// TODO: 构造函数无返回值，无法返回error code；
PartitionInstance::PartitionInstance(int nNode,
									int nState,
									int nSitePattern,
									int nRateCategory,
									int nEigenDecomposition,
									int nNodeForTransitionMatrix,
									int nTipStateArray,
									int nTipCondlikeArray,
									int nInternalNodeForCondlike)
{
	specifyParams(nNode,
						nState,
						nSitePattern,
						nRateCategory,
						nEigenDecomposition,
						nNodeForTransitionMatrix,
						nTipStateArray,
						nTipCondlikeArray,
						nInternalNodeForCondlike);
}
*/



// 析构函数:
PartitionInstance::~PartitionInstance()
{
#ifdef DEBUG_DESTROY
	printf("Entering ~PartitionInstance()...\n");
#endif

	CuLErrorCode returnState = finalize();
	if(returnState != CUL_SUCCESS)
		printErrorCode(returnState);

#ifdef DEBUG_DESTROY
	printf("Leaving ~PartitionInstance()...\n");
#endif
}


void PartitionInstance::setPMatKernelInfor(void)
{
	_PMat_blockDim_z = 1;
	_nPMatArrayPerBlock = 0;
	_nBlockPerPMatArray = 0;

	if(4 == _nPaddedState){
		_PMat_blockDim_x = BLOCK_DIMENSION_X_PMAT_4STATE_BASELINE;
		_PMat_blockDim_y = BLOCK_DIMENSION_Y_PMAT_4STATE_BASELINE;
	}
	else if(20 == _nPaddedState){
		_PMat_blockDim_x = BLOCK_DIMENSION_X_PMAT_20STATE_BASELINE;
		_PMat_blockDim_y = BLOCK_DIMENSION_Y_PMAT_20STATE_BASELINE;
	}
	else if(64 == _nPaddedState){
		_PMat_blockDim_x = N_THREAD_PER_BLOCK_PMAT_CODEML;
		_PMat_blockDim_y = 1;

		_nBlockPerPMatArray = _nPaddedState * _nPaddedState / (N_ELEMENT_PER_THREAD_PMAT_CODEML * _PMat_blockDim_x);

		assert(_nPaddedState == _nBlockPerPMatArray * N_ELEMENT_PER_THREAD_PMAT_CODEML);
	}
	else{		
		// For padded version of state == X:
		if(8 == _nPaddedState){
			_PMat_blockDim_x = BLOCK_DIMENSION_X_PMAT_XSTATE_8_BASELINE;
			_PMat_blockDim_y = BLOCK_DIMENSION_Y_PMAT_XSTATE_8_BASELINE;
		}
		else if(16 == _nPaddedState){
			_PMat_blockDim_x = BLOCK_DIMENSION_X_PMAT_XSTATE_16_BASELINE;
			_PMat_blockDim_y = BLOCK_DIMENSION_Y_PMAT_XSTATE_16_BASELINE;
		}
		else if(24 == _nPaddedState){
			_PMat_blockDim_x = BLOCK_DIMENSION_X_PMAT_XSTATE_24_BASELINE;
			_PMat_blockDim_y = BLOCK_DIMENSION_Y_PMAT_XSTATE_24_BASELINE;
		}
		else if(32 == _nPaddedState){
			_PMat_blockDim_x = BLOCK_DIMENSION_X_PMAT_XSTATE_32_BASELINE;
			_PMat_blockDim_y = BLOCK_DIMENSION_Y_PMAT_XSTATE_32_BASELINE;
		}
		else if(40 == _nPaddedState){
			_PMat_blockDim_x = BLOCK_DIMENSION_X_PMAT_XSTATE_40_BASELINE;
			_PMat_blockDim_y = BLOCK_DIMENSION_Y_PMAT_XSTATE_40_BASELINE;
		}
		else if(48 == _nPaddedState){
			_PMat_blockDim_x = BLOCK_DIMENSION_X_PMAT_XSTATE_48_BASELINE;
			_PMat_blockDim_y = BLOCK_DIMENSION_Y_PMAT_XSTATE_48_BASELINE;
		}
		else if(56 == _nPaddedState){
			_PMat_blockDim_x = BLOCK_DIMENSION_X_PMAT_XSTATE_56_BASELINE;
			_PMat_blockDim_y = BLOCK_DIMENSION_Y_PMAT_XSTATE_56_BASELINE;
		}
		else{
			// For nPaddedState > 64:
			_PMat_blockDim_x = (_nPaddedState + N_ELEMENT_PER_THREAD_PMAT_XSTATE_LARGE_STATE - 1) / N_ELEMENT_PER_THREAD_PMAT_XSTATE_LARGE_STATE;
			_PMat_blockDim_y = BLOCK_DIMENSION_Y_PMAT_XSTATE_LARGE_STATE_BASELINE;
			_nBlockPerPMatArray = (_nPaddedState + _PMat_blockDim_y - 1) / _PMat_blockDim_y;
		}
	}

	if(0 == _nBlockPerPMatArray)
		_nPMatArrayPerBlock = _PMat_blockDim_y * _PMat_blockDim_z;

	// Make sure only one of the two settings is valid;
	assert((0 == _nPMatArrayPerBlock && 0 != _nBlockPerPMatArray) || (0 != _nPMatArrayPerBlock && 0 == _nBlockPerPMatArray));
}



void PartitionInstance::setCondlikeKernelInfor(void)
{
	int nBlockPerArray_pad, nBlockPerArray_noPad, nSitePerBlock;
	bool useCodeml = false;

	_nBlockPerClArray_nonCodeml = 0;

	_condlike_blockDim_y = 1;
	_condlike_blockDim_z = 1;
	_condlike_blockDim_y_pad = 1;
	_condlike_blockDim_z_pad = 1;
	_condlike_blockDim_y_noPad = 1;
	_condlike_blockDim_z_noPad = 1;

	// For nPaddedState == 4 / 20 / 64:
	if(4 == _nPaddedState){
#ifdef USE_OLD_VERSION
		// The old schedule scheme: for nPaddedState = 4, each thread is responsible for 4 site patterns, and block dimension is: (128, 1);
		_condlike_blockDim_x_noPad = _condlike_blockDim_x_pad = N_THREAD_PER_BLOCK_CONDLIKE_4STATE_BASELINE;
		nSitePerBlock = N_SITE_PER_THREAD_CONDLIKE_4STATE_BASELINE * N_THREAD_PER_BLOCK_CONDLIKE_4STATE_BASELINE;
#else
		// The new schedule scheme: calculate the maximum block count for each array
		nSitePerBlock = N_SITE_PER_THREAD_CONDLIKE_4STATE_VERSION2 * N_THREAD_PER_BLOCK_CONDLIKE_4STATE_VERSION2;
#endif
	}
	else if(20 == _nPaddedState){
		// For nPaddedState == 20, each thread is responsible for 5 states, and 4 threads are responsible for a site pattern
		// a thread block is for 64 site patterns, and block dimension is: (4, 64);
		_condlike_blockDim_x_noPad = _condlike_blockDim_x_pad = _nPaddedState / N_STATE_PER_THREAD_CONDLIKE_20STATE_BASELINE;
		nSitePerBlock = N_SITE_PER_BLOCK_CONDLIKE_20STATE_BASELINE;
		_condlike_blockDim_y_noPad = _condlike_blockDim_y_pad = nSitePerBlock;
	}
	else if(64 == _nPaddedState){
		// For nPaddedState == 64, each thread is responsible for 16 elements, and 2 thread blocks are responsible for 32 site patterns;
		// the block dimension is: (8, 8);
		useCodeml = true;
		_condlike_blockDim_x = _condlike_blockDim_x_pad = _condlike_blockDim_x_noPad = BLOCK_DIMENSION_X_CONDLIKE_CODEML;
		_condlike_blockDim_y = _condlike_blockDim_y_pad = _condlike_blockDim_y_noPad = BLOCK_DIMENSION_Y_CONDLIKE_CODEML;
	}
	else{
		// For nPaddedState = x:
		// For padded version, use shared memory for the tile of PMat/condlike;
		if(8 == _nPaddedState){
			_condlike_blockDim_x_pad = BLOCK_DIMENSION_X_CONDLIKE_XSTATE_8_BASELINE;
			_condlike_blockDim_y_pad = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE;
		}
		else if(16 == _nPaddedState){
			_condlike_blockDim_x_pad = BLOCK_DIMENSION_X_CONDLIKE_XSTATE_16_BASELINE;
			_condlike_blockDim_y_pad = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE;
		}
		else if(24 == _nPaddedState){
			_condlike_blockDim_x_pad = BLOCK_DIMENSION_X_CONDLIKE_XSTATE_24_BASELINE;
			_condlike_blockDim_y_pad = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE;
		}
		else if(32 == _nPaddedState){
			_condlike_blockDim_x_pad = _nPaddedState / N_STATE_PER_THREAD_CONDLIKE_XSTATE_OTHER_BASELINE;
			_condlike_blockDim_y_pad = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_32_BASELINE;
		}
		else{
			_condlike_blockDim_x_pad = _nPaddedState / N_STATE_PER_THREAD_CONDLIKE_XSTATE_OTHER_BASELINE;
			_condlike_blockDim_y_pad = BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_OTHER_BASELINE;
		}
		
		int nPatternPerBlock = _condlike_blockDim_y_pad;
		_nBlockPerClArray_nonCodeml = (_nPaddedSitePattern + nPatternPerBlock - 1) / nPatternPerBlock;
		nBlockPerArray_pad = _nBlockPerClArray_nonCodeml;
		
		
		// For no-pad version, just pad the state count to a multiply of 8, and do not use shared memory for tile of PMat/condlike;
		// If nPaddedState < 32, the block dimension is: (128, 1) and each thread is for 16 states, else (256, 1) and each thread is for 8 states;
		int nElemPerThread;
		if(_nPaddedState < XSTATE_THRESHOLD){
			nElemPerThread = N_STATE_PER_THREAD_CONDLIKE_XSTATE_SMALL_BASELINE;
			_condlike_blockDim_x_noPad = N_THREAD_PER_BLOCK_CONDLIKE_XSTATE_SMALL_BASELINE;
		}
		else{
			nElemPerThread = N_STATE_PER_THREAD_CONDLIKE_XSTATE_LARGE_BASELINE;
			_condlike_blockDim_x_noPad = N_THREAD_PER_BLOCK_CONDLIKE_XSTATE_LARGE_BASELINE;
		}

		int nElemPerArray = _nSitePattern * _nPaddedState;		// The padded site patterns don't need to calculate;
		int nElemPerBlock = nElemPerThread * (_condlike_blockDim_x_noPad * _condlike_blockDim_y_noPad * _condlike_blockDim_z_noPad);
		nBlockPerArray_noPad = (nElemPerArray + nElemPerBlock - 1) / nElemPerBlock;
		_nBlockPerClArray_nonCodeml = nBlockPerArray_noPad;
	}
	
	if(!useCodeml){
		// For nPaddedState != 64:
		if(0 == _nBlockPerClArray_nonCodeml){
			// For nPaddedState != x:
			nBlockPerArray_pad = (_nSitePattern + nSitePerBlock - 1) / nSitePerBlock;
			_nBlockPerClArray_nonCodeml = nBlockPerArray_noPad = nBlockPerArray_pad;
		}
	}
	else{
		// For nPaddedState == 64, assert nPaddedSitePattern is a multiply of 32;
		_nBlockPerClArray_codeml = nBlockPerArray_pad = nBlockPerArray_noPad = _nPaddedSitePattern / 32 * 2;
	}

	_nThreadBlockPerClArray_pad = nBlockPerArray_pad;
	_nThreadBlockPerClArray_noPad = nBlockPerArray_noPad;
	_nThreadBlockPerClArray = max(nBlockPerArray_pad, nBlockPerArray_noPad);
}



void PartitionInstance::setScaleKernelInfor(void)
{
	// For use of scaling factors, block dimension is: (128, 1)
	_scale_blockDim_x = N_THREAD_PER_BLOCK_SCALE_BASELINE;

	// assert each thread is responsible for the scaling of a site pattern:
	_nThreadBlockPerScaleNode = (_nSitePattern + _scale_blockDim_x - 1) / _scale_blockDim_x;
}



void PartitionInstance::setLikelihoodKernelInfor(void)
{
	int nSitePerBlock;
	_siteLnL_blockDim_y = 1;
	_siteLnL_blockDim_z = 1;

	_siteLnL_gridDim_y = 1;
	_siteLnL_gridDim_z = 1;

	if(4 == _nPaddedState){
		// For nPaddedState == 4, block dimension is: (2, 32), and 2 threads are responsible for a site pattern, a block is responsible for 32 site patterns;
		_siteLnL_blockDim_x = BLOCK_DIMENSION_X_SITE_LNL_4STATE_BASELINE;
		_siteLnL_blockDim_y = BLOCK_DIMENSION_Y_SITE_LNL_4STATE_BASELINE;

		nSitePerBlock = _siteLnL_blockDim_y;
	}
	else if(20 == _nPaddedState){
		// For nPaddedState == 20, block dimension is: (8, 16), and 8 threads are responsible for a site pattern, a block is responsible for 16 site patterns;
		_siteLnL_blockDim_x = BLOCK_DIMENSION_X_SITE_LNL_20STATE_BASELINE;
		_siteLnL_blockDim_y = BLOCK_DIMENSION_Y_SITE_LNL_20STATE_BASELINE;
		
		nSitePerBlock = _siteLnL_blockDim_y;
	}
	else if(64 == _nPaddedState){
		// For nPaddedState == 64, block dimension is: (16, 8), and 16 threads are responsible for a site pattern, a block is responsible for 8 site patterns;
		_siteLnL_blockDim_x = BLOCK_DIMENSION_X_SITE_LNL_64STATE_BASELINE;
		_siteLnL_blockDim_y = BLOCK_DIMENSION_Y_SITE_LNL_64STATE_BASELINE;
		
		nSitePerBlock = _siteLnL_blockDim_y;
	}
	else{			
		// For nPaddedState != 4 / 20 / 64
		int nCategory = _nEigenDecomposition * _nRateCategory;
		if((_nPaddedState <= 8 && nCategory <= 4) || (_nPaddedState <= 16 && nCategory <= 2)){
			// For small state count && small category count, set the block dimension to: (4, 32); every 4 threads are responsible for a site pattern;
			// and a thread block is for 32 site patterns;
			_siteLnL_blockDim_x = BLOCK_DIMENSION_X_SITE_LNL_XSTATE_BASELINE_SMALL;
			_siteLnL_blockDim_y = BLOCK_DIMENSION_Y_SITE_LNL_XSTATE_BASELINE_SMALL;
		}
		else{
			// For large state count || large category count, set the block dimension to: (8, 16); every 8 threads are responsible for a site pattern;
			// and a thread block is for 16 site patterns;
			_siteLnL_blockDim_x = BLOCK_DIMENSION_X_SITE_LNL_XSTATE_BASELINE_LARGE;
			_siteLnL_blockDim_y = BLOCK_DIMENSION_Y_SITE_LNL_XSTATE_BASELINE_LARGE;
		}

		nSitePerBlock = _siteLnL_blockDim_y;
	}

	_siteLnL_gridDim_x = (_nSitePattern + nSitePerBlock - 1) / nSitePerBlock;


	// For reduction of site likelihoods, thread block dimension is: (128, 1);
	_reduceLnL_blockDim_x = N_THREAD_PER_BLOCK_REDUCE_BASELINE;
	
	// If the nSitePattern < 10000, each thread is responsible for the reduction of 2 elements, else 4 elements; 
	int nElementPerBlock;
	if(_nSitePattern < SITE_PATTERN_THRESHOLD_REDUCE)
		nElementPerBlock = _reduceLnL_blockDim_x * N_ELEMENT_PER_THREAD_REDUCE_BASELINE_SMALL;
	else
		nElementPerBlock = _reduceLnL_blockDim_x * N_ELEMENT_PER_THREAD_REDUCE_BASELINE_LARGE;
	
	_reduceLnL_gridDim_x = (_nSitePattern + nElementPerBlock - 1) / nElementPerBlock;
}



// TODO: 错误检查机制以及函数返回值???
// Specify the paramters of the current partition instance;
// 假设每个node的PMat/condlike的数目为：nEigenDecomposition * nRateCategory;
CuLErrorCode PartitionInstance::specifyParams(int nNode,
											int nState,
											int nSitePattern,
											int nRateCategory,
											int nEigenDecomposition,
											int nNodeForTransitionMatrix,
											int nTipStateArray,
											int nTipCondlikeArray,
											int nInternalNodeForCondlike,
											int nNodeScaler,
											bool isRootedTree)
{
#ifdef DEBUG
	printf("Entering specifyParams()...\n");
#endif

	_nNode = nNode;
	_nState = nState;
	_nSitePattern = nSitePattern;
	_nRateCategory = nRateCategory;
	_nEigenDecomposition = nEigenDecomposition;
	_nNodeForTransitionMatrix = nNodeForTransitionMatrix;
	_nTipStateArray = nTipStateArray;
	_nTipCondlikeArray = nTipCondlikeArray;
	_nInternalNodeForCondlike = nInternalNodeForCondlike;
	_curNodeScalerCnt = _nNodeScaler = nNodeScaler;

	_isRootedTree = isRootedTree;

	_nArrayPerNode = _nRateCategory * _nEigenDecomposition;			// how many transition matrices / condlike arrays are allocated for each node;

	// TODO: 对state/site pattern进行pad，另外，应该对哪些matrix进行pad，U/V/R/PMat/conP...?
	// TODO: 对nState=20的是否进行pad，对其余state数目如何进行pad???
	
	// pad the state to a multiply of PAD_SIZE:
	if(_nState <= 4){
		_nPaddedState = 4;
	}
	else if(20 == _nState){
		_nPaddedState = 20;
	}
	else{
		_nPaddedState = (_nState + PAD_SIZE - 1) / PAD_SIZE * PAD_SIZE;
	}
	
	if(4 == _nPaddedState){
		_nPaddedSitePattern = _nSitePattern;
	}
	else if(8 == _nPaddedState){
		_nPaddedSitePattern = (_nSitePattern + BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE - 1) / BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_8_BASELINE;
	}
	else if(16 == _nPaddedState){
		_nPaddedSitePattern = (_nSitePattern + BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE - 1) / BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_16_BASELINE;
	}
	else if(20 == _nPaddedState){
		_nPaddedSitePattern = (_nSitePattern + N_SITE_PER_BLOCK_CONDLIKE_20STATE_BASELINE - 1) / N_SITE_PER_BLOCK_CONDLIKE_20STATE_BASELINE * N_SITE_PER_BLOCK_CONDLIKE_20STATE_BASELINE;
	}
	else if(24 == _nPaddedState){
		_nPaddedSitePattern = (_nSitePattern + BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE - 1) / BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_24_BASELINE;
	}
	else if(32 == _nPaddedState){
		_nPaddedSitePattern = (_nSitePattern + BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_32_BASELINE - 1) / BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_32_BASELINE * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_32_BASELINE;
	}
	else if(64 == _nPaddedState){
		_nPaddedSitePattern = (_nSitePattern + N_SITE_PER_BLOCK_CONDLIKE_64STATE_BASELINE - 1) / N_SITE_PER_BLOCK_CONDLIKE_64STATE_BASELINE * N_SITE_PER_BLOCK_CONDLIKE_64STATE_BASELINE;
	}
	else{
		_nPaddedSitePattern = (_nSitePattern + BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_OTHER_BASELINE - 1) / BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_OTHER_BASELINE * BLOCK_DIMENSION_Y_CONDLIKE_XSTATE_OTHER_BASELINE;
	}

	//printf("====================\nPad infor: nState = %d, nPaddedState = %d, nSitePattern = %d, nPaddedSitePattern = %d\n====================\n", _nState, _nPaddedState, _nSitePattern, _nPaddedSitePattern);
	

#ifdef DEBUG_VALUE
	printf("_nState = %d, _nPaddedState = %d, _nSitePattern = %d, _nPaddedSitePattern = %d\n", _nState, _nPaddedState, _nSitePattern, _nPaddedSitePattern);
#endif

	_PMat_size = _nState * _nState;
	_PMat_pad_size = _nPaddedState * _nPaddedState;
	_condlike_size = _nState * _nSitePattern;
	_condlike_pad_size = _nPaddedState * _nPaddedSitePattern;

	// For PMat/condlike offset:
	_maxPMatOpCount = _nNodeForTransitionMatrix * _nArrayPerNode;
	_maxCondlikeOpCount = _nInternalNodeForCondlike * _nArrayPerNode;
	_maxScaleOpCount = max(_nNode, _nNodeScaler);

	 // Set block/grid dimensions of kernels:
	setPMatKernelInfor();

	setCondlikeKernelInfor();

	setScaleKernelInfor();

	setLikelihoodKernelInfor();
	

	// Calculate the memory needed:
	int memory_needed_host = (_nNodeForTransitionMatrix * _nArrayPerNode * _PMat_pad_size + \						// _host_PMat
						_nTipCondlikeArray * _condlike_pad_size + \												// _host_tipCondlike
						_nInternalNodeForCondlike * _nArrayPerNode * _condlike_pad_size) * sizeof(CUFlt) + \	// _host_intCondlike
						_nTipStateArray * _nPaddedSitePattern * sizeof(int);									// _host_tipState

	int memory_needed_host_pinned = (_nRateCategory + \									// _host_rate
								_nArrayPerNode + \									// _host_rateCatWeight
								_nPaddedState + \									// _host_stateFreq
								_nPaddedSitePattern + \								// _host_sitePatternWeight
								_nRateCategory * _nNodeForTransitionMatrix + \		// _host_brLen
								_nEigenDecomposition * _PMat_pad_size * 2 + \		// _host_U && _host_V
								_nEigenDecomposition * _nPaddedState + \			// _host_R
								_nPaddedSitePattern * 2) * sizeof(CUFlt) + \		// _host_siteLnL && _host_reduceLnL
								(_nNodeForTransitionMatrix * _nArrayPerNode + \		// _host_PMat_offset
								_nInternalNodeForCondlike * _nArrayPerNode * 2 + \	// _host_condlike_op && _host_condlike_opStartBlkInd
								_maxCondlikeOpCount * _nThreadBlockPerClArray + \	// _host_condlike_blkIndToOpInd
								_maxScaleOpCount * _nThreadBlockPerScaleNode * 3) * sizeof(int);		// __host_scale_blkIndToClOffset && _host_scale_blkIndToScaleOffset && _host_scale_startBlkInd

	int memory_needed_device = memory_needed_host + \
							memory_needed_host_pinned + 
							_maxScaleOpCount * _nPaddedSitePattern * sizeof(CUFlt);		// _dev_nodeScaleFactor
	
	if(64 == _nPaddedState)
		memory_needed_device += _nArrayPerNode * _nPaddedState * sizeof(CUFlt);


	size_t memory_total_device, memory_free_device;
	getDeviceMemoryInfo(&memory_total_device, &memory_free_device);

#ifdef DEBUG_VALUE
	printf("\nTotal device memory: %d\nAvailable device memory: %d\nMemory needed host: %d\nMemory needed host pinned: %d\nMemory needed device: %d\n", memory_total_device, memory_free_device, memory_needed_host, memory_needed_host_pinned, memory_needed_device);
#endif

	if(memory_free_device < memory_needed_device)
		return CUL_ERROR_OUT_OF_DEVICE_MEMORY;

	// Allocate host page lock memory:
	callocHostPinnedMemory(_host_rate, _nRateCategory);
	if(NULL == _host_rate)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_rateCatWeight, _nArrayPerNode);
	if(NULL == _host_rateCatWeight)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_stateFreq, _nPaddedState);
	if(NULL == _host_stateFreq)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_sitePatternWeight, _nPaddedSitePattern);
	if(NULL == _host_sitePatternWeight)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_brLen, _nRateCategory * _nNodeForTransitionMatrix);
	if(NULL == _host_brLen)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_U, _nEigenDecomposition * _PMat_pad_size);
	if(NULL == _host_U)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_V, _nEigenDecomposition * _PMat_pad_size);
	if(NULL == _host_V)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_R, _nEigenDecomposition * _nPaddedState);
	if(NULL == _host_R)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_siteLnL, _nPaddedSitePattern);
	if(NULL == _host_siteLnL)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_reduceLnL, _nPaddedSitePattern);
	if(NULL == _host_reduceLnL)
		return CUL_ERROR_BAD_ALLOC;


	// Allocate common pageable memory:
	_host_PMat =(CUFlt *) calloc(_nNodeForTransitionMatrix * _nArrayPerNode * _PMat_pad_size, sizeof(CUFlt));
	if(NULL == _host_PMat)
		return CUL_ERROR_BAD_ALLOC;

	if(_nTipStateArray > 0){
		_host_tipState =(int *) calloc(_nTipStateArray * _nPaddedSitePattern, sizeof(int));
		if(NULL == _host_tipState)
			return CUL_ERROR_BAD_ALLOC;
	}
	else
		_host_tipState = NULL;

	if(_nTipCondlikeArray > 0){
		_host_tipCondlike =(CUFlt *) calloc(_nTipCondlikeArray * _condlike_pad_size, sizeof(CUFlt));
		if(NULL == _host_tipCondlike)
			return CUL_ERROR_BAD_ALLOC;
	}
	else
		_host_tipCondlike = NULL;

	_host_intCondlike =(CUFlt *) calloc(_nInternalNodeForCondlike * _nArrayPerNode * _condlike_pad_size, sizeof(CUFlt));
	if(NULL == _host_intCondlike)
		return CUL_ERROR_BAD_ALLOC;

	_nodesToScale = (int *) calloc(_maxScaleOpCount, sizeof(int));
	if(NULL == _nodesToScale)
		return CUL_ERROR_BAD_ALLOC;


	// For offset purpose, allocate host page locked memory:

	callocHostPinnedMemory(_host_PMat_offset, _maxPMatOpCount);
	if(NULL == _host_PMat_offset)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_condlike_op, _maxCondlikeOpCount);
	if(NULL == _host_condlike_op)
		return CUL_ERROR_BAD_ALLOC;

	// TODO: blkIndToOpInd的大小需要根据site pattern以及state的数目精确计算

	callocHostPinnedMemory(_host_condlike_blkIndToOpInd, _maxCondlikeOpCount * _nThreadBlockPerClArray);
	if(NULL == _host_condlike_blkIndToOpInd)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_condlike_opStartBlkInd, _maxCondlikeOpCount);
	if(NULL == _host_condlike_opStartBlkInd)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_scale_blkIndToClOffset, _maxScaleOpCount * _nThreadBlockPerScaleNode);
	if(NULL == _host_scale_blkIndToClOffset)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_scale_blkIndToScaleOffset, _maxScaleOpCount * _nThreadBlockPerScaleNode);
	if(NULL == _host_scale_blkIndToScaleOffset)
		return CUL_ERROR_BAD_ALLOC;

	callocHostPinnedMemory(_host_scale_startBlkInd, _maxScaleOpCount * _nThreadBlockPerScaleNode);
	if(NULL == _host_scale_startBlkInd)
		return CUL_ERROR_BAD_ALLOC;


	// Allocate device memory for tiP, condLike...
	callocDeviceMemory(_dev_rate, _nRateCategory);
	callocDeviceMemory(_dev_rateCatWeight, _nArrayPerNode);
	callocDeviceMemory(_dev_stateFreq, _nPaddedState);
	callocDeviceMemory(_dev_sitePatternWeight, _nPaddedSitePattern);
	callocDeviceMemory(_dev_brLen, _nRateCategory * _nNodeForTransitionMatrix);
	callocDeviceMemory(_dev_U, _nEigenDecomposition * _PMat_pad_size);
	callocDeviceMemory(_dev_V, _nEigenDecomposition * _PMat_pad_size);
	callocDeviceMemory(_dev_R, _nEigenDecomposition * _nPaddedState);
	callocDeviceMemory(_dev_PMat, _nNodeForTransitionMatrix * _nArrayPerNode * _PMat_pad_size);
	if(_nTipStateArray > 0)
		callocDeviceMemory(_dev_tipState, _nTipStateArray * _nPaddedSitePattern);
	else
		_dev_tipState = NULL;
	if(_nTipCondlikeArray > 0)
		callocDeviceMemory(_dev_tipCondlike, _nTipCondlikeArray * _condlike_pad_size);
	else
		_dev_tipCondlike = NULL;
	callocDeviceMemory(_dev_intCondlike, _nInternalNodeForCondlike * _nArrayPerNode * _condlike_pad_size);
	callocDeviceMemory(_dev_nodeScaleFactor, _maxScaleOpCount * _nPaddedSitePattern);
	callocDeviceMemory(_dev_siteLnL, _nPaddedSitePattern);
	callocDeviceMemory(_dev_reduceLnL, _nPaddedSitePattern);
	
	// TODO: (state==64)CuCodeML在计算PMat时分成了两步，第一步计算exp(t * r)，考虑合并???
	// 若不合并，检查_dev_exptRoot的大小;
	if(64 == _nPaddedState){
		callocDeviceMemory(_dev_exptRoot, _maxPMatOpCount * _nPaddedState);
	}
		
	callocDeviceMemory(_dev_PMat_offset, _maxPMatOpCount);
	callocDeviceMemory(_dev_condlike_op, _maxCondlikeOpCount);

	callocDeviceMemory(_dev_condlike_blkIndToOpInd, _maxCondlikeOpCount * _nThreadBlockPerClArray);
	callocDeviceMemory(_dev_condlike_opStartBlkInd, _maxCondlikeOpCount);

	callocDeviceMemory(_dev_scale_blkIndToClOffset, _maxScaleOpCount * _nThreadBlockPerScaleNode);
	callocDeviceMemory(_dev_scale_blkIndToScaleOffset, _maxScaleOpCount * _nThreadBlockPerScaleNode);
	callocDeviceMemory(_dev_scale_startBlkInd, _maxScaleOpCount * _nThreadBlockPerScaleNode);

#ifdef DEBUG_VALUE
	printf("\n_nNode = %d, _nState = %d, _nSitePattern = %d, _nRateCategory = %d, _nEigenDecomposition = %d, _nNodeForTransitionMatrix = %d, _nTipStateArray = %d, _nTipCondlikeArray = %d, _nInternalNodeForCondlike = %d\n", _nNode, _nState, _nSitePattern, _nRateCategory, _nEigenDecomposition, _nNodeForTransitionMatrix, _nTipStateArray, _nTipCondlikeArray, _nInternalNodeForCondlike);
#endif
#ifdef DEBUG
	printf("Leaving specifyParams()...\n");
#endif
	
	return CUL_SUCCESS;
}



// Specify the tree topology of the current partition;
// TODO: 此处是将tree复制了一份，是否有必要??? 另外，复制的新的node是否需要释放空间???
CuLErrorCode PartitionInstance::specifyTreeTopology(CuLTreeNode *root)
{
#ifdef DEBUG
	printf("Entering specifyTreeTopology()...\n");
#endif

	_rootLabel = root->label;
	_nLayerOfTree = 0;
	_nodeLayerMap.clear();

	std::map<CuLTreeNode*, CuLTreeNode*> nodeMap;
	std::queue<CuLTreeNode*> nodeQue;
	nodeQue.push(root);

	while(!nodeQue.empty()){
		int curSize = nodeQue.size();
		vector<CuLTreeNode*> nodeLayerVec;

		for(int iNode = 0; iNode < curSize; iNode ++){
			CuLTreeNode *curNode = nodeQue.front();
			nodeQue.pop();
			
			CuLTreeNode *newNode;
			if(nodeMap.end() == nodeMap.find(curNode)){
				newNode = new CuLTreeNode(curNode->label, curNode->nChild);
				nodeMap.insert(pair<CuLTreeNode*, CuLTreeNode*>(curNode, newNode));
			}
			else{
				newNode = nodeMap[curNode];
			}

			nodeLayerVec.push_back(newNode);

			for(int iChild = 0; iChild < curNode->nChild; iChild ++){
				CuLTreeNode *curChild = curNode->child[iChild];
				if(nodeMap.end() == nodeMap.find(curChild)){
					CuLTreeNode* newChild = new CuLTreeNode(curChild->label, curChild->nChild);
					
					newNode->child[iChild] = newChild;
					nodeMap.insert(pair<CuLTreeNode*, CuLTreeNode*>(curChild, newChild));
				}
				else{
					newNode->child[iChild] = nodeMap[curNode];
				}
			}
			
			for(int iChild = 0; iChild < curNode->nChild; iChild ++)
				nodeQue.push(curNode->child[iChild]);
		}
		
		_nodeLayerMap[_nLayerOfTree] = nodeLayerVec;
		_nLayerOfTree ++;
	}

#ifdef DEBUG
	printf("Leaving specifyTreeTopology()...\n");
#endif

	return CUL_SUCCESS;
}



// Specify the state frequencies of the current partition;
// TODO: host to device的copy的size大小应该为_nState还是_nPaddedState???
CuLErrorCode PartitionInstance::specifyStateFrequency(const double *inStateFreq,
														cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering specifyStateFrequency()...\n");
#endif

	memcpyHostToHost(_host_stateFreq, inStateFreq, _nState);

	memcpyHostToDeviceAsync(_dev_stateFreq, _host_stateFreq, _nState, stream);		// TODO: copy的大小为nState还是nPaddedState???

#ifdef DEBUG
	printf("Leaving specifyStateFrequency()...\n");
#endif

	return CUL_SUCCESS;
}


CuLErrorCode PartitionInstance::getStateFrequency(double *outStateFreq,
													cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering getStateFrequency()...\n");
#endif
	
	memcpyDeviceToHostAsync(_host_stateFreq, _dev_stateFreq, _nState, stream);		// TODO: copy的大小为nState还是nPaddedState???
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	memcpyHostToHost(outStateFreq, _host_stateFreq, _nState);

#ifdef DEBUG
	printf("Leaving getStateFrequency()...\n");
#endif

	return CUL_SUCCESS;
}



// Specify the site pattern weights of the current partition;
// TODO: host to device的copy的size应该为_nSitePattern还是_nPaddedSitePattern???
CuLErrorCode PartitionInstance::specifySitePatternWeight(const double *inPatternWeight,
														cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering specifySitePatternWeight()...\n");
#endif
	
	memcpyHostToHost(_host_sitePatternWeight, inPatternWeight, _nSitePattern);
	memcpyHostToDeviceAsync(_dev_sitePatternWeight, _host_sitePatternWeight, _nSitePattern, stream);

#ifdef DEBUG
	printf("Leaving specifySitePatternWeight()...\n");
#endif

	return CUL_SUCCESS;
}


// Get the site pattern weights of the current partition;
CuLErrorCode PartitionInstance::getSitePatternWeight(double *outPatternWeight,
														cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering getSitePatternWeight()...\n");
#endif

	memcpyDeviceToHostAsync(_host_sitePatternWeight, _dev_sitePatternWeight, _nSitePattern, stream);		// TODO: copy的大小为nState还是nPaddedState???
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	memcpyHostToHost(outPatternWeight, _host_sitePatternWeight, _nSitePattern);

#ifdef DEBUG
	printf("Leaving getSitePatternWeight()...\n");
#endif
	return CUL_SUCCESS;
}



// Specify the rates of the current partition;
CuLErrorCode PartitionInstance::specifyRate(const double *inRate,
											cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering specifyRate()...\n");
#endif

	memcpyHostToHost(_host_rate, inRate, _nRateCategory);
	memcpyHostToDeviceAsync(_dev_rate, _host_rate, _nRateCategory, stream);
	
#ifdef DEBUG
	printf("Leaving specifyRate()...\n");
#endif

	return CUL_SUCCESS;
}



// Get the rates of the current partition;
CuLErrorCode PartitionInstance::getRate(double *outRate,
										cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering getRate()...\n");
#endif

	memcpyDeviceToHostAsync(_host_rate, _dev_rate, _nRateCategory, stream);		// TODO: copy的大小为nState还是nPaddedState???
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	memcpyHostToHost(outRate, _host_rate, _nRateCategory);

#ifdef DEBUG
	printf("Leaving getRate()...\n");
#endif

	return CUL_SUCCESS;
}



// Specify the rate category weights of the current partition;
CuLErrorCode PartitionInstance::specifyRateCategoryWeight(const int nCategory,
														const int *categoryId,
														const double **inRateCategoryWeight,
														cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering specifyRateCategoryWeight()...\n");
#endif

	for(int iCategory = 0; iCategory < nCategory; iCategory ++){
		if(categoryId[iCategory] < 0 || categoryId[iCategory] >= _nEigenDecomposition)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		memcpyHostToHost(_host_rateCatWeight + categoryId[iCategory] * _nRateCategory, inRateCategoryWeight[iCategory], _nRateCategory);
	}

	memcpyHostToDeviceAsync(_dev_rateCatWeight, _host_rateCatWeight, _nEigenDecomposition * _nRateCategory, stream);

#ifdef DEBUG
	printf("Leaving specifyRateCategoryWeight()...\n");
#endif

	return CUL_SUCCESS;
}



// Get the rate category weights of the current partition;
CuLErrorCode PartitionInstance::getRateCategoryWeight(const int nCategory,
												const int *categoryId,
												double **outRateCategoryWeight,
												cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering getRateCategoryWeight()...\n");
#endif

	memcpyDeviceToHostAsync(_host_rateCatWeight, _dev_rateCatWeight, _nEigenDecomposition * _nRateCategory, stream);
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	for(int iCategory = 0; iCategory < nCategory; iCategory ++){
		if(categoryId[iCategory] < 0 || categoryId[iCategory] >= _nEigenDecomposition)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		memcpyHostToHost(outRateCategoryWeight[iCategory], _host_rateCatWeight + categoryId[iCategory] * _nRateCategory, _nRateCategory);
	}

#ifdef DEBUG
	printf("Leaving getRateCategoryWeight()...\n");
#endif

	return CUL_SUCCESS;
}



// Specify the eigen decomposition of the current partition;
CuLErrorCode PartitionInstance::specifyEigenDecomposition(const int nEigenDecomp,
														const int *eigenDecompId,
														const double **inEigenVector,
														const double **inInverEigenVector,
														const double **inEigenValue,
														cudaStream_t stream)
{	
#ifdef DEBUG
	printf("Entering specifyEigenDecomposition()...\n");
#endif

	CUFlt *pU = NULL, *pV = NULL, *pR = NULL;
	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		if(eigenDecompId[iEigen] < 0 || eigenDecompId[iEigen] >= _nEigenDecomposition)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		pU = _host_U + eigenDecompId[iEigen] * _PMat_pad_size;
		pV = _host_V + eigenDecompId[iEigen] * _PMat_pad_size;
		pR = _host_R + eigenDecompId[iEigen] * _nPaddedState;

		memcpyHostToHost(pR, inEigenValue[iEigen], _nState);

		if(_nPaddedState == _nState){
			// Copy the whole matrix:
			memcpyHostToHost(pU, inEigenVector[iEigen], _PMat_pad_size);
			memcpyHostToHost(pV, inInverEigenVector[iEigen], _PMat_pad_size);
		}
		else{
			// Copy one state by another:
			for(int iState = 0; iState < _nState; iState ++){
				memcpyHostToHost(pU + iState * _nPaddedState, inEigenVector[iEigen] + iState * _nState, _nState);
				memcpyHostToHost(pV + iState * _nPaddedState, inInverEigenVector[iEigen] + iState * _nState, _nState);
			}
		}
	}

	memcpyHostToDeviceAsync(_dev_U, _host_U, _nEigenDecomposition * _PMat_pad_size, stream);
	memcpyHostToDeviceAsync(_dev_V, _host_V, _nEigenDecomposition * _PMat_pad_size, stream);
	memcpyHostToDeviceAsync(_dev_R, _host_R, _nEigenDecomposition * _nPaddedState, stream);

#ifdef DEBUG
	printf("Leaving specifyEigenDecomposition()...\n");
#endif

	return CUL_SUCCESS;
}



// Get the eigen decomposition of the current partition;
CuLErrorCode PartitionInstance::getEigenDecomposition(const int nEigenDecomp,
												const int *eigenDecompId,
												double **outEigenVector,
												double **outInverEigenVector,
												double **outEigenValue,
												cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering getEigenDecomposition()...\n");
#endif

	memcpyDeviceToHostAsync(_host_U, _dev_U, _nEigenDecomposition * _PMat_pad_size, stream);
	memcpyDeviceToHostAsync(_host_V, _dev_V, _nEigenDecomposition * _PMat_pad_size, stream);
	memcpyDeviceToHostAsync(_host_R, _dev_R, _nEigenDecomposition * _nPaddedState, stream);
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	CUFlt *pU = NULL, *pV = NULL, *pR = NULL;
	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		if(eigenDecompId[iEigen] < 0 || eigenDecompId[iEigen] >= _nEigenDecomposition)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		pU = _host_U + eigenDecompId[iEigen] * _PMat_pad_size;
		pV = _host_V + eigenDecompId[iEigen] * _PMat_pad_size;
		pR = _host_R + eigenDecompId[iEigen] * _nPaddedState;

		memcpyHostToHost(outEigenValue[iEigen], pR, _nState);

		if(_nPaddedState == _nState){
			memcpyHostToHost(outEigenVector[iEigen], pU, _PMat_pad_size);
			memcpyHostToHost(outInverEigenVector[iEigen], pV, _PMat_pad_size);
		}
		else{
			for(int iState = 0; iState < _nState; iState ++){
				memcpyHostToHost(outEigenVector[iEigen] + iState * _nState, pU + iState * _nPaddedState, _nState);
				memcpyHostToHost(outInverEigenVector[iEigen] + iState * _nState, pV + iState * _nPaddedState, _nState);
			}
		}
	}

#ifdef DEBUG
	printf("Leaving getEigenDecomposition()...\n");
#endif

	return CUL_SUCCESS;
}



// Specify the transition matrix of the current partition;
CuLErrorCode PartitionInstance::specifyTransitionMatrixMulti(const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														const double **inMatrix)
{
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg("cudaDeviceSynchronize() failed");
	printf("Entering specifyTransitionMatrixMulti()...\n");
#endif

	const int PMat_size = _PMat_size;
	const int PMat_eigen_size = _nRateCategory * PMat_size;
	const int PMat_pad_size = _PMat_pad_size;
	const int PMat_pad_eigen_size = _nRateCategory * PMat_pad_size;
	const int PMat_pad_node_size = _nEigenDecomposition * PMat_pad_eigen_size;

	//memset(_host_PMat, 0, _nNodeForTransitionMatrix * PMat_pad_node_size * sizeof(CUFlt));

	CUFlt *pPMat = NULL;
	double *pInPMat = NULL;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(nodeId[iNode] < 0 || nodeId[iNode] >= _nNodeForTransitionMatrix)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		//memset(pPMat, 0, PMat_pad_node_size * sizeof(CUFlt));

		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			if(eigenDecompId[iEigen] < 0 || eigenDecompId[iEigen] >= _nEigenDecomposition)
				return CUL_ERROR_INDEX_OUT_OF_RANGE;

			pPMat = _host_PMat + nodeId[iNode] * PMat_pad_node_size + eigenDecompId[iEigen] * PMat_pad_eigen_size;
			pInPMat = (double *)inMatrix[iNode] + iEigen * PMat_eigen_size;

			if(_nPaddedState == _nState){
				// Directly copy the whole matrix:
				memcpyHostToHost(pPMat, pInPMat, PMat_eigen_size);
			}
			else{
				// Copy one state by another:
				for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pPMat += PMat_pad_size, pInPMat += PMat_size){
					for(int iState = 0; iState < _nState; iState ++)
						memcpyHostToHost(pPMat + iState * _nPaddedState, pInPMat + iState * _nState, _nState);
				}
			}
		}
	}

	memcpyHostToDeviceSync(_dev_PMat, _host_PMat, _nNodeForTransitionMatrix * PMat_pad_node_size);

#ifdef DEBUG
	printf("Leaving specifyTransitionMatrixMulti()...\n");
#endif

	return CUL_SUCCESS;
}



// Specify the transition matrix of the current partition;
// 假设nodeId[]中指定的index即为该node对应的PMat array的index;
CuLErrorCode PartitionInstance::specifyTransitionMatrixAll(const int nNode,
														const int *nodeId,
														const double **inMatrix)
{
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg("cudaDeviceSynchronize() failed");
	printf("Entering specifyTransitionMatrixAll()...\n");
#endif

	const int PMat_size = _PMat_size;
	const int PMat_pad_size = _PMat_pad_size;
	const int PMat_pad_eigen_size = _nRateCategory * PMat_pad_size;
	const int PMat_pad_node_size = _nEigenDecomposition * PMat_pad_eigen_size;

	//memset(_host_PMat, 0, _nNodeForTransitionMatrix * PMat_pad_node_size * sizeof(CUFlt));

	CUFlt *pPMat = NULL;
	double *pInPMat = NULL;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(nodeId[iNode] < 0 || nodeId[iNode] >= _nNodeForTransitionMatrix)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		pPMat = _host_PMat + nodeId[iNode] * PMat_pad_node_size;
		pInPMat = (double *)inMatrix[iNode];

		if(_nPaddedState == _nState){
			// Directly copy the whole matrix:
			memcpyHostToHost(pPMat, pInPMat, PMat_pad_node_size);
		}
		else{
			// Copy one state by another:
			for(int iEigen = 0; iEigen < _nEigenDecomposition; iEigen ++){
				for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pPMat += PMat_pad_size, pInPMat += PMat_size){
					for(int iState = 0; iState < _nState; iState ++)
						memcpyHostToHost(pPMat + iState * _nPaddedState, pInPMat + iState * _nState, _nState);
				}
			}
		}
	}

	memcpyHostToDeviceSync(_dev_PMat, _host_PMat, _nNodeForTransitionMatrix * PMat_pad_node_size);

#ifdef DEBUG
	printf("Leaving specifyTransitionMatrixAll()...\n");
#endif

	return CUL_SUCCESS;
}



inline void callKernelPMat(CUFlt *P, CUFlt *U, CUFlt *V, CUFlt *R, CUFlt *brLen, CUFlt *exptRootAll, CuLPMatOffset *offset, const int nMatrix, const int nState, const int nPaddedState, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t stream)
{
#ifdef DEBUG_TIME
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	timeBegin();
	for(int itr = 0; itr < 1; itr ++){
#endif

	// For nPaddedState == 64, use codeml version else use baseline version;
	if(nPaddedState != 64)
		callKernelPMat_baseline(P, U, V, R, brLen, offset, nMatrix, nState, nPaddedState, nBlockPerGrid, nThreadPerBlock, stream);
	else
		callKernelPMat_codeml(P, U, V, R, brLen, exptRootAll, offset, nMatrix, nState, nPaddedState, nBlockPerGrid, nThreadPerBlock, stream);
	
#ifdef DEBUG_TIME
	}
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	PMat_time = timeEnd();

	FILE *fout = fopen("PMat_time.txt", "a");
	fprintf(fout, "for nState = %d, nPMat = %d, nIteration = %d: %lld us (%lld.%06lld s)\n\n", nState, nMatrix, nIteration, PMat_time, PMat_time / multiple, PMat_time % multiple);
	fclose(fout);

	FILE *fout2 = fopen("PMat_time_format.txt", "a");
	fprintf(fout2, "%lld.%06lld\t", PMat_time / multiple, PMat_time % multiple);
	fclose(fout2);
#endif
}


// Calculate the transition matrices of the current partition;
// TODO: nState = 4/20以及其他state数目；nState = 61 => nState = 64时暂时用的CuCodeML的方法，也即每个thread负责16个matrix element的计算，block大小为(64, 4)，也即一个block负责一个node，一共需要nMatrix个block，需要事先计算好每个block的offset以及brLen * rate的值；
// 注意：此处假设_nTransitionMatrixPerNode = _nEigenDecomposition * _nRateCategory，核对是否正确，另外，假设PMat的组织形式为：同一个node的所有eigen decomposition的所有rate category的PMat在一起，也即同一个eigen decomposition的所有rate category的PMat在一起；
// Calculate the transition matrices of the current partition;
CuLErrorCode PartitionInstance::calculateTransitionMatrixMulti(const int nNode,
															const int *nodeId,
															const int nEigenDecomp,
															const int *eigenDecompId,
															const double *brLen,
															cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering calculateTransitionMatrixMulti()...\n");
#endif

	const int PMat_pad_size = _PMat_pad_size;
	const int PMat_pad_eigen_size = _nRateCategory * PMat_pad_size;
	const int PMat_pad_node_size = _nEigenDecomposition * PMat_pad_eigen_size;

	resetHostMemory(_host_PMat_offset, _maxPMatOpCount);

	int cntMatrix = 0, brLen_offset, PMat_offset, UV_offset, R_offset;
	for(int iNode = 0; iNode < nNode; iNode ++){
		int curNode = nodeId[iNode];
		if(curNode < 0 || curNode >= _nNodeForTransitionMatrix)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			if(eigenDecompId[iEigen] < 0 || eigenDecompId[iEigen] >= _nEigenDecomposition)
				return CUL_ERROR_INDEX_OUT_OF_RANGE;

			PMat_offset = curNode * PMat_pad_node_size + eigenDecompId[iEigen] * PMat_pad_eigen_size;
			UV_offset = eigenDecompId[iEigen] * PMat_pad_size;
			R_offset = eigenDecompId[iEigen] * _nPaddedState;

			for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, PMat_offset += PMat_pad_size){
				brLen_offset = iRateCat * _nNodeForTransitionMatrix + curNode;
				_host_brLen[brLen_offset] = brLen[iNode] * _host_rate[iRateCat];

				_host_PMat_offset[cntMatrix].brLen_offset = brLen_offset;
				_host_PMat_offset[cntMatrix].UV_offset = UV_offset;
				_host_PMat_offset[cntMatrix].R_offset = R_offset;
				_host_PMat_offset[cntMatrix].P_offset = PMat_offset;

				cntMatrix ++;
			}
		}
	}

	assert(nNode * nEigenDecomp * _nRateCategory == cntMatrix);

	memcpyHostToDeviceAsync(_dev_PMat_offset, _host_PMat_offset, cntMatrix, stream);
	memcpyHostToDeviceAsync(_dev_brLen, _host_brLen, _nRateCategory * _nNodeForTransitionMatrix, stream);

	int gridDim_x = 1, gridDim_y = 1, gridDim_z = 1;
	
	// If _nPaddedState is small, a thread block is responsible for k PMat matrices, if _nPaddedState is large, m thread blocks is responsible for a PMat matrix;
	if(0 != _nPMatArrayPerBlock){
		gridDim_x = (cntMatrix + _nPMatArrayPerBlock - 1) / _nPMatArrayPerBlock;
	}
	else{
		gridDim_x = _nBlockPerPMatArray;
		gridDim_y = cntMatrix;
	}


	dim3 nBlockPerGrid(gridDim_x, gridDim_y, gridDim_z);
	dim3 nThreadPerBlock(_PMat_blockDim_x, _PMat_blockDim_y, _PMat_blockDim_z);

	//printf("===============\nPMat kernel infor: \nblock dimension: (%d, %d, %d)\ngrid dimension: (%d, %d, %d)\n===================\n", nThreadPerBlock.x, nThreadPerBlock.y, nThreadPerBlock.z, nBlockPerGrid.x, nBlockPerGrid.y, nBlockPerGrid.z);

	callKernelPMat(_dev_PMat, _dev_U, _dev_V, _dev_R, _dev_brLen, _dev_exptRoot, _dev_PMat_offset, cntMatrix, _nState, _nPaddedState, nBlockPerGrid, nThreadPerBlock, stream);

#ifdef DEBUG
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	printf("Leaving calculateTransitionMatrixMulti()...\n");
#endif

	return CUL_SUCCESS;
}



// Calculate the transition matrices of the current partition;
CuLErrorCode PartitionInstance::calculateTransitionMatrixAll(const int nNode,
															const int *nodeId,
															const double *brLen,
															cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering calculateTransitionMatrixAll()...\n");
#endif

	const int PMat_pad_size = _PMat_pad_size;
	const int PMat_pad_eigen_size = _nRateCategory * PMat_pad_size;
	const int PMat_pad_node_size = _nEigenDecomposition * PMat_pad_eigen_size;

	resetHostMemory(_host_PMat_offset, _nNodeForTransitionMatrix * _nArrayPerNode);

	int cntMatrix = 0, brLen_offset, PMat_offset, UV_offset, R_offset;
	for(int iNode = 0; iNode < nNode; iNode ++){
		int curNode = nodeId[iNode];
		if(curNode < 0 || curNode >= _nNodeForTransitionMatrix)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		PMat_offset = curNode * PMat_pad_node_size;

		for(int iEigen = 0; iEigen < _nEigenDecomposition; iEigen ++){
			UV_offset = iEigen * PMat_pad_size;
			R_offset = iEigen * _nPaddedState;

			for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, PMat_offset += PMat_pad_size){
				brLen_offset = iRateCat * _nNodeForTransitionMatrix + curNode;
				_host_brLen[brLen_offset] = brLen[iNode] * _host_rate[iRateCat];

				_host_PMat_offset[cntMatrix].brLen_offset = brLen_offset;
				_host_PMat_offset[cntMatrix].UV_offset = UV_offset;
				_host_PMat_offset[cntMatrix].R_offset = R_offset;
				_host_PMat_offset[cntMatrix].P_offset = PMat_offset;

				cntMatrix ++;
			}
		}
	}

	assert(nNode * _nEigenDecomposition * _nRateCategory == cntMatrix);

	memcpyHostToDeviceAsync(_dev_PMat_offset, _host_PMat_offset, cntMatrix, stream);
	memcpyHostToDeviceAsync(_dev_brLen, _host_brLen, _nRateCategory * _nNodeForTransitionMatrix, stream);

	int gridDim_x = 1, gridDim_y = 1, gridDim_z = 1;
	
	// If _nPaddedState is small, a thread block is responsible for k PMat matrices, if _nPaddedState is large, m thread blocks is responsible for a PMat matrix;
	if(0 != _nPMatArrayPerBlock){
		gridDim_x = (cntMatrix + _nPMatArrayPerBlock - 1) / _nPMatArrayPerBlock;
	}
	else{
		gridDim_x = _nBlockPerPMatArray;
		gridDim_y = cntMatrix;
	}

	dim3 nBlockPerGrid(gridDim_x, gridDim_y, gridDim_z);
	dim3 nThreadPerBlock(_PMat_blockDim_x, _PMat_blockDim_y, _PMat_blockDim_z);

	printf("===============\nPMat kernel infor: \nblock dimension: (%d, %d, %d)\ngrid dimension: (%d, %d, %d)\n===================\n", nThreadPerBlock.x, nThreadPerBlock.y, nThreadPerBlock.z, nBlockPerGrid.x, nBlockPerGrid.y, nBlockPerGrid.z);

	callKernelPMat(_dev_PMat, _dev_U, _dev_V, _dev_R, _dev_brLen, _dev_exptRoot, _dev_PMat_offset, cntMatrix, _nState, _nPaddedState, nBlockPerGrid, nThreadPerBlock, stream);


#ifdef DEBUG
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	printf("Leaving calculateTransitionMatrixAll()...\n");
#endif

	return CUL_SUCCESS;
}



// Get transition matrix of a partition;
CuLErrorCode PartitionInstance::getTransitionMatrixMulti(const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														double **outMatrix)
{
#ifdef DEBUG
	printf("Entering getTransitionMatrixMulti()...\n");
#endif

	const int PMat_size = _PMat_size;
	const int PMat_eigen_size = _nRateCategory * PMat_size;
	const int PMat_pad_size = _PMat_pad_size;
	const int PMat_pad_eigen_size = _nRateCategory * PMat_pad_size;
	const int PMat_pad_node_size = _nEigenDecomposition * PMat_pad_eigen_size;
	
	memcpyDeviceToHostSync(_host_PMat, _dev_PMat, _nNodeForTransitionMatrix * PMat_pad_node_size);

	/*
#ifdef USE_CODEML_PMAT
	transposeMatrix(_host_PMat, _nNodeForTransitionMatrix * _nArrayPerNode, _nPaddedState, _nPaddedState);
#endif
	*/

	CUFlt *pPMat = NULL;
	double *pOutPMat = NULL;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(nodeId[iNode] < 0 || nodeId[iNode] >= _nNodeForTransitionMatrix)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			if(eigenDecompId[iEigen] < 0 || eigenDecompId[iEigen] >= _nEigenDecomposition)
				return CUL_ERROR_INDEX_OUT_OF_RANGE;

			pPMat = _host_PMat + nodeId[iNode] * PMat_pad_node_size + eigenDecompId[iEigen] * PMat_pad_eigen_size;
			pOutPMat = outMatrix[iNode] + iEigen * PMat_eigen_size;

			if(_nPaddedState == _nState){
				// Directly copy the whole matrix:
				memcpyHostToHost(pOutPMat, pPMat, PMat_eigen_size);
			}
			else{
				// Copy one state by another:
				for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pPMat += PMat_pad_size, pOutPMat += PMat_size){
					for(int iState = 0; iState < _nState; iState ++)
						memcpyHostToHost(pOutPMat + iState * _nState, pPMat + iState * _nPaddedState, _nState);
				}
			}
		}
	}

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg("cudaDeviceSynchronize() failed");
	printf("Leaving getTransitionMatrixMulti()...\n");
#endif

	return CUL_SUCCESS;
}


// Get the transition matrices of the current partition;
CuLErrorCode PartitionInstance::getTransitionMatrixAll(const int nNode,
														const int *nodeId,
														double **outMatrix)
{
#ifdef DEBUG
	printf("Entering getTransitionMatrixAll()...\n");
#endif

	const int PMat_size = _PMat_size;
	const int PMat_pad_size = _PMat_pad_size;
	const int PMat_pad_eigen_size = _nRateCategory * PMat_pad_size;
	const int PMat_pad_node_size = _nEigenDecomposition * PMat_pad_eigen_size;
	
	memcpyDeviceToHostSync(_host_PMat, _dev_PMat, _nNodeForTransitionMatrix * PMat_pad_node_size);

	/*
#ifdef USE_CODEML_PMAT
	transposeMatrix(_host_PMat, _nNodeForTransitionMatrix * _nArrayPerNode, _nPaddedState, _nPaddedState);
#endif
	*/

	CUFlt *pPMat = NULL;
	double *pOutPMat = NULL;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(nodeId[iNode] < 0 || nodeId[iNode] >= _nNodeForTransitionMatrix)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		pPMat = _host_PMat + nodeId[iNode] * PMat_pad_node_size;
		pOutPMat = outMatrix[iNode];

		if(_nPaddedState == _nState){
			// Directly copy the whole matrix:
			memcpyHostToHost(pOutPMat, pPMat, PMat_pad_node_size);
		}
		else{
			// Copy one state by another:
			for(int iEigen = 0; iEigen < _nEigenDecomposition; iEigen ++){
				for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pPMat += PMat_pad_size, pOutPMat += PMat_size){
					for(int iState = 0; iState < _nState; iState ++)
						memcpyHostToHost(pOutPMat + iState * _nState, pPMat + iState * _nPaddedState, _nState);
				}
			}
		}
	}

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg("cudaDeviceSynchronize() failed");
	printf("Leaving getTransitionMatrixAll()...\n");
#endif

	return CUL_SUCCESS;
}



// TODO: 想明白这里的node id还是tree中的label，若为tree中的label，则还需经过nodeIndToArrayInd[]的转换;
// Specify the tip states of the current partition;
// 目前假设tipNodeId[]即为该node对应的tipState array的index;
CuLErrorCode PartitionInstance::specifyTipState(const int nTipNode,
												const int *tipNodeId,
												const int **inTipState)
{	
#ifdef DEBUG
	printf("Entering specifyTipState()...\n");
#endif
	//memset(_host_tipState, 0, _nTipStateArray * state_pad_size * sizeof(int));
	
	int *pTipState = NULL, *pInTipState = NULL;
	for(int iNode = 0; iNode < nTipNode; iNode ++){
		if(tipNodeId[iNode] < 0 || tipNodeId[iNode] >= _nTipStateArray)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		pTipState = _host_tipState + tipNodeId[iNode] * _nPaddedSitePattern;
		pInTipState = (int *)inTipState[iNode];
		
		memcpyHostToHost(pTipState, pInTipState, _nSitePattern);
	}

	memcpyHostToDeviceSync(_dev_tipState, _host_tipState, _nTipStateArray * _nPaddedSitePattern);

#ifdef DEBUG
	printf("Leaving specifyTipState()...\n");
#endif

	return CUL_SUCCESS;
}


CuLErrorCode PartitionInstance::getTipState(const int nTipNode,
											const int *tipNodeId,
											int **outTipState)
{
#ifdef DEBUG
	printf("Entering getTipState()...\n");
#endif

	memcpyDeviceToHostSync(_host_tipState, _dev_tipState, _nTipStateArray * _nPaddedSitePattern);


	int *pTipState = NULL, *pOutTipState = NULL;
	for(int iNode = 0; iNode < nTipNode; iNode ++){
		if(tipNodeId[iNode] < 0 || tipNodeId[iNode] >= _nTipStateArray)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		pTipState = _host_tipState + tipNodeId[iNode] * _nPaddedSitePattern;
		pOutTipState = outTipState[iNode];
		
		memcpyHostToHost(pOutTipState, pTipState, _nSitePattern);
	}

#ifdef DEBUG
	printf("Leaving getTipState()...\n");
#endif

	return CUL_SUCCESS;
}



// Specify the tip conditional likelihoods of the current partition;
// 目前假设tipNodeId[]即为该node在tipCondlike array中的index;
CuLErrorCode PartitionInstance::specifyTipCondlike(const int nTipNode,
												const int *tipNodeId,
												const double **inTipCondlike)
{
#ifdef DEBUG
	printf("Entering specifyTipCondlike()...\n");
#endif

	const int condlike_size = _condlike_size;
	const int condlike_pad_size = _condlike_pad_size;

	//memset(_host_tipCondlike, 0, _nTipCondlikeArray * condlike_pad_size * sizeof(CUFlt));
	
	CUFlt *pTipCondlike = NULL;
	double *pInTipCondlike = NULL;
	for(int iNode = 0; iNode < nTipNode; iNode ++){
		if(tipNodeId[iNode] < 0 || tipNodeId[iNode] >= _nTipCondlikeArray)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		pTipCondlike = _host_tipCondlike + tipNodeId[iNode] * condlike_pad_size;
		pInTipCondlike = (double *)inTipCondlike[iNode];

		if(_nPaddedState == _nState){
			// Directly copy the whole matrix:
			memcpyHostToHost(pTipCondlike, pInTipCondlike, condlike_size);
		}
		else{
			// Copy one pattern by another:
			for(int iPattern = 0; iPattern < _nSitePattern; iPattern ++)
				memcpyHostToHost(pTipCondlike + iPattern * _nPaddedState, pInTipCondlike + iPattern * _nState, _nState);
		}
	}

	memcpyHostToDeviceSync(_dev_tipCondlike, _host_tipCondlike, _nTipCondlikeArray * condlike_pad_size);

#ifdef DEBUG
	printf("Leaving specifyTipCondlike()...\n");
#endif

	return CUL_SUCCESS;
}


CuLErrorCode PartitionInstance::getTipCondlike(const int nTipNode,
											const int *tipNodeId,
											double **outTipCondlike)
{
#ifdef DEBUG
	printf("Entering getTipCondlike()...\n");
#endif

	memcpyDeviceToHostSync(_host_tipCondlike, _dev_tipCondlike, _nTipCondlikeArray * _condlike_pad_size);

	const int condlike_size = _condlike_size;
	const int condlike_pad_size = _condlike_pad_size;

	//memset(_host_tipCondlike, 0, _nTipCondlikeArray * condlike_pad_size * sizeof(CUFlt));
	
	CUFlt *pTipCondlike = NULL;
	double *pOutTipCondlike = NULL;
	for(int iNode = 0; iNode < nTipNode; iNode ++){
		if(tipNodeId[iNode] < 0 || tipNodeId[iNode] >= _nTipCondlikeArray)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		pTipCondlike = _host_tipCondlike + tipNodeId[iNode] * condlike_pad_size;
		pOutTipCondlike = outTipCondlike[iNode];

		if(_nPaddedState == _nState){
			memcpyHostToHost(pOutTipCondlike, pTipCondlike, condlike_size);
		}
		else{
			for(int iPattern = 0; iPattern < _nSitePattern; iPattern ++)
				memcpyHostToHost(pOutTipCondlike + iPattern * _nState, pTipCondlike + iPattern * _nPaddedState, _nState);
		}
	}

#ifdef DEBUG
	printf("Leaving getTipCondlike()...\n");
#endif

	return CUL_SUCCESS;
}



// 假设此处的nodeId为tree中的label;
// Specify the conditional likelihoods of internal nodes of the current partition;
CuLErrorCode PartitionInstance::specifyInternalCondlikeMulti(const int nNode,
															const int *nodeId,
															const int nEigenDecomp,
															const int *eigenDecompId,
															const double **inCondlike)
{
#ifdef DEBUG
	printf("Entering specifyInternalCondlikeMulti()...\n");
#endif

	const int condlike_size = _condlike_size;
	const int condlike_eigen_size = _nRateCategory * condlike_size;
	const int condlike_pad_size = _condlike_pad_size;
	const int condlike_pad_eigen_size = _nRateCategory * condlike_pad_size;
	const int condlike_pad_node_size = _nEigenDecomposition * condlike_pad_eigen_size;

	const int nTipNode = _nTipStateArray + _nTipCondlikeArray;

	CUFlt *pCondlike = NULL;
	double *pInCondlike = NULL;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(nodeId[iNode] < 0 || _nodeIndToArrayInd.end() == _nodeIndToArrayInd.find(nodeId[iNode]))
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		int arrayId = _nodeIndToArrayInd[nodeId[iNode]] - nTipNode;
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			if(eigenDecompId[iEigen] < 0 || eigenDecompId[iEigen] >= _nEigenDecomposition)
				return CUL_ERROR_INDEX_OUT_OF_RANGE;

			pCondlike = _host_intCondlike + arrayId * condlike_pad_node_size + eigenDecompId[iEigen] * condlike_pad_eigen_size;
			pInCondlike = (double *)inCondlike[iNode] + iEigen * condlike_eigen_size;

			if(_nPaddedState == _nState){
				if(_nPaddedSitePattern == _nSitePattern){
					// Directly copy the whole matrix:
					memcpyHostToHost(pCondlike, pInCondlike, condlike_eigen_size);
				}
				else{
					// Copy one rate category by another:
					for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pCondlike += condlike_pad_size, pInCondlike += condlike_size){
						memcpyHostToHost(pCondlike, pInCondlike, condlike_size);
					}
				}
			}
			else{
				// Copy one site pattern by another:
				for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pCondlike += condlike_pad_size, pInCondlike += condlike_size){
					for(int iPattern = 0; iPattern < _nSitePattern; iPattern ++)
						memcpyHostToHost(pCondlike + iPattern * _nPaddedState, pInCondlike + iPattern * _nState, _nState);
				}
			}
		}
	}

	memcpyHostToDeviceSync(_dev_intCondlike, _host_intCondlike, _nInternalNodeForCondlike * condlike_pad_node_size);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg("cudaDeviceSynchronize() failed");
	printf("Leaving specifyInternalCondlikeMulti()...\n");
#endif

	return CUL_SUCCESS; 
}



// Specify the conditional likelihoods of internal nodes of the current partition; 
CuLErrorCode PartitionInstance::specifyInternalCondlikeAll(const int nNode,
															const int *nodeId,
															const double **inCondlike)
{
#ifdef DEBUG
	printf("Entering specifyInternalCondlikeAll()...\n");
#endif

	const int condlike_size = _condlike_size;
	const int condlike_node_size = _nArrayPerNode * condlike_size;
	const int condlike_pad_size = _condlike_pad_size;
	const int condlike_pad_node_size = _nArrayPerNode * condlike_pad_size;

	const int nTipNode = _nTipStateArray + _nTipCondlikeArray;

	CUFlt *pCondlike = NULL;
	double *pInCondlike = NULL;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(nodeId[iNode] < 0 || _nodeIndToArrayInd.end() == _nodeIndToArrayInd.find(nodeId[iNode]))
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		int arrayId = _nodeIndToArrayInd[nodeId[iNode]] - nTipNode;

		pCondlike = _host_intCondlike + arrayId * condlike_pad_node_size;
		pInCondlike = (double *)inCondlike[iNode];

		if(_nPaddedState == _nState){
			if(_nPaddedSitePattern == _nSitePattern){
				// Directly copy the whole matrix:
				memcpyHostToHost(pCondlike, pInCondlike, condlike_node_size);
			}
			else{
				// Copy one rate category by another:
				for(int iEigen = 0; iEigen < _nEigenDecomposition; iEigen ++){
					for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pCondlike += condlike_pad_size, pInCondlike += condlike_size){
						memcpyHostToHost(pCondlike, pInCondlike, condlike_size);
					}
				}
			}
		}
		else{
			// Copy one site pattern by another:
			for(int iEigen = 0; iEigen < _nEigenDecomposition; iEigen ++){
				for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pCondlike += condlike_pad_size, pInCondlike += condlike_size){
					for(int iPattern = 0; iPattern < _nSitePattern; iPattern ++)
						memcpyHostToHost(pCondlike + iPattern * _nPaddedState, pInCondlike + iPattern * _nState, _nState);
				}
			}
		}
	}

	memcpyHostToDeviceSync(_dev_intCondlike, _host_intCondlike, _nInternalNodeForCondlike * condlike_pad_node_size);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg("cudaDeviceSynchronize() failed");
	printf("Leaving specifyInternalCondlikeAll()...\n");
#endif

	return CUL_SUCCESS;
}



/* Specify the map from node index to array id. The array id is 0 to _nTipStateArray - 1 if the node is an tip node and is specified through tip state array, 
else _nTipStateArray to _nTipStateArray + _nTipCondlikeArray - 1 if the node is a tip node and is specified through tip conditional likelihood array, 
else _nTipStateArray + _nTipCondlikeArray to _nTipStateArray + _nTipCondlikeArray + _nInternalNodeForCondlike - 1 if the node is an internal node;
Input:
  indMap: key为node在tree中的label，value为node在tipState/tipCondlike/intCondlike array中的index;
*/
CuLErrorCode PartitionInstance::mapNodeIndToArrayInd(const int nNode,
													const int *indMap)
{
#ifdef DEBUG
	printf("Entering mapNodeIndToArrayInd()...\n");
#endif

	if(nNode != _nNode)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;

	const int nArray = _nTipStateArray + _nTipCondlikeArray + _nInternalNodeForCondlike;

	for(int iNode = 0; iNode < nNode; iNode ++){
		if(indMap[iNode] < 0 || indMap[iNode] >= nArray)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		_nodeIndToArrayInd[iNode] = indMap[iNode];
	}

#ifdef DEBUG
	printf("Leaving mapNodeIndToArrayInd()...\n");
#endif

	return CUL_SUCCESS;
}


// Set which nodes are to be scaled
// 假设此处的node index为node在tree中的label;
// TODO: 考虑清楚是否需要移除之前设置的node scaler的标志???  目前的方法是调用该函数会导致此前设置的node scaler全部失效;
CuLErrorCode PartitionInstance::specifyNodeScalerIndex(const int nNodeScaler,
														const int *nodeId)
{
	if(nNodeScaler < 0 || nNodeScaler > _nNodeScaler)
		return CUL_ERROR_BAD_PARAM_VALUE;

	for(int iNode = 0; iNode < _nNode; iNode ++)
		_nodeIndToScaleInd[iNode] = -1;

	_curNodeScalerCnt = nNodeScaler;
	for(int iNode = 0; iNode < nNodeScaler; iNode ++){
		if(nodeId[iNode] < 0 || nodeId[iNode] >= _nNode)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		_nodeIndToScaleInd[nodeId[iNode]] = iNode;
	}

	return CUL_SUCCESS;
}


int PartitionInstance::setCondlikeBlockIndToArrayInd(const bool isRooted,
													const bool usePadVersion,		// whether to use pad version for nPaddedState != 4 / 20 / 64
													int *pHostBlkIndToArrayInd, 
													int *pHostStartBlkInd,
													CuLCondlikeOp *condlikeOp,
													const int nOp,
													int &nThreadPerArray_baseline)
{
	// (Optionally) reset the host memory;
	//resetHostMemory(_host_condlike_opStartBlkInd, _maxCondlikeOpCount);
	//resetHostMemory(_host_condlike_blkIndToOpInd, _maxCondlikeOpCount * _nThreadBlockPerClArray);

#ifdef DEBUG
	printf("Entering setCondlikeBlockIndToArrayInd()...\n");
#endif


	int totalBlock = 0, nBlockPerArray;
	if(_nPaddedState != 64){
		// For nPaddedState != 64:
#ifndef USE_OLD_VERSION
		if(4 == _nPaddedState){
			// The new scheme: adjust the task granulity according to the current value of nOp * nSitePattern
			_condlike_blockDim_y = 1;
			_condlike_blockDim_z = 1;

			int totalPattern = nOp * _nSitePattern, nSitePerThread;
			if(totalPattern < TOTAL_PATTERN_THRESHOLD_4STATE_SMALL){
				_condlike_blockDim_x = N_THREAD_PER_BLOCK_CONDLIKE_4STATE_VERSION2;
				nSitePerThread = N_SITE_PER_THREAD_CONDLIKE_4STATE_VERSION2;
			}
			else if(totalPattern < TOTAL_PATTERN_THRESHOLD_4STATE_MEDIUM){
				_condlike_blockDim_x = N_THREAD_PER_BLOCK_CONDLIKE_4STATE_VERSION3;
				nSitePerThread = N_SITE_PER_THREAD_CONDLIKE_4STATE_VERSION3_SMALL;
			}
			else{
				_condlike_blockDim_x = N_THREAD_PER_BLOCK_CONDLIKE_4STATE_VERSION3;
				if(_nSitePattern < PATTERN_THRESHOLD_4STATE_LARGE){
					nSitePerThread = N_SITE_PER_THREAD_CONDLIKE_4STATE_VERSION3_MEDIUM;
				}
				else{
					nSitePerThread = N_SITE_PER_THREAD_CONDLIKE_4STATE_VERSION3_LARGE;
				}
			}

			int nSitePerBlock = nSitePerThread * _condlike_blockDim_x;
			nBlockPerArray = (_nSitePattern + nSitePerBlock - 1) / nSitePerBlock;
			nThreadPerArray_baseline = nBlockPerArray * _condlike_blockDim_x;
		}
		else{
#endif
			if(usePadVersion){
				_condlike_blockDim_x = _condlike_blockDim_x_pad;
				_condlike_blockDim_y = _condlike_blockDim_y_pad;
				_condlike_blockDim_z = _condlike_blockDim_z_pad;

				nBlockPerArray = _nThreadBlockPerClArray_pad;
				nThreadPerArray_baseline = nBlockPerArray * (_condlike_blockDim_x_pad * _condlike_blockDim_y_pad * _condlike_blockDim_z_pad);
			}
			else{
				_condlike_blockDim_x = _condlike_blockDim_x_noPad;
				_condlike_blockDim_y = _condlike_blockDim_y_noPad;
				_condlike_blockDim_z = _condlike_blockDim_z_noPad;

				nBlockPerArray = _nThreadBlockPerClArray_noPad;
				nThreadPerArray_baseline = nBlockPerArray * (_condlike_blockDim_x_noPad * _condlike_blockDim_y_noPad * _condlike_blockDim_z_noPad);
			}
#ifndef USE_OLD_VERSION
		}
#endif
		//assert(nOp <= _maxCondlikeOpCount);
		//assert(nOp * nBlockPerArray <= _maxCondlikeOpCount * _nThreadBlockPerClArray);
		//assert(nOp * nBlockPerArray == totalBlock);
	}
	else{
		// For nPaddedState == 64, no matter which case is, nBlockPerArray is the same;
		nBlockPerArray = _nBlockPerClArray_codeml;
	}

	// Set the map of block index to condlike array index:
	for(int iArray = 0; iArray < nOp; iArray ++){
		pHostStartBlkInd[iArray] = totalBlock;
		for(int iBlock = 0; iBlock < nBlockPerArray; iBlock ++, totalBlock ++){
			pHostBlkIndToArrayInd[totalBlock] = iArray;
		}
	}


#ifdef DEBUG
	printf("Leaving setCondlikeBlockIndToArrayInd()...\n");
#endif

	_condlike_gridDim_x = totalBlock;
	_condlike_gridDim_y = 1;
	_condlike_gridDim_z = 1;

	return totalBlock;
}



int PartitionInstance::setScaleBlockIndToOffset(int nNodeToScale, 
												int *nodeId, 
												int *pHostBlkIndToClOffset,
												int *pHostBlkIndToScaleOffset,
												int *pHostStartBlkInd)
{
	assert(nNodeToScale <= _nNodeScaler);

	int totalBlock = 0, preTotalBlock = 0, nTipNode = _nTipStateArray + _nTipCondlikeArray, condlike_pad_node_size = _nArrayPerNode * _condlike_pad_size;
	for(int iNode = 0; iNode < nNodeToScale; iNode ++){
		int condlikeBufferInd = _nodeIndToArrayInd[nodeId[iNode]] - nTipNode;
		if(condlikeBufferInd < 0)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		int scaleBufferInd = _nodeIndToScaleInd[nodeId[iNode]];
		if(scaleBufferInd == -1)
			printErrorCode(CUL_ERROR_INTERNAL);

		scaleBufferInd *= _nPaddedSitePattern;
		int condlikeOffset = condlikeBufferInd * condlike_pad_node_size;
		int nStatePerBlock = _scale_blockDim_x * _nPaddedState;
		
		preTotalBlock = totalBlock;
		for(int iBlock = 0, offset1 = 0, offset2 = 0; iBlock < _nThreadBlockPerScaleNode; iBlock ++, totalBlock ++, offset1 += nStatePerBlock, offset2 += _scale_blockDim_x){
			pHostBlkIndToClOffset[totalBlock] = condlikeOffset + offset1;
			pHostBlkIndToScaleOffset[totalBlock] = scaleBufferInd + offset2;
			pHostStartBlkInd[totalBlock] = preTotalBlock;
		}
	}

	return totalBlock;
}



inline void callKernelCondlike(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const bool usePadVersion, const int nSitePattern, const int nPaddedState, const int nState, const int nThreadPerArray_baseline, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream)
{
	//printf("\nGoing to use the rooted version...\n\n");
	//printf("\nBlock dimension: (%d, %d)\nGrid dimension: (%d, %d)\n", nThreadPerBlock.x, nThreadPerBlock.y, nBlockPerGrid.x, nBlockPerGrid.y);

#ifdef DEBUG_TIME
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	timeBegin();
	for(int itr = 0; itr < 100; itr ++){
#endif

	// For nPaddedState != 64, use the baseline version else use the codeml version:
	if(nPaddedState != 64)
		callKernelCondlike_baseline(intCondlike, tipState, tipCondlike, PMat, nOp, condlikeOp, blkIndToOpInd, opStartBlkInd, usePadVersion, nSitePattern, nPaddedState, nState, nThreadPerArray_baseline, nBlockPerGrid, nThreadPerBlock, stream);
	else
		callKernelCondlike_codeml(intCondlike, tipState, tipCondlike, PMat, nOp, condlikeOp, blkIndToOpInd, opStartBlkInd, nSitePattern, nPaddedState, nState, nBlockPerGrid, nThreadPerBlock, stream);


#ifdef DEBUG_TIME
	}
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	condlike_time = timeEnd();

#ifdef TRANSPOSE_PMAT
	FILE *fout = fopen("condlike_time_rooted_transpose.txt", "a");
#else
	FILE *fout = fopen("condlike_time_rooted_noTranspose.txt", "a");
#endif

	fprintf(fout, "for nState = %d, nSitePattern = %d, nOp = %d, nIteration = %d: %lld us (%lld.%06lld s)\n\n", nState, nSitePattern, nOp, nIteration / 10, condlike_time, condlike_time / multiple, condlike_time % multiple);
	fclose(fout);

#ifdef TRANSPOSE_PMAT
	FILE *fout2 = fopen("condlike_time_rooted_transpose_format.txt", "a");
#else
	FILE *fout2 = fopen("condlike_time_rooted_noTranspose_format.txt", "a");
#endif
	fprintf(fout2, "%lld.%06lld\t", condlike_time / multiple, condlike_time % multiple);
	fclose(fout2);

#endif
}


inline void callKernelNodeScale(CUFlt *nodeScaleFactor, CUFlt *intCondlike, int *blkIndToCondlikeOffset, int *blkIndToScaleBufferOffset, int *startBlkInd, int nCategory, int nSitePattern, int nPaddedSitePattern, int nState, int nPaddedState, int nBlockPerGrid, int nThreadPerBlock, cudaStream_t &stream)
{
#ifdef DEBUG_TIME
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	timeBegin();
	for(int itr = 0; itr < nIteration / 10; itr ++){
#endif

	callKernelNodeScale_baseline(nodeScaleFactor,intCondlike, blkIndToCondlikeOffset, blkIndToScaleBufferOffset, startBlkInd, nCategory, nSitePattern, nPaddedSitePattern, nState, nPaddedState, nBlockPerGrid, nThreadPerBlock, stream);

#ifdef DEBUG_TIME
	}
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	scale_time = timeEnd();

	FILE *fout = fopen("scale_time.txt", "a");
	fprintf(fout, "Baseline version:\n");
	fprintf(fout, "for nState = %d, nSitePattern = %d, nCategory = %d, nIteration = %d: %lld us (%lld.%06lld s)\n\n", nState, nSitePattern, nCategory, nIteration / 10, lnL_time, lnL_time / multiple, lnL_time % multiple);
	fclose(fout);

	fout = fopen("scale_time_format.txt", "a");
	fprintf(fout, "%lld.%06lld\t", lnL_time / multiple, lnL_time % multiple);
	fclose(fout);
#endif
}


inline void callKernelCondlike_unrooted(CUFlt *intCondlike, int *tipState, CUFlt *tipCondlike, CUFlt *PMat, const int nOp, CuLCondlikeOp *condlikeOp, int *blkIndToOpInd, int *opStartBlkInd, const bool usePadVersion, const int nSitePattern, const int nPaddedSitePattern, const int nPaddedState, const int nState, const int nThreadPerArray_baseline, dim3 nBlockPerGrid, dim3 nThreadPerBlock, cudaStream_t &stream)
{
	printf("\nGoing to use the unrooted version...\n\n");
#ifdef DEBUG_TIME
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	timeBegin();
	for(int itr = 0; itr < 1; itr ++){
#endif

	// For nPaddedState != 64, use the baseline version else use the codeml version:
	if(nPaddedState != 64)
		callKernelCondlike_baseline_unrooted(intCondlike, tipState, tipCondlike, PMat, nOp, condlikeOp, blkIndToOpInd, opStartBlkInd, usePadVersion, nSitePattern, nPaddedState, nState, nThreadPerArray_baseline, nBlockPerGrid, nThreadPerBlock, stream);
	else
		callKernelCondlike_codeml_unrooted(intCondlike, tipState, tipCondlike, PMat, nOp, condlikeOp, blkIndToOpInd, opStartBlkInd, nSitePattern, nPaddedSitePattern, nPaddedState, nState, nBlockPerGrid, nThreadPerBlock, stream);


#ifdef DEBUG_TIME
	}
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	condlike_time = timeEnd();

#ifdef TRANSPOSE_PMAT
	FILE *fout = fopen("condlike_time_unrooted_transpose.txt", "a");
#else
	FILE *fout = fopen("condlike_time_unrooted_noTranspose.txt", "a");
#endif

	fprintf(fout, "for nState = %d, nSitePattern = %d, nOp = %d, nIteration = %d: %lld us (%lld.%06lld s)\n\n", nState, nSitePattern, nOp, nIteration / 10, condlike_time, condlike_time / multiple, condlike_time % multiple);
	fclose(fout);

	FILE *fout2 = fopen("condlike_time_unrooted_transpose_format.txt", "a");
	fprintf(fout2, "%lld.%06lld\t", condlike_time / multiple, condlike_time % multiple);
	fclose(fout2);
#endif
}



// Sort the arrays according to the case value:
void sortByCase(int nChild, int *childCase, int *isTip, int *condlikeOffset, int *PMatOffset)
{
	for(int iChild = 0; iChild < nChild; iChild ++){
		for(int jChild = iChild + 1; jChild < nChild; jChild ++){
			if(childCase[iChild] > childCase[jChild]){
				swap(childCase[iChild], childCase[jChild]);
				swap(isTip[iChild], isTip[jChild]);
				swap(condlikeOffset[iChild], condlikeOffset[jChild]);
				swap(PMatOffset[iChild], PMatOffset[jChild]);
			}
		}
	}
}


// For unrooted tree:
CuLErrorCode PartitionInstance::calculateCondlikeMulti(const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering calculateCondlikeMulti_unrooted()...\n");
#endif

	//(Optionally) reset the condlike operation and offsets;
	resetHostMemory(_host_condlike_op, _maxCondlikeOpCount);
	resetHostMemory(_host_condlike_opStartBlkInd, _maxCondlikeOpCount);
	resetHostMemory(_host_condlike_blkIndToOpInd, _maxCondlikeOpCount * _nThreadBlockPerClArray);
	resetHostMemory(_host_scale_blkIndToClOffset, _maxScaleOpCount * _nThreadBlockPerScaleNode);
	resetHostMemory(_host_scale_blkIndToScaleOffset, _maxScaleOpCount * _nThreadBlockPerScaleNode);
	resetHostMemory(_host_scale_startBlkInd, _maxScaleOpCount * _nThreadBlockPerScaleNode);

	const int condlike_pad_size = _condlike_pad_size;
	const int condlike_pad_eigen_size = _nRateCategory * condlike_pad_size;
	const int condlike_pad_node_size = _nEigenDecomposition * condlike_pad_eigen_size;

	const int PMat_pad_size = _PMat_pad_size;
	const int PMat_pad_eigen_size = _nRateCategory * PMat_pad_size;
	const int PMat_pad_node_size = _nEigenDecomposition * PMat_pad_eigen_size;

	const int nTipNode = _nTipStateArray + _nTipCondlikeArray;
	const int nTotalNode = _nTipStateArray + _nTipCondlikeArray + _nInternalNodeForCondlike;

	int cntOp, cntNode, cntScaler, preTotalOp = 0, preTotalBlock = 0, preTotalScaleBlock = 0;
	int cntCase1;
	int condlike_offset_F, condlike_offset_S[3], PMat_offset_S[3], whichCase[3], isTip[3], condlike_cat_offset, PMat_cat_offset, curCase;
	bool rootedVersion, usePadVersion;

	std::set<int> nodeSet;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(nodeId[iNode] < 0 || nodeId[iNode] >= nTotalNode)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		nodeSet.insert(nodeId[iNode]);
	}

	CuLCondlikeOp *pHostOp = NULL;
	int *pHostBlkIndToArrayInd = NULL, *pHostStartBlkInd = NULL, *pHostBlkIndToClOffset = NULL, *pHostBlkIndToScaleOffset = NULL, *pHostScaleStartBlkInd = NULL;
	for(int iLayer = _nLayerOfTree - 1; iLayer >= 0; iLayer --){
		// (Optionally) reset the condlike operation:
		//resetHostMemory(_host_condlike_op, _maxCondlikeOpCount);
#ifdef DEBUG_CONDLIKE
		printf("\nGoing to process node in layer %d:\n", iLayer);
#endif

		pHostOp = _host_condlike_op + preTotalOp;
		pHostBlkIndToArrayInd = _host_condlike_blkIndToOpInd + preTotalBlock;
		pHostStartBlkInd = _host_condlike_opStartBlkInd + preTotalOp;

		pHostBlkIndToClOffset = _host_scale_blkIndToClOffset + preTotalScaleBlock;
		pHostBlkIndToScaleOffset = _host_scale_blkIndToScaleOffset + preTotalScaleBlock;
		pHostScaleStartBlkInd = _host_scale_startBlkInd + preTotalScaleBlock;

		cntOp = 0;
		cntNode = 0;
		cntScaler = 0;
		cntCase1 = 0;
		rootedVersion = true;			// By default, use the rooted version, that is, two children is assumed;
		usePadVersion = false;
		int nNodeInLayer = _nodeLayerMap[iLayer].size();

		for(int iNode = 0; iNode < nNodeInLayer; iNode ++){
			CuLTreeNode *curNode = _nodeLayerMap[iLayer][iNode];

			if(nodeSet.end() != nodeSet.find(curNode->label)){
				if(curNode->nChild < 2 || curNode->nChild > 3 || _nodeIndToArrayInd.end() == _nodeIndToArrayInd.find(curNode->label))
					return CUL_ERROR_INDEX_OUT_OF_RANGE;

				cntNode ++;

				// Judge whether the current node need scaling:
				if(_nodeIndToScaleInd[curNode->label] > -1){
					_nodesToScale[cntScaler] = curNode->label;
					cntScaler ++;
				}

				if(curNode->nChild > 2)
					rootedVersion = false;

				int arrayInd_F = _nodeIndToArrayInd[curNode->label];
				condlike_offset_F = (arrayInd_F - nTipNode) * condlike_pad_node_size;
#ifdef DEBUG_CONDLIKE
				printf("\n\tFor node %d: arrayInd_F = %d\ncondlike_offset_F = %d\n", curNode->label, arrayInd_F, condlike_offset_F);
#endif
				
				for(int iChild = 0; iChild < curNode->nChild; iChild ++){
					CuLTreeNode *curChild = curNode->child[iChild];
					int childLabel = curChild->label;
					if(_nodeIndToArrayInd.end() == _nodeIndToArrayInd.find(childLabel))
						return CUL_ERROR_INDEX_OUT_OF_RANGE;

					int arrayInd_S = _nodeIndToArrayInd[childLabel];
					PMat_offset_S[iChild] = childLabel * PMat_pad_node_size;

					if(arrayInd_S < _nTipStateArray){
						isTip[iChild] = 1;
						whichCase[iChild] = 1;
						condlike_offset_S[iChild] = arrayInd_S * _nPaddedSitePattern;
					}
					else if(arrayInd_S < nTipNode){
						isTip[iChild] = 1;
						whichCase[iChild] = 2;
						condlike_offset_S[iChild] = (arrayInd_S - _nTipStateArray) * condlike_pad_size;
					}
					else{
						isTip[iChild] = 0;
						whichCase[iChild] = 3;
						condlike_offset_S[iChild] = (arrayInd_S - nTipNode) * condlike_pad_node_size;
					}
				}

				sortByCase(curNode->nChild, whichCase, isTip, condlike_offset_S, PMat_offset_S);
				curCase = 1;
				for(int iChild = 0; iChild < curNode->nChild; iChild ++){
					if(whichCase[iChild] > 1)
						curCase ++;
				}

				if(1 == curCase)
					cntCase1 += nEigenDecomp * _nRateCategory;

				for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
					if(eigenDecompId[iEigen] < 0 || eigenDecompId[iEigen] >= _nEigenDecomposition)
						return CUL_ERROR_INDEX_OUT_OF_RANGE;

					condlike_cat_offset = eigenDecompId[iEigen] * condlike_pad_eigen_size;
					PMat_cat_offset = eigenDecompId[iEigen] * PMat_pad_eigen_size;

					for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, condlike_cat_offset += condlike_pad_size, PMat_cat_offset += PMat_pad_size){
						pHostOp[cntOp].nChild = curNode->nChild;
						pHostOp[cntOp].whichCase = curCase;
						pHostOp[cntOp].father_condlike_offset = condlike_offset_F + condlike_cat_offset;

						for(int iChild = 0; iChild < curNode->nChild; iChild ++){
							pHostOp[cntOp].child_case[iChild] = whichCase[iChild];
							pHostOp[cntOp].child_P_offset[iChild] = PMat_offset_S[iChild] + PMat_cat_offset;
							pHostOp[cntOp].isChildTip[iChild] = isTip[iChild];

							if(whichCase[iChild] <= 2)
								pHostOp[cntOp].child_condlike_offset[iChild] = condlike_offset_S[iChild];
							else
								pHostOp[cntOp].child_condlike_offset[iChild] = condlike_offset_S[iChild] + condlike_cat_offset;
						}
#ifdef DEBUG_CONDLIKE
						printf("\n\t\tFor cat %d: nChild = %d, curCase = %d, isChildTip[0] = %d, isChildTip[1] = %d, father_condlike_offset = %d, child_P_offset[0] = %d, child_P_offset[1] = %d, child_condlike_offset[0] = %d, child_condlike_offset[1] = %d\n", iEigen * _nRateCategory + iRateCat, pHostOp[cntOp].nChild, pHostOp[cntOp].whichCase, pHostOp[cntOp].isChildTip[0], pHostOp[cntOp].isChildTip[1], pHostOp[cntOp].father_condlike_offset, pHostOp[cntOp].child_P_offset[0], pHostOp[cntOp].child_P_offset[1], pHostOp[cntOp].child_condlike_offset[0], pHostOp[cntOp].child_condlike_offset[1]);
#endif

						cntOp ++;
					}
				}
			}
		}	// end of for(int iNode = 0; iNode < nNodeInLayer; iNode ++);

#ifdef DEBUG
		printf("Going to calculate condlike of nodes in layer %d, cntOp = %d\n\n", iLayer, cntOp);
#endif

		if(cntOp > 0){

			assert(cntNode * nEigenDecomp * _nRateCategory == cntOp);

			memcpyHostToDeviceAsync(_dev_condlike_op + preTotalOp, pHostOp, cntOp, stream);

			printf("\n===============\ncntCase1 = %d, cntOp = %d\n===============\n", cntCase1, cntOp);
			if(_nPaddedState != 4 && _nPaddedState != 20 && _nPaddedState != 64){
				if(rootedVersion){
					// For rooted version:
					if(cntCase1 < cntOp){
						if(_nPaddedState <= 8 || _nPaddedState > 48 || cntOp * _nSitePattern < CONDLIKE_XSTATE_USE_PAD_THRESHOLD)
							usePadVersion = true;
					}
				}
				else{
					// For unrooted version:
					if(cntCase1 == cntOp){
						if(_nPaddedState > 24)
							usePadVersion = true;
					}
					else{
						if(_nPaddedState <= 8 || _nPaddedState > 32 || cntOp * _nSitePattern < CONDLIKE_XSTATE_USE_PAD_THRESHOLD)
							usePadVersion = true;
					}
				}
			}

			int nThreadPerArray_baseline = 0;
			
			int cntBlock = setCondlikeBlockIndToArrayInd(rootedVersion,
														usePadVersion,
														pHostBlkIndToArrayInd, 
														pHostStartBlkInd,
														pHostOp,
														cntOp,
														nThreadPerArray_baseline);		// Set the block index to condlike array index;

			
			memcpyHostToDeviceAsync(_dev_condlike_blkIndToOpInd + preTotalBlock, pHostBlkIndToArrayInd, cntBlock, stream);
			memcpyHostToDeviceAsync(_dev_condlike_opStartBlkInd + preTotalOp, pHostStartBlkInd, cntOp, stream);

			//blockSize = _condlike_blockDim_x * _condlike_blockDim_y * _condlike_blockDim_z;

			dim3 nThreadPerBlock(_condlike_blockDim_x, _condlike_blockDim_y, _condlike_blockDim_z);
			dim3 nBlockPerGrid(_condlike_gridDim_x, _condlike_gridDim_y, _condlike_gridDim_z);
#ifdef DEBUG
			printf("\nGrid dimension: (%d, %d, %d)\nBlock dimension: (%d, %d, %d)\n\n", _condlike_gridDim_x, _condlike_gridDim_y, _condlike_gridDim_z, _condlike_blockDim_x, _condlike_blockDim_y, _condlike_blockDim_z);
#endif

			if(rootedVersion)
				callKernelCondlike(_dev_intCondlike, _dev_tipState, _dev_tipCondlike, _dev_PMat, cntOp, _dev_condlike_op + preTotalOp, _dev_condlike_blkIndToOpInd + preTotalBlock, _dev_condlike_opStartBlkInd + preTotalOp, usePadVersion, _nSitePattern, _nPaddedState, _nState, nThreadPerArray_baseline, nBlockPerGrid, nThreadPerBlock, stream);
			else
				callKernelCondlike_unrooted(_dev_intCondlike, _dev_tipState, _dev_tipCondlike, _dev_PMat, cntOp, _dev_condlike_op + preTotalOp, _dev_condlike_blkIndToOpInd + preTotalBlock, _dev_condlike_opStartBlkInd + preTotalOp, usePadVersion, _nSitePattern, _nPaddedSitePattern, _nPaddedState, _nState, nThreadPerArray_baseline, nBlockPerGrid, nThreadPerBlock, stream);

			preTotalOp += cntOp;
			preTotalBlock += cntBlock;


			// Optionally do node scaling:
			if(cntScaler > 0){
				cntBlock = setScaleBlockIndToOffset(cntScaler, 
													_nodesToScale, 
													pHostBlkIndToClOffset,
													pHostBlkIndToScaleOffset,
													pHostScaleStartBlkInd);

				memcpyHostToDeviceAsync(_dev_scale_blkIndToClOffset + preTotalScaleBlock, pHostBlkIndToClOffset, cntBlock, stream);
				memcpyHostToDeviceAsync(_dev_scale_blkIndToScaleOffset + preTotalScaleBlock, pHostBlkIndToScaleOffset, cntBlock, stream);
				memcpyHostToDeviceAsync(_dev_scale_startBlkInd + preTotalScaleBlock, pHostScaleStartBlkInd, cntBlock, stream);

				callKernelNodeScale(_dev_nodeScaleFactor, _dev_intCondlike, _dev_scale_blkIndToClOffset + preTotalScaleBlock, _dev_scale_blkIndToScaleOffset + preTotalScaleBlock, _dev_scale_startBlkInd + preTotalScaleBlock, _nArrayPerNode, _nSitePattern, _nPaddedSitePattern, _nState, _nPaddedState, cntBlock, _scale_blockDim_x, stream);

				preTotalScaleBlock += cntBlock;
			}

#ifdef DEBUG
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
#endif
		}
	}

#ifdef DEBUG
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	printf("Leaving calculateCondlikeMulti()...\n");
#endif

	return CUL_SUCCESS;
}


// Calculate the conditional likelihood of a partition;
CuLErrorCode PartitionInstance::calculateCondlikeAll(const int nNode,
													const int *nodeId,
													cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering calculateCondlikeAll()...\n");
#endif

	//(Optionally) reset the condlike operation and offsets;
	resetHostMemory(_host_condlike_op, _maxCondlikeOpCount);
	resetHostMemory(_host_condlike_opStartBlkInd, _maxCondlikeOpCount);
	resetHostMemory(_host_condlike_blkIndToOpInd, _maxCondlikeOpCount * _nThreadBlockPerClArray);
	resetHostMemory(_host_scale_blkIndToClOffset, _maxScaleOpCount * _nThreadBlockPerScaleNode);
	resetHostMemory(_host_scale_blkIndToScaleOffset, _maxScaleOpCount * _nThreadBlockPerScaleNode);
	resetHostMemory(_host_scale_startBlkInd, _maxScaleOpCount * _nThreadBlockPerScaleNode);

	const int condlike_pad_size = _condlike_pad_size;
	const int condlike_pad_node_size = _nArrayPerNode * condlike_pad_size;

	const int PMat_pad_size = _PMat_pad_size;
	const int PMat_pad_node_size = _nArrayPerNode * PMat_pad_size;

	const int nTipNode = _nTipStateArray + _nTipCondlikeArray;
	const int nTotalNode = _nTipStateArray + _nTipCondlikeArray + _nInternalNodeForCondlike;

	int cntOp, cntNode, cntScaler, preTotalOp = 0, preTotalBlock = 0, preTotalScaleBlock = 0;
	int cntCase1;
	int condlike_offset_F, condlike_offset_S[3], PMat_offset_S[3], whichCase[3], isTip[3], condlike_cat_offset, PMat_cat_offset, curCase;
	bool rootedVersion, usePadVersion;

	set<int> nodeSet;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(nodeId[iNode] < 0 || nodeId[iNode] >= nTotalNode)
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		nodeSet.insert(nodeId[iNode]);
	}

	CuLCondlikeOp *pHostOp = NULL;
	int *pHostBlkIndToArrayInd = NULL, *pHostStartBlkInd = NULL, *pHostBlkIndToClOffset = NULL, *pHostBlkIndToScaleOffset = NULL, *pHostScaleStartBlkInd = NULL;
	for(int iLayer = _nLayerOfTree - 1; iLayer >= 0; iLayer --){
#ifdef DEBUG_STREAM
		printf("\nGoing to calculate condlike of nodes in layer %d...\n", iLayer);
		cutilSafeCall(cudaStreamSynchronize(stream));
		cutilCheckMsg("cudaStreamSynchronize() failed");
#endif

		// (Optionally) reset the condlike operation:
		//resetHostMemory(_host_condlike_op, _maxCondlikeOpCount);

		pHostOp = _host_condlike_op + preTotalOp;
		pHostBlkIndToArrayInd = _host_condlike_blkIndToOpInd + preTotalBlock;
		pHostStartBlkInd = _host_condlike_opStartBlkInd + preTotalOp;

		pHostBlkIndToClOffset = _host_scale_blkIndToClOffset + preTotalScaleBlock;
		pHostBlkIndToScaleOffset = _host_scale_blkIndToScaleOffset + preTotalScaleBlock;
		pHostScaleStartBlkInd = _host_scale_startBlkInd + preTotalScaleBlock;

		cntOp = 0;
		cntNode = 0;
		cntScaler = 0;
		cntCase1 = 0;
		rootedVersion = true;
		usePadVersion = false;
		int nNodeInLayer = _nodeLayerMap[iLayer].size();

		for(int iNode = 0; iNode < nNodeInLayer; iNode ++){
			CuLTreeNode *curNode = _nodeLayerMap[iLayer][iNode];

			if(nodeSet.end() != nodeSet.find(curNode->label)){		
				if(curNode->nChild < 2 || curNode->nChild > 3 || _nodeIndToArrayInd.end() == _nodeIndToArrayInd.find(curNode->label))
					return CUL_ERROR_INDEX_OUT_OF_RANGE;

				cntNode ++;

				// Judge whether the current node need scaling:
				if(_nodeIndToScaleInd[curNode->label] > -1){
					_nodesToScale[cntScaler] = curNode->label;
					cntScaler ++;
				}

				if(curNode->nChild > 2)
					rootedVersion = false;

				int arrayInd_F = _nodeIndToArrayInd[curNode->label];
				condlike_offset_F = (arrayInd_F - nTipNode) * condlike_pad_node_size;
				
				for(int iChild = 0; iChild < curNode->nChild; iChild ++){
					CuLTreeNode *curChild = curNode->child[iChild];
					int childLabel = curChild->label;
					if(_nodeIndToArrayInd.end() == _nodeIndToArrayInd.find(childLabel))
						return CUL_ERROR_INDEX_OUT_OF_RANGE;

					int arrayInd_S = _nodeIndToArrayInd[childLabel];
					PMat_offset_S[iChild] = childLabel * PMat_pad_node_size;

					if(arrayInd_S < _nTipStateArray){
						isTip[iChild] = 1;
						whichCase[iChild] = 1;
						condlike_offset_S[iChild] = arrayInd_S * _nPaddedSitePattern;
					}
					else if(arrayInd_S < nTipNode){
						isTip[iChild] = 1;
						whichCase[iChild] = 2;
						condlike_offset_S[iChild] = (arrayInd_S - _nTipStateArray) * condlike_pad_size;
					}
					else{
						isTip[iChild] = 0;
						whichCase[iChild] = 3;
						condlike_offset_S[iChild] = (arrayInd_S - nTipNode) * condlike_pad_node_size;
					}
				}

				sortByCase(curNode->nChild, whichCase, isTip, condlike_offset_S, PMat_offset_S);
				curCase = 1;
				for(int iChild = 0; iChild < curNode->nChild; iChild ++){
					if(whichCase[iChild] > 1)
						curCase ++;
				}

				if(1 == curCase)
					cntCase1 += _nEigenDecomposition * _nRateCategory;

				condlike_cat_offset = 0;
				PMat_cat_offset = 0;

				for(int iEigen = 0; iEigen < _nEigenDecomposition; iEigen ++){
					for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, condlike_cat_offset += condlike_pad_size, PMat_cat_offset += PMat_pad_size){
						pHostOp[cntOp].nChild = curNode->nChild;
						pHostOp[cntOp].whichCase = curCase;
						pHostOp[cntOp].father_condlike_offset = condlike_offset_F + condlike_cat_offset;

						for(int iChild = 0; iChild < curNode->nChild; iChild ++){
							pHostOp[cntOp].child_case[iChild] = whichCase[iChild];
							pHostOp[cntOp].child_P_offset[iChild] = PMat_offset_S[iChild] + PMat_cat_offset;
							pHostOp[cntOp].isChildTip[iChild] = isTip[iChild];

							if(whichCase[iChild] <= 2)
								pHostOp[cntOp].child_condlike_offset[iChild] = condlike_offset_S[iChild];
							else
								pHostOp[cntOp].child_condlike_offset[iChild] = condlike_offset_S[iChild] + condlike_cat_offset;
						}

						cntOp ++;
					}
				}
			}
		}	// end of for(int iNode = 0; iNode < nNodeInLayer; iNode ++);

		if(cntOp > 0){

			assert(cntNode * _nEigenDecomposition * _nRateCategory == cntOp);

			memcpyHostToDeviceAsync(_dev_condlike_op + preTotalOp, pHostOp, cntOp, stream);

			if(_nPaddedState != 4 && _nPaddedState != 20 && _nPaddedState != 64){
				if(rootedVersion){
					// For rooted version:
					if(cntCase1 < cntOp){
						if(_nPaddedState <= 8 || _nPaddedState > 48 || cntOp * _nSitePattern < CONDLIKE_XSTATE_USE_PAD_THRESHOLD)
							usePadVersion = true;
					}
				}
				else{
					// For unrooted version:
					if(cntCase1 == cntOp){
						if(_nPaddedState > 24)
							usePadVersion = true;
					}
					else{
						if(_nPaddedState <= 8 || _nPaddedState > 32 || cntOp * _nSitePattern < CONDLIKE_XSTATE_USE_PAD_THRESHOLD)
							usePadVersion = true;
					}
				}
			}

			usePadVersion = true;

			int nThreadPerArray_baseline = 0;
			
			int cntBlock = setCondlikeBlockIndToArrayInd(rootedVersion,
														usePadVersion,
														pHostBlkIndToArrayInd, 
														pHostStartBlkInd, 
														pHostOp,
														cntOp,
														nThreadPerArray_baseline);		// Set the block index to condlike array index;

			memcpyHostToDeviceAsync(_dev_condlike_blkIndToOpInd + preTotalBlock, pHostBlkIndToArrayInd, cntBlock, stream);
			memcpyHostToDeviceAsync(_dev_condlike_opStartBlkInd + preTotalOp, pHostStartBlkInd, cntOp, stream);

			//blockSize = _condlike_blockDim_x * _condlike_blockDim_y * _condlike_blockDim_z;

			dim3 nThreadPerBlock(_condlike_blockDim_x, _condlike_blockDim_y, _condlike_blockDim_z);
			dim3 nBlockPerGrid(_condlike_gridDim_x, _condlike_gridDim_y, _condlike_gridDim_z);
#ifdef DEBUG
			printf("\nGrid dimension: (%d, %d, %d)\nBlock dimension: (%d, %d, %d)\n\n", nBlockPerGrid.x, nBlockPerGrid.y, nBlockPerGrid.z, nThreadPerBlock.x, nThreadPerBlock.y, nThreadPerBlock.z);
#endif
	
			if(rootedVersion)
				callKernelCondlike(_dev_intCondlike, _dev_tipState, _dev_tipCondlike, _dev_PMat, cntOp, _dev_condlike_op + preTotalOp, _dev_condlike_blkIndToOpInd + preTotalBlock, _dev_condlike_opStartBlkInd + preTotalOp, usePadVersion, _nSitePattern, _nPaddedState, _nState, nThreadPerArray_baseline, nBlockPerGrid, nThreadPerBlock, stream);
			else
				callKernelCondlike_unrooted(_dev_intCondlike, _dev_tipState, _dev_tipCondlike, _dev_PMat, cntOp, _dev_condlike_op + preTotalOp, _dev_condlike_blkIndToOpInd + preTotalBlock, _dev_condlike_opStartBlkInd + preTotalOp, usePadVersion, _nSitePattern, _nPaddedSitePattern, _nPaddedState, _nState, nThreadPerArray_baseline, nBlockPerGrid, nThreadPerBlock, stream);


			preTotalOp += cntOp;
			preTotalBlock += cntBlock;

			// Optionally do node scaling:
			if(cntScaler > 0){
				cntBlock = setScaleBlockIndToOffset(cntScaler, 
													_nodesToScale, 
													pHostBlkIndToClOffset,
													pHostBlkIndToScaleOffset,
													pHostScaleStartBlkInd);

				memcpyHostToDeviceAsync(_dev_scale_blkIndToClOffset + preTotalScaleBlock, pHostBlkIndToClOffset, cntBlock, stream);
				memcpyHostToDeviceAsync(_dev_scale_blkIndToScaleOffset + preTotalScaleBlock, pHostBlkIndToScaleOffset, cntBlock, stream);
				memcpyHostToDeviceAsync(_dev_scale_startBlkInd + preTotalScaleBlock, pHostScaleStartBlkInd, cntBlock, stream);

				callKernelNodeScale(_dev_nodeScaleFactor, _dev_intCondlike, _dev_scale_blkIndToClOffset + preTotalScaleBlock, _dev_scale_blkIndToScaleOffset + preTotalScaleBlock, _dev_scale_startBlkInd + preTotalScaleBlock, _nArrayPerNode, _nSitePattern, _nPaddedSitePattern, _nState, _nPaddedState, cntBlock, _scale_blockDim_x, stream);

				preTotalScaleBlock += cntBlock;
			}
		}
	}

#ifdef DEBUG
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	printf("Leaving calculateCondlikeAll()...\n");
#endif

	return CUL_SUCCESS;
}




// Get the conditional likelihood of a partition;
// TODO: 考虑condlike的哪种组织方式更合理：是所有node的所有eigen decomposition的所有rate category的condlike在一起，还是同一eigen的所有node的所有rate category在一起，后者在下面这一类函数中似乎更好(需要拷贝的数据更少)；
// 假设这里的nodeId[]为node在tree中的label;
CuLErrorCode PartitionInstance::getIntCondlikeMulti(const int nNode,
												const int *nodeId,
												const int nEigenDecomp,
												const int *eigenDecompId,
												double **outCondlike)
{
#ifdef DEBUG
	printf("Entering getIntCondlikeMulti()...\n");
#endif

	const int nTipNode = _nTipStateArray + _nTipCondlikeArray;
	const int condlike_size = _condlike_size;
	const int condlike_eigen_size = _nRateCategory * condlike_size;
	const int condlike_pad_size = _condlike_pad_size;
	const int condlike_pad_eigen_size = _nRateCategory * condlike_pad_size;
	const int condlike_pad_node_size = _nEigenDecomposition * condlike_pad_eigen_size;

	memcpyDeviceToHostSync(_host_intCondlike, _dev_intCondlike, _nInternalNodeForCondlike * condlike_pad_node_size);

	CUFlt *pCondlike = NULL;
	double *pOutCondlike = NULL;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(nodeId[iNode] < 0 || nodeId[iNode] >= _nNode || _nodeIndToArrayInd.end() == _nodeIndToArrayInd.find(nodeId[iNode]))
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		int arrayId = _nodeIndToArrayInd[nodeId[iNode]] - nTipNode;

		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			if(eigenDecompId[iEigen] < 0 || eigenDecompId[iEigen] >= _nEigenDecomposition)
				return CUL_ERROR_INDEX_OUT_OF_RANGE;

			pCondlike = _host_intCondlike + arrayId * condlike_pad_node_size + eigenDecompId[iEigen] * condlike_pad_eigen_size;
			pOutCondlike = outCondlike[iNode] + iEigen * condlike_eigen_size;
			
			if(_nPaddedState == _nState){
				if(_nPaddedSitePattern == _nSitePattern){
					// Directly copy the whole matrix:
					memcpyHostToHost(pOutCondlike, pCondlike, condlike_eigen_size);
				}
				else{
					// Copy one rate category by another:
					for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pCondlike += condlike_pad_size, pOutCondlike += condlike_size){
						memcpyHostToHost(pOutCondlike, pCondlike, condlike_size);
					}
				}
			}
			else{
				// Copy one state by another:
				for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pCondlike += condlike_pad_size, pOutCondlike += condlike_size){
					for(int iPattern = 0; iPattern < _nSitePattern; iPattern ++)
						memcpyHostToHost(pOutCondlike + iPattern * _nState, pCondlike + iPattern * _nPaddedState, _nState);
				}
			}
		}
	}

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg("cudaDeviceSynchronize() failed");
	printf("Leaving getIntCondlikeMulti()...\n");
#endif

	return CUL_SUCCESS;
}



// Get the conditional likelihood of a partition;
CuLErrorCode PartitionInstance::getIntCondlikeAll(const int nNode,
												const int *nodeId,
												double **outCondlike)
{
#ifdef DEBUG
	printf("Entering getIntCondlikeAll()...\n");
#endif

	const int nTipNode = _nTipStateArray + _nTipCondlikeArray;
	const int condlike_size = _condlike_size;
	const int condlike_node_size = _nArrayPerNode * condlike_size;
	const int condlike_pad_size = _condlike_pad_size;
	const int condlike_pad_node_size = _nArrayPerNode * condlike_pad_size;

	memcpyDeviceToHostSync(_host_intCondlike, _dev_intCondlike, _nInternalNodeForCondlike * condlike_pad_node_size);

	CUFlt *pCondlike = NULL;
	double *pOutCondlike = NULL;

	for(int iNode = 0; iNode < nNode; iNode ++){
		if(nodeId[iNode] < 0 || nodeId[iNode] >= _nNode || _nodeIndToArrayInd.end() == _nodeIndToArrayInd.find(nodeId[iNode]))
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		pCondlike = _host_intCondlike + (_nodeIndToArrayInd[nodeId[iNode]] - nTipNode) * condlike_pad_node_size;
		pOutCondlike = outCondlike[iNode];

		if(_nPaddedState == _nState){
			if(_nPaddedSitePattern == _nSitePattern){
				// Directly copy the whole matrix:
				memcpyHostToHost(pOutCondlike, pCondlike, condlike_node_size);
			}
			else{
				// Copy one rate category by another:
				for(int iEigen = 0; iEigen < _nEigenDecomposition; iEigen ++){
					for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pCondlike += condlike_pad_size, pOutCondlike += condlike_size){
						memcpyHostToHost(pOutCondlike, pCondlike, condlike_size);
					}
				}
			}
		}
		else{
			// Copy one state by another:
			for(int iEigen = 0; iEigen < _nEigenDecomposition; iEigen ++){
				for(int iRateCat = 0; iRateCat < _nRateCategory; iRateCat ++, pCondlike += condlike_pad_size, pOutCondlike += condlike_size){
					for(int iPattern = 0; iPattern < _nSitePattern; iPattern ++){
						memcpyHostToHost(pOutCondlike + iPattern * _nState, pCondlike + iPattern * _nPaddedState, _nState);
					}
				}
			}
		}
	}

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg("cudaDeviceSynchronize() failed");
	printf("Leaving getIntCondlikeAll()...\n");
#endif

	return CUL_SUCCESS;
}



inline int callKernelLikelihood(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *rootCondlike, CUFlt *rateCatWeight, CUFlt *stateFreq, CUFlt *sitePatternWeight, int nNodeScaler, CUFlt *scaleFactor, const int nPaddedState, const int nState, const int nEigenDecomp, const int nRateCategory, const int nSitePattern, const int nPaddedSitePattern, dim3 nBlockPerGrid_siteLnL, dim3 nThreadPerBlock_siteLnL, int nBlockPerGrid_reduce, int nThreadPerBlock_reduce, cudaStream_t &stream)
{
	int nLeft;

	//printf("For likelihood kernel, block dimension is: (%d, %d)\ngrid dimension is: (%d, %d)\n", nThreadPerBlock_siteLnL.x, nThreadPerBlock_siteLnL.y, nBlockPerGrid_siteLnL.x, nBlockPerGrid_siteLnL.y);

#ifdef DEBUG_TIME
	timeBegin();
	for(int itr = 0; itr < 1; itr ++){
#endif

	nLeft = callKernelLikelihood_baseline(reduceLnL, siteLnL, rootCondlike, rateCatWeight, stateFreq, sitePatternWeight, nNodeScaler, scaleFactor, nPaddedState, nState, nEigenDecomp, nRateCategory, nSitePattern, nPaddedSitePattern, nBlockPerGrid_siteLnL, nThreadPerBlock_siteLnL, nBlockPerGrid_reduce, nThreadPerBlock_reduce, stream);

#ifdef DEBUG_TIME
	}
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	lnL_time = timeEnd();

	FILE *fout = fopen("lnL_time.txt", "a");
	fprintf(fout, "for nState = %d, nSitePattern = %d, nCategory = %d, nIteration = %d: %lld us (%lld.%06lld s)\n\n", nState, nSitePattern, nEigenDecomp * nRateCategory, nIteration, lnL_time, lnL_time / multiple, lnL_time % multiple);
	fclose(fout);

	fout = fopen("lnL_time_format.txt", "a");
	fprintf(fout, "%lld.%06lld\t", lnL_time / multiple, lnL_time % multiple);
	fclose(fout);
#endif
	return nLeft;
}


inline CUFlt finalReduction(CUFlt *tempLnL, int count)
{
	CUFlt lnL = 0.0f;
	for(int i = 0; i < count; i ++)
		lnL += tempLnL[i];

	return lnL;
}


// Calculate the likelihood of a partition;
// First calculate site likelihood, next reduction of site likelihoods, and the last part of reduction is finished by CPU or directly on GPU(by another kernel call); 实际上一般最后剩下的lnL的数目也不多(等于归约时block的数目，假设一个block中包含256个thread，则一个partition剩下的值一般不会超过100，应该在CPU端做效果更好);
CuLErrorCode PartitionInstance::calculateLikelihoodSync(double &lnL,
														cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering calculateLikelihoodSync()...\n");
#endif

	if(_rootLabel < 0 || _rootLabel >= _nNode || _nodeIndToArrayInd.end() == _nodeIndToArrayInd.find(_rootLabel))
		return CUL_ERROR_INDEX_OUT_OF_RANGE;

	const int condlike_pad_node_size = _nArrayPerNode * _condlike_pad_size;
	CUFlt *pRootCondlike = _dev_intCondlike + (_nodeIndToArrayInd[_rootLabel] - _nTipStateArray - _nTipCondlikeArray) * condlike_pad_node_size;

#ifdef DEBUG_CONDLIKE
	printf("_condlike_pad_size = %d, condlike_pad_node_size = %d\nroot label: %d, root array index: %d, root condlike offset: %d\n", _condlike_pad_size, condlike_pad_node_size,  _rootLabel, _nodeIndToArrayInd[_rootLabel], (_nodeIndToArrayInd[_rootLabel] - _nTipStateArray - _nTipCondlikeArray) * condlike_pad_node_size);
#endif

	// call kernel to calculate site likelihood...
	dim3 nBlockPerGrid_siteLnL(_siteLnL_gridDim_x, _siteLnL_gridDim_y, _siteLnL_gridDim_z);
	dim3 nThreadPerBlock_siteLnL(_siteLnL_blockDim_x, _siteLnL_blockDim_y, _siteLnL_blockDim_z);

	//printf("block dimension: (%d, %d)\ngrid dimension: (%d, %d)\n", nThreadPerBlock_siteLnL.x, nThreadPerBlock_siteLnL.y, nBlockPerGrid_siteLnL.x, nBlockPerGrid_siteLnL.y);

	_reduce_lnL_size = callKernelLikelihood(_dev_reduceLnL, _dev_siteLnL, pRootCondlike, _dev_rateCatWeight, _dev_stateFreq, _dev_sitePatternWeight, _curNodeScalerCnt, _dev_nodeScaleFactor, _nPaddedState, _nState, _nEigenDecomposition, _nRateCategory, _nSitePattern, _nPaddedSitePattern, nBlockPerGrid_siteLnL, nThreadPerBlock_siteLnL, _reduceLnL_gridDim_x, _reduceLnL_blockDim_x, stream);

	memcpyDeviceToHostAsync(_host_reduceLnL, _dev_reduceLnL, _reduce_lnL_size, stream);

	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	lnL = (double) finalReduction(_host_reduceLnL, _reduce_lnL_size);

#ifdef DEBUG
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	printf("Leaving calculateLikelihoodSync()...\n");
#endif

	return CUL_SUCCESS;
}



// Calculate the likelihood of a partition; 
CuLErrorCode PartitionInstance::calculateLikelihoodAsync(cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering calculateLikelihoodAsync()...\n");
#endif

	if(_rootLabel < 0 || _rootLabel >= _nNode || _nodeIndToArrayInd.end() == _nodeIndToArrayInd.find(_rootLabel))
		return CUL_ERROR_INDEX_OUT_OF_RANGE;

	const int condlike_pad_node_size = _nArrayPerNode * _condlike_pad_size;
	CUFlt *pRootCondlike = _dev_intCondlike + (_nodeIndToArrayInd[_rootLabel] - _nTipStateArray - _nTipCondlikeArray) * condlike_pad_node_size;

	// call kernel to calculate site likelihood...
	dim3 nBlockPerGrid_siteLnL(_siteLnL_gridDim_x, _siteLnL_gridDim_y, _siteLnL_gridDim_z);
	dim3 nThreadPerBlock_siteLnL(_siteLnL_blockDim_x, _siteLnL_blockDim_y, _siteLnL_blockDim_z);

	_reduce_lnL_size = callKernelLikelihood(_dev_reduceLnL, _dev_siteLnL, pRootCondlike, _dev_rateCatWeight, _dev_stateFreq, _dev_sitePatternWeight, _curNodeScalerCnt, _dev_nodeScaleFactor, _nPaddedState, _nState, _nEigenDecomposition, _nRateCategory, _nSitePattern, _nPaddedSitePattern, nBlockPerGrid_siteLnL, nThreadPerBlock_siteLnL, _reduceLnL_gridDim_x, _reduceLnL_blockDim_x, stream);

#ifdef DEBUG
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	printf("Leaving calculateLikelihoodAsync()...\n");
#endif

	return CUL_SUCCESS;
}



// Get the likelihood of a partition; 
// TODO: cudaMemcpy()阻塞的是stream中的所有operation还是整个device的operation，若阻塞的是整个device的operation，则需要将相关的host端的array声明为page-locked memory；
// 另外，是否有需要在cudaMemcpy()/cudaMemcpyAsync()之前调用cudaStreamSynchronize()? 应该没有必要吧???
CuLErrorCode PartitionInstance::getLikelihood(double &lnL,
											cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering getLikelihood()...\n");
#endif

	memcpyDeviceToHostAsync(_host_reduceLnL, _dev_reduceLnL, _reduce_lnL_size, stream);

	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	lnL = (double) finalReduction(_host_reduceLnL, _reduce_lnL_size);

#ifdef DEBUG
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	printf("Leaving getLikelihood()...\n");
#endif

	return CUL_SUCCESS;
}



// Get the site likelihood of a partition;
CuLErrorCode PartitionInstance::getSiteLikelihood(double *outSiteLikelihood,
													cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering getSiteLikelihood()...\n");
#endif

	memcpyDeviceToHostAsync(_host_siteLnL, _dev_siteLnL, _nSitePattern, stream);
	
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	memcpyHostToHost(outSiteLikelihood, _host_siteLnL, _nSitePattern);

#ifdef DEBUG
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	printf("Leaving getSiteLikelihood()...\n");
#endif

	return CUL_SUCCESS;
}



// Set the site likelihood of a partition;
CuLErrorCode PartitionInstance::setSiteLikelihood(double *inSiteLikelihood,
													cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering setSiteLikelihood()...\n");
#endif

	memcpyHostToHost(_host_siteLnL, inSiteLikelihood, _nSitePattern);
	memcpyHostToDeviceAsync(_dev_siteLnL, _host_siteLnL, _nPaddedSitePattern, stream);


#ifdef DEBUG
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");
	printf("Leaving setSiteLikelihood()...\n");
#endif

	return CUL_SUCCESS;
}


inline int callKernelLikelihoodFromSiteLnL(CUFlt *reduceLnL, CUFlt *siteLnL, CUFlt *sitePatternWeight, const int nSitePattern, int nBlockPerGrid, int nThreadPerBlock, cudaStream_t &stream)
{
	int nLeft = callKernelLikelihoodFromSiteLnL_baseline(reduceLnL, siteLnL, sitePatternWeight, nSitePattern, nBlockPerGrid, nThreadPerBlock, stream);

	return nLeft;
}



// Calculate log likelihood of a partition from specified site likelihood, that is: reduction of site likelihood;
CuLErrorCode PartitionInstance::calculateLikelihoodFromSiteLnLSync(double &lnL,
																	cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering calculateLikelihoodFromSiteLnLSync()...\n");
#endif

	_reduce_lnL_size = callKernelLikelihoodFromSiteLnL(_dev_reduceLnL, _dev_siteLnL, _dev_sitePatternWeight, _nSitePattern, _reduceLnL_gridDim_x, _reduceLnL_blockDim_x, stream);

	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	memcpyDeviceToHostAsync(_host_reduceLnL, _dev_reduceLnL, _reduce_lnL_size, stream);

	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilCheckMsg("cudaStreamSynchronize() failed");

	lnL = (double) finalReduction(_host_reduceLnL, _reduce_lnL_size);

#ifdef DEBUG
	printf("Leaving calculateLikelihoodFromSiteLnLSync()...\n");
#endif

	return CUL_SUCCESS;
}



// Calculate log likelihood of a partition from specified site likelihood, that is: reduction of site likelihood;
CuLErrorCode PartitionInstance::calculateLikelihoodFromSiteLnLAsync(cudaStream_t stream)
{
#ifdef DEBUG
	printf("Entering calculateLikelihoodFromSiteLnLAsync()...\n");
#endif

	_reduce_lnL_size = callKernelLikelihoodFromSiteLnL(_dev_reduceLnL, _dev_siteLnL, _dev_sitePatternWeight, _nSitePattern, _reduceLnL_gridDim_x, _reduceLnL_blockDim_x, stream);

#ifdef DEBUG
	printf("Leaving calculateLikelihoodFromSiteLnLAsync()...\n");
#endif

	return CUL_SUCCESS;
}



// Finalize the partition instance;
CuLErrorCode PartitionInstance::finalize(void)
{
#ifdef DEBUG_DESTROY
	printf("Entering finalize()...\n");
#endif

	// Free host page locked memory:
	freeHostPinnedMemory(_host_rate);
	freeHostPinnedMemory(_host_rateCatWeight);
	freeHostPinnedMemory(_host_stateFreq);
	freeHostPinnedMemory(_host_sitePatternWeight);
	freeHostPinnedMemory(_host_brLen);
	freeHostPinnedMemory(_host_U);
	freeHostPinnedMemory(_host_V);
	freeHostPinnedMemory(_host_R);
	freeHostPinnedMemory(_host_siteLnL);
	freeHostPinnedMemory(_host_reduceLnL);

	// Free host pageable memory:
	freeHostMemory(_host_PMat);
	freeHostMemory(_host_tipState); 
	freeHostMemory(_host_tipCondlike);
	freeHostMemory(_host_intCondlike);
	freeHostMemory(_nodesToScale);

	// Free offset arrays(page locked memory):
	freeHostPinnedMemory(_host_PMat_offset);
	freeHostPinnedMemory(_host_condlike_op);
	freeHostPinnedMemory(_host_condlike_blkIndToOpInd);
	freeHostPinnedMemory(_host_condlike_opStartBlkInd);
	freeHostPinnedMemory(_host_scale_blkIndToClOffset);
	freeHostPinnedMemory(_host_scale_blkIndToScaleOffset);
	freeHostPinnedMemory(_host_scale_startBlkInd);


	// Free device memory:
	freeDeviceMemory(_dev_rate);
	freeDeviceMemory(_dev_rateCatWeight);
	freeDeviceMemory(_dev_stateFreq);
	freeDeviceMemory(_dev_sitePatternWeight);
	freeDeviceMemory(_dev_brLen);
	freeDeviceMemory(_dev_U);
	freeDeviceMemory(_dev_V); 
	freeDeviceMemory(_dev_R);
	freeDeviceMemory(_dev_PMat);
	freeDeviceMemory(_dev_tipState); 
	freeDeviceMemory(_dev_tipCondlike); 
	freeDeviceMemory(_dev_intCondlike);
	freeDeviceMemory(_dev_nodeScaleFactor);
	freeDeviceMemory(_dev_siteLnL);
	freeDeviceMemory(_dev_reduceLnL);

	freeDeviceMemory(_dev_PMat_offset);
	freeDeviceMemory(_dev_condlike_op);
	freeDeviceMemory(_dev_condlike_blkIndToOpInd);
	freeDeviceMemory(_dev_condlike_opStartBlkInd);
	freeDeviceMemory(_dev_scale_blkIndToClOffset);
	freeDeviceMemory(_dev_scale_blkIndToScaleOffset);
	freeDeviceMemory(_dev_scale_startBlkInd);

	if(64 == _nPaddedState)
		freeDeviceMemory(_dev_exptRoot);

#ifdef DEBUG_DESTROY
	printf("Leaving finalize()...\n");
#endif

	return CUL_SUCCESS;
}
