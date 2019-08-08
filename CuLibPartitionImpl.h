#ifndef CULIB_PARTITION_H
#define CULIB_PARTITION_H

#include "CuLibCommon.h"
#include "CuLibCommon-cuda.h"


// TODO: 加上构造函数和析构函数
// TODO: 是否应该再在host端开辟一块空间以实现host to device的一次性copy，而不是多次的copy，另外，是否使用page-locked memory?
class PartitionInstance
{
public:
	/*
	// 仅仅在模块测试时需要，之后移除!!!
	CuLErrorCode createCUDAStream(int nStream,
								int streamId);

	CuLErrorCode destroyCUDAStream(int nStream);
	*/


	// 构造函数:
	PartitionInstance(){};

	// 析构函数:
	~PartitionInstance();

	/*
	PartitionInstance(int nNode,
					int nState,
					int nSitePattern,
					int nRateCategory,
					int nEigenDecomposition,
					int nNodeForTransitionMatrix,
					int nTipStateArray,
					int nTipCondlikeArray,
					int nInternalNodeForCondlike);
					*/

	// Set grid & block dimension for calculation of PMat;
	void setPMatKernelInfor(void);

	// Set grid & block dimension for calculation of condlike;
	void setCondlikeKernelInfor(void);

	// Set grid & block dimension for scaling of condlike;
	void setScaleKernelInfor(void);

	// Set grid & block dimension for calculation of likelihood;
	void setLikelihoodKernelInfor(void);

	// Specify the paramters of the current partition instance;
	CuLErrorCode specifyParams(int nNode,
								int nState,
								int nSitePattern,
								int nRateCategory,
								int nEigenDecomposition,
								int nNodeForTransitionMatrix,
								int nTipStateArray,
								int nTipCondlikeArray,
								int nInternalNodeForCondlike,
								int nNodeScaler,
								bool isRootedTree);

	// Specify the tree topology of the current partition;
	CuLErrorCode specifyTreeTopology(CuLTreeNode *root);

	// Specify the state frequencies of the current partition;
	CuLErrorCode specifyStateFrequency(const double *inStateFreq,
										cudaStream_t stream);

	// Get the state frequencies of the current partition;
	CuLErrorCode getStateFrequency(double *outStateFreq,
									cudaStream_t stream);

	// Specify the site pattern weights of the current partition;
	CuLErrorCode specifySitePatternWeight(const double *inPatternWeight,
											cudaStream_t stream);

	// Get the site pattern weights of the current partition;
	CuLErrorCode getSitePatternWeight(double *outPatternWeight,
										cudaStream_t stream);

	// Specify the rates of the current partition;
	CuLErrorCode specifyRate(const double *inRate,
							cudaStream_t stream);

	// Get the rates of the current partition;
	CuLErrorCode getRate(double *outRate,
						cudaStream_t stream);

	// Specify the rate category weights of the current partition;
	CuLErrorCode specifyRateCategoryWeight(const int nCategory,
											const int *categoryId,
											const double **inRateCategoryWeight,
											cudaStream_t stream);

	// Get the rate category weights of the current partition;
	CuLErrorCode getRateCategoryWeight(const int nCategory,
										const int *categoryId,
										double **outRateCategoryWeight,
										cudaStream_t stream);

	// Specify the eigen decomposition of the current partition;
	CuLErrorCode specifyEigenDecomposition(const int nEigenDecomp,
											const int *eigenDecompId,
											const double **inEigenVector,
											const double **inInverEigenVector,
											const double **inEigenValue,
											cudaStream_t stream);

	// Get the eigen decomposition of the current partition;
	CuLErrorCode getEigenDecomposition(const int nEigenDecomp,
										const int *eigenDecompId,
										double **outEigenVector,
										double **outInverEigenVector,
										double **outEigenValue,
										cudaStream_t stream);

	// Specify the transition matrix of the current partition;
	CuLErrorCode specifyTransitionMatrixMulti(const int nNode,
											const int *nodeId,
											const int nEigenDecomp,
											const int *eigenDecompId,
											const double **inMatrix);

	// Specify the transition matrix of the current partition;
	CuLErrorCode specifyTransitionMatrixAll(const int nNode,
											const int *nodeId,
											const double **inMatrix);

	// Calculate the transition matrices of the current partition;
	CuLErrorCode calculateTransitionMatrixMulti(const int nNode,
												const int *nodeId,
												const int nEigenDecomp,
												const int *eigenDecompId,
												const double *brLen,
												cudaStream_t stream);

	// Calculate the transition matrices of the current partition;
	CuLErrorCode calculateTransitionMatrixAll(const int nNode,
											const int *nodeId,
											const double *brLen,
											cudaStream_t stream);

	// Get transition matrix of a partition;
	CuLErrorCode getTransitionMatrixMulti(const int nNode,
										const int *nodeId,
										const int nEigenDecomp,
										const int *eigenDecompId,
										double **outMatrix);

	// Get the transition matrices of the current partition;
	CuLErrorCode getTransitionMatrixAll(const int nNode,
										const int *nodeId,
										double **outMatrix);

	// Specify the tip states of the current partition;
	CuLErrorCode specifyTipState(const int nTipNode,
								const int *tipNodeId,
								const int **inTipState);

	// Get the tip states of the current partition;
	CuLErrorCode getTipState(const int nTipNode,
							const int *tipNodeId,
							int **outTipState);

	// Specify the tip conditional likelihoods of the current partition;
	CuLErrorCode specifyTipCondlike(const int nTipNode,
									const int *tipNodeId,
									const double **inTipCondlike);

	// Get the tip conditional likelihoods of the current partition;
	CuLErrorCode getTipCondlike(const int nTipNode,
								const int *tipNodeId,
								double **outTipCondlike);

	// Specify the conditional likelihoods of internal nodes of the current partition;
	CuLErrorCode specifyInternalCondlikeMulti(const int nNode,
											const int *nodeId,
											const int nEigenDecomp,
											const int *eigenDecompId,
											const double **inCondlike);

	// Specify the conditional likelihoods of internal nodes of the current partition; 
	CuLErrorCode specifyInternalCondlikeAll(const int nNode,
											const int *nodeId,
											const double **inCondlike);

	// Map the node index to tipState / tipCondlike / intCondlike array index;
	CuLErrorCode mapNodeIndToArrayInd(const int nNode,
									const int *indMap);

	// Map the thread block index to condlike operation index, assert at least one thread block is responsible for an operation;
	int setCondlikeBlockIndToArrayInd(const bool isRooted,
									const bool usePadVersion,
									int *pHostBlkIndToArrayInd, 
									int *pHostStartBlkInd,
									CuLCondlikeOp *condlikeOp,
									const int nOp,
									int &nThreadPerArray_baseline);

	// Specify which nodes need scaling:
	CuLErrorCode specifyNodeScalerIndex(const int nNodeScaler,
										const int *nodeId);

	// Set offset of each thread block for scaling:
	int setScaleBlockIndToOffset(int nNodeToScale, 
								int *nodeId, 
								int *pHostBlkIndToClOffset,
								int *pHostBlkIndToScaleOffset,
								int *pHostStartBlkInd);

	// Calculate the conditional likelihood of a partition;
	CuLErrorCode calculateCondlikeMulti(const int nNode,
										const int *nodeId,
										const int nEigenDecomp,
										const int *eigenDecompId,
										cudaStream_t stream);

	CuLErrorCode calculateCondlikeMulti_unrooted(const int nNode,
										const int *nodeId,
										const int nEigenDecomp,
										const int *eigenDecompId,
										cudaStream_t stream);

	// Calculate the conditional likelihood of a partition;
	CuLErrorCode calculateCondlikeAll(const int nNode,
									const int *nodeId,
									cudaStream_t stream);

	CuLErrorCode calculateCondlikeAll_unrooted(const int nNode,
									const int *nodeId,
									cudaStream_t stream);

	// Get the conditional likelihood of a partition;
	CuLErrorCode getIntCondlikeMulti(const int nNode,
								const int *nodeId,
								const int nEigenDecomp,
								const int *eigenDecompId,
								double **outCondlike);

	// Get the conditional likelihood of a partition;
	CuLErrorCode getIntCondlikeAll(const int nNode,
								const int *nodeId,
								double **outCondlike);

	// Calculate the likelihood of a partition;
	CuLErrorCode calculateLikelihoodSync(double &lnL,
										cudaStream_t stream);

	// Calculate the likelihood of a partition; 
	CuLErrorCode calculateLikelihoodAsync(cudaStream_t stream);

	// Get the likelihood of a partition; 
	CuLErrorCode getLikelihood(double &lnL,
							cudaStream_t stream);

	// Get the site likelihood of a partition;
	CuLErrorCode getSiteLikelihood(double *outSiteLikelihood,
									cudaStream_t stream);

	// Set the site likelihood of a partition;
	CuLErrorCode setSiteLikelihood(double *inSiteLikelihood,
									cudaStream_t stream);

	// Calculate the log likelihood value of a partition from pre-specified site likelihood values;
	CuLErrorCode calculateLikelihoodFromSiteLnLSync(double &lnL,
													cudaStream_t stream);

	// Calculate the log likelihood value of a partition from pre-specified site likelihood values;
	CuLErrorCode calculateLikelihoodFromSiteLnLAsync(cudaStream_t stream);

	// Finalize the partition instance;
	CuLErrorCode finalize(void);

private:

	int _nNode;
	int _nState;
	int _nSitePattern;
	int _nRateCategory;				// CHECK: category rates只有一组，但是category weights有nEigenDecomposition组；
	int _nEigenDecomposition;
	int _nNodeForTransitionMatrix;
	int _nTipStateArray;
	int _nTipCondlikeArray;
	int _nInternalNodeForCondlike;
	int _nNodeScaler;
	int _curNodeScalerCnt;

	int _nArrayPerNode;
	int _maxPMatOpCount, _maxCondlikeOpCount;		// 注意：分配与operation有关的内存时是按照该值进行分配的，在实际计算PMat/condlike的时候需要核对operation的数目是否超过了分配的内存大小!!!
	int _maxScaleOpCount;
	int _reduce_lnL_size;			// 注意：reduce_lnL_size也即分配的_dev_reduceLnL的大小;
	
	//int _streamId;
	int _nPaddedState;
	int _nPaddedSitePattern;
	int _condlike_size, _condlike_pad_size;
	int _PMat_size, _PMat_pad_size;

	// TODO: 哪些memory应该分配为page locked/pinned memory以支持异步的copy??? 应该是处理_host_PMat和_host_intCondlike的都可以分配为page locked memory，因为不太大，而且需要在host和device之间进行copy，根据MrBayes调用beagle的情况，PMat和intCondlike并不需要在host和device之间进行copy，一般是直接计算得到，因此没必要分配为page locked memory;
	// 与condlike/PMat operation有关的数组也应该分配为page locked memory;
	CUFlt *_host_rate, *_host_rateCatWeight;
	CUFlt *_host_stateFreq;
	CUFlt *_host_sitePatternWeight;
	CUFlt *_host_brLen;		// 假设brLen的组织方式为：同一个rateCategory的所有node的brLen在一起；
	CUFlt *_host_U, *_host_V, *_host_R;
	CUFlt *_host_PMat;
	int *_host_tipState;
	CUFlt *_host_tipCondlike, *_host_intCondlike;
	CUFlt *_host_siteLnL;
	CUFlt *_host_reduceLnL;

	// For offset:
	CuLPMatOffset *_host_PMat_offset;
	CuLCondlikeOp *_host_condlike_op;

	int *_host_condlike_blkIndToOpInd, *_host_condlike_opStartBlkInd;
	int *_host_scale_blkIndToClOffset, *_host_scale_blkIndToScaleOffset, *_host_scale_startBlkInd;

	// Device:
	CUFlt *_dev_rate, *_dev_rateCatWeight;
	CUFlt *_dev_stateFreq;
	CUFlt *_dev_sitePatternWeight;
	CUFlt *_dev_brLen;
	CUFlt *_dev_U, *_dev_V, *_dev_R;
	CUFlt *_dev_exptRoot;			// For state == 64;  
	CUFlt *_dev_PMat;
	int *_dev_tipState;
	CUFlt *_dev_tipCondlike, *_dev_intCondlike;
	CUFlt *_dev_nodeScaleFactor;
	CUFlt *_dev_siteLnL;
	CUFlt *_dev_reduceLnL;

	CuLPMatOffset *_dev_PMat_offset;
	CuLCondlikeOp *_dev_condlike_op;

	int *_dev_condlike_blkIndToOpInd, *_dev_condlike_opStartBlkInd;
	int *_dev_scale_blkIndToClOffset, *_dev_scale_blkIndToScaleOffset, *_dev_scale_startBlkInd;


	bool _isRootedTree;
	int _rootLabel;
	int _nLayerOfTree;					// the maximum depth
	std::map<int, std::vector<CuLTreeNode*> > _nodeLayerMap;		// key: layer id, 0 means the root, value: the nodes in the corresponding layer;
	std::map<int, int> _nodeIndToArrayInd;		// map the node id(tree node label) to the array id(condlike array index)
	std::map<int, int> _nodeIndToScaleInd;		// -1 if the node is not a scaler node, else the index of the scale buffer;
	int *_nodesToScale;

	// For use of calling PMat kernels:
	int _PMat_blockDim_x, _PMat_blockDim_y, _PMat_blockDim_z;
	int _PMat_gridDim_x, _PMat_gridDim_y, _PMat_gridDim_z;
	int _nPMatArrayPerBlock;
	int _nBlockPerPMatArray;

	// For use of calling condlike kernels:
	int _condlike_blockDim_x, _condlike_blockDim_y, _condlike_blockDim_z;
	int _condlike_blockDim_x_pad, _condlike_blockDim_y_pad, _condlike_blockDim_z_pad;
	int _condlike_blockDim_x_noPad, _condlike_blockDim_y_noPad, _condlike_blockDim_z_noPad;
	int _condlike_gridDim_x, _condlike_gridDim_y, _condlike_gridDim_z;
	int _nBlockPerClArray_nonCodeml, _nBlockPerClArray_codeml;
	int _nThreadBlockPerClArray, _nThreadBlockPerClArray_pad, _nThreadBlockPerClArray_noPad;

	// For use of calling scaling kernels:
	int _scale_blockDim_x;
	int _nThreadBlockPerScaleNode;

	// For use of calling site lnl kernels:
	int _siteLnL_blockDim_x, _siteLnL_blockDim_y, _siteLnL_blockDim_z;
	int _siteLnL_gridDim_x, _siteLnL_gridDim_y, _siteLnL_gridDim_z;

	// For use of reduction of site likelihoods:
	int _reduceLnL_blockDim_x;
	int _reduceLnL_gridDim_x;
};



inline void getDeviceMemoryInfo(size_t *totalMemory, size_t *freeMemory)
{
	cutilSafeCall(cudaMemGetInfo(totalMemory, freeMemory));
	cutilCheckMsg("cudaGetMemInfo() failed");
}


inline void* callocHostMemory(size_t nElement, size_t typeSize)
{
	void *pMemory = calloc(nElement, typeSize);
	
	return pMemory;
}


inline void* mallocHostMemory(size_t nElement, size_t typeSize)
{
	void *pMemory = malloc(nElement * typeSize);
	
	return pMemory;
}


template<typename T>
inline void callocHostPinnedMemory(T *&pArray, size_t nElement)
{
	cutilSafeCall(cudaHostAlloc((void **)&pArray, nElement * sizeof(T), cudaHostAllocDefault));
	cutilCheckMsg("cudaHostAlloc() failed");

	memset(pArray, 0, nElement * sizeof(T));
}


template<typename T>
inline void mallocHostPinnedMemory(T *&pArray, size_t nElement)
{
	cutilSafeCall(cudaHostAlloc((void **)&pArray, nElement * sizeof(T), cudaHostAllocDefault));
	cutilCheckMsg("cudaHostAlloc() failed");
}


template<typename T>
inline void callocDeviceMemory(T *&pMemory, size_t nElement)
{
	cutilSafeCall(cudaMalloc((void**)&pMemory, nElement * sizeof(T)));
	cutilCheckMsg("cudaMalloc() failed");
	cutilSafeCall(cudaMemset(pMemory, 0, nElement * sizeof(T)));
	cutilCheckMsg("cudaMemset() failed");
}

template<typename T>
inline void mallocDeviceMemory(T *&pMemory, size_t nElement)
{
	cutilSafeCall(cudaMalloc((void**)&pMemory, nElement * sizeof(T)));
	cutilCheckMsg("cudaMalloc() failed");
}

template<typename T1, typename T2>
inline void memcpyHostToHost(T1 *dst, T2 *src, size_t nElement)
{
	for(size_t iElem = 0; iElem < nElement; iElem ++)
		dst[iElem] = (T1) src[iElem];
}


template<typename T>
inline void memcpyHostToHost(T *dst, T *src, size_t nElement)
{
	memcpy(dst, src, nElement * sizeof(T));
}


template<typename T>
inline void memcpyHostToHostAndReplicate(T *dst, T *src, size_t nElement, size_t step, int nTimes)
{
	memcpyHostToHost(dst, src, nElement);

	int offset = 0;
	for(int itr = 0; itr < nTimes; itr ++, offset += step){
		memcpy(dst + offset + step, dst + offset, nElement);
	}
}


template<typename T1, typename T2>
inline void memcpyHostToHostAndReplicate(T1 *dst, T2 *src, size_t nElement, size_t step, int nTimes)
{
	memcpyHostToHost(dst, src, nElement);

	int offset = 0;
	for(int itr = 0; itr < nTimes; itr ++, offset += step){
		memcpy(dst + offset + step, dst + offset, nElement);
	}
}


template<typename T>
inline void memcpyHostToDeviceSync(T *dst, T *src, size_t nElement)
{
	cutilSafeCall(cudaMemcpy(dst, src, nElement * sizeof(T), cudaMemcpyHostToDevice));
	cutilCheckMsg("cudaMemcpy() host to device failed");
}

template<typename T>
inline void memcpyHostToDeviceAsync(T *dst, T *src, size_t nElement, cudaStream_t stream)
{
	cutilSafeCall(cudaMemcpyAsync(dst, src, nElement * sizeof(T), cudaMemcpyHostToDevice, stream));
	cutilCheckMsg("cudaMemcpy() host to device failed");
}

template<typename T>
inline void memcpyDeviceToHostSync(T *dst, T *src, size_t nElement)
{
	cutilSafeCall(cudaMemcpy(dst, src, nElement * sizeof(T), cudaMemcpyDeviceToHost));
	cutilCheckMsg("cudaMemcpy() device to host failed");
}


template<typename T>
inline void memcpyDeviceToHostAsync(T *dst, T *src, size_t nElement, cudaStream_t stream)
{
	cutilSafeCall(cudaMemcpyAsync(dst, src, nElement * sizeof(T), cudaMemcpyDeviceToHost, stream));
	cutilCheckMsg("cudaMemcpy() device to host failed");
}


template<typename T>
inline void freeHostMemory(T *&pArray)
{
	if(pArray){
		free(pArray);
		pArray = NULL;
	}
}


template<typename T>
inline void freeHostPinnedMemory(T *&pArray)
{
	if(pArray){
		cutilSafeCall(cudaFreeHost(pArray));
		cutilCheckMsg("cudaFreeHost() failed");
		pArray = NULL;
	}
}


template<typename T>
inline void freeDeviceMemory(T *&pArray)
{
	if(pArray){
		cutilSafeCall(cudaFree(pArray));
		cutilCheckMsg("cudaFree() failed");
		pArray = NULL;
	}
}


template<typename T>
inline void resetHostMemory(T *pArray, size_t nElement)
{
	memset(pArray, 0, nElement * sizeof(T));
}


template<typename T>
inline void resetDeviceMemory(T *pArray, size_t nElement)
{
	cutilSafeCall(cudaMemset(pArray, 0, nElement * sizeof(T)));
	cutilCheckMsg("cudaMemset() failed");
}

#endif