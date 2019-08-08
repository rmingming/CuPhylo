#include "CuLibInstanceImpl.h"

//#define DEBUG

/*
// 构造函数:
// TODO: 构造函数由于没有返回值，无法知道deviceId是否合法;
CuLInstance::CuLInstance(const int nPartition,
						const int deviceId)
{
	createCuLInstance(nPartition, deviceId);
}
*/

// 析构函数:
CuLInstance::~CuLInstance()
{
#ifdef DEBUG_DESTROY
	printf("Entering ~CuLInstance()...\n");
#endif

	CuLErrorCode returnState = finalizeInstance();
	if(returnState != CUL_SUCCESS)
		printErrorCode(returnState);

#ifdef DEBUG_DESTROY
	printf("Leaving ~CuLInstance()...\n");
#endif
}



CuLErrorCode CheckDeviceId(int deviceId){
	if(deviceId < 0)
		return CUL_ERROR_DEVICE_NOT_AVALAIABLE;

	int deviceCount;
	cutilSafeCall(cudaGetDeviceCount(&deviceCount));
	cutilCheckMsg("cudaGetDeviceCount() failed");

	if(deviceId >= deviceCount)
		return CUL_ERROR_DEVICE_NOT_AVALAIABLE;
	else
		return CUL_SUCCESS;
}



int CuLInstance::CheckAndGetStreamId(const int partitionId)
{
	int streamId;
	if(_streamIndMap.find(partitionId) != _streamIndMap.end()){
		streamId = _streamIndMap[partitionId];
	}
	else{
		streamId = _nStreamInUse;
		_streamIndMap.insert(pair<int, int>(partitionId, streamId));			// TODO: 当streamIndMap中未找到当前partition index时，是应该报错(没有调用specifyParams())还是应该分配stream???

		_nStreamInUse = (_nStreamInUse + 1) % MAX_STREAM;
	}

	return streamId;
}



// Create a cuLibrary instance
CuLErrorCode CuLInstance::createCuLInstance(int nPartition,
											int deviceId)
{
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		PartitionInstance *pParInstance = new PartitionInstance();
		if(NULL == pParInstance)
			return CUL_ERROR_COMMON;

		_partitionVec.push_back(pParInstance);
	}

	_nPartition = nPartition;

	CuLErrorCode returnState = CheckDeviceId(deviceId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	// TODO: 设置的device的作用范围??? 若在一个程序中创建多个instance，则会不会出现混淆的情况???
	cutilSafeCall(cudaSetDevice(deviceId));
	cutilCheckMsg("cudaSetDevice() failed");
	_deviceId = deviceId;

	for(int iStream = 0; iStream < MAX_STREAM; iStream ++){
		cutilSafeCall(cudaStreamCreate(&_stream[iStream]));
		cutilCheckMsg("cudaStreamCreate() failed");
	}

	_nStreamInUse = 0;

	return CUL_SUCCESS;
}



// Return the count of partition of the current instance;
int CuLInstance::getPartitionCount()
{
	return _nPartition;
}



// Return the device id of the current instance;
int CuLInstance::getDeviceId()
{
	return _deviceId;
}



// Specify the parameters of a partition;
CuLErrorCode CuLInstance::specifyPartitionParams(const int partitionId,
												const int nNode,
												const int nState,
												const int nSitePattern,
												const int nRateCategory,
												const int nEigenDecomposition,
												const int nNodeForTransitionMatrix,
												const int nTipStateArray,
												const int nTipCondlikeArray,
												const int nInternalNodeForCondlike,
												const int nNodeScaler,
												const bool isRooted)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyParams(nNode,
																		nState,
																		nSitePattern,
																		nRateCategory,
																		nEigenDecomposition,
																		nNodeForTransitionMatrix,
																		nTipStateArray,
																		nTipCondlikeArray,
																		nInternalNodeForCondlike,
																		nNodeScaler,
																		isRooted);
	if(returnState != CUL_SUCCESS)
		return returnState;

	// TODO: 允许对同一个partition多次指定不同的参数???
	if(_streamIndMap.end() == _streamIndMap.find(partitionId)){
		_streamIndMap.insert(pair<int, int>(partitionId, _nStreamInUse));
	}
	else{
		_streamIndMap[partitionId] = _nStreamInUse;
	}
	_nStreamInUse = (_nStreamInUse + 1) % MAX_STREAM;


	return CUL_SUCCESS;
}



// Specify the tree topology of a partition;
CuLErrorCode CuLInstance::specifyPartitionTreeTopology(const int partitionId,
														CuLTreeNode *root)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyTreeTopology(root);

	return returnState;
}



// Specify the state frequency of a partition;
CuLErrorCode CuLInstance::specifyPartitionStateFrequency(const int partitionId,
														const double *inStateFreq)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);

	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
	*/

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyStateFrequency(inStateFreq, _stream[streamId]);

	return returnState;
}



// Specify the site pattern weight of a partition;
CuLErrorCode CuLInstance::specifyPartitionSitePatternWeight(const int partitionId, 
															const double *inPatternWeight)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->specifySitePatternWeight(inPatternWeight, _stream[streamId]);

	return returnState;
}



// Specify the rates of a partition;
CuLErrorCode CuLInstance::specifyPartitionRate(const int partitionId, 
												const double *inRate)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyRate(inRate, _stream[streamId]);

	return returnState;
}



// Specify the partition rate category weights;
CuLErrorCode CuLInstance::specifyPartitionRateCategoryWeight(const int partitionId, 
															const int nCategory,
															const int *categoryId,
															const double **inRateCategoryWeight)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyRateCategoryWeight(nCategory,
																					categoryId,
																					inRateCategoryWeight,
																					_stream[streamId]);

	return returnState;
}



// Specify the eigen vectors and eigen values of a partition; 
CuLErrorCode CuLInstance::specifyPartitionEigenDecomposition(const int partitionId,
															const int nEigenDecomp,
															const int *eigenDecompId,
															const double **inEigenVector,
															const double **inInverEigenVector,
															const double **inEigenValue)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyEigenDecomposition(nEigenDecomp,
																					eigenDecompId,
																					inEigenVector,
																					inInverEigenVector,
																					inEigenValue,
																					_stream[streamId]);

	return returnState;
}



// Specify the transition matrix of a partition;
CuLErrorCode CuLInstance::specifyPartitionTransitionMatrixMulti(const int partitionId,
																const int nNode,
																const int *nodeId,
																const int nEigenDecomp,
																const int *eigenDecompId,
																const double **inMatrix)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyTransitionMatrixMulti(nNode,
																					nodeId,
																					nEigenDecomp,
																					eigenDecompId,
																					inMatrix);

	return returnState;
}



// Specify the transition matrix of a partition;
CuLErrorCode CuLInstance::specifyPartitionTransitionMatrixAll(const int partitionId,
															const int nNode,
															const int *nodeId,
															const double **inMatrix)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyTransitionMatrixAll(nNode,
																					nodeId,
																					inMatrix);

	return returnState;
}



// Calculate transition matrix of a partition;
CuLErrorCode CuLInstance::calculatePartitionTransitionMatrixMulti(const int partitionId,
																	const int nNode,
																	const int *nodeId,
																	const int nEigenDecomp,
																	const int *eigenDecompId,
																	const double *brLen)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->calculateTransitionMatrixMulti(nNode,
																						nodeId,
																						nEigenDecomp,
																						eigenDecompId,
																						brLen,
																						_stream[streamId]);

	return returnState;
}



// Calculate transition matrix of a partition;
CuLErrorCode CuLInstance::calculatePartitionTransitionMatrixAll(const int partitionId,
																const int nNode,
																const int *nodeId,
																const double *brLen)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->calculateTransitionMatrixAll(nNode,
																						nodeId,
																						brLen,
																						_stream[streamId]);

	return returnState;
}



// Get transition matrix of a partition;
CuLErrorCode CuLInstance::getPartitionTransitionMatrixMulti(const int partitionId,
															const int nNode,
															const int *nodeId,
															const int nEigenDecomp,
															const int *eigenDecompId,
															double **outMatrix)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->getTransitionMatrixMulti(nNode,
																					nodeId,
																					nEigenDecomp,
																					eigenDecompId,
																					outMatrix);

	return returnState;
}



// Get transition matrix of a partition;
CuLErrorCode CuLInstance::getPartitionTransitionMatrixAll(const int partitionId,
														const int nNode,
														const int *nodeId,
														double **outMatrix)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->getTransitionMatrixAll(nNode,
																				nodeId,
																				outMatrix);

	return returnState;
}



// Specify the tip states of a partition;
CuLErrorCode CuLInstance::specifyPartitionTipState(const int partitionId,
												const int nTipNode,
												const int *tipNodeId,
												const int **inTipState)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyTipState(nTipNode,
																		tipNodeId,
																		inTipState);

	return returnState;
}



// Specify the tip conditional likelihoods of a partition;
CuLErrorCode CuLInstance::specifyPartitionTipCondlike(const int partitionId,
													const int nTipNode,
													const int *tipNodeId,
													const double **inTipCondlike)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyTipCondlike(nTipNode,
																			tipNodeId,
																			inTipCondlike);

	return returnState;
}



// Specify the conditional likelihoods of internal nodes;
CuLErrorCode CuLInstance::specifyPartitionInternalCondlikeMulti(const int partitionId,
																const int nNode,
																const int *nodeId,
																const int nEigenDecomp,
																const int *eigenDecompId,
																const double **inCondlike)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyInternalCondlikeMulti(nNode,
																						nodeId,
																						nEigenDecomp,
																						eigenDecompId,
																						inCondlike);

	return returnState;
}



// Specify the conditional likelihoods of internal nodes;
CuLErrorCode CuLInstance::specifyPartitionInternalCondlikeAll(const int partitionId,
															const int nNode,
															const int *nodeId,
															const double **inCondlike)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyInternalCondlikeAll(nNode,
																					nodeId,
																					inCondlike);

	return returnState;
}



// Map the node label to condlike array index;
CuLErrorCode CuLInstance::mapPartitionNodeIndToArrayInd(const int partitionId,
														const int nNode,
														const int *indMap)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->mapNodeIndToArrayInd(nNode, indMap);

	return returnState;
}



// Specify which nodes need scaling;
CuLErrorCode CuLInstance::specifyPartitionNodeScalerIndex(const int partitionId,
														const int nNodeScaler,
														const int *nodeId)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->specifyNodeScalerIndex(nNodeScaler, nodeId);

	return returnState;
}



// Calculate the conditional likelihoods of internal nodes;
CuLErrorCode CuLInstance::calculatePartitionCondlikeMulti(const int partitionId,
														const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->calculateCondlikeMulti(nNode,
																				nodeId,
																				nEigenDecomp,
																				eigenDecompId,
																				_stream[streamId]);

	return returnState;
}



// Calculate the conditional likelihoods of internal nodes;
CuLErrorCode CuLInstance::calculatePartitionCondlikeAll(const int partitionId,
														const int nNode,
														const int *nodeId)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->calculateCondlikeAll(nNode, 
																				nodeId,
																				_stream[streamId]);

	return returnState;
}



// Get the conditional likelihoods of a partition;
CuLErrorCode CuLInstance::getPartitionIntCondlikeMulti(const int partitionId,
													const int nNode,
													const int *nodeId,
													const int nEigenDecomp,
													const int *eigenDecompId,
													double **outCondlike)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->getIntCondlikeMulti(nNode,
																			nodeId,
																			nEigenDecomp,
																			eigenDecompId,
																			outCondlike);

	return returnState;
}



// Get the conditional likelihoods of a partition;
CuLErrorCode CuLInstance::getPartitionIntCondlikeAll(const int partitionId,
												const int nNode,
												const int *nodeId,
												double **outCondlike)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = _partitionVec[partitionId]->getIntCondlikeAll(nNode,
																			nodeId,
																			outCondlike);
	
	return returnState;
}



// Calculate the log likelihood value of a partition;
CuLErrorCode CuLInstance::calculatePartitionLikelihoodSync(const int partitionId,
															double &lnL)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->calculateLikelihoodSync(lnL, _stream[streamId]);

	return returnState;
}



// Calculate the log likelihood value of a partition;
CuLErrorCode CuLInstance::calculatePartitionLikelihoodAsync(const int partitionId)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->calculateLikelihoodAsync(_stream[streamId]);

	return returnState;
}



// Get the log likelihood value of a partition, should be called after function calculatePartitionLikelihoodAsync();
CuLErrorCode CuLInstance::getPartitionLikelihood(const int partitionId,
												double &lnL)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->getLikelihood(lnL, _stream[streamId]);

	return returnState;
}



// Get the site likelihood values of a partition;
CuLErrorCode CuLInstance::getPartitionSiteLikelihood(const int partitionId,
													double *outSiteLikelihood)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->getSiteLikelihood(outSiteLikelihood, _stream[streamId]);

	return returnState;
}



// Set the site likelihood values of a partition;
CuLErrorCode CuLInstance::setPartitionSiteLikelihood(const int partitionId,
													double *inSiteLikelihood)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->setSiteLikelihood(inSiteLikelihood, _stream[streamId]);

	return returnState;
}



// Calculate the log likelihood value of a partition from pre-specified site likelihood values;
CuLErrorCode CuLInstance::calculatePartitionLikelihoodFromSiteLnLSync(const int partitionId,
																		double &lnL)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->calculateLikelihoodFromSiteLnLSync(lnL, _stream[streamId]);

	return returnState;
}



// Calculate the log likelihood value of a partition from pre-specified site likelihood values;
CuLErrorCode CuLInstance::calculatePartitionLikelihoodFromSiteLnLAsync(const int partitionId)
{
	if(NULL == _partitionVec[partitionId])
		return CUL_ERROR_BAD_INSTANCE;

	int streamId = CheckAndGetStreamId(partitionId);
	/*
	if(streamId < 0 || streamId >= MAX_STREAM)
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
		*/

	CuLErrorCode returnState = _partitionVec[partitionId]->calculateLikelihoodFromSiteLnLAsync(_stream[streamId]);

	return returnState;
}



// Remove some partitions from the current instance;
CuLErrorCode CuLInstance::removePartition(const int nPartition,
										const int *partitionId)
{
	//CuLErrorCode returnState;
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		if(partitionId[iPartition] < 0 || partitionId[iPartition] >= _nPartition || NULL == _partitionVec[partitionId[iPartition]])
			return CUL_ERROR_INDEX_OUT_OF_RANGE;

		/*
		returnState = _partitionVec[partitionId[iPartition]]->finalize();
		if(CUL_SUCCESS != returnState)
			return returnState;
			*/

		if(NULL != _partitionVec[partitionId[iPartition]]){
			delete _partitionVec[partitionId[iPartition]];		// TODO: delete会调用PartitionInstance的析构函数，析构函数中会调用finalize()来释放空间，是否多余???
			_partitionVec[partitionId[iPartition]] = NULL;
		}
	}
	
	return CUL_SUCCESS;
}



// Finalize the current instance;
CuLErrorCode CuLInstance::finalizeInstance(void)
{
#ifdef DEBUG_DESTROY
	printf("Entering finalizeInstance()...\n");
#endif

	for(int iStream = 0; iStream < MAX_STREAM; iStream ++){
		cutilSafeCall(cudaStreamDestroy(_stream[iStream]));
		cutilCheckMsg("cudaStreamDestroy() failed");
	}

	//CuLErrorCode returnState;
	for(int iPartition = 0; iPartition < _nPartition; iPartition ++){
		/*
		returnState = _partitionVec[iPartition]->finalize();
		if(CUL_SUCCESS != returnState){
			return returnState;
		}
		*/

		if(NULL != _partitionVec[iPartition]){
			delete _partitionVec[iPartition];		// TODO: delete会调用PartitionInstance的析构函数，析构函数中会调用finalize()来释放空间，是否多余???
			_partitionVec[iPartition] = NULL;
		}
	}
	
#ifdef DEBUG_DESTROY
	printf("Leaving finalizeInstance()...\n");
#endif

	return CUL_SUCCESS;
}
