#include "CuLibrary.h"

//#define DEBUG

std::vector<CuLInstance*> _cuLInstanceVec;


inline CuLErrorCode CheckInstanceAndParitionId(const int instanceId,
											const int partitionId)
{
	if(instanceId < 0 || instanceId >= _cuLInstanceVec.size() || NULL == _cuLInstanceVec[instanceId] || partitionId < 0 || partitionId >= _cuLInstanceVec[instanceId]->getPartitionCount())
		return CUL_ERROR_INDEX_OUT_OF_RANGE;
	else
		return CUL_SUCCESS;
}



// TODO: 检查deviceId是否合法，也即是否有对应的GPU存在，若不存在，或者非N卡，则设置error code为CUL_ERROR_DEVICE_NOT_AVALAIABLE，否则设置error code为CUL_SUCCESS
CuLErrorCode CuLInitializeCuLInstance(const int nPartition, 
									const int deviceId, 
									int &cuLInstanceId)
{
	CuLInstance *pCuLInstance = new CuLInstance();
	if(NULL == pCuLInstance)
		return CUL_ERROR_BAD_INSTANCE;

	CuLErrorCode returnState = pCuLInstance->createCuLInstance(nPartition, 
																deviceId);
	if(CUL_SUCCESS == returnState){
		_cuLInstanceVec.push_back(pCuLInstance);
		cuLInstanceId = _cuLInstanceVec.size() - 1;			// CuLInstance index starts from 0
	}
	else{
		cuLInstanceId = -1;
	}
	
	return returnState;
}



CuLErrorCode CuLSpecifyPartitionParams(const int cuLInstanceId, 
										const int partitionId, 
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
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId,
															partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionParams(partitionId, 
																		nNode,
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

	return returnState;
}



CuLErrorCode CuLSpecifyPartitionTreeTopology(const int cuLInstanceId,
											const int partitionId,
											CuLTreeNode* root)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionTreeTopology(partitionId, root);
	
	return returnState;
}



CuLErrorCode CuLSpecifyPartitionStateFrequency(const int cuLInstanceId,
												const int partitionId,
												const double *inStateFreq)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionStateFrequency(partitionId,
																				inStateFreq);

	return returnState;
}


CuLErrorCode CuLSpecifyPartitionSitePatternWeight(const int cuLInstanceId,
												const int partitionId, 
												const double *inPatternWeight)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionSitePatternWeight(partitionId, 
																				inPatternWeight);

	return returnState;
}



CuLErrorCode CuLSpecifyPartitionRate(const int cuLInstanceId,
										const int partitionId, 
										const double *inRate)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionRate(partitionId, inRate);

	return returnState;
}



CuLErrorCode CuLSpecifyPartitionRateCategoryWeight(const int cuLInstanceId,
													const int partitionId, 
													const int nCategory,
													const int *categoryId,
													const double **inRateCategoryWeight)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionRateCategoryWeight(partitionId, 
																					nCategory,
																					categoryId,
																					inRateCategoryWeight);

	return returnState;
}


CuLErrorCode CuLSpecifyPartitionEigenDecomposition(const int cuLInstanceId,
													const int partitionId,
													const int nEigenDecomp,
													const int *eigenDecompId,
													const double **inEigenVector,
													const double **inInverEigenVector,
													const double **inEigenValue)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionEigenDecomposition(partitionId,
																					nEigenDecomp,
																					eigenDecompId,
																					inEigenVector,
																					inInverEigenVector,
																					inEigenValue);

	return returnState;
}


CuLErrorCode CuLSpecifyPartitionTransitionMatrixMulti(const int cuLInstanceId,
														const int partitionId,
														const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														const double **inMatrix)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionTransitionMatrixMulti(partitionId,
																						nNode,
																						nodeId,
																						nEigenDecomp,
																						eigenDecompId,
																						inMatrix);

	return returnState;
}



CuLErrorCode CuLSpecifyPartitionTransitionMatrixAll(const int cuLInstanceId,
													const int partitionId,
													const int nNode,
													const int *nodeId,
													const double **inMatrix)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionTransitionMatrixAll(partitionId,
																						nNode,
																						nodeId,
																						inMatrix);

	return returnState;
}



CuLErrorCode CuLCalculatePartitionTransitionMatrixMulti(const int cuLInstanceId,
														const int partitionId,
														const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														const double *brLen)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->calculatePartitionTransitionMatrixMulti(partitionId,
																						nNode,
																						nodeId,
																						nEigenDecomp,
																						eigenDecompId,
																						brLen);

	return returnState;
}


CuLErrorCode CuLCalculatePartitionTransitionMatrixAll(const int cuLInstanceId,
														const int partitionId,
														const int nNode,
														const int *nodeId,
														const double *brLen)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->calculatePartitionTransitionMatrixAll(partitionId,
																						nNode,
																						nodeId,
																						brLen);

	return returnState;
}


CuLErrorCode CuLGetPartitionTransitionMatrixMulti(const int cuLInstanceId,
													const int partitionId,
													const int nNode,
													const int *nodeId,
													const int nEigenDecomp,
													const int *eigenDecompId,
													double **outMatrix)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->getPartitionTransitionMatrixMulti(partitionId,
																					nNode,
																					nodeId,
																					nEigenDecomp,
																					eigenDecompId,
																					outMatrix);

	return returnState;
}


CuLErrorCode CuLGetPartitionTransitionMatrixAll(const int cuLInstanceId,
												const int partitionId,
												const int nNode,
												const int *nodeId,
												double **outMatrix)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->getPartitionTransitionMatrixAll(partitionId,
																					nNode,
																					nodeId,
																					outMatrix);

	return returnState;
}



CuLErrorCode CuLSpecifyPartitionTipState(const int cuLInstanceId,
										const int partitionId,
										const int nTipNode,
										const int *tipNodeId,
										const int **inTipState)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionTipState(partitionId,
																			nTipNode,
																			tipNodeId,
																			inTipState);

	return returnState;
}


CuLErrorCode CuLSpecifyPartitionTipCondlike(const int cuLInstanceId,
											const int partitionId,
											const int nTipNode,
											const int *tipNodeId,
											const double **inTipCondlike)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionTipCondlike(partitionId,
																			nTipNode,
																			tipNodeId,
																			inTipCondlike);

	return returnState;
}


CuLErrorCode CuLSpecifyPartitionInternalCondlikeMulti(const int cuLInstanceId,
														const int partitionId,
														const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														const double **inCondlike)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionInternalCondlikeMulti(partitionId,
																						nNode,
																						nodeId,
																						nEigenDecomp,
																						eigenDecompId,
																						inCondlike);

	return returnState;
}



CuLErrorCode CuLSpecifyPartitionInternalCondlikeAll(const int cuLInstanceId,
													const int partitionId,
													const int nNode,
													const int *nodeId,
													const double **inCondlike)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionInternalCondlikeAll(partitionId,
																						nNode,
																						nodeId,
																						inCondlike);

	return returnState;
}



CuLErrorCode CuLMapPartitionNodeIndToArrayInd(const int cuLInstanceId,
											const int partitionId,
											const int nNode,
											const int *indMap)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->mapPartitionNodeIndToArrayInd(partitionId,
																				nNode,
																				indMap);

	return returnState;
}



CuLErrorCode CuLSpecifyPartitionNodeScalerIndex(const int cuLInstanceId,
												const int partitionId,
												const int nNodeScaler,
												const int *nodeId)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->specifyPartitionNodeScalerIndex(partitionId,
																				nNodeScaler,
																				nodeId);
	
	return returnState;
}



CuLErrorCode CuLCalculatePartitionCondlikeMulti(const int cuLInstanceId,
												const int partitionId,
												const int nNode,
												const int *nodeId,
												const int nEigenDecomp,
												const int *eigenDecompId)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->calculatePartitionCondlikeMulti(partitionId,
																					nNode,
																					nodeId,
																					nEigenDecomp,
																					eigenDecompId);

	return returnState;
}


CuLErrorCode CuLCalculatePartitionCondlikeAll(const int cuLInstanceId,
												const int partitionId,
												const int nNode,
												const int *nodeId)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->calculatePartitionCondlikeAll(partitionId,
																					nNode,
																					nodeId);

	return returnState;
}


CuLErrorCode CuLGetPartitionIntCondlikeMulti(const int cuLInstanceId,
											const int partitionId,
											const int nNode,
											const int *nodeId,
											const int nEigenDecomp,
											const int *eigenDecompId,
											double **outCondlike)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->getPartitionIntCondlikeMulti(partitionId,
																				nNode,
																				nodeId,
																				nEigenDecomp,
																				eigenDecompId,
																				outCondlike);

	return returnState;
}



CuLErrorCode CuLGetPartitionIntCondlikeAll(const int cuLInstanceId,
										const int partitionId,
										const int nNode,
										const int *nodeId,
										double **outCondlike)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->getPartitionIntCondlikeAll(partitionId,
																			nNode,
																			nodeId,
																			outCondlike);
		
	return returnState;
}



CuLErrorCode CuLCalculatePartitionLikelihoodSync(const int cuLInstanceId,
												const int partitionId,
												double &lnL)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->calculatePartitionLikelihoodSync(partitionId, lnL);

	return returnState;
}



CuLErrorCode CuLCalculatePartitionLikelihoodAsync(const int cuLInstanceId,
												const int partitionId)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->calculatePartitionLikelihoodAsync(partitionId);

	return returnState;
}



CuLErrorCode CuLGetPartitionLikelihood(const int cuLInstanceId,
										const int partitionId,
										double &lnL)
{
#ifdef DEBUG
	printf("Entering CuLGetPartitionLikelihood()...\n");
#endif

	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->getPartitionLikelihood(partitionId, lnL);

#ifdef DEBUG
	printf("Leaving CuLGetPartitionLikelihood()...\n");
#endif

	return returnState;
}



CuLErrorCode CuLGetPartitionSiteLikelihood(const int cuLInstanceId,
											const int partitionId,
											double *outSiteLikelihood)
{
#ifdef DEBUG
	printf("Entering CuLGetPartitionSiteLikelihood()...\n");
#endif

	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->getPartitionSiteLikelihood(partitionId, outSiteLikelihood);

#ifdef DEBUG
	printf("Leaving CuLGetPartitionSiteLikelihood()...\n");
#endif

	return returnState;
}



CuLErrorCode CuLSetPartitionSiteLikelihood(const int cuLInstanceId,
											const int partitionId,
											double *inSiteLikelihood)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->setPartitionSiteLikelihood(partitionId, inSiteLikelihood);

	return returnState;
}



CuLErrorCode CuLCalculatePartitionLikelihoodFromSiteLnLSync(const int cuLInstanceId,
															const int partitionId,
															double &lnL)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->calculatePartitionLikelihoodFromSiteLnLSync(partitionId, lnL);

	return returnState;
}



CuLErrorCode CuLCalculatePartitionLikelihoodFromSiteLnLAsync(const int cuLInstanceId,
															const int partitionId)
{
	CuLErrorCode returnState = CheckInstanceAndParitionId(cuLInstanceId, 
														partitionId);
	if(returnState != CUL_SUCCESS)
		return returnState;

	returnState = _cuLInstanceVec[cuLInstanceId]->calculatePartitionLikelihoodFromSiteLnLAsync(partitionId);

	return returnState;
}



CuLErrorCode CuLRemovePartition(const int cuLInstanceId,
								const int nPartition,
								const int *partitionId)
{
	if(cuLInstanceId < 0 || cuLInstanceId >= _cuLInstanceVec.size() || NULL == _cuLInstanceVec[cuLInstanceId])
		return CUL_ERROR_INDEX_OUT_OF_RANGE;

	CuLErrorCode returnState = _cuLInstanceVec[cuLInstanceId]->removePartition(nPartition, partitionId);

	return returnState;
}


CuLErrorCode CuLFinalizeInstance(const int cuLInstanceId)
{
#ifdef DEBUG_DESTROY
	printf("Entering CuLFinalizeInstance()...\n");
#endif

	if(cuLInstanceId < 0 || cuLInstanceId >= _cuLInstanceVec.size() || NULL == _cuLInstanceVec[cuLInstanceId])
		return CUL_ERROR_INDEX_OUT_OF_RANGE;

	/*
	CuLErrorCode returnState = _cuLInstanceVec[cuLInstanceId]->finalizeInstance();
	if(CUL_SUCCESS != returnState)
		return returnState;
		*/

	if(NULL != _cuLInstanceVec[cuLInstanceId]){
		delete _cuLInstanceVec[cuLInstanceId];
		_cuLInstanceVec[cuLInstanceId] = NULL;
	}

#ifdef DEBUG_DESTROY
	printf("Leaving CuLFinalizeInstance()...\n");
#endif

	return CUL_SUCCESS;
}

