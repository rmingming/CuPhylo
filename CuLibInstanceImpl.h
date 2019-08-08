#ifndef CULIB_INSTANCE_IMPL_H
#define CULIB_INSTANCE_IMPL_H

#include "CuLibPartitionImpl.h"
#include "CuLibCommon.h"
#include "CuLibCommon-cuda.h"


// TODO: 加上构造函数和析构函数???
class CuLInstance
{
public:
	// 构造函数：
	CuLInstance(){};

	// 析构函数:
	~CuLInstance();

	/*
	CuLInstance(const int nPartition,
				const int deviceId);
				*/

	// Create a cuLibrary instance
	CuLErrorCode createCuLInstance(const int nPartition,
									const int deviceId);

	// Return the count of partition of the current instance;
	int getPartitionCount();

	// Return the device id of the current instance;
	int getDeviceId();

	// Get stream index;
	int CheckAndGetStreamId(const int partitionId);

	// Specify the parameters of a partition;
	CuLErrorCode specifyPartitionParams(const int partitionId,
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
										const bool isRooted);

	// Specify the tree topology of a partition;
	CuLErrorCode specifyPartitionTreeTopology(const int partitionId,
												CuLTreeNode *root);

	// Specify the state frequency of a partition;
	CuLErrorCode specifyPartitionStateFrequency(const int partitionId,
												const double *inStateFreq);

	// Specify the site pattern weight of a partition;
	CuLErrorCode specifyPartitionSitePatternWeight(const int partitionId, 
													const double *inPatternWeight);

	// Specify the rates of a partition;
	CuLErrorCode specifyPartitionRate(const int partitionId, 
									const double *inRate);

	// Specify the partition rate category weights;
	CuLErrorCode specifyPartitionRateCategoryWeight(const int partitionId, 
													const int nCategory,
													const int *categoryId,
													const double **inRateCategoryWeight);

	// Specify the eigen vectors and eigen values of a partition; 
	CuLErrorCode specifyPartitionEigenDecomposition(const int partitionId,
													const int nEigenDecomp,
													const int *eigenDecompId,
													const double **inEigenVector,
													const double **inInverEigenVector,
													const double **inEigenValue);

	// Specify the transition matrix of a partition;
	CuLErrorCode specifyPartitionTransitionMatrixMulti(const int partitionId,
														const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														const double **inMatrix);

	// Specify the transition matrix of a partition;
	CuLErrorCode specifyPartitionTransitionMatrixAll(const int partitionId,
														const int nNode,
														const int *nodeId,
														const double **inMatrix);

	// Calculate transition matrix of a partition;
	CuLErrorCode calculatePartitionTransitionMatrixMulti(const int partitionId,
														const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														const double *brLen);

	// Calculate transition matrix of a partition;
	CuLErrorCode calculatePartitionTransitionMatrixAll(const int partitionId,
														const int nNode,
														const int *nodeId,
														const double *brLen);

	// Get transition matrix of a partition;
	CuLErrorCode getPartitionTransitionMatrixMulti(const int partitionId,
													const int nNode,
													const int *nodeId,
													const int nEigenDecomp,
													const int *eigenDecompId,
													double **outMatrix);

	// Get transition matrix of a partition;
	CuLErrorCode getPartitionTransitionMatrixAll(const int partitionId,
													const int nNode,
													const int *nodeId,
													double **outMatrix);

	// Specify the tip states of a partition;
	CuLErrorCode specifyPartitionTipState(const int partitionId,
											const int nTipNode,
											const int *tipNodeId,
											const int **inTipState);

	// Specify the tip conditional likelihoods of a partition;
	CuLErrorCode specifyPartitionTipCondlike(const int partitionId,
											const int nTipNode,
											const int *tipNodeId,
											const double **inTipCondlike);

	// Specify the conditional likelihoods of internal nodes;
	CuLErrorCode specifyPartitionInternalCondlikeMulti(const int partitionId,
														const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														const double **inCondlike);

	// Specify the conditional likelihoods of internal nodes;
	CuLErrorCode specifyPartitionInternalCondlikeAll(const int partitionId,
													const int nNode,
													const int *nodeId,
													const double **inCondlike);

	// Map the node label to condlike array index;
	CuLErrorCode mapPartitionNodeIndToArrayInd(const int partitionId,
											const int nNode,
											const int *indMap);

	// Specify which nodes need scaling;
	CuLErrorCode specifyPartitionNodeScalerIndex(const int partitionId,
												const int nNodeScaler,
												const int *nodeId);

	// Calculate the conditional likelihoods of internal nodes;
	CuLErrorCode calculatePartitionCondlikeMulti(const int partitionId,
												const int nNode,
												const int *nodeId,
												const int nEigenDecomp,
												const int *eigenDecompId);

	// Calculate the conditional likelihoods of internal nodes;
	CuLErrorCode calculatePartitionCondlikeAll(const int partitionId,
												const int nNode,
												const int *nodeId);

	// Get the conditional likelihoods of a partition;
	CuLErrorCode getPartitionIntCondlikeMulti(const int partitionId,
											const int nNode,
											const int *nodeId,
											const int nEigenDecomp,
											const int *eigenDecompId,
											double **outCondlike);

	// Get the conditional likelihoods of a partition;
	CuLErrorCode getPartitionIntCondlikeAll(const int partitionId,
										const int nNode,
										const int *nodeId,
										double **outCondlike);

	// Calculate the log likelihood value of a partition;
	CuLErrorCode calculatePartitionLikelihoodSync(const int partitionId,
													double &lnL);

	// Calculate the log likelihood value of a partition;
	CuLErrorCode calculatePartitionLikelihoodAsync(const int partitionId);

	// Get the log likelihood value of a partition, should be called after function calculatePartitionLikelihoodAsync();
	CuLErrorCode getPartitionLikelihood(const int partitionId,
										double &lnL);

	// Get the site likelihood values of a partition;
	CuLErrorCode getPartitionSiteLikelihood(const int partitionId,
											double *outSiteLikelihood);

	// Set the site likelihood values of a partition;
	CuLErrorCode setPartitionSiteLikelihood(const int partitionId,
											double *inSiteLikelihood);

	// Calculate the log likelihood value of a partition from pre-specified site likelihood values;
	CuLErrorCode calculatePartitionLikelihoodFromSiteLnLSync(const int partitionId,
															double &lnL);

	// Calculate the log likelihood value of a partition from pre-specified site likelihood values;
	CuLErrorCode calculatePartitionLikelihoodFromSiteLnLAsync(const int partitionId);

	// Remove some partitions from the current instance;
	CuLErrorCode removePartition(const int nPartition,
								const int *partitionId);

	// Finalize the current instance;
	CuLErrorCode finalizeInstance(void);

private:
	int _deviceId;
	int _nPartition;
	int _nStreamInUse;
	std::vector<PartitionInstance*> _partitionVec;		// 应该用指针还是不用指针，应该差不多？
	cudaStream_t _stream[MAX_STREAM];
	std::map<int, int> _streamIndMap;
};

#endif