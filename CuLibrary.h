// The first set of APIs: regard each partition as seperate partition, can't shared parameter or memory among partitions;
#ifndef CU_LIBRARY_H
#define CU_LIBRARY_H

#include "CuLibCommon.h"
#include "CuLibInstanceImpl.h"



/*
 TODO:
   注意： 在设置PMat / state array / conditional likelihood时，nodeId对应的是实际的节点index还是分配的array的index?；
   应该是实际的node id，这样才方便追踪依赖关系；
   另外，一个问题是：tip state/condlike以及internal condlike是分开分配空间的，因此，如何建立地址与tree node label之间的映射关系，也即在计算condlike时，如果只指定待计算的node的tree label，则如何指定该node对应的internal node的condlike数组的偏移以及其两个孩子节点对应的tip state/condlike或者internal condlike数组的偏移???
   让用户再指定node label与tip state/tip condlike/internal condlike偏移之间的映射关系？？？
   这样的话，指定PMat/tip state/tip condlike/internal condlike时的nodeId[]为对应的数组的偏移还是实际节点的label？注意说明清楚；
   另外一个问题：在指定PMat / tip state / tip condlike/internal condlike时，应不应该将host端数组置为0后再设置，若设置为0后再进行copy，则会导致之前的设置完全失效； => 暂时先改为在分配空间的同时设置为0，之后设置时不清0，直接进行copy；
   另外，copy应该是一次性的copy还是多次copy?
   也即是否应该用整个host端的数组覆盖device端的数组？若device端有更新，而host端没有同步的更新，则会出错，除非每次proposal被接受时再将device端的condlike copy回host端(可以一试，因为若proposal被拒绝，可以通过将host端的数组重新copy到device端实现reset)；
   */

/*
enum CuLErrorCode
{
	CUL_SUCCESS						= 0;			// No error;
	CUL_ERROR_COMMON				= -1;			// General error;
	CUL_ERROR_BAD_ALLOC				= -2
	CUL_ERROR_OUT_OF_DEVICE_MEMORY	= -3;			// memory allocate error;
	CUL_ERROR_DEVICE_NOT_AVALAIABLE = -4;	// The device doesn't exist or is not available
	CUL_ERROR_INDEX_OUT_OF_RANGE	= -5;
};

// MAY: todo: 注意无根树的情况：root有三个child?
struct CuLTreeNode{
	int label;			// the index of the current node
	int nChild;
	CuLTreeNode *child[3];

	CuLTreeNode(int l, int n):label(l), nChild(n){
		child[0] = child[1] = child[2] = NULL;
	}
} CuLTreeNode;
*/

/*
	Create(Initialize) a CuLibrary instance, the instance can be used to manage multiple partition instances.
	Input:
		nPartition: how many partitions are managed by the current instance;
		deviceId: the device id for this instance;
	Return:
		error code;
		cuLInstanceId: the id of the newly created CuLibrary instance.
*/
CuLErrorCode CuLInitializeCuLInstance(const int nPartition, 
									const int deviceId, 
									int &cuLInstanceId);


// TODO: state freq究竟可不可能有多组；
/*
	Specify the parameters of the specified partition, a partition instance will be created for the specified partition.
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified, notice, partitionId starts from 0, that's in range: [0, nPartition - 1];
		nNode: count of nodes in the tree;
		nState: state count for the current partition, different partition can have different data type thus different state count;
		nSitePattern: site pattern count for the specified partition;
		nRateCategory: rate category count;
		nEigenDecomposition: count of eigen decomposition;
		nNodeForTransitionMatrix: count of transition matrix buffers (each of size: nTransitionMatrixPerNode * nState * nState);
		nTipStateArray: count of tip state array, each state array is of size nSitePattern;
		nTipCondlikeArray: count of tip conditional likelihood array, each conditional likelihood array of the tip node is of size nState * nSitePattern;
		nInternalNodeForCondlike: count of conditional likelihood buffers (each of size: nCondlikePerInternalNode * nSitePattern * nState);
	Return:
		CuLErrorCode;
*/
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
										const bool isRooted);


/*
	Specify the tree topology for the specified partition of the specified CuLInstance;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		root: the root of the tree for the specified partition;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionTreeTopology(const int cuLInstanceId,
											const int partitionId,
											CuLTreeNode* root);


/*
	Specify the state frequency for the specified partition of the specified CuLInstance;
	Input: 
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		inStateFreq: the input state freqency values, of size nState;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionStateFrequency(const int cuLInstanceId,
												const int partitionId,
												const double *inStateFreq);


/*
	Specify the site pattern weight for the specified partition of the specified CuLInstance;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		inPatternWeight: the input site pattern weights, should of size nSitePattern;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionSitePatternWeight(const int cuLInstanceId,
												const int partitionId, 
												const double *inPatternWeight);


/*
	Specify the rates for the specified partition of the specified CuLInstance;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		inRates: the input rates, should of size nRateCategory;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionRate(const int cuLInstanceId,
										const int partitionId, 
										const double *inRate);


/*
	Specify the rate category weights for the specified partition of the specified CuLInstance;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nCategory: how many categories of rate category weight to specify, nEigenDecomposition categories in total;
		categoryId: the category index;
		inRateCategoryWeight: the input rate category weights, nCategory in total, each should of size nRateCategory;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionRateCategoryWeight(const int cuLInstanceId,
													const int partitionId, 
													const int nCategory,
													const int *categoryId,
													const double **inRateCategoryWeight);

/*
	Specify the eigen decomposition vector and values for the specified partition of the secified CuLInstance;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nEigenDecomp: how many eigen decomposition buffers to specify;
		eigenDecompId: the eigen decomposition buffers to specify, should be of size nEigenDecomp;
		inEigenVector: the input eigen vectors, nEigenDecomp in total, each should be of size nState * nState;
		inInverEigenVector: the input inverse eigen vectors, nEigenDecomp in total, each should be of size nState * nState;
		inEigenValue: the input eigen values, nEigenDecomp in total, each should be of size nState;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionEigenDecomposition(const int cuLInstanceId,
													const int partitionId,
													const int nEigenDecomp,
													const int *eigenDecompId,
													const double **inEigenVector,
													const double **inInverEigenVector,
													const double **inEigenValue);


/*
	Specify the transition matries of the specified partition for the specified CuLInstance;
	there are nTransitionMatrixPerNode transition matries for each node, 
	you can specify k (k <= nTransitionMatrixPerNode) matries for a node at one time, 
	thus each of the input matries should be of size nState * nState.
	Or you can set all matries of a node at one time using the the API - CuLSpecifyPartitionTransitionMatrixMulti(),
	in which case, the each of the input matries should be of size nTransitionMatrixPerNode * nState * nState;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: how many nodes' matries to specify;
		nodeId: the node index;
		nEigenDecomp: how many eigen decomposition categories of transition matrices to specify;
		eigenDecompId: the index of eigen decomposition categories;
		inMatrix: the input matrix values, nMatrix in total, each of size nState * nState;
	Return:
	CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionTransitionMatrixMulti(const int cuLInstanceId,
														const int partitionId,
														const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														const double **inMatrix);


/*
	Specify the transition matries of the specified partition for the specified CuLInstance;
	there are nTransitionMatrixPerNode transition matries for each node.
	You can set all matries of a node at one time using this API,
	just make sure each of the input matries should be of size nTransitionMatrixPerNode * nState * nState;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of the input matrices;
		nodeId: the node indexs of the input transition matrices;
		inMatrix: the input transition matrices, nNode in total, each of size nTransitionMatrixPerNode * nState * nState;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionTransitionMatrixAll(const int cuLInstanceId,
													const int partitionId,
													const int nNode,
													const int *nodeId,
													const double **inMatrix);


/*
	Calculate transition matrices of the specified partition of the specified CuLInstance;
	For each node, there are nTransitionMatrixPerNode transition matrices (each of size nState * nState), 
	this function will calculate partial of the nTransitionMatrixPerNode transition matrices you specify in param nodeId and categoryId;
	If you want to calculate all transition matrices of some nodes, use function CuLCalculatePartitionTransitionMatrixMulti() instead;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of nodes;
		nodeId: the index of the nodes to calculate transition matrices;
		nEigenDecomp: how many eigen decomposition categories of transition matrices to calculate;
		eigenDecompId: the index of eigen decomposition categories;
		brLen: the branch lengths;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLCalculatePartitionTransitionMatrixMulti(const int cuLInstanceId,
														const int partitionId,
														const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														const double *brLen);


/*
	Calculate transition matrices of the specified partition of the specified CuLInstance;
	For each node, there are nTransitionMatrixPerNode transition matrices (each of size nState * nState), 
	this function will calculate all of the nTransitionMatrixPerNode transition matrices you specify in param nodeId;
	If you want to calculate partial of the nTransitionMatrixPerNode transition matrices of some nodes, use function CuLCalculatePartitionTransitionMatrixSingle() instead;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of the transition matrices to calculate;
		nodeId: the indexs of the nodes to calculate transition matrices;
		brLen: the branch lengths;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLCalculatePartitionTransitionMatrixAll(const int cuLInstanceId,
														const int partitionId,
														const int nNode,
														const int *nodeId,
														const double *brLen);


/*
	Get the transition matrices of the specified partition of the specified CuLInstance;
	For each node, there are nTransitionMatrixPerNode matrices, and
	this function will return partial of the nTransitionMatrixPerNode matrices of the specified node according to nodeId and matricId;
	If you want to get all nTransitionMatrixPerNode matrices of some nodes, use CuLGetPartitionTransitionMatrixMulti() instead;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of nodes;
		nodeId: node indexs for which to get transition matrices;
		nEigenDecomp: how many eigen decomposition categories to get;
		eigenDecompId: the eigen decomposition category index;
		outMatrix: the output transition matrices, nMatrix in total, each of size nState * nState;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLGetPartitionTransitionMatrixMulti(const int cuLInstanceId,
													const int partitionId,
													const int nNode,
													const int *nodeId,
													const int nEigenDecomp,
													const int *eigenDecompId,
													double **outMatrix);


/*
	Get the transition matrices of the specified partition of the specified CuLInstance;
	For each node, there are nTransitionMatrixPerNode matrices, and
	this function will return all the nTransitionMatrixPerNode matrices of the specified node according to param nodeId;
	If you want to get partial of the nTransitionMatrixPerNode matrices of some nodes, use CuLGetPartitionTransitionMatrixSingle() instead;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of node's matrices to output;
		nodeId: node indexs for which to get transition matrices;
		outMatrix: the output transition matrices, nNode in total, each of size nTransitionMatrixPerNode * nState * nState;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLGetPartitionTransitionMatrixAll(const int cuLInstanceId,
												const int partitionId,
												const int nNode,
												const int *nodeId,
												double **outMatrix);


/*
	Specify the tip states for the specified partition of the specified CuLInstance.
	There are nTipStateArray in total, each of size nSitePattern;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nTipNode: count of tip nodes to specify state arrays;
		tipNodeId: indexs of the tip nodes;
		inTipState: input state array of the tip nodes, nTipNode in total, each of size nSitePattern;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionTipState(const int cuLInstanceId,
										const int partitionId,
										const int nTipNode,
										const int *tipNodeId,
										const int **inTipState);


/*
	Specify the conditional likelihood values of the specified partition of the specified CuLInstance;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nTipNode: count of tip nodes to specify conditional likelihoods;
		tipNodeId: indexs of the nodes to specify conditional likelihoods;
		inTipCondlike: input conditional likelihoods, nTipNode in total, each of size nSitePattern * nState;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionTipCondlike(const int cuLInstanceId,
											const int partitionId,
											const int nTipNode,
											const int *tipNodeId,
											const double **inTipCondlike);


/*
	Specify the conditional likelihood values of the internal nodes of the specified partition of the specified CuLInstance;
	For each internal node, there are nCondlikePerInternalNode conditional likelihood arrays (each of size nSitePattern * nState), and 
	this function will specify partial of the nCondlikePerInternalNode arrays for the specified nodes;
	If you want to specify all nCondlikePerInternalNode arrays at one time, use function CuLSpecifyPartitionInternalCondlikeMulti() instead;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of nodes;
		nodeId: the indexs of nodes to specify conditional likelihood values;
		nEigenDecomp: how many eigen decomposition categories to specify;
		eigenDecompId: the index of eigen decomposition categories;
		inCondlike: the input conditional likelihood array, nArray in total, each of size nSitePattern * nState;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionInternalCondlikeMulti(const int cuLInstanceId,
														const int partitionId,
														const int nNode,
														const int *nodeId,
														const int nEigenDecomp,
														const int *eigenDecompId,
														const double **inCondlike);


/*
	Specify the conditional likelihood values of the internal nodes of the specified partition of the specified CuLInstance;
	For each internal node, there are nCondlikePerInternalNode conditional likelihood arrays (each of size nSitePattern * nState), and 
	this function will specify all the nCondlikePerInternalNode arrays for the specified nodes at one time;
	If you want to specify partial of the nCondlikePerInternalNode arrays at one time, use function CuLSpecifyPartitionInternalCondlikeSingle() instead;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of conditional likelihood arrays to specify;
		nodeId: the indexs of nodes to specify conditional likelihood values;
		inCondlike: the input conditional likelihood array, nNode in total, each of size nCondlikePerInternalNode * nSitePattern * nState;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionInternalCondlikeAll(const int cuLInstanceId,
													const int partitionId,
													const int nNode,
													const int *nodeId,
													const double **inCondlike);



/*
	Map the node label to the index of condlike array;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of nodes in the tree;
		indMap: the map of node label to condlike array index;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLMapPartitionNodeIndToArrayInd(const int cuLInstanceId,
											const int partitionId,
											const int nNode,
											const int *indMap);


/*
	Specify which nodes need scaling;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNodeScaler: how many nodes need scaling;
		nodeId: label of nodes which need scaling;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSpecifyPartitionNodeScalerIndex(const int cuLInstanceId,
												const int partitionId,
												const int nNodeScaler,
												const int *nodeId);


/*
	Calculate conditional likelihood of the specified node;
	For each internal node, there are nCondlikePerInternalNode conditional likelihood arrays in total, each of size nSitePattern * nState, and
	this function will calculate partial of the nCondlikePerInternalNode conditional likelihood arrays of the specified node;
	If you want to calculate all nCondlikePerInternalNode conditional likelihood arrays, use function CuLCalculatePartitionCondlikeMulti() instead;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of nodes;
		nodeId: which node to calculate, you can specify all the nodes which need updating at one time, the library will trace the dependency;
		nEigenDecomp: how many eigen decomposition categories to calculate;
		eigenDecompId: the index of eigen decomposition categories;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLCalculatePartitionCondlikeMulti(const int cuLInstanceId,
												const int partitionId,
												const int nNode,
												const int *nodeId,
												const int nEigenDecomp,
												const int *eigenDecompId);


/*
	Calculate conditional likelihood of the specified node;
	For each internal node, there are nCondlikePerInternalNode conditional likelihood arrays in total, each of size nSitePattern * nState, and
	this function will calculate all the nCondlikePerInternalNode conditional likelihood arrays of the specified node;
	If you want to calculate partial of the nCondlikePerInternalNode conditional likelihood arrays, use function CuLCalculatePartitionCondlikeSingle() instead;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of conditional likelihood arrays to calculate;
		nodeId: which node to calculate, you can specify all the nodes which need updating at one time, the library will trace the dependency;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLCalculatePartitionCondlikeAll(const int cuLInstanceId,
												const int partitionId,
												const int nNode,
												const int *nodeId);


/*
	Get the conditional likelihood values of the specified nodes;
	For each node, there are nCondlikePerInternalNode conditional likelihood arrays, each of size nSitePattern * nState, and 
	this function will return partial of the nCondlikePerInternalNode arrays of the specified nodes;
	If you want to get all the nCondlikePerInternalNode arrays at one time, use function CuLGetPartitionCondlikeMulti() instead;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of nodes;
		nodeId: indexs of the nodes to get conditional likelihood arrays;
		nEigenDecomp: how many eigen decomposition categories;
		eigenDecompId: the index of eigen decomposition categories;
		outCondlike: the output conditional likelihood array, nArray in total, each of size nSitePattern * nState;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLGetPartitionIntCondlikeMulti(const int cuLInstanceId,
											const int partitionId,
											const int nNode,
											const int *nodeId,
											const int nEigenDecomp,
											const int *eigenDecompId,
											double **outCondlike);



/*
	Get the conditional likelihood values of the specified nodes;
	For each node, there are nCondlikePerInternalNode conditional likelihood arrays, each of size nSitePattern * nState, and 
	this function will return all the nCondlikePerInternalNode arrays of the specified nodes at one time;
	If you want to get partial of the nCondlikePerInternalNode arrays, use function CuLGetPartitionCondlikeSingle() instead;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		nNode: count of nodes' conditional likelihood arrays to get;
		nodeId: indexs of the nodes to get conditional likelihood arrays;
		outCondlike: the output conditional likelihood array, nArray in total, each of size nSitePattern * nState;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLGetPartitionIntCondlikeAll(const int cuLInstanceId,
										const int partitionId,
										const int nNode,
										const int *nodeId,
										double **outCondlike);


/*
	Calculate the log likelihood of the specified partition;
	This function will synchronize until the final log likelihood value is got, 
	if you want to return right away after the call, use function  CuLCalculatePartitionLikelihoodAsync() instead;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		lnL: the output log likelihood of the specified partition;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLCalculatePartitionLikelihoodSync(const int cuLInstanceId,
												const int partitionId,
												double &lnL);



/*
	Calculate the log likelihood of the specified partition;
	This function will return right away before the log likelihood value is got, 
	later, you can call CuLGetPartitionLikelihood() to get the log likelihood value of the specified partition.
	The synchronize version of this function is CuLCalculatePartitionLikelihoodAsync().
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLCalculatePartitionLikelihoodAsync(const int cuLInstanceId,
												const int partitionId);


/*
	Get the log likelihood value of the specified partition, this function should be called after
	the CuLCalculatePartitionLikelihoodAsync() is called;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		lnL: the calculated log likelihood value of the specified partition;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLGetPartitionLikelihood(const int cuLInstanceId,
										const int partitionId,
										double &lnL);



/*
	Get the site likelihoods of the specified partition;
	This function can be called alone or after function CuLCalculatePartitionLikelihoodSync() or CuLCalculatePartitionLikelihoodAsync() is called;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		outSiteLikelihood: the output site likelihood values, of size nSitePattern;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLGetPartitionSiteLikelihood(const int cuLInstanceId,
											const int partitionId,
											double *outSiteLikelihood);



/*
	Set the site likelihoods of the specified partition;
	This function can be called alone or after function CuLCalculatePartitionLikelihoodSync() or CuLCalculatePartitionLikelihoodAsync() is called;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		inSiteLikelihood: the input site likelihood values, of size nSitePattern;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLSetPartitionSiteLikelihood(const int cuLInstanceId,
											const int partitionId,
											double *inSiteLikelihood);


/*
	Calculate likelihood value from pre-specified site likelihood values;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
		lnL: the output likelihood value of the specified partition;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLCalculatePartitionLikelihoodFromSiteLnLSync(const int cuLInstanceId,
															const int partitionId,
															double &lnL);



/*
	Calculate likelihood value from pre-specified site likelihood values;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		partitionId: the partition id which is specified;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLCalculatePartitionLikelihoodFromSiteLnLAsync(const int cuLInstanceId,
															const int partitionId);



/*
	Remove some partition from the specified CuLInstance;
	Input:
		cuLInstanceId: CuLInstance which manages the specified partition;
		nPartition: count of partitions to be removed from the instance;
		partitionId: which partitions of the specified CuLInstance to remove;
*/
CuLErrorCode CuLRemovePartition(const int cuLInstanceId,
								const int nPartition,
								const int *partitionId);


/*
	Finalize the specified CuLInstance;
	The memory allocated will be freed;
	Input:
		cuLInstanceId: which CuLInstance to finalize;
	Return:
		CuLErrorCode;
*/
CuLErrorCode CuLFinalizeInstance(const int cuLInstanceId);

#endif