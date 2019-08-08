#include "CuLibrary.h"
#include "CuLibKernel-codemlAndMrBayes-rooted.h"
#include <set>
#include <queue>


#define EPS 0.00001

bool transposePMat;
bool doScale, isRooted;

//#define DEBUG_CONDLIKE
//#define DEBUG

/*
const int maxStreamCount = 10;
cudaStream_t stream[maxStreamCount];
*/

double *inRate, *inStateFreq, **inRateCatWeight, *inSitePatternWeight, *inBrLen;
double **inTipCondlike, **inIntCondlike, **inU, **inV, **inR, **inPMat;
double *outRate, *outStateFreq, **outRateCatWeight, *outSitePatternWeight;
double **outTipCondlike, **outU, **outV, **outR;
double *scaleFactor;
int **inTipState;
int **outTipState;
int *nodeIdToArrayId;
int *nodeScalerInd;
int *scalerIndMap;
double **outPMat, **outIntCondlike, *outSiteLnL, *inSiteLnL;
int *tipStateNodeId, *tipCondlikeNodeId, *intCondlikeNodeId;
double outLnL;


//int stateCountArray[] = {4, 20, 61};
//int stateCountArray[] = {2, 4, 10, 20, 40, 61};
//int stateCountArray[] = {2, 5, 7, 10, 13, 15, 16, 24, 28, 30, 32, 40, 45, 56};
//int stateCountArray[] = {5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 19, 22, 24, 27, 30, 32, 35, 38, 41, 44, 47, 50, 53, 56, 67, 70, 73, 76, 79, 82};
int stateCountArray[] = {25, 31, 37, 43, 49, 55, 67, 73, 79};
//int eigenDecompCountArray[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
int eigenDecompCountArray[] = {1};
//int eigenDecompCountArray[] = {2, 2, 2, 2, 2};
//int eigenDecompCountArray[] = {3, 3, 3, 3, 3};
//int rateCatCountArray[] = {1, 1, 1, 1, 1};
int rateCatCountArray[] = {1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48};
//int rateCatCountArray[] = {1, 1, 1, 2, 2, 2, 4, 4, 4};
//int rateCatCountArray[] = {4, 4, 4, 4, 4};
//int rateCatCountArray[] = {8, 8, 8, 8, 8};


int sitePatternCountArray[] = {500, 1000, 3000, 5000, 8000, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 15000};
//int sitePatternCountArray[] = {8000, 10000, 15000, 20000, 25000, 30000};
//int sitePatternCountArray[] = {35000, 40000, 45000, 50000};
int tipStateCountArray[] = {3, 4, 6};
int tipCondlikeCountArray[] = {6, 5, 3};

int nPartition, deviceId;
int nNode, nTipState, nTipCondlike, nIntCondlike, nState, nEigenDecomp, nRateCategory, nSitePattern;
int nNodeScaler;


// Allocate memory for host memory:
void allocateMemory()
{
	const int nNode = nTipState + nTipCondlike + nIntCondlike;
	const int nArrayPerNode = nEigenDecomp * nRateCategory;
	const int condlikeSize = nSitePattern * nState;
	const int PMat_size = nState * nState;
	const int PMat_node_size = nArrayPerNode * PMat_size;

	int nMaxNodeScaler = max(nNode, nNodeScaler);

	nodeScalerInd = (int*) calloc(nMaxNodeScaler, sizeof(int));
	scalerIndMap = (int*) calloc(nMaxNodeScaler, sizeof(int));
	scaleFactor = (double*) calloc(nMaxNodeScaler * nSitePattern, sizeof(double));

	nodeIdToArrayId = (int*) calloc(nNode, sizeof(int));

	inRate = (double *) calloc(nRateCategory, sizeof(double));
	if(NULL == inRate)
		printError("Error in allocating memory for inRate");

	outRate = (double *) calloc(nRateCategory, sizeof(double));
	if(NULL == outRate)
		printError("Error in allocating memory for outRate");

	inStateFreq = (double *) calloc(nState, sizeof(double));
	if(NULL == inStateFreq)
		printError("Error in allocating memory for inStateFreq");

	outStateFreq = (double *) calloc(nState, sizeof(double));
	if(NULL == outStateFreq)
		printError("Error in allocating memory for outStateFreq");

	inRateCatWeight = (double **) calloc(nEigenDecomp, sizeof(double*));
	if(NULL == inRateCatWeight)
		printError("Error in allocating memory for inRateCatWeight");
	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		inRateCatWeight[iEigen] = (double *) calloc(nRateCategory, sizeof(double));
		if(NULL == inRateCatWeight[iEigen])
			printError("Error in allocating memory for inRateCatWeight");
	}

	outRateCatWeight = (double **) calloc(nEigenDecomp, sizeof(double*));
	if(NULL == outRateCatWeight)
		printError("Error in allocating memory for outRateCatWeight");
	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		outRateCatWeight[iEigen] = (double *) calloc(nRateCategory, sizeof(double));
		if(NULL == outRateCatWeight[iEigen])
			printError("Error in allocating memory for outRateCatWeight");
	}

	inSitePatternWeight = (double *) calloc(nSitePattern, sizeof(double));
	if(NULL == inSitePatternWeight)
		printError("Error in allocating memory for inSitePatternWeight");

	outSitePatternWeight = (double *) calloc(nSitePattern, sizeof(double));
	if(NULL == outSitePatternWeight)
		printError("Error in allocating memory for outSitePatternWeight");

	inBrLen = (double *) calloc(nNode, sizeof(double));
	if(NULL == inBrLen)
		printError("Error in allocating memory for inBrLen");

	inTipState = (int **) calloc(nTipState, sizeof(int *));
	if(NULL == inTipState)
		printError("Error in allocating memory for inTipState");
	for(int iTip = 0; iTip < nTipState; iTip ++){
		inTipState[iTip] = (int *) calloc(nSitePattern, sizeof(int));
		if(NULL == inTipState[iTip])
			printError("Error in allocating memory for inTipState");
	}

	outTipState = (int **) calloc(nTipState, sizeof(int *));
	if(NULL == outTipState)
		printError("Error in allocating memory for outTipState");
	for(int iTip = 0; iTip < nTipState; iTip ++){
		outTipState[iTip] = (int *) calloc(nSitePattern, sizeof(int));
		if(NULL == outTipState[iTip])
			printError("Error in allocating memory for outTipState");
	}

	inTipCondlike = (double **) calloc(nTipCondlike, sizeof(double *));
	if(NULL == inTipCondlike)
		printError("Error in allocating memory for inTipCondlike");
	for(int iTip = 0; iTip < nTipCondlike; iTip ++){
		inTipCondlike[iTip] = (double *) calloc(condlikeSize, sizeof(double));
		if(NULL == inTipCondlike[iTip])
			printError("Error in allocating memory for inTipCondlike");
	}

	outTipCondlike = (double **) calloc(nTipCondlike, sizeof(double *));
	if(NULL == outTipCondlike)
		printError("Error in allocating memory for outTipCondlike");
	for(int iTip = 0; iTip < nTipCondlike; iTip ++){
		outTipCondlike[iTip] = (double *) calloc(condlikeSize, sizeof(double));
		if(NULL == outTipCondlike[iTip])
			printError("Error in allocating memory for outTipCondlike");
	}

	inIntCondlike = (double **) calloc(nIntCondlike, sizeof(double *));
	if(NULL == inIntCondlike)
		printError("Error in allocating memory for inIntCondlike");
	for(int iInt = 0; iInt < nIntCondlike; iInt ++){
		inIntCondlike[iInt] = (double *) calloc(nArrayPerNode * condlikeSize, sizeof(double));
		if(NULL == inIntCondlike[iInt])
			printError("Error in allocating memory for inIntCondlike");
	}

	outIntCondlike = (double **) calloc(nIntCondlike, sizeof(double *));
	if(NULL == outIntCondlike)
		printError("Error in allocating memory for outIntCondlike");
	for(int iInt = 0; iInt < nIntCondlike; iInt ++){
		outIntCondlike[iInt] = (double *) calloc(nArrayPerNode * condlikeSize, sizeof(double));
		if(NULL == outIntCondlike[iInt])
			printError("Error in allocating memory for outIntCondlike");
	}

	inU = (double **) calloc(nEigenDecomp, sizeof(double *));
	if(NULL == inU)
		printError("Error in allocating memory for inU");
	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		inU[iEigen] = (double *) calloc(PMat_size, sizeof(double));
		if(NULL == inU[iEigen])
			printError("Error in allocating memory for inU");
	}

	outU = (double **) calloc(nEigenDecomp, sizeof(double *));
	if(NULL == outU)
		printError("Error in allocating memory for outU");
	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		outU[iEigen] = (double *) calloc(PMat_size, sizeof(double));
		if(NULL == outU[iEigen])
			printError("Error in allocating memory for outU");
	}

	inV = (double **) calloc(nEigenDecomp, sizeof(double *));
	if(NULL == inV)
		printError("Error in allocating memory for inV");
	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		inV[iEigen] = (double *) calloc(PMat_size, sizeof(double));
		if(NULL == inV[iEigen])
			printError("Error in allocating memory for inV");
	}

	outV = (double **) calloc(nEigenDecomp, sizeof(double *));
	if(NULL == outV)
		printError("Error in allocating memory for outV");
	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		outV[iEigen] = (double *) calloc(PMat_size, sizeof(double));
		if(NULL == outV[iEigen])
			printError("Error in allocating memory for outV");
	}

	inR = (double **) calloc(nEigenDecomp, sizeof(double *));
	if(NULL == inR)
		printError("Error in allocating memory for inR");
	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		inR[iEigen] = (double *) calloc(nState, sizeof(double));
		if(NULL == inR[iEigen])
			printError("Error in allocating memory for inR");
	}

	outR = (double **) calloc(nEigenDecomp, sizeof(double *));
	if(NULL == outR)
		printError("Error in allocating memory for inR");
	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		outR[iEigen] = (double *) calloc(nState, sizeof(double));
		if(NULL == outR[iEigen])
			printError("Error in allocating memory for outR");
	}

	inPMat = (double **) calloc(nNode, sizeof(double *));
	if(NULL == inPMat)
		printError("Error in allocating memory for inPMat");
	for(int iNode = 0; iNode < nNode; iNode ++){
		inPMat[iNode] = (double *) calloc(PMat_node_size, sizeof(double));
		if(NULL == inPMat[iNode])
			printError("Error in allocating memory for inPMat");
	}

	outPMat = (double **) calloc(nNode, sizeof(double *));
	if(NULL == outPMat)
		printError("Error in allocating memory for outPMat");
	for(int iNode = 0; iNode < nNode; iNode ++){
		outPMat[iNode] = (double *) calloc(PMat_node_size, sizeof(double));
		if(NULL == outPMat[iNode])
			printError("Error in allocating memory for outPMat");
	}

	tipStateNodeId = (int *) calloc(nTipState, sizeof(int));
	if(NULL == tipStateNodeId)
		printError("Error in allocating memory for tipStateNodeId");

	tipCondlikeNodeId = (int *) calloc(nTipCondlike, sizeof(int));
	if(NULL == tipCondlikeNodeId)
		printError("Error in allocating memory for tipCondlikeNodeId");

	intCondlikeNodeId = (int *) calloc(nIntCondlike, sizeof(int));
	if(NULL == intCondlikeNodeId)
		printError("Error in allocating memory for intCondlikeNodeId");

	inSiteLnL = (double *) calloc(nSitePattern, sizeof(double));
	if(NULL == inSiteLnL)
		printError("Error in allocating memory for inSiteLnL");

	outSiteLnL = (double *) calloc(nSitePattern, sizeof(double));
	if(NULL == outSiteLnL)
		printError("Error in allocating memory for outSiteLnL");
}


void freeMemory()
{
	free(nodeScalerInd);
	nodeScalerInd = NULL;

	free(scalerIndMap);
	scalerIndMap = NULL;

	free(scaleFactor);
	scaleFactor = NULL;

	free(nodeIdToArrayId);
	nodeIdToArrayId = NULL;

	free(inRate);
	inRate = NULL;

	free(outRate);
	outRate = NULL;

	free(inStateFreq);
	inStateFreq = NULL;

	free(outStateFreq);
	outStateFreq = NULL;

	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		free(inRateCatWeight[iEigen]);
		inRateCatWeight[iEigen] = NULL;
	}
	free(inRateCatWeight);
	inRateCatWeight = NULL;

	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		free(outRateCatWeight[iEigen]);
		outRateCatWeight[iEigen] = NULL;
	}
	free(outRateCatWeight);
	outRateCatWeight = NULL;
	
	free(inSitePatternWeight);
	inSitePatternWeight = NULL;

	free(outSitePatternWeight);
	outSitePatternWeight = NULL;

	free(inBrLen);
	inBrLen = NULL;

	for(int iTip = 0; iTip < nTipCondlike; iTip ++){
		free(inTipCondlike[iTip]);
		inTipCondlike[iTip] = NULL;
	}
	free(inTipCondlike);
	inTipCondlike = NULL;

	for(int iTip = 0; iTip < nTipCondlike; iTip ++){
		free(outTipCondlike[iTip]);
		outTipCondlike[iTip] = NULL;
	}
	free(outTipCondlike);
	outTipCondlike = NULL;
	
	for(int iNode = 0; iNode < nIntCondlike; iNode ++){
		free(inIntCondlike[iNode]);
		inIntCondlike[iNode] = NULL;
	}
	free(inIntCondlike);
	inIntCondlike = NULL;

	for(int iNode = 0; iNode < nIntCondlike; iNode ++){
		free(outIntCondlike[iNode]);
		outIntCondlike[iNode] = NULL;
	}
	free(outIntCondlike);
	outIntCondlike = NULL;

	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		free(inU[iEigen]);
		inU[iEigen] = NULL;
	}
	free(inU);
	inU = NULL;

	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		free(outU[iEigen]);
		outU[iEigen] = NULL;
	}
	free(outU);
	outU = NULL;

	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		free(inV[iEigen]);
		inV[iEigen] = NULL;
	}
	free(inV);
	inV = NULL;

	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		free(outV[iEigen]);
		outV[iEigen] = NULL;
	}
	free(outV);
	outV = NULL;

	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		free(inR[iEigen]);
		inR[iEigen] = NULL;
	}
	free(inR);
	inR = NULL;

	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		free(outR[iEigen]);
		outR[iEigen] = NULL;
	}
	free(outR);
	outR = NULL;

	for(int iTip = 0; iTip < nTipState; iTip ++){
		free(inTipState[iTip]);
		inTipState[iTip] = NULL;
	}
	free(inTipState);
	inTipState = NULL;

	for(int iTip = 0; iTip < nTipState; iTip ++){
		free(outTipState[iTip]);
		outTipState[iTip] = NULL;
	}
	free(outTipState);
	outTipState = NULL;

	for(int iNode = 0; iNode < nNode; iNode ++){
		free(inPMat[iNode]);
		inPMat[iNode] = NULL;
	}
	free(inPMat);
	inPMat = NULL;

	for(int iNode = 0; iNode < nNode; iNode ++){
		free(outPMat[iNode]);
		outPMat[iNode] = NULL;
	}
	free(outPMat);
	outPMat = NULL;

	free(tipStateNodeId);
	tipStateNodeId = NULL;

	free(tipCondlikeNodeId);
	tipCondlikeNodeId = NULL;

	free(intCondlikeNodeId);
	intCondlikeNodeId = NULL;

	free(inSiteLnL);
	inSiteLnL = NULL;

	free(outSiteLnL);
	outSiteLnL = NULL;
}


void initValues()
{
	srand(int(time(0)));

	for(int iState = 0; iState < nState; iState ++)
		inStateFreq[iState] = (double)(rand() % nState + 1) / nState;

	for(int iRate = 0; iRate < nRateCategory; iRate ++)
		inRate[iRate] = (double)(rand() % 10 + 1) / (100.0);

	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++)
		inSitePatternWeight[iPattern] = (double)(rand() % 5 + 1);

	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		for(int iRateCat = 0; iRateCat < nRateCategory; iRateCat ++)
			inRateCatWeight[iEigen][iRateCat] = (double)(rand() % 10 + 1) / 11.0;
	}

	const int nNode = nTipState + nTipCondlike + nIntCondlike;
	for(int iNode = 0; iNode < nNode; iNode ++){
		inBrLen[iNode] = (double)(rand() % 100 + 1) / 1001.0;
		//inBrLen[iNode] = 0.0f;
	}

	for(int iTip = 0; iTip < nTipState; iTip ++){
		for(int iPattern = 0; iPattern < nSitePattern; iPattern ++)
			inTipState[iTip][iPattern] = (int)(rand() % nState);
	}

	for(int iTip = 0; iTip < nTipCondlike; iTip ++){
		for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
			for(int iState = 0; iState < nState; iState ++)
				inTipCondlike[iTip][iPattern * nState + iState] = (double)(rand() % 100 + 1) / (100 * nState);
		}
	}

	/*
	const int condlike_size = nSitePattern * nState;
	int offset;
	for(int iInt = 0; iInt < nIntCondlike; iInt ++){
		offset = 0;
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			for(int iRateCat = 0; iRateCat < nRateCategory; iRateCat ++, offset += condlike_size){
				for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
					for(int iState = 0; iState < nState; iState ++)
						inIntCondlike[iInt][offset + iPattern * nState + iState] = (double)(rand() % 10 + 1) / 101.0;
				}
			}
		}
	}
	*/
	

	const int PMat_size = nState * nState;
	for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
		for(int iElem = 0; iElem < nState; iElem ++){
			inR[iEigen][iElem] = (double)(rand() % 10 + 1) / 100.0;
			//inR[iEigen][iElem] = 0.0f;
		}

		for(int iElem = 0; iElem < PMat_size; iElem ++){
			inU[iEigen][iElem] = (double)(rand() % 10 + 1) / 101.0;
			inV[iEigen][iElem] = (double)(rand() % 10 + 1) / 101.0;
			//inU[iEigen][iElem] = 1.0;
			//inV[iEigen][iElem] = 1.0;
			//inU[iEigen][iElem] = double(iElem % nState + 1) * (iEigen + 1);
			//inV[iEigen][iElem] = double(iElem % nState + 1) * (iEigen + 1);
		}
	}

	// Initialize PMat:
	/*
	double *pPMat;
	for(int iNode = 0; iNode < nNode; iNode ++){
		pPMat = inPMat[iNode];
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			for(int iRateCat = 0; iRateCat < nRateCategory; iRateCat ++, pPMat += PMat_size){
				for(int iElem = 0; iElem < PMat_size; iElem ++){
					pPMat[iElem] = double(rand() % 10 + 1) / 101.0;
				}
			}
		}
	}
	*/
}


void initSiteLnL()
{
	srand(int(time(0)));
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		inSiteLnL[iPattern] = double(rand() % 1000 + 1) / 1000.0;
	}
}


// Tree 1: the leaf node is only at the bottom layer, and each node has two children;
CuLTreeNode* buildTree1(int nNode)
{
	std::vector<CuLTreeNode*> nodeVec;
	// for leaf nodes:
	for(int iNode = 0; iNode < nNode / 2 + 1; iNode ++){
		CuLTreeNode *curNode = new CuLTreeNode(iNode, 0);
		for(int iChild = 0; iChild < 3; iChild ++)
			curNode->child[iChild] = NULL;
		nodeVec.push_back(curNode);
	}

	// for internal nodes:
	CuLTreeNode *root = NULL;
	int leafThreshold = nNode / 2 + 1;
	for(int iNode = nNode / 2 + 1; iNode < nNode; iNode ++){
		CuLTreeNode *curNode = new CuLTreeNode(iNode, 2);
		int leftChildLabel = (iNode - leafThreshold) * 2;
		curNode->child[0] = nodeVec[leftChildLabel];
		curNode->child[1] = nodeVec[leftChildLabel + 1];
		curNode->child[2] = NULL;

		root = curNode;
		nodeVec.push_back(curNode);
	}

#ifdef DEBUG
	printf("nodeIdToArrayId[] is:\n");
#endif
	int cntIntNode = 0;
	for(int iNode = 0; iNode < nNode; iNode ++){
		nodeIdToArrayId[iNode] = iNode;

		if(iNode >= leafThreshold){
			intCondlikeNodeId[cntIntNode] = iNode;
			if(cntIntNode < nNodeScaler){
				nodeScalerInd[cntIntNode] = iNode;
				scalerIndMap[iNode] = cntIntNode;
			}
			cntIntNode ++;
		}

#ifdef DEBUG
		printf("%d: %d\n", iNode, nodeIdToArrayId[iNode]);
#endif
	}

	assert(cntIntNode == nIntCondlike);
#ifdef DEBUG
	printf("\n\n");
#endif

	return root;
}


CuLTreeNode* buildTree1_unrooted(int nNode)
{
	std::vector<CuLTreeNode*> nodeVec;
	// for leaf nodes:
	for(int iNode = 0; iNode < nNode; iNode ++){
		CuLTreeNode *curNode = new CuLTreeNode(iNode, 0);
		for(int iChild = 0; iChild < 3; iChild ++)
			curNode->child[iChild] = NULL;
		nodeVec.push_back(curNode);
	}

	CuLTreeNode *root = NULL;
	int nTip = nNode / 2 + 1;
	std::queue<CuLTreeNode *> nodeQue;
	for(int iNode = 0; iNode < nTip - 1; iNode ++)
		nodeQue.push(nodeVec[iNode]);

	int ind = nTip;
	while(ind < nNode){
		CuLTreeNode *curNode = nodeVec[ind];
		curNode->nChild = 2;
		curNode->child[0] = nodeQue.front();
		nodeQue.pop();
		curNode->child[1] = nodeQue.front();
		nodeQue.pop();

		nodeQue.push(curNode);
		root = curNode;
		ind ++;
	}

	root->nChild = 3;
	root->child[2] = nodeVec[nTip - 1];
	
#ifdef DEBUG
	printf("nodeIdToArrayId[] is:\n");
#endif
	int cntIntNode = 0;
	for(int iNode = 0; iNode < nNode; iNode ++){
		nodeIdToArrayId[iNode] = iNode;

		if(iNode >= nTip){
			intCondlikeNodeId[cntIntNode] = iNode;
			if(cntIntNode < nNodeScaler){
				nodeScalerInd[cntIntNode] = iNode;
				scalerIndMap[iNode] = cntIntNode;
			}
			cntIntNode ++;
		}

#ifdef DEBUG
		printf("%d: %d\n", iNode, nodeIdToArrayId[iNode]);
#endif
	}

	assert(cntIntNode == nIntCondlike);
#ifdef DEBUG
	printf("\n\n");
#endif

	return root;
}


// Tree 2: 
// each node has two children and one of the children is a leaf node;
CuLTreeNode* buildTree2(int nNode)
{
	std::vector<CuLTreeNode *> nodeVec;
	CuLTreeNode *root = NULL;
	for(int iNode = 0; iNode < nNode; iNode ++){
		CuLTreeNode *curNode = new CuLTreeNode(iNode, 0);
		if(iNode > 0 && iNode % 2 == 0){			// Internal node;
			curNode->nChild = 2;
			curNode->child[0] = nodeVec[iNode-2];
			curNode->child[1] = nodeVec[iNode-1];
		}
		else{
			curNode->nChild = 0;
			curNode->child[0] = curNode->child[1] = NULL;
		}
		curNode->child[2] = NULL;

		nodeVec.push_back(curNode);
		root = curNode;
	}

#ifdef DEBUG_CONDLIKE
	printf("nodeIdToArrayId is:\n");
#endif
	int cntTipNode = 0, cntIntNode = 0;
	int nTipNode = nNode / 2 + 1;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(iNode == 0 || iNode % 2 == 1){
			nodeIdToArrayId[iNode] = cntTipNode;
			cntTipNode ++;
		}
		else{
			nodeIdToArrayId[iNode] = nTipNode + cntIntNode;
			intCondlikeNodeId[cntIntNode] = iNode;

			if(cntIntNode < nNodeScaler){
				nodeScalerInd[cntIntNode] = iNode;
				scalerIndMap[iNode] = cntIntNode;
			}
			cntIntNode ++;
		}
#ifdef DEBUG_CONDLIKE
		printf("%d: %d\n", iNode, nodeIdToArrayId[iNode]);
#endif
	}
#ifdef DEBUG_CONDLIKE
	printf("\n\n");
#endif

	assert(cntTipNode == nTipNode && cntTipNode + cntIntNode == nNode);

	return root;
}


CuLTreeNode* buildTree2_unrooted(int nNode)
{
	std::vector<CuLTreeNode *> nodeVec;
	CuLTreeNode *root = NULL;
	for(int iNode = 0; iNode < nNode; iNode ++){
		CuLTreeNode *curNode = new CuLTreeNode(iNode, 0);
		if(iNode > 0 && iNode % 2 == 0){			// Internal node;
			curNode->nChild = 2;
			curNode->child[0] = nodeVec[iNode-2];
			curNode->child[1] = nodeVec[iNode-1];
		}
		else{
			curNode->nChild = 0;
			curNode->child[0] = curNode->child[1] = NULL;
		}
		curNode->child[2] = NULL;

		nodeVec.push_back(curNode);
		//root = curNode;
	}

	root = nodeVec[nNode - 2];
	root->nChild = 3;
	root->child[2] = nodeVec[nNode - 1];

#ifdef DEBUG_CONDLIKE
	printf("nodeIdToArrayId is:\n");
#endif
	int cntTipNode = 0, cntIntNode = 0;
	int nTipNode = nNode / 2 + 1;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(iNode == 0 || iNode % 2 == 1){
			nodeIdToArrayId[iNode] = cntTipNode;
			cntTipNode ++;
		}
		else{
			nodeIdToArrayId[iNode] = nTipNode + cntIntNode;
			intCondlikeNodeId[cntIntNode] = iNode;

			if(cntIntNode < nNodeScaler){
				nodeScalerInd[cntIntNode] = iNode;
				scalerIndMap[iNode] = cntIntNode;
			}
			cntIntNode ++;
		}
#ifdef DEBUG_CONDLIKE
		printf("%d: %d\n", iNode, nodeIdToArrayId[iNode]);
#endif
	}
#ifdef DEBUG_CONDLIKE
	printf("\n\n");
#endif

	assert(cntTipNode == nTipNode && cntTipNode + cntIntNode == nNode);

	return root;
}



// Tree 3: the leaf node is only at the bottom layer, and each node has two children, one is tip state, the other is tip condlike
CuLTreeNode* buildTree3(int nNode)
{
	std::vector<CuLTreeNode*> nodeVec;
	// for leaf nodes:
	for(int iNode = 0; iNode < nNode / 2 + 1; iNode ++){
		CuLTreeNode *curNode = new CuLTreeNode(iNode, 0);
		for(int iChild = 0; iChild < 3; iChild ++)
			curNode->child[iChild] = NULL;
		nodeVec.push_back(curNode);
	}

	// for internal nodes:
	CuLTreeNode *root = NULL;
	int leafThreshold = nNode / 2 + 1;
	for(int iNode = nNode / 2 + 1; iNode < nNode; iNode ++){
		CuLTreeNode *curNode = new CuLTreeNode(iNode, 2);
		int leftChildLabel = (iNode - leafThreshold) * 2;
		curNode->child[0] = nodeVec[leftChildLabel];
		curNode->child[1] = nodeVec[leftChildLabel + 1];
		curNode->child[2] = NULL;

		root = curNode;
		nodeVec.push_back(curNode);
	}

#ifdef DEBUG
	printf("nodeIdToArrayId[] is:\n");
#endif
	int cntIntNode = 0, cntTipState = 0, cntTipCondlike = 0;
	for(int iNode = 0; iNode < nNode; iNode ++){
		nodeIdToArrayId[iNode] = iNode;
		if(iNode < leafThreshold){
			if(iNode % 2 == 0){
				nodeIdToArrayId[iNode] = cntTipState;
				cntTipState ++;
			}
			else{
				nodeIdToArrayId[iNode] = nTipState + cntTipCondlike;
				cntTipCondlike ++;
			}
		}
		else{
			intCondlikeNodeId[cntIntNode] = iNode;
			if(cntIntNode < nNodeScaler){
				nodeScalerInd[cntIntNode] = iNode;
				scalerIndMap[iNode] = cntIntNode;
			}
			cntIntNode ++;
		}

#ifdef DEBUG
		printf("%d: %d\n", iNode, nodeIdToArrayId[iNode]);
#endif
	}

	assert(cntIntNode == nIntCondlike && cntTipState == nTipState && cntTipCondlike == nTipCondlike);
#ifdef DEBUG
	printf("\n\n");
#endif

	return root;
}


// Unrooted version of tree 3
CuLTreeNode* buildTree3_unrooted(int nNode)
{
	std::vector<CuLTreeNode*> nodeVec;
	// for leaf nodes:
	for(int iNode = 0; iNode < nNode; iNode ++){
		CuLTreeNode *curNode = new CuLTreeNode(iNode, 0);
		for(int iChild = 0; iChild < 3; iChild ++)
			curNode->child[iChild] = NULL;
		nodeVec.push_back(curNode);
	}

	CuLTreeNode *root = NULL;
	int nTip = nNode / 2 + 1;
	std::queue<CuLTreeNode *> nodeQue;
	for(int iNode = 0; iNode < nTip - 1; iNode ++)
		nodeQue.push(nodeVec[iNode]);

	int ind = nTip;
	while(ind < nNode){
		CuLTreeNode *curNode = nodeVec[ind];
		curNode->nChild = 2;
		curNode->child[0] = nodeQue.front();
		nodeQue.pop();
		curNode->child[1] = nodeQue.front();
		nodeQue.pop();

		nodeQue.push(curNode);
		root = curNode;
		ind ++;
	}

	root->nChild = 3;
	root->child[2] = nodeVec[nTip - 1];
	
#ifdef DEBUG
	printf("nodeIdToArrayId[] is:\n");
#endif
	int cntIntNode = 0, cntTipState = 0, cntTipCondlike = 0;
	for(int iNode = 0; iNode < nNode; iNode ++){
		nodeIdToArrayId[iNode] = iNode;
		if(iNode < nTip){
			if(iNode % 2 == 0){
				nodeIdToArrayId[iNode] = cntTipState;
				cntTipState ++;
			}
			else{
				nodeIdToArrayId[iNode] = nTipState + cntTipCondlike;
				cntTipCondlike ++;
			}
		}
		else{
			intCondlikeNodeId[cntIntNode] = iNode;
			if(cntIntNode < nNodeScaler){
				nodeScalerInd[cntIntNode] = iNode;
				scalerIndMap[iNode] = cntIntNode;
			}
			cntIntNode ++;
		}

#ifdef DEBUG
		printf("%d: %d\n", iNode, nodeIdToArrayId[iNode]);
#endif
	}

	assert(cntIntNode == nIntCondlike && cntTipState == nTipState && cntTipCondlike == nTipCondlike);
#ifdef DEBUG
	printf("\n\n");
#endif

	return root;
}



void printTree(CuLTreeNode *root)
{
	std::queue<CuLTreeNode *> nodeQue;
	nodeQue.push(root);
	while(!nodeQue.empty()){
		CuLTreeNode *curNode = nodeQue.front();
		nodeQue.pop();
		printf("node %d: ", curNode->label);
		if(curNode->nChild == 0)
			printf("leaf node\n");
		else{
			for(int iChild = 0; iChild < curNode->nChild; iChild ++){
				printf("%d ", curNode->child[iChild]->label);
				nodeQue.push(curNode->child[iChild]);
			}
			printf("\n");
		}
	}
}


void calculatePMat_CPU(const int nNodeForPMat, const int *nodeId, const int nEigen, const int *eigenDecompId)
{
	double *newBrLen = new double[nRateCategory * nNode];
	int ind = 0;
	const int nNode = nTipState + nTipCondlike + nIntCondlike;
	for(int iRate = 0; iRate < nRateCategory; iRate ++){
		for(int iNode = 0; iNode < nNode; iNode ++, ind ++){
			newBrLen[ind] = inRate[iRate] * inBrLen[iNode];
		}
	}

	double *pPMat = NULL, *pU, *pV, *pR;
	const int PMat_size = nState * nState;
	const int PMat_eigen_size = nRateCategory * PMat_size;
	for(int iNode = 0; iNode < nNodeForPMat; iNode ++){
		//int curNode = nodeId[iNode];
		for(int iEigen = 0; iEigen < nEigen; iEigen ++){
			pU = inU[eigenDecompId[iEigen]];
			pV = inV[eigenDecompId[iEigen]];
			pR = inR[eigenDecompId[iEigen]];
			pPMat = inPMat[nodeId[iNode]] + iEigen * PMat_eigen_size;

			for(int iRate = 0; iRate < nRateCategory; iRate ++, pPMat += PMat_size){
				double curBrLen = newBrLen[iRate * nNode + nodeId[iNode]];

				for(int row = 0; row < nState; row ++){
					for(int col = 0; col < nState; col ++){
						double sum = 0.0f;
						for(int i = 0; i < nState; i ++){
							sum += pU[row * nState + i] * exp(pR[i] * curBrLen) * pV[i * nState + col];
						}
						if(transposePMat)
							pPMat[col * nState + row] = sum;
						else
							pPMat[row * nState + col] = sum;
					}
				}
			}
		}
	}

	if(newBrLen){
		free(newBrLen);
		newBrLen = NULL;
	}

#ifdef USE_CODEML_PMAT
	// Transpose PMat:
	for(int iNode = 0; iNode < nNodeForPMat; iNode ++){
		transposeMatrix(inPMat[nodeId[iNode]], nEigenDecomp * nRateCategory, nState, nState);
	}
#endif
}


void comparePMat(const int nNodeForPMat, const int nEigen)
{
	const int PMat_size = nState * nState;
	int offset = 0, cntError = 0;
	double *pPMat_GPU, *pPMat_CPU;
	for(int iNode = 0; iNode < nNodeForPMat; iNode ++){
		offset = 0;
		for(int iEigen = 0; iEigen < nEigen; iEigen ++){
			for(int iRate = 0; iRate < nRateCategory; iRate ++, offset += PMat_size){
				pPMat_GPU = outPMat[iNode] + offset;
				pPMat_CPU = inPMat[iNode] + offset;

				for(int row = 0; row < nState; row ++){
					for(int col = 0; col < nState; col ++){
						if(fabs(pPMat_GPU[row * nState + col] - pPMat_CPU[row * nState + col]) > EPS){
							cntError ++;
							printf("Error! pPMat_CPU[%d][%d][%d][%d][%d] = %f, pPMat_GPU[%d][%d][%d][%d][%d] = %f\n", iNode, iEigen, iRate, row, col, pPMat_CPU[row * nState + col], iNode, iEigen, iRate, row, col, pPMat_GPU[row * nState + col]);

							if(cntError == 10)
								return;
						}
					}
				}
			}
		}
	}

	printf("PMat right!\n");
}


// Transpose version: row * col;
void calculateCondlike_CPU_case1_transpose(double *intCondlike, int *tipState_L, int *tipState_R, double *PMat_L, double *PMat_R, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		int offset_L = tipState_L[iPattern] * nState;
		int offset_R = tipState_R[iPattern] * nState;
		for(int iState = 0; iState < nState; iState ++){
			intCondlike[iPattern * nState + iState] = PMat_L[iState + offset_L] * PMat_R[iState + offset_R];
		}
	}
}


// Non-transpose version: row * row;
void calculateCondlike_CPU_case1_noTranspose(double *intCondlike, int *tipState_L, int *tipState_R, double *PMat_L, double *PMat_R, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		int offset_L = tipState_L[iPattern];
		int offset_R = tipState_R[iPattern];
		for(int iState = 0; iState < nState; iState ++){
			intCondlike[iPattern * nState + iState] = PMat_L[iState * nState + offset_L] * PMat_R[iState * nState + offset_R];
		}
	}
}


// Transpose version:
void calculateCondlike_CPU_case2_transpose(double *intCondlike, int *tipState_L, double *tipCondlike_R, double *PMat_L, double *PMat_R, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		int offset_L = tipState_L[iPattern] * nState;
		
		for(int iState = 0; iState < nState; iState ++){
			CUFlt sum_R = 0.0f;
			for(int i = 0; i < nState; i ++)
				sum_R += tipCondlike_R[iPattern * nState + i] * PMat_R[i * nState + iState];

			intCondlike[iPattern * nState + iState] = PMat_L[iState + offset_L] * sum_R;
		}
	}
}


// Non-transpose version:
void calculateCondlike_CPU_case2_noTranspose(double *intCondlike, int *tipState_L, double *tipCondlike_R, double *PMat_L, double *PMat_R, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		int offset_L = tipState_L[iPattern];
		
		for(int iState = 0; iState < nState; iState ++){
			CUFlt sum_R = 0.0f;
			for(int i = 0; i < nState; i ++)
				sum_R += tipCondlike_R[iPattern * nState + i] * PMat_R[iState * nState + i];

			intCondlike[iPattern * nState + iState] = PMat_L[iState * nState + offset_L] * sum_R;
		}
	}
}


// Transpose version:
void calculateCondlike_CPU_case3_transpose(double *intCondlike, double *tipCondlike_L, double *tipCondlike_R, double *PMat_L, double *PMat_R, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		for(int iState = 0; iState < nState; iState ++){
			CUFlt sum_R = 0.0f, sum_L = 0.0f;
			for(int i = 0; i < nState; i ++){
				sum_L += tipCondlike_L[iPattern * nState + i] * PMat_L[i * nState + iState];
				sum_R += tipCondlike_R[iPattern * nState + i] * PMat_R[i * nState + iState];
			}
			intCondlike[iPattern * nState + iState] = sum_L * sum_R;
		}
	}
}


// Non-transpose version:
void calculateCondlike_CPU_case3_noTranspose(double *intCondlike, double *tipCondlike_L, double *tipCondlike_R, double *PMat_L, double *PMat_R, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		for(int iState = 0; iState < nState; iState ++){
			CUFlt sum_R = 0.0f, sum_L = 0.0f;
			for(int i = 0; i < nState; i ++){
				sum_L += tipCondlike_L[iPattern * nState + i] * PMat_L[iState * nState + i];
				sum_R += tipCondlike_R[iPattern * nState + i] * PMat_R[iState * nState + i];
			}
			intCondlike[iPattern * nState + iState] = sum_L * sum_R;
		}
	}
}


// Unrooted version:
void calculateCondlike_CPU_case1_first_transpose(double *intCondlike, int *tipState_S, double *PMat_S, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		int offset_S = tipState_S[iPattern] * nState;
		
		for(int iState = 0; iState < nState; iState ++){
			intCondlike[iPattern * nState + iState] = PMat_S[iState + offset_S];
		}
	}
}


void calculateCondlike_CPU_case1_notFirst_transpose(double *intCondlike, int *tipState_S, double *PMat_S, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		int offset_S = tipState_S[iPattern] * nState;
		
		for(int iState = 0; iState < nState; iState ++){
			intCondlike[iPattern * nState + iState] *= PMat_S[iState + offset_S];
		}
	}
}


// Non-transpose version: row * row;
void calculateCondlike_CPU_case1_first_noTranspose(double *intCondlike, int *tipState_S, double *PMat_S, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		int offset_S = tipState_S[iPattern];

		for(int iState = 0; iState < nState; iState ++){
			intCondlike[iPattern * nState + iState] = PMat_S[iState * nState + offset_S];
		}
	}
}


void calculateCondlike_CPU_case1_notFirst_noTranspose(double *intCondlike, int *tipState_S, double *PMat_S, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		int offset_S = tipState_S[iPattern];

		for(int iState = 0; iState < nState; iState ++){
			intCondlike[iPattern * nState + iState] *= PMat_S[iState * nState + offset_S];
		}
	}
}


void calculateCondlike_CPU_case2_first_transpose(double *condlike_F, double *condlike_S, double *PMat_S, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		for(int iState = 0; iState < nState; iState ++){
			CUFlt sum_S = 0.0f;
			for(int i = 0; i < nState; i ++){
				sum_S += condlike_S[iPattern * nState + i] * PMat_S[i * nState + iState];
			}
			condlike_F[iPattern * nState + iState] = sum_S;
		}
	}
}


void calculateCondlike_CPU_case2_notFirst_transpose(double *condlike_F, double *condlike_S, double *PMat_S, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		for(int iState = 0; iState < nState; iState ++){
			CUFlt sum_S = 0.0f;
			for(int i = 0; i < nState; i ++){
				sum_S += condlike_S[iPattern * nState + i] * PMat_S[i * nState + iState];
			}
			condlike_F[iPattern * nState + iState] *= sum_S;
		}
	}
}


void calculateCondlike_CPU_case2_first_noTranspose(double *condlike_F, double *condlike_S, double *PMat_S, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		for(int iState = 0; iState < nState; iState ++){
			CUFlt sum_S = 0.0f;
			for(int i = 0; i < nState; i ++){
				sum_S += condlike_S[iPattern * nState + i] * PMat_S[iState * nState + i];
			}
			condlike_F[iPattern * nState + iState] = sum_S;
		}
	}
}


void calculateCondlike_CPU_case2_notFirst_noTranspose(double *condlike_F, double *condlike_S, double *PMat_S, const int nSitePattern, const int nState)
{
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		for(int iState = 0; iState < nState; iState ++){
			CUFlt sum_S = 0.0f;
			for(int i = 0; i < nState; i ++){
				sum_S += condlike_S[iPattern * nState + i] * PMat_S[iState * nState + i];
			}
			condlike_F[iPattern * nState + iState] *= sum_S;
		}
	}
}


void postOrderTraversal(CuLTreeNode *root, std::vector<CuLTreeNode *> &nodeVec)
{
	if(root == NULL)
		return;

	for(int iChild = 0; iChild < root->nChild; iChild ++)
		postOrderTraversal(root->child[iChild], nodeVec);

	if(root->nChild > 0)
		nodeVec.push_back(root);
}


void nodeScale_CPU(double *condlike, double *scaleFactor)
{
	double *pCondlike;
	const int condlike_size = nSitePattern * nState;
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		double maxValue = 0.0;
		pCondlike = condlike + iPattern * nState;
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			for(int iRateCat = 0; iRateCat < nRateCategory; iRateCat ++, pCondlike += condlike_size){
				for(int iState = 0; iState < nState; iState ++){
					maxValue = max(maxValue, pCondlike[iState]);
				}
			}
		}

		if(maxValue <= 0.0)
			maxValue = 1.0;

		scaleFactor[iPattern] = log(maxValue);
		maxValue = 1.0 / maxValue;
		pCondlike = condlike + iPattern * nState;
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			for(int iRateCat = 0; iRateCat < nRateCategory; iRateCat ++, pCondlike += condlike_size){
				for(int iState = 0; iState < nState; iState ++){
					pCondlike[iState] *= maxValue;
				}
			}
		}	
	}
}



void calculateCondlike_CPU(CuLTreeNode *root, const int nIntCondlike, const int *nodeId, const int nEigen, const int *eigenDecompId)
{
	std::set<int> nodeSet;
	for(int iNode = 0; iNode < nIntCondlike; iNode ++){
		nodeSet.insert(nodeId[iNode]);
	}

	std::set<int> scalerNodeSet;
	for(int iNode = 0; iNode < nNodeScaler; iNode ++)
		scalerNodeSet.insert(nodeScalerInd[iNode]);

	std::vector<CuLTreeNode*> nodeVec;
	postOrderTraversal(root, nodeVec);

	const int condlike_size = nSitePattern * nState;
	const int condlike_eigen_size = nRateCategory * condlike_size;
	//const int condlike_node_size = nEigenDecomp * condlike_eigen_size;
	const int nTipNode = nTipState + nTipCondlike;
	const int PMat_size = nState * nState;
	const int PMat_eigen_size = nRateCategory * PMat_size;
	//const int PMat_node_size = nEigenDecomp * PMat_eigen_size;

	double *pPMat_L, *pPMat_R, *pPMat_S, *pCondlike_F;
	void *pCondlike_L, *pCondlike_R, *pCondlike_S;
	int whichCase[3], curCase, PMat_id[3], condlike_id[3]; 
	for(int iNode = 0; iNode < nodeVec.size(); iNode ++){
		CuLTreeNode *curNode = nodeVec[iNode];
		if(nodeSet.find(curNode->label) == nodeSet.end() || curNode->nChild > 3)
			continue;

		//condlike_offset_F = nodeIdToArrayId[curNode->label] * condlike_node_size;
		int arrayId_F = nodeIdToArrayId[curNode->label] - nTipNode;
		for(int iChild = 0; iChild < curNode->nChild; iChild ++){
			CuLTreeNode *curChild = curNode->child[iChild];
			int childId = nodeIdToArrayId[curChild->label];
			if(childId < nTipState){
				whichCase[iChild] = 1;
				PMat_id[iChild] = curChild->label;
				condlike_id[iChild] = childId;
			}
			else if(childId < nTipNode){
				whichCase[iChild] = 2;
				PMat_id[iChild] = curChild->label;
				condlike_id[iChild] = childId - nTipState;
			}
			else{
				whichCase[iChild] = 3;
				PMat_id[iChild] = curChild->label;
				condlike_id[iChild] = childId - nTipNode;
			}
		}

		if(curNode->nChild == 2){
			if(whichCase[0] == 1){
				if(whichCase[1] == 1){
					curCase = 1;
				}
				else{
					curCase = 2;
				}
			}
			else{
				if(whichCase[1] == 1){
					curCase = 2;
					swap(PMat_id[0], PMat_id[1]);
					swap(whichCase[0], whichCase[1]);
					swap(condlike_id[0], condlike_id[1]);
				}
				else
					curCase = 3;
			}
		}

		int PMat_cat_offset, condlike_cat_offset;
		for(int iEigen = 0; iEigen < nEigen; iEigen ++){
			PMat_cat_offset = eigenDecompId[iEigen] * PMat_eigen_size;
			condlike_cat_offset = eigenDecompId[iEigen] * condlike_eigen_size;

			for(int iRate = 0; iRate < nRateCategory; iRate ++, condlike_cat_offset += condlike_size, PMat_cat_offset += PMat_size){
				pCondlike_F = inIntCondlike[arrayId_F] + condlike_cat_offset;
				
				if(curNode->nChild == 2){
					pPMat_L = inPMat[PMat_id[0]] + PMat_cat_offset;
					pPMat_R = inPMat[PMat_id[1]] + PMat_cat_offset;

					if(whichCase[0] == 1)
						pCondlike_L = inTipState[condlike_id[0]];
					else if(whichCase[0] == 2)
						pCondlike_L = inTipCondlike[condlike_id[0]];
					else
						pCondlike_L = inIntCondlike[condlike_id[0]] + condlike_cat_offset;

					if(whichCase[1] == 1)
						pCondlike_R = inTipState[condlike_id[1]];
					else if(whichCase[1] == 2)
						pCondlike_R = inTipCondlike[condlike_id[1]];
					else
						pCondlike_R = inIntCondlike[condlike_id[1]] + condlike_cat_offset;

					if(curCase == 1){
						if(transposePMat)
							calculateCondlike_CPU_case1_transpose(pCondlike_F, (int *)pCondlike_L, (int *)pCondlike_R, pPMat_L, pPMat_R, nSitePattern, nState);
						else
							calculateCondlike_CPU_case1_noTranspose(pCondlike_F, (int *)pCondlike_L, (int *)pCondlike_R, pPMat_L, pPMat_R, nSitePattern, nState);
					}
					else if(curCase == 2){
						if(transposePMat)
							calculateCondlike_CPU_case2_transpose(pCondlike_F, (int *)pCondlike_L, (double *)pCondlike_R, pPMat_L, pPMat_R, nSitePattern, nState);
						else
							calculateCondlike_CPU_case2_noTranspose(pCondlike_F, (int *)pCondlike_L, (double *)pCondlike_R, pPMat_L, pPMat_R, nSitePattern, nState);
					}
					else{
						if(transposePMat)
							calculateCondlike_CPU_case3_transpose(pCondlike_F, (double *)pCondlike_L, (double *)pCondlike_R, pPMat_L, pPMat_R, nSitePattern, nState);
						else
							calculateCondlike_CPU_case3_noTranspose(pCondlike_F, (double *)pCondlike_L, (double *)pCondlike_R, pPMat_L, pPMat_R, nSitePattern, nState);
					}
				}
				else{
					for(int iChild = 0; iChild < curNode->nChild; iChild ++){
						pPMat_S = inPMat[PMat_id[iChild]] + PMat_cat_offset;
						if(whichCase[iChild] == 1){
							pCondlike_S = inTipState[condlike_id[iChild]];
							if(iChild == 0){
								if(transposePMat)
									calculateCondlike_CPU_case1_first_transpose(pCondlike_F, (int *)pCondlike_S, pPMat_S, nSitePattern, nState);
								else
									calculateCondlike_CPU_case1_first_noTranspose(pCondlike_F, (int *)pCondlike_S, pPMat_S, nSitePattern, nState);
							}
							else{
								if(transposePMat)
									calculateCondlike_CPU_case1_notFirst_transpose(pCondlike_F, (int *)pCondlike_S, pPMat_S, nSitePattern, nState);
								else
									calculateCondlike_CPU_case1_notFirst_noTranspose(pCondlike_F, (int *)pCondlike_S, pPMat_S, nSitePattern, nState);
							}
						}
						else{
							if(whichCase[iChild] == 2)
								pCondlike_S = inTipCondlike[condlike_id[iChild]];
							else
								pCondlike_S = inIntCondlike[condlike_id[iChild]] + condlike_cat_offset;

							if(iChild == 0){
								if(transposePMat)
									calculateCondlike_CPU_case2_first_transpose(pCondlike_F, (double *)pCondlike_S, pPMat_S, nSitePattern, nState);
								else
									calculateCondlike_CPU_case2_first_noTranspose(pCondlike_F, (double *)pCondlike_S, pPMat_S, nSitePattern, nState);
							}
							else{
								if(transposePMat)
									calculateCondlike_CPU_case2_notFirst_transpose(pCondlike_F, (double *)pCondlike_S, pPMat_S, nSitePattern, nState);
								else
									calculateCondlike_CPU_case2_notFirst_noTranspose(pCondlike_F, (double *)pCondlike_S, pPMat_S, nSitePattern, nState);
							}
						}
					}
				}
			}
		}

		if(doScale && scalerNodeSet.find(curNode->label) != scalerNodeSet.end()){
			//printf("Goint to scale node %d...\n", arrayId_F);
			nodeScale_CPU(inIntCondlike[arrayId_F], scaleFactor + scalerIndMap[curNode->label] * nSitePattern);
		}
	}
}


void compareCondlike(const int nNode, const int *nodeId, const int nEigen, const int *eigenDecompId)
{
	const int nTipNode = nTipState + nTipCondlike;
	const int condlike_size = nSitePattern * nState;
	const int condlike_eigen_size = nRateCategory * condlike_size;
	//const int condlike_node_size = nEigenDecomp * condlike_eigen_size;

	double *pCondlike_CPU, *pCondlike_GPU;
	int cntError = 0;
	for(int iNode = 0; iNode < nNode; iNode ++){
		int arrayId = nodeIdToArrayId[nodeId[iNode]] - nTipNode;
		//printf("nodeIdToArrayId[%d] = %d, arrayId = %d\n", iNode, nodeIdToArrayId[iNode], arrayId);

		for(int iEigen = 0; iEigen < nEigen; iEigen ++){
			int offset = eigenDecompId[iEigen] * condlike_eigen_size;

			for(int iRate = 0; iRate < nRateCategory; iRate ++, offset += condlike_size){
				pCondlike_CPU = inIntCondlike[arrayId] + offset;
				pCondlike_GPU = outIntCondlike[arrayId] + offset;

				for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
					for(int iState = 0; iState < nState; iState ++){
						if(fabs(pCondlike_CPU[iPattern * nState + iState] - pCondlike_GPU[iPattern * nState + iState]) > EPS){
							cntError ++;
							printf("Error! condlike_CPU[%d][%d][%d][%d][%d] = %f, condlike_GPU[%d][%d][%d][%d][%d] = %f\n", arrayId, iEigen, iRate, iPattern, iState, pCondlike_CPU[iPattern * nState + iState], arrayId, iEigen, iRate, iPattern, iState, pCondlike_GPU[iPattern * nState + iState]);

							if(cntError == 10)
								return;
						}
					}
				}
			}
		}
	}

	printf("Condlike right!\n");
}


double calculateLikelihood_CPU(double *rootCondlike)
{
	const int condlike_size = nSitePattern * nState;
	//const int condlike_eigen_size = nRateCategory * condlike_size;

	double *pRootCondlike;
	double lnL = 0.0f, catSum;
	
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		pRootCondlike = rootCondlike + iPattern * nState;
		catSum = 0.0f;
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			for(int iRateCat = 0; iRateCat < nRateCategory; iRateCat ++, pRootCondlike += condlike_size){
				double stateSum = 0.0f;
				for(int iState = 0; iState < nState; iState ++){
					stateSum += pRootCondlike[iState] * inStateFreq[iState];
				}
				
				catSum += stateSum * inRateCatWeight[iEigen][iRateCat];
			}
		}

		if(catSum <= 0)
			catSum = CUFLT_MIN;
		catSum = log(catSum);

		if(doScale){
			for(int iNode = 0; iNode < nNodeScaler; iNode ++){
				catSum += scaleFactor[iPattern + iNode * nSitePattern];
			}
		}

		inSiteLnL[iPattern] = catSum;
		lnL += catSum * inSitePatternWeight[iPattern];
	}

	return lnL;
}


double reductionOfSiteLikelihood(double *pSiteLnL, double *patternWeight, int nPattern)
{
	double lnL = 0.0f;
	for(int iPattern = 0; iPattern < nPattern; iPattern ++){
		lnL += pSiteLnL[iPattern] * patternWeight[iPattern];
	}

	return lnL;
}


void compareSiteLikelihood()
{
	int cntError = 0;
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		if(fabs(inSiteLnL[iPattern] - outSiteLnL[iPattern]) > EPS){
			printf("Error! siteLnL_CPU[%d] = %.12lf, siteLnL_GPU[%d] = %.12lf\n", iPattern, exp(inSiteLnL[iPattern]), iPattern, exp(outSiteLnL[iPattern]));
			cntError ++;
			if(cntError == 10)
				return;
		}
	}

	printf("Site likelihood right!\n");
}


void compareLikelihood(double lnL_CPU, double lnL_GPU)
{
	if(fabs(lnL_CPU - lnL_GPU) > EPS)
		printf("Error! lnL_CPU = %f, lnL_GPU = %f\n", lnL_CPU, lnL_GPU);
	else
		printf("Likelihood right! lnL_CPU = %f, lnL_GPU = %f\n", lnL_CPU, lnL_GPU);
}


template<typename T>
void printValuesToFile(T *pArray, int nRow, int nCol, const char *filename)
{
	FILE *fout = fopen(filename, "a");

	for(int iRow = 0; iRow < nRow; iRow ++){
		for(int iCol = 0; iCol < nCol; iCol ++){
			fprintf(fout, "%f ", pArray[iRow * nCol + iCol]);
		}
		fprintf(fout, "\n");
	}
	fprintf(fout, "\n");

	fclose(fout);
}


template<typename T>
void printValuesToFile(T **pArray, int nArray, int nCat, int nRow, int nCol, const char *filename)
{
	FILE *fout = fopen(filename, "a");

	T *pCur = NULL;
	const int cat_offset = nRow * nCol;
	for(int iArray = 0; iArray < nArray; iArray ++){
		pCur = pArray[iArray];
		fprintf(fout, "\nnode %d:\n", iArray);
		for(int iCat = 0; iCat < nCat; iCat ++, pCur += cat_offset){
			fprintf(fout, "category %d:\n", iCat);
			for(int iRow = 0; iRow < nRow; iRow ++){
				for(int iCol = 0; iCol < nCol; iCol ++){
					fprintf(fout, "%f ", pCur[iRow * nCol + iCol]);
				}
				fprintf(fout, "\n");
			}
			fprintf(fout, "\n");
		}
	}

	fclose(fout);
}


template<typename T>
void printSingleValueToFile(T value_CPU, T value_GPU, const char *filename)
{
	FILE *fout = fopen(filename, "a");
	fprintf(fout, "\nFor nNode = %d, nTipState = %d, nTipCondlike = %d, nIntCondlike = %d, nState = %d, nEigenDecomp = %d, nRateCategory = %d, nSitePattern = %d:\n", nNode, nTipState, nTipCondlike, nIntCondlike, nState, nEigenDecomp, nRateCategory, nSitePattern);
	fprintf(fout, "CPU: %f, GPU: %f\n", value_CPU, value_GPU);
	fclose(fout);
}


template<typename T>
void compareInputAndOutputArray(T *inArray, T *outArray, const int nElem)
{
	int cntError = 0;
	for(int iElem = 0; iElem < nElem; iElem ++){
		if(fabs(inArray[iElem] - outArray[iElem]) > EPS){
			printf("Error! inArray[%d] = %f, outArray[%d] = %f\n", iElem, inArray[iElem], iElem, outArray[iElem]);
			cntError ++;
			if(cntError == 10)
				return;
		}
	}

	printf("Right!\n");
}


template<typename T>
void compareInputAndOutputArray(T **inArray, T **outArray, const int nRow, const int nCol)
{
	int cntError = 0;
	for(int iRow = 0; iRow < nRow; iRow ++){
		for(int iCol = 0; iCol < nCol; iCol ++){
			if(fabs(inArray[iRow][iCol] - outArray[iRow][iCol]) > EPS){
				printf("Error! inArray[%d][%d] = %f, outArray[%d][%d] = %f\n", iRow, iCol, inArray[iRow][iCol], iRow, iCol, outArray[iRow][iCol]);
				cntError ++;
				if(cntError == 10)
					return;
			}
		}
	}

	printf("Right!\n");
}

template<typename T>
void checkValue(T *pArray, int nRow, int nCol, T threshold)
{
	int cntError = 0;
	for(int iRow = 0; iRow < nRow; iRow ++){
		for(int iCol = 0; iCol < nCol; iCol ++){
			if(pArray[iRow * nCol + iCol] > threshold){
				cntError ++;

				if(cntError <= 10)
					printf("Error in value: array[%d][%d] = %f\n", iRow, iCol, pArray[iRow * nCol + iCol]);
			}
		}
	}

	if(cntError == 0)
		printf("Values all right!\n");
}


template<typename T>
void checkValue(T **pArray, int nArrayToCheck, int* arrayId, int nCat, int nRow, int nCol, T threshold)
{
	T *pCur = NULL;
	const int cat_offset = nRow * nCol;
	int cntError = 0;
	for(int iArray = 0; iArray < nArrayToCheck; iArray ++){
		pCur = pArray[arrayId[iArray]];
		for(int iCat = 0; iCat < nCat; iCat ++, pCur += cat_offset){
			for(int iRow = 0; iRow < nRow; iRow ++){
				for(int iCol = 0; iCol < nCol; iCol ++){
					if(pCur[iRow * nCol + iCol] > threshold){
						cntError ++;

						if(cntError <= 10)
							printf("Error in value: array[%d][%d][%d][%d] = %f\n", iArray, iCat, iRow, iCol, pCur[iRow * nCol + iCol]);
					}
				}
			}
		}
	}
	
	if(cntError == 0)
		printf("Values all right!\n");
}


// TODO: 目前暂时只考虑一个partition，一条chain，之后考虑若有多条chain，如何调用???
void testCuLibraryImpl(int curState)
{

	// Initialize instance:
	CuLErrorCode returnState;
	int instanceId;

	returnState = CuLInitializeCuLInstance(nPartition, 
											deviceId, 
											instanceId);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
	printf("instanceId = %d\n", instanceId);
#ifdef DEBUG
	printf("No error in CuLInitializeCuLInstance()\n");
#endif

	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		 //nEigenDecomp = eigenDecompCountArray[iPartition % nPartition];
		 //nEigenDecomp = 1;
		 
		 //nRateCategory = rateCatCountArray[iPartition % nPartition];
		 //nRateCategory = 1;

		 //nSitePattern = sitePatternCountArray[iPartition % nPartition];
#ifndef USE_CODEML_CONDLIKE
		 nState = stateCountArray[iPartition % nPartition];
#else
		 nState = 61;
#endif
#ifdef USE_CODEML_PMAT
		 nState = 61;
#endif
#ifdef USE_CODEML_LIKELIHOOD
		 nState = 61;
#endif

		nState = curState;
		 
		//nTipState = tipStateCountArray[iPartition % 3];
		//nTipCondlike = tipCondlikeCountArray[iPartition % 3];
		//nIntCondlike = nNode - nTipState - nTipCondlike;

		nTipState = 0;
		nTipCondlike = 2;
		nIntCondlike = nNode - nTipState - nTipCondlike;


		nNodeScaler = 0;
		//nNodeScaler = nIntCondlike - 1;

		printf("\nFor %d-th partition, nState = %d, nTipState = %d, nTipCondlike = %d, nEigenDecomp = %d, nRateCategory = %d\n\n", iPartition, nState, nTipState, nTipCondlike, nEigenDecomp, nRateCategory);

		//int nNode = nTipState + nTipCondlike + nIntCondlike;

		allocateMemory();

		initValues();

		CuLTreeNode *root = NULL;
		/*
		if(isRooted)
			root = buildTree1(nNode);
		else
			root = buildTree1_unrooted(nNode);
		printTree(root);
		*/
		
		/*
		if(iPartition % 2 == 0){
			if(isRooted)
				root = buildTree1(nNode);
			else
				root = buildTree1_unrooted(nNode);
		}
		else{
			if(isRooted)
				root = buildTree2(nNode);
			else
				root = buildTree2_unrooted(nNode);
		}
		printTree(root);
			*/
		if(isRooted)
			root = buildTree1(nNode);
		else
			root = buildTree1_unrooted(nNode);
		printTree(root);

	// Specify params:
	returnState = CuLSpecifyPartitionParams(instanceId, 
											iPartition, 
											nNode,
											nState,
											nSitePattern,
											nRateCategory,
											nEigenDecomp,
											nNode,
											nTipState,
											nTipCondlike,
											nIntCondlike,
											nNodeScaler,
											isRooted);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionParams()\n");
#endif


	// Set tree topology:
	returnState = CuLSpecifyPartitionTreeTopology(instanceId,
												iPartition,
												root);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionTreeTopology()\n");
#endif


	// Set state frequency:
	returnState = CuLSpecifyPartitionStateFrequency(instanceId,
												iPartition,
												inStateFreq);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionStateFrequency()\n");
#endif

	/*
	// Get state frequency:
	returnState = parInstance.getStateFrequency(outStateFreq);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in specifyStateFrequency()\n");
#endif
	printf("\nFor state freq:\n");
	compareInputAndOutputArray(inStateFreq, outStateFreq, nState);

	//printValuesToFile(inStateFreq, 1, nState, "stateFreq.txt");
	//printValuesToFile(outStateFreq, 1, nState, "stateFreq.txt");
	*/


	// Set pattern weight:
	returnState = CuLSpecifyPartitionSitePatternWeight(instanceId,
													iPartition,
													inSitePatternWeight);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionSitePatternWeight()\n");
#endif

	/*
	// Get pattern weight:
	returnState = parInstance.getSitePatternWeight(outSitePatternWeight);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in getSitePatternWeight()\n");
#endif
	printf("\nFor site pattern weight:\n");
	compareInputAndOutputArray(inSitePatternWeight, outSitePatternWeight, nSitePattern);

	//printValuesToFile(inSitePatternWeight, 1, nSitePattern, "patternWeight.txt");
	//printValuesToFile(outSitePatternWeight, 1, nSitePattern, "patternWeight.txt");
	*/


	// Set rates:
	returnState = CuLSpecifyPartitionRate(instanceId,
											iPartition,
											inRate);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionRate()\n");
#endif

	/*
	// Get rates:
	returnState = parInstance.getRate(outRate);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in getRate()\n");
#endif
	printf("\nFor rate:\n");
	compareInputAndOutputArray(inRate, outRate, nRateCategory);

	//printValuesToFile(inRate, 1, nRateCategory, "rate.txt");
	//printValuesToFile(outRate, 1, nRateCategory, "rate.txt");
	*/


	// Set rate category weights:
	int nEigen = nEigenDecomp, *eigenDecompId = new int[nEigen];
	for(int iEigen = 0; iEigen < nEigen; iEigen ++)
		eigenDecompId[iEigen] = iEigen;
	returnState = CuLSpecifyPartitionRateCategoryWeight(instanceId,
														iPartition,
														nEigen,	
														eigenDecompId, 
														(const double**)inRateCatWeight);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionRateCategoryWeight()\n");
#endif

	/*
	// Get rate category weight:
	returnState = parInstance.getRateCategoryWeight(nEigen,	eigenDecompId, outRateCatWeight);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in getRateCategoryWeight()\n");
#endif
	printf("\nFor rate category weight:\n");
	compareInputAndOutputArray(inRateCatWeight, outRateCatWeight, nEigen, nRateCategory);

	//printValuesToFile(inRateCatWeight, nEigen, 1, 1, nRateCategory, "rateCatWeight.txt");
	//printValuesToFile(outRateCatWeight, nEigen, 1, 1, nRateCategory, "rateCatWeight.txt");
	*/

	
	// Set eigen decomposition:
	returnState = CuLSpecifyPartitionEigenDecomposition(instanceId,
														iPartition,
														nEigen, 
														eigenDecompId, 
														(const double**)inU, 
														(const double **)inV, 
														(const double **)inR);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionEigenDecomposition()\n");
#endif

	/*
	// Get eigen decomposition:
	returnState = parInstance.getEigenDecomposition(nEigen, eigenDecompId, outU, outV, outR);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in getEigenDecomposition()\n");
#endif
	printf("\nFor U:\n");
	compareInputAndOutputArray(inU, outU, nEigen, nState * nState);

	//printValuesToFile(inU, nEigen, 1, nState, nState, "U.txt");
	//printValuesToFile(outU, nEigen, 1, nState, nState, "U.txt");

	printf("\nFor V:\n");
	compareInputAndOutputArray(inV, outV, nEigen, nState * nState);

	//printValuesToFile(inV, nEigen, 1, nState, nState, "V.txt");
	//printValuesToFile(outV, nEigen, 1, nState, nState, "V.txt");

	printf("\nFor R:\n");
	compareInputAndOutputArray(inR, outR, nEigen, nState);

	//printValuesToFile(inR, nEigen, 1, 1, nState, "R.txt");
	//printValuesToFile(outR, nEigen, 1, 1, nState, "R.txt");
	*/


	// Calculate transition matrix:
	int nNodeForPMat = nNode - 1, *nodeId = new int[nNodeForPMat], rootId = root->label;
	double *curBrLen = new double[nNodeForPMat];
	int curNode = 0;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(iNode == rootId)
			continue;
		
		nodeId[curNode] = iNode;
		curBrLen[curNode] = inBrLen[iNode];
		curNode ++;
	}

	/*
	// Specify transition matrices:
	returnState = CuLSpecifyPartitionTransitionMatrixMulti(instanceId,
														iPartition,
														nNodeForPMat, 
														nodeId, 
														nEigen, 
														eigenDecompId, 
														(const double **)inPMat);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionTransitionMatrixMulti()\n");
#endif
	
	
	returnState = CuLSpecifyPartitionTransitionMatrixAll(instanceId,
														iPartition,
														nNodeForPMat, 
														nodeId, 
														(const double **)inPMat);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionTransitionMatrixAll()\n");
#endif
	*/

	const int nNodeToCalc_PMat = 6;
	int pNodeId_PMat[nNodeToCalc_PMat] = {4, 5, 6, 9, 10, 13};

	
	//for(int i = 0; i < 3; i ++)
	returnState = CuLCalculatePartitionTransitionMatrixMulti(instanceId,
														iPartition,
														nNodeForPMat, 
														nodeId, 
														nEigen, 
														eigenDecompId, 
														(const double *)curBrLen);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLCalculatePartitionTransitionMatrixMulti()\n");
#endif
	
	/*
	
	returnState = CuLCalculatePartitionTransitionMatrixAll(instanceId,
														iPartition,
														nNodeForPMat, 
														(const int *)nodeId, 
														inBrLen);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLCalculatePartitionTransitionMatrixAll()\n");
#endif	
	*/

	/*
	// Copy transition matrix back:
	returnState = CuLGetPartitionTransitionMatrixMulti(instanceId,
														iPartition,
														nNodeForPMat, 
														nodeId, 
														nEigen, 
														eigenDecompId, 
														outPMat);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLGetPartitionTransitionMatrixMulti()\n");
#endif
	*/
	
	
	returnState = CuLGetPartitionTransitionMatrixAll(instanceId,
														iPartition,
														nNodeForPMat, 
														nodeId, 
														outPMat);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLGetPartitionTransitionMatrixAll()\n");
#endif
	printValuesToFile(outPMat, nNodeForPMat, nEigenDecomp * nRateCategory, nState, nState, "PMat_GPU.txt");

	calculatePMat_CPU(nNodeForPMat, nodeId, nEigen, eigenDecompId);
	printValuesToFile(inPMat, nNodeForPMat, nEigenDecomp * nRateCategory, nState, nState, "PMat_CPU.txt");
#ifdef DEBUG
	printf("No error in calculatePMat_CPU()\n");
#endif

	comparePMat(nNodeForPMat, nEigen);
#ifdef DEBUG
	printf("No error in comparePMat()\n");
#endif

	/*
	freeMemory();

	if(curBrLen){
		free(curBrLen);
		curBrLen = NULL;
	}

	if(eigenDecompId){
		free(eigenDecompId);
		eigenDecompId = NULL;
	}

	if(nodeId){
		free(nodeId);
		nodeId = NULL;
	}

	continue;
	*/

	// Specify tip state and tip condlike:
	for(int iNode = 0; iNode < nTipState; iNode ++)
		tipStateNodeId[iNode] = iNode;
	for(int iNode = 0; iNode < nTipCondlike; iNode ++)
		tipCondlikeNodeId[iNode] = iNode;

	if(nTipState > 0){
		returnState = CuLSpecifyPartitionTipState(instanceId,
											iPartition,
											nTipState, 
											tipStateNodeId, 
											(const int **)inTipState);
		if(CUL_SUCCESS != returnState)
			printErrorCode(returnState);
#ifdef DEBUG
		printf("No error in CuLSpecifyPartitionTipState()\n");
#endif
	}

	/*
	// Get tip state:
	returnState = parInstance.getTipState(nTipState, tipStateNodeId, outTipState);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in getTipState()\n");
#endif

	printf("\nFor tip state:\n");
	compareInputAndOutputArray(inTipState, outTipState, nTipState, nSitePattern);

	//printValuesToFile(inTipState, nTipState, 1, 1, nSitePattern, "tipState.txt");
	//printValuesToFile(outTipState, nTipState, 1, 1, nSitePattern, "tipState.txt");
	*/


	// Set tip condlike:
	if(nTipCondlike > 0){
		returnState = CuLSpecifyPartitionTipCondlike(instanceId,
												iPartition,
												nTipCondlike, 
												tipCondlikeNodeId, 
												(const double **)inTipCondlike);
		if(CUL_SUCCESS != returnState)
			printErrorCode(returnState);
#ifdef DEBUG
		printf("No error in CuLSpecifyPartitionTipCondlike()\n");
#endif
	}

	/*
	// Set tip condlike:
	returnState = parInstance.getTipCondlike(nTipCondlike, tipCondlikeNodeId, outTipCondlike);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in getTipCondlike()\n");
#endif

	printf("\nFor tip condlike:\n");
	compareInputAndOutputArray(inTipCondlike, outTipCondlike, nTipCondlike, nSitePattern * nState);

	//printValuesToFile(inTipCondlike, nTipCondlike, 1, nSitePattern, nState, "tipCondlike.txt");
	//printValuesToFile(outTipCondlike, nTipCondlike, 1, nSitePattern, nState, "tipCondlike.txt");
	*/


	// Set node label to array index map:
	returnState = CuLMapPartitionNodeIndToArrayInd(instanceId,
												iPartition,
												nNode, 
												nodeIdToArrayId);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLMapPartitionNodeIndToArrayInd()\n");
#endif


	// Set node label to array index map:
	returnState = CuLSpecifyPartitionNodeScalerIndex(instanceId,
													iPartition,
													nNodeScaler,
													nodeScalerInd);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionNodeScalerIndex()\n");
#endif

	/*
	// Set internal condlike:
	returnState = CuLSpecifyPartitionInternalCondlikeMulti(instanceId,
														iPartition,
														nIntCondlike, 
														intCondlikeNodeId, 
														nEigen, 
														eigenDecompId, 
														(const double **)inIntCondlike);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionInternalCondlikeMulti()\n");
#endif
	*/

	/*
	returnState = CuLSpecifyPartitionInternalCondlikeAll(instanceId,
														iPartition,
														nIntCondlike, 
														intCondlikeNodeId, 
														(const double **)inIntCondlike);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSpecifyPartitionInternalCondlikeAll()\n");
#endif
	*/
	
	
	// Calculate condlike of all internal nodes:
	const int nNodeToCalc_condlike = 5;
	int pNodeId_condlike[nNodeToCalc_condlike] = {6, 8, 10, 12, 14};
	
	/*
	returnState = CuLCalculatePartitionCondlikeMulti(instanceId,
													iPartition,
													nIntCondlike, 
													intCondlikeNodeId, 
													nEigen, 
													eigenDecompId);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLCalculatePartitionCondlikeMulti()\n");
#endif
	*/
	
	returnState = CuLCalculatePartitionCondlikeAll(instanceId,
													iPartition,
													nIntCondlike, 
													intCondlikeNodeId);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLCalculatePartitionCondlikeAll()\n");
#endif
	
	
	// Copy internal condlike back:
	/*
	returnState = CuLGetPartitionIntCondlikeMulti(instanceId,
													iPartition,
													nIntCondlike, 
													intCondlikeNodeId, 
													nEigen, 
													eigenDecompId, 
													outIntCondlike);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLGetPartitionIntCondlikeMulti()\n");
#endif
	*/
	
	
	returnState = CuLGetPartitionIntCondlikeAll(instanceId,
												iPartition,
												nIntCondlike, 
												intCondlikeNodeId,
												outIntCondlike);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLGetPartitionIntCondlikeAll()\n");
#endif

	//if(nState == 61)
		//printValuesToFile(outIntCondlike, nIntCondlike, nEigen * nRateCategory, nSitePattern, nState, "condlike_GPU.txt");
	
	printf("Going to check condlike_GPU...\n");
	int *checkInd = new int[nNode];
	for(int iNode = 0; iNode < nNodeScaler; iNode ++)
		checkInd[iNode] = nodeIdToArrayId[nodeScalerInd[iNode]] - nTipState - nTipCondlike;
	checkValue(outIntCondlike, nNodeScaler, checkInd, nEigen * nRateCategory, nSitePattern, nState, 1.0);
#ifdef DEBUG
	printf("No error in checkValue()\n");
#endif

	calculateCondlike_CPU(root, nIntCondlike, intCondlikeNodeId, nEigen, eigenDecompId);
#ifdef DEBUG
	printf("No error in calculateCondlike_CPU()\n");
#endif

	
	//if(nState == 61)
		//printValuesToFile(inIntCondlike, nIntCondlike, nEigen * nRateCategory, nSitePattern, nState, "condlike_CPU.txt");
	
	if(doScale){
		printf("Going to check condlike_CPU...\n");
		checkValue(inIntCondlike, nNodeScaler, checkInd, nEigen * nRateCategory, nSitePattern, nState, 1.0);
	}

#ifdef DEBUG
	printf("No error in calculateCondlike_CPU()\n");
#endif

	compareCondlike(nIntCondlike, intCondlikeNodeId, nEigen, eigenDecompId);
#ifdef DEBUG
	printf("No error in compareCondlike()\n");
#endif
	
	/*
	initSiteLnL();
	returnState = CuLSetPartitionSiteLikelihood(instanceId,
												iPartition,
												inSiteLnL);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLSetPartitionSiteLikelihood()\n");
#endif

	
	returnState = CuLCalculatePartitionLikelihoodFromSiteLnLSync(instanceId,
																iPartition,
																outLnL);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLCalculatePartitionLikelihoodFromSiteLnLSync()\n");
#endif
	
	
	returnState = CuLCalculatePartitionLikelihoodFromSiteLnLAsync(instanceId,
																iPartition);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLCalculatePartitionLikelihoodFromSiteLnLAsync()\n");
#endif

	returnState = CuLGetPartitionLikelihood(instanceId,
											iPartition,
											outLnL);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLGetPartitionLikelihood()\n");
#endif
		

	double lnL_CPU = reductionOfSiteLikelihood(inSiteLnL, inSitePatternWeight, nSitePattern);
	compareLikelihood(lnL_CPU, outLnL);
	*/

	/*
	returnState = CuLCalculatePartitionLikelihoodSync(instanceId,
													iPartition,
													outLnL);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLCalculatePartitionLikelihoodSync()\n");
#endif
	*/

	
	returnState = CuLCalculatePartitionLikelihoodAsync(instanceId,
													iPartition);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLCalculatePartitionLikelihoodAsync()\n");
#endif

	returnState = CuLGetPartitionLikelihood(instanceId,
											iPartition,
											outLnL);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLGetPartitionLikelihood()\n");
#endif

	
#ifdef DEBUG_CONDLIKE
	printf("root->label = %d, nodeIdToArrayId[%d] = %d\n", root->label, nodeIdToArrayId[root->label]);
#endif
	double *rootCondlike = inIntCondlike[nodeIdToArrayId[root->label] - nTipState - nTipCondlike];
	double lnL_CPU = calculateLikelihood_CPU(rootCondlike);
#ifdef DEBUG
	printf("No error in calculateLikelihood_CPU()\n");
#endif

	compareLikelihood(lnL_CPU, outLnL);

	//printSingleValueToFile(lnL_CPU, outLnL, "lnL.txt");


	// Copy site likelihood back:
	returnState = CuLGetPartitionSiteLikelihood(instanceId,
												iPartition,
												outSiteLnL);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLGetPartitionSiteLikelihood()\n");
#endif

	compareSiteLikelihood();

	//printValuesToFile(inSiteLnL, 1, nSitePattern, "siteLnL_CPU.txt");
	//printValuesToFile(outSiteLnL, 1, nSitePattern, "siteLnL_GPU.txt");

	freeMemory();

	if(curBrLen){
		free(curBrLen);
		curBrLen = NULL;
	}

	if(eigenDecompId){
		free(eigenDecompId);
		eigenDecompId = NULL;
	}

	if(nodeId){
		free(nodeId);
		nodeId = NULL;
	}

	if(checkInd){
		free(checkInd);
		checkInd = NULL;
	}

	}

	// Finalize the instance:
	returnState = CuLFinalizeInstance(instanceId);
	printf("instanceId = %d\n", instanceId);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLFinalizeInstance()\n");
#endif

}


int main()
{
	nPartition = 1;
	deviceId = 0;
	nNode = 3;

#ifndef TRANSPOSE_PMAT
	transposePMat = false;
#else
	transposePMat = true;
#endif

	if(transposePMat)
		printf("Going to transpose PMat for CPU...\n");
	else
		printf("Not going to transpose PMat for CPU...\n");

	doScale = true;
	isRooted = true;

	for(int iState = 4; iState < 5; iState ++){
		//if(iState == 20 || (iState >= 57 && iState <= 64))
			//continue;
		printf("For state = %d:\n", iState);
		//printf("For state = %d:\n", stateCountArray[iState]);

		if(isRooted){
			FILE *fout = fopen("condlike_time_rooted_transpose.txt", "a");
			fprintf(fout, "\n\nFor state = %d:\n", iState);
			fclose(fout);
		}
		else{
			FILE *fout = fopen("condlike_time_unrooted_transpose.txt", "a");
			fprintf(fout, "\n\nFor state = %d:\n", iState);
			fclose(fout);
		}

		for(int iEigen = 0; iEigen < 1; iEigen ++){
			nEigenDecomp = eigenDecompCountArray[iEigen];
			for(int iRate = 0; iRate < 14; iRate ++){
				nRateCategory = rateCatCountArray[iRate];
				printf("\n======================\nFor nOp = %d:\n\n", nEigenDecomp * nRateCategory);
				for(int iPattern = 0; iPattern < 13; iPattern ++){
					nSitePattern = sitePatternCountArray[iPattern];
					printf("For nSitePattern = %d:\n\n", nSitePattern);
					testCuLibraryImpl(iState);
				}

				if(isRooted){
					FILE *fconp = fopen("condlike_time_rooted_transpose_format.txt", "a");
					fprintf(fconp, "\t");
					fclose(fconp);
				}
				else{
					FILE *fconp = fopen("condlike_time_unrooted_transpose_format.txt", "a");
					fprintf(fconp, "\t");
					fclose(fconp);
				}
			}
		}
		//testCuLibraryImpl(stateCountArray[iState]);

		FILE *fout2 = fopen("PMat_time_format.txt", "a");
		fprintf(fout2, "\n\n\n");
		fclose(fout2);

		if(isRooted){
			fout2 = fopen("condlike_time_rooted_transpose_format.txt", "a");
			fprintf(fout2, "\n");
			fclose(fout2);
		}
		else{
			fout2 = fopen("condlike_time_unrooted_transpose_format.txt", "a");
			fprintf(fout2, "\n");
			fclose(fout2);
		}

		fout2 = fopen("scale_time_format.txt", "a");
		fprintf(fout2, "\n\n\n");
		fclose(fout2);

		fout2 = fopen("lnL_time_format.txt", "a");
		fprintf(fout2, "\n\n\n\n");
		fclose(fout2);

		fout2 = fopen("lnL_reduction_time_format.txt", "a");
		fprintf(fout2, "\n\n");
		fclose(fout2);
	}
	/*
	for(int iTipState = 0; iTipState < 1; iTipState ++){
		nTipState = tipStateCountArray[iTipState];
		nTipCondlike = tipCondlikeCountArray[iTipState];
		nIntCondlike = nNode - nTipState - nTipCondlike;
		//nTipState = 3;
		//nTipCondlike = 5;
		//nIntCondlike = 7;

		for(int iState = 0; iState < 1; iState ++){
			//nState = stateCountArray[iState];
			nState = 61;

			for(int iEigen = 0; iEigen < 1; iEigen ++){
				nEigenDecomp = eigenDecompCountArray[iEigen];
				//nEigenDecomp = 1;

				for(int iRate = 0; iRate < 1; iRate ++){
					nRateCategory = rateCatCountArray[iRate];
					//nRateCategory = 3;

					for(int iPattern = 0; iPattern < 1; iPattern ++){
						nSitePattern = sitePatternCountArray[iPattern];
						//nSitePattern = 1000;

						printf("\nFor deviceId = %d, nPartition = %d, nNode = %d, nTipState = %d, nTipCondlike = %d, nIntCondlike = %d, nState = %d, nEigenDecomp = %d, nRateCategory = %d, nSitePattern = %d:\n", deviceId, nPartition, nNode, nTipState, nTipCondlike, nIntCondlike, nState, nEigenDecomp, nRateCategory, nSitePattern);

						testCuLibraryImpl();
					}
				}
			}
		}
	}
	*/

	return 0;
}