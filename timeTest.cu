#include "CuLibrary.h"
#include "CuLibKernel-codemlAndMrBayes-rooted.h"
#include <set>
#include <queue>
#include <sys/time.h>
#include <math.h>
#include "/home/hxm/beagle/include/libhmsbeagle-1/libhmsbeagle/beagle.h"


#define EPS 0.00001

bool transposePMat;
bool doScale, isRooted;

double **inRate, **inStateFreq, ***inRateCatWeight, **inSitePatternWeight, **inBrLen;
double ***inTipCondlike, ***inIntCondlike, ***inU, ***inV, ***inR, ***inPMat;
//double *outRate, *outStateFreq, **outRateCatWeight, *outSitePatternWeight;
//double **outTipCondlike, **outU, **outV, **outR;
double **scaleFactor;
int ***inTipState;
//int **outTipState;
int **nodeIdToArrayId;
int **nodeScalerInd;
int **scalerIndMap;
double ***outPMat, ***outIntCondlike, **outSiteLnL, **inSiteLnL;
int **tipStateNodeId, **tipCondlikeNodeId, **intCondlikeNodeId;
double *lnL_GPU, *lnL_CPU, *lnL_beagle;

int **eigenDecompId, *rootId;
int ***sonId;


CuLTreeNode **partitionRoot;

int stateCountArray[] = {4, 20, 61};
int eigenDecompCountArray[10] = {4, 4, 4, 4, 4};
int rateCatCountArray[10] = {1, 1, 1, 1, 1};

int patternCount[] = {500, 1000, 5000, 10000, 15000};
int sitePatternCountArray[10] = {500, 500, 500, 500, 500};
int tipStateCountArray[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
int tipCondlikeCountArray[10] = {32, 32, 32, 32, 32, 32, 32, 32, 32, 32};
int intCondlikeCountArray[10] = {31, 31, 31, 31, 31, 31, 31, 31, 31, 31};

int nPartition, deviceId, nIteration;
int nNode, nTipState, nTipCondlike, nIntCondlike, nState, nEigenDecomp, nRateCategory, nSitePattern;
int nNodeScaler;


long long time_lnL_beagle, time_lnL_cuLib;
long long multiple = 1000000ll;
//static int conp_cnt=0;
void timeBegin(struct timeval *tBegin){
	gettimeofday(tBegin, NULL);
}

long long timeEnd(struct timeval tBegin){
	 struct timeval tEnd;
     gettimeofday(&tEnd, NULL);
	
	 long long usec = tEnd.tv_sec * multiple + tEnd.tv_usec - (tBegin.tv_sec * multiple + tBegin.tv_usec);
	 return usec;
}


// Allocate memory for host memory:
void allocateMemory()
{
	// Allocate memory for the whole nPartition partitions:
	nodeScalerInd = (int **) calloc(nPartition, sizeof(int*));
	scalerIndMap = (int**) calloc(nPartition, sizeof(int*));
	scaleFactor = (double**) calloc(nPartition, sizeof(double*));
	nodeIdToArrayId = (int**) calloc(nPartition, sizeof(int*));
	inRate = (double **) calloc(nPartition, sizeof(double*));
	inStateFreq = (double **) calloc(nPartition, sizeof(double*));
	inRateCatWeight = (double ***) calloc(nPartition, sizeof(double**));
	inSitePatternWeight = (double **) calloc(nPartition, sizeof(double*));
	inBrLen = (double **) calloc(nPartition, sizeof(double*));
	inTipState = (int ***) calloc(nPartition, sizeof(int **));
	inTipCondlike = (double ***) calloc(nPartition, sizeof(double **));
	inIntCondlike = (double ***) calloc(nPartition, sizeof(double **));
	outIntCondlike = (double ***) calloc(nPartition, sizeof(double **));
	inU = (double ***) calloc(nPartition, sizeof(double **));
	inV = (double ***) calloc(nPartition, sizeof(double **));
	inR = (double ***) calloc(nPartition, sizeof(double **));
	inPMat = (double ***) calloc(nPartition, sizeof(double **));
	outPMat = (double ***) calloc(nPartition, sizeof(double **));
	tipStateNodeId = (int **) calloc(nPartition, sizeof(int *));
	tipCondlikeNodeId = (int **) calloc(nPartition, sizeof(int *));
	intCondlikeNodeId = (int **) calloc(nPartition, sizeof(int *));

	inSiteLnL = (double **) calloc(nPartition, sizeof(double *));
	outSiteLnL = (double **) calloc(nPartition, sizeof(double *));

	lnL_GPU = (double *) calloc(nPartition, sizeof(double));
	lnL_CPU = (double *) calloc(nPartition, sizeof(double));
	lnL_beagle = (double *) calloc(nPartition, sizeof(double));

	eigenDecompId = (int **) calloc(nPartition, sizeof(int *));

	sonId = (int ***) calloc(nPartition, sizeof(int **));
	rootId = (int *) calloc(nPartition, sizeof(int));
	partitionRoot = (CuLTreeNode **) calloc(nPartition, sizeof(CuLTreeNode *));

	int nArrayPerNode, condlikeSize, PMat_size, PMat_node_size, nMaxNodeScaler;
	
	// Allocate memory for each partition:
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		nEigenDecomp = eigenDecompCountArray[iPartition % nPartition];
		 //nEigenDecomp = 1;
		nRateCategory = rateCatCountArray[iPartition % nPartition];
		//nRateCategory = 1;
		nSitePattern = sitePatternCountArray[iPartition % nPartition];

		nTipState = tipStateCountArray[iPartition % nPartition];
		nTipCondlike = tipCondlikeCountArray[iPartition % nPartition];
		nIntCondlike = intCondlikeCountArray[iPartition % nPartition];
		assert(nTipState + nTipCondlike + nIntCondlike == nNode);

		nArrayPerNode = nEigenDecomp * nRateCategory;
		condlikeSize = nSitePattern * nState;
		PMat_size = nState * nState;
		PMat_node_size = nArrayPerNode * PMat_size;

		nMaxNodeScaler = max(nNode, nNodeScaler);


		// Allocate memory for scale:
		nodeScalerInd[iPartition] = (int*) calloc(nMaxNodeScaler, sizeof(int));
		scalerIndMap[iPartition] = (int*) calloc(nMaxNodeScaler, sizeof(int));
		scaleFactor[iPartition] = (double*) calloc(nMaxNodeScaler * nSitePattern, sizeof(double));

		nodeIdToArrayId[iPartition] = (int*) calloc(nNode, sizeof(int));

		inRate[iPartition] = (double *) calloc(nRateCategory, sizeof(double));
	
		inStateFreq[iPartition] = (double *) calloc(nState, sizeof(double));
	
		inRateCatWeight[iPartition] = (double **) calloc(nEigenDecomp, sizeof(double*));
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			inRateCatWeight[iPartition][iEigen] = (double *) calloc(nRateCategory, sizeof(double));
		}

		inSitePatternWeight[iPartition] = (double *) calloc(nSitePattern, sizeof(double));
	
		inBrLen[iPartition] = (double *) calloc(nNode, sizeof(double));
	
		inTipState[iPartition] = (int **) calloc(nTipState, sizeof(int *));
		for(int iTip = 0; iTip < nTipState; iTip ++){
			inTipState[iPartition][iTip] = (int *) calloc(nSitePattern, sizeof(int));
		}

		inTipCondlike[iPartition] = (double **) calloc(nTipCondlike, sizeof(double *));
		for(int iTip = 0; iTip < nTipCondlike; iTip ++){
			inTipCondlike[iPartition][iTip] = (double *) calloc(condlikeSize, sizeof(double));
		}

		inIntCondlike[iPartition] = (double **) calloc(nIntCondlike, sizeof(double *));
		for(int iInt = 0; iInt < nIntCondlike; iInt ++){
			inIntCondlike[iPartition][iInt] = (double *) calloc(nArrayPerNode * condlikeSize, sizeof(double));
		}

		outIntCondlike[iPartition] = (double **) calloc(nIntCondlike, sizeof(double *));
		for(int iInt = 0; iInt < nIntCondlike; iInt ++){
			outIntCondlike[iPartition][iInt] = (double *) calloc(nArrayPerNode * condlikeSize, sizeof(double));
		}


		inU[iPartition] = (double **) calloc(nEigenDecomp, sizeof(double *));
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			inU[iPartition][iEigen] = (double *) calloc(PMat_size, sizeof(double));
		}

		inV[iPartition] = (double **) calloc(nEigenDecomp, sizeof(double *));
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			inV[iPartition][iEigen] = (double *) calloc(PMat_size, sizeof(double));
		}

		inR[iPartition] = (double **) calloc(nEigenDecomp, sizeof(double *));
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			inR[iPartition][iEigen] = (double *) calloc(nState, sizeof(double));
		}

		inPMat[iPartition] = (double **) calloc(nNode, sizeof(double *));
		for(int iNode = 0; iNode < nNode; iNode ++){
			inPMat[iPartition][iNode] = (double *) calloc(PMat_node_size, sizeof(double));
		}

		outPMat[iPartition] = (double **) calloc(nNode, sizeof(double *));
		for(int iNode = 0; iNode < nNode; iNode ++){
			outPMat[iPartition][iNode] = (double *) calloc(PMat_node_size, sizeof(double));
		}

		inSiteLnL[iPartition] = (double *) calloc(nSitePattern, sizeof(double));
		outSiteLnL[iPartition] = (double *) calloc(nSitePattern, sizeof(double));

		tipStateNodeId[iPartition] = (int *) calloc(nTipState, sizeof(int));
		tipCondlikeNodeId[iPartition] = (int *) calloc(nTipCondlike, sizeof(int));
		intCondlikeNodeId[iPartition] = (int *) calloc(nIntCondlike, sizeof(int));

		eigenDecompId[iPartition] = (int *) calloc(nEigenDecomp, sizeof(int));

		sonId[iPartition] = (int **) calloc(nNode, sizeof(int *));
		for(int iNode = 0; iNode < nNode; iNode ++)
			sonId[iPartition][iNode] = (int *) calloc(3, sizeof(int));
	}
}


void freeMemory()
{
	int nEigenDecomp, nTipState, nTipCondlike, nIntCondlike, nNode;
	
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		
		nEigenDecomp = eigenDecompCountArray[iPartition % nPartition];
	    nTipState = tipStateCountArray[iPartition % nPartition];
		nTipCondlike = tipCondlikeCountArray[iPartition % nPartition];
		nIntCondlike = intCondlikeCountArray[iPartition % nPartition];
		nNode = nTipState + nTipCondlike + nIntCondlike;

		free(nodeScalerInd[iPartition]);
		
		free(scalerIndMap[iPartition]);
		
		free(scaleFactor[iPartition]);
		
		free(nodeIdToArrayId[iPartition]);
		
		free(inRate[iPartition]);
		
		free(inStateFreq[iPartition]);
		
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++)
			free(inRateCatWeight[iPartition][iEigen]);
		free(inRateCatWeight[iPartition]);
		
		free(inSitePatternWeight[iPartition]);

		free(inBrLen[iPartition]);

		for(int iTipState = 0; iTipState < nTipState; iTipState ++)
			free(inTipState[iPartition][iTipState]);
		free(inTipState[iPartition]);

		for(int iTipCondlike = 0; iTipCondlike < nTipCondlike; iTipCondlike ++)
			free(inTipCondlike[iPartition][iTipCondlike]);
		free(inTipCondlike[iPartition]);

		for(int iIntCondlike = 0; iIntCondlike < nIntCondlike; iIntCondlike ++){
			free(inIntCondlike[iPartition][iIntCondlike]);
			free(outIntCondlike[iPartition][iIntCondlike]);
		}
		free(inIntCondlike[iPartition]);
		free(outIntCondlike[iPartition]);

		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			free(inU[iPartition][iEigen]);
			free(inV[iPartition][iEigen]);
			free(inR[iPartition][iEigen]);
		}
		free(inU[iPartition]);
		free(inV[iPartition]);
		free(inR[iPartition]);

		for(int iNode = 0; iNode < nNode; iNode ++){
			free(inPMat[iPartition][iNode]);
			free(outPMat[iPartition][iNode]);
		}
		free(inPMat[iPartition]);
		free(outPMat[iPartition]);

		free(tipStateNodeId[iPartition]);
		free(tipCondlikeNodeId[iPartition]);
		free(intCondlikeNodeId[iPartition]);

		free(inSiteLnL[iPartition]);
		free(outSiteLnL[iPartition]);

		free(eigenDecompId[iPartition]);

		for(int iNode = 0; iNode < nNode; iNode ++)
			free(sonId[iPartition][iNode]);
		free(sonId[iPartition]);
	}
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

	free(inStateFreq);
	inStateFreq = NULL;

	free(inRateCatWeight);
	inRateCatWeight = NULL;

	free(inSitePatternWeight);
	inSitePatternWeight = NULL;

	free(inBrLen);
	inBrLen = NULL;

	free(inTipState);
	inTipState = NULL;

	free(inTipCondlike);
	inTipCondlike = NULL;

	free(inIntCondlike);
	inIntCondlike = NULL;
	free(outIntCondlike);
	outIntCondlike = NULL;

	free(inU);
	inU = NULL;

	free(inV);
	inV = NULL;

	free(inR);
	inR = NULL;
	
	free(inPMat);
	inPMat = NULL;
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

	free(lnL_GPU);
	lnL_GPU = NULL;

	free(lnL_CPU);
	lnL_CPU = NULL;

	free(lnL_beagle);
	lnL_beagle = NULL;

	free(eigenDecompId);
	eigenDecompId = NULL;

	free(rootId);
	rootId = NULL;

	free(partitionRoot);
	partitionRoot = NULL;

	free(sonId);
	sonId = NULL;
}


void initValues()
{
	srand(int(time(0)));
	//int nRateCategory, nEigenDecomp, nTipState, nTipCondlike, nIntCondlike, nNode;
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		nRateCategory = rateCatCountArray[iPartition % nPartition];
		nEigenDecomp = eigenDecompCountArray[iPartition % nPartition];
	    
		nTipState = tipStateCountArray[iPartition % nPartition];
		nTipCondlike = tipCondlikeCountArray[iPartition % nPartition];
		nIntCondlike = intCondlikeCountArray[iPartition % nPartition];
		assert(nTipState + nTipCondlike + nIntCondlike == nNode);

		nSitePattern = sitePatternCountArray[iPartition % nPartition];

		for(int iState = 0; iState < nState; iState ++)
			inStateFreq[iPartition][iState] = (double)(rand() % nState + 1) / (nState * 10);

		for(int iRate = 0; iRate < nRateCategory; iRate ++)
			inRate[iPartition][iRate] = (double)(rand() % 10 + 1) / (100.0);

		for(int iPattern = 0; iPattern < nSitePattern; iPattern ++)
			inSitePatternWeight[iPartition][iPattern] = (double)(rand() % 5 + 1);

		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			for(int iRateCat = 0; iRateCat < nRateCategory; iRateCat ++)
				inRateCatWeight[iPartition][iEigen][iRateCat] = (double)(rand() % 10 + 1) / 11.0;
		}

		for(int iNode = 0; iNode < nNode; iNode ++){
			inBrLen[iPartition][iNode] = (double)(rand() % 10 + 1) / 100.0;
			//inBrLen[iPartition][iNode] = 1.0f;
		}

		for(int iTip = 0; iTip < nTipState; iTip ++){
			for(int iPattern = 0; iPattern < nSitePattern; iPattern ++)
				inTipState[iPartition][iTip][iPattern] = (int)(rand() % nState);
		}

		for(int iTip = 0; iTip < nTipCondlike; iTip ++){
			for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
				for(int iState = 0; iState < nState; iState ++)
					inTipCondlike[iPartition][iTip][iPattern * nState + iState] = (double)(rand() % 100 + 1) / (200 * nState);
			}
		}

		/*
		int condlike_size = nSitePattern * nState;
		int offset;
		for(int iInt = 0; iInt < nIntCondlike; iInt ++){
			offset = 0;
			for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
				for(int iRateCat = 0; iRateCat < nRateCategory; iRateCat ++, offset += condlike_size){
					for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
						for(int iState = 0; iState < nState; iState ++)
							inIntCondlike[iPartition][iInt][offset + iPattern * nState + iState] = (double)(rand() % 10 + 1) / 101.0;
					}
				}
			}
		}
		*/
	

		int PMat_size = nState * nState;
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
			for(int iElem = 0; iElem < nState; iElem ++){
				inR[iPartition][iEigen][iElem] = (double)(rand() % 10 + 1) / 50.0;
				//inR[iPartition][iEigen][iElem] = 1.0f;
			}

			for(int iElem = 0; iElem < PMat_size; iElem ++){
				//inU[iPartition][iEigen][iElem] = (double)(rand() % 10 + 1) / 101.0;
				//inV[iPartition][iEigen][iElem] = (double)(rand() % 10 + 1) / 101.0;
				//inU[iPartition][iEigen][iElem] = 1.0;
				//inV[iPartition][iEigen][iElem] = 0.1;
				inU[iPartition][iEigen][iElem] = double(rand() % nState + 1) * (iEigen + 1) * 0.1;
				inV[iPartition][iEigen][iElem] = double(rand() % nState + 1) * (iEigen + 1) * 0.1;
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
						pPMat[iPartition][iElem] = double(rand() % 10 + 1) / 101.0;
					}
				}
			}
		}
		*/
	}
}



// Tree 1: the leaf node is only at the bottom layer, and each node has two children;
CuLTreeNode* buildTree1(int iPartition, int nNode)
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

		if(curNode->child[0] != NULL){
			sonId[iPartition][curNode->label][0] = curNode->child[0]->label;
			sonId[iPartition][curNode->label][1] = curNode->child[1]->label;
		}

		root = curNode;
		nodeVec.push_back(curNode);
	}

#ifdef DEBUG
	printf("nodeIdToArrayId[] is:\n");
#endif
	int cntIntNode = 0;
	for(int iNode = 0; iNode < nNode; iNode ++){
		nodeIdToArrayId[iPartition][iNode] = iNode;

		if(iNode >= leafThreshold){
			intCondlikeNodeId[iPartition][cntIntNode] = iNode;
			if(cntIntNode < nNodeScaler){
				nodeScalerInd[iPartition][cntIntNode] = iNode;
				scalerIndMap[iPartition][iNode] = cntIntNode;
			}
			cntIntNode ++;
		}

#ifdef DEBUG
		printf("%d: %d\n", iNode, nodeIdToArrayId[iPartition][iNode]);
#endif
	}

	assert(cntIntNode == nIntCondlike);
#ifdef DEBUG
	printf("\n\n");
#endif

	return root;
}


CuLTreeNode* buildTree1_unrooted(int iPartition, int nNode)
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

		if(curNode->child[0] != NULL){
			sonId[iPartition][curNode->label][0] = curNode->child[0]->label;
			sonId[iPartition][curNode->label][1] = curNode->child[1]->label;
		}

		nodeQue.push(curNode);
		root = curNode;
		ind ++;
	}

	root->nChild = 3;
	root->child[2] = nodeVec[nTip - 1];

	sonId[iPartition][root->label][2] = root->child[2]->label;
	
#ifdef DEBUG
	printf("nodeIdToArrayId[] is:\n");
#endif
	int cntIntNode = 0;
	for(int iNode = 0; iNode < nNode; iNode ++){
		nodeIdToArrayId[iPartition][iNode] = iNode;

		if(iNode >= nTip){
			intCondlikeNodeId[iPartition][cntIntNode] = iNode;
			if(cntIntNode < nNodeScaler){
				nodeScalerInd[iPartition][cntIntNode] = iNode;
				scalerIndMap[iPartition][iNode] = cntIntNode;
			}
			cntIntNode ++;
		}

#ifdef DEBUG
		printf("%d: %d\n", iNode, nodeIdToArrayId[iPartition][iNode]);
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
CuLTreeNode* buildTree2(int iPartition, int nNode)
{
	std::vector<CuLTreeNode *> nodeVec;
	CuLTreeNode *root = NULL;
	for(int iNode = 0; iNode < nNode; iNode ++){
		CuLTreeNode *curNode = new CuLTreeNode(iNode, 0);
		if(iNode > 0 && iNode % 2 == 0){			// Internal node;
			curNode->nChild = 2;
			curNode->child[0] = nodeVec[iNode-2];
			curNode->child[1] = nodeVec[iNode-1];

			sonId[iPartition][curNode->label][0] = curNode->child[0]->label;
			sonId[iPartition][curNode->label][1] = curNode->child[1]->label;
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
			nodeIdToArrayId[iPartition][iNode] = cntTipNode;
			cntTipNode ++;
		}
		else{
			nodeIdToArrayId[iPartition][iNode] = nTipNode + cntIntNode;
			intCondlikeNodeId[iPartition][cntIntNode] = iNode;

			if(cntIntNode < nNodeScaler){
				nodeScalerInd[iPartition][cntIntNode] = iNode;
				scalerIndMap[iPartition][iNode] = cntIntNode;
			}
			cntIntNode ++;
		}
#ifdef DEBUG_CONDLIKE
		printf("%d: %d\n", iNode, nodeIdToArrayId[iPartition][iNode]);
#endif
	}
#ifdef DEBUG_CONDLIKE
	printf("\n\n");
#endif

	assert(cntTipNode == nTipNode && cntTipNode + cntIntNode == nNode);

	return root;
}


CuLTreeNode* buildTree2_unrooted(int iPartition, int nNode)
{
	std::vector<CuLTreeNode *> nodeVec;
	CuLTreeNode *root = NULL;
	for(int iNode = 0; iNode < nNode; iNode ++){
		CuLTreeNode *curNode = new CuLTreeNode(iNode, 0);
		if(iNode > 0 && iNode % 2 == 0){			// Internal node;
			curNode->nChild = 2;
			curNode->child[0] = nodeVec[iNode-2];
			curNode->child[1] = nodeVec[iNode-1];

			sonId[iPartition][curNode->label][0] = curNode->child[0]->label;
			sonId[iPartition][curNode->label][1] = curNode->child[1]->label;
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

	sonId[iPartition][root->label][2] = root->child[2]->label;

#ifdef DEBUG_CONDLIKE
	printf("nodeIdToArrayId is:\n");
#endif
	int cntTipNode = 0, cntIntNode = 0;
	int nTipNode = nNode / 2 + 1;
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(iNode == 0 || iNode % 2 == 1){
			nodeIdToArrayId[iPartition][iNode] = cntTipNode;
			cntTipNode ++;
		}
		else{
			nodeIdToArrayId[iPartition][iNode] = nTipNode + cntIntNode;
			intCondlikeNodeId[iPartition][cntIntNode] = iNode;

			if(cntIntNode < nNodeScaler){
				nodeScalerInd[iPartition][cntIntNode] = iNode;
				scalerIndMap[iPartition][iNode] = cntIntNode;
			}
			cntIntNode ++;
		}
#ifdef DEBUG_CONDLIKE
		printf("%d: %d\n", iNode, nodeIdToArrayId[iPartition][iNode]);
#endif
	}
#ifdef DEBUG_CONDLIKE
	printf("\n\n");
#endif

	assert(cntTipNode == nTipNode && cntTipNode + cntIntNode == nNode);

	return root;
}



// Tree 3: the leaf node is only at the bottom layer, and each node has two children, one is tip state, the other is tip condlike
CuLTreeNode* buildTree3(int iPartition, int nNode)
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

		if(curNode->child[0] != NULL){
			sonId[iPartition][curNode->label][0] = curNode->child[0]->label;
			sonId[iPartition][curNode->label][1] = curNode->child[1]->label;
		}

		root = curNode;
		nodeVec.push_back(curNode);
	}

#ifdef DEBUG
	printf("nodeIdToArrayId[] is:\n");
#endif
	int cntIntNode = 0, cntTipState = 0, cntTipCondlike = 0;
	for(int iNode = 0; iNode < nNode; iNode ++){
		nodeIdToArrayId[iPartition][iNode] = iNode;
		if(iNode < leafThreshold){
			if(iNode % 2 == 0){
				nodeIdToArrayId[iPartition][iNode] = cntTipState;
				cntTipState ++;
			}
			else{
				nodeIdToArrayId[iPartition][iNode] = nTipState + cntTipCondlike;
				cntTipCondlike ++;
			}
		}
		else{
			intCondlikeNodeId[iPartition][cntIntNode] = iNode;
			if(cntIntNode < nNodeScaler){
				nodeScalerInd[iPartition][cntIntNode] = iNode;
				scalerIndMap[iPartition][iNode] = cntIntNode;
			}
			cntIntNode ++;
		}

#ifdef DEBUG
		printf("%d: %d\n", iNode, nodeIdToArrayId[iPartition][iNode]);
#endif
	}

	assert(cntIntNode == nIntCondlike && cntTipState == nTipState && cntTipCondlike == nTipCondlike);
#ifdef DEBUG
	printf("\n\n");
#endif

	return root;
}


// Unrooted version of tree 3
CuLTreeNode* buildTree3_unrooted(int iPartition, int nNode)
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

		if(curNode->child[0] != NULL){
			sonId[iPartition][curNode->label][0] = curNode->child[0]->label;
			sonId[iPartition][curNode->label][1] = curNode->child[1]->label;
		}

		nodeQue.push(curNode);
		root = curNode;
		ind ++;
	}

	root->nChild = 3;
	root->child[2] = nodeVec[nTip - 1];

	sonId[iPartition][root->label][2] = root->child[2]->label;
	
#ifdef DEBUG
	printf("nodeIdToArrayId[] is:\n");
#endif
	int cntIntNode = 0, cntTipState = 0, cntTipCondlike = 0;
	for(int iNode = 0; iNode < nNode; iNode ++){
		nodeIdToArrayId[iPartition][iNode] = iNode;
		if(iNode < nTip){
			if(iNode % 2 == 0){
				nodeIdToArrayId[iPartition][iNode] = cntTipState;
				cntTipState ++;
			}
			else{
				nodeIdToArrayId[iPartition][iNode] = nTipState + cntTipCondlike;
				cntTipCondlike ++;
			}
		}
		else{
			intCondlikeNodeId[iPartition][cntIntNode] = iNode;
			if(cntIntNode < nNodeScaler){
				nodeScalerInd[iPartition][cntIntNode] = iNode;
				scalerIndMap[iPartition][iNode] = cntIntNode;
			}
			cntIntNode ++;
		}

#ifdef DEBUG
		printf("%d: %d\n", iNode, nodeIdToArrayId[iPartition][iNode]);
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


void printNodeRelation(int iPartition)
{
	printf("\n==============\nNode relation for %d-th partition:\n", iPartition);
	for(int iNode = 0; iNode < nNode; iNode ++){
		if(sonId[iPartition][iNode][0] == 0 && sonId[iPartition][iNode][1] == 0)
			continue;

		printf("Son of node %d: %d, %d", iNode, sonId[iPartition][iNode][0], sonId[iPartition][iNode][1]);
		if(!isRooted && iNode == rootId[iPartition])
			printf(", %d\n", sonId[iPartition][iNode][2]);
		else
			printf("\n");
	}
	printf("==================\n");
}


void calculatePMat_CPU(int iPartition, const int nNodeForPMat, const int *nodeId, const int nEigen, const int nRateCategory, const int *eigenDecompId)
{
	double *newBrLen = new double[nRateCategory * nNode];
	int ind = 0;
	const int nNode = nTipState + nTipCondlike + nIntCondlike;
	//printf("nNodeForPMat = %d, nEigen = %d, nRateCat = %d, nTipState = %d, nTipCondlike = %d, nIntCondlike = %d\n", nNodeForPMat, nEigen, nRateCategory, nTipState, nTipCondlike, nIntCondlike);
	for(int iRate = 0; iRate < nRateCategory; iRate ++){
		for(int iNode = 0; iNode < nNode; iNode ++, ind ++){
			newBrLen[ind] = inRate[iPartition][iRate] * inBrLen[iPartition][iNode];
		}
	}

	double *pPMat = NULL, *pU, *pV, *pR;
	const int PMat_size = nState * nState;
	const int PMat_eigen_size = nRateCategory * PMat_size;
	for(int iNode = 0; iNode < nNodeForPMat; iNode ++){
		//int curNode = nodeId[iNode];
		for(int iEigen = 0; iEigen < nEigen; iEigen ++){
			pU = inU[iPartition][eigenDecompId[iEigen]];
			pV = inV[iPartition][eigenDecompId[iEigen]];
			pR = inR[iPartition][eigenDecompId[iEigen]];
			pPMat = inPMat[iPartition][nodeId[iNode]] + iEigen * PMat_eigen_size;

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

	/*
#ifdef USE_CODEML_PMAT
	// Transpose PMat:
	for(int iNode = 0; iNode < nNodeForPMat; iNode ++){
		transposeMatrix(inPMat[iPartition][nodeId[iNode]], nEigenDecomp * nRateCategory, nState, nState);
	}
#endif
	*/
}


void comparePMat(int iPartition, const int nNodeForPMat, const int nEigen)
{
	const int PMat_size = nState * nState;
	int offset = 0, cntError = 0;
	double *pPMat_GPU, *pPMat_CPU;
	for(int iNode = 0; iNode < nNodeForPMat; iNode ++){
		offset = 0;
		for(int iEigen = 0; iEigen < nEigen; iEigen ++){
			for(int iRate = 0; iRate < nRateCategory; iRate ++, offset += PMat_size){
				pPMat_GPU = outPMat[iPartition][iNode] + offset;
				pPMat_CPU = inPMat[iPartition][iNode] + offset;

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



void calculateCondlike_CPU(int iPartition, CuLTreeNode *root, const int nIntCondlike, const int *nodeId, const int nEigen, const int *eigenDecompId)
{
	std::set<int> nodeSet;
	for(int iNode = 0; iNode < nIntCondlike; iNode ++){
		nodeSet.insert(nodeId[iNode]);
	}

	std::set<int> scalerNodeSet;
	for(int iNode = 0; iNode < nNodeScaler; iNode ++)
		scalerNodeSet.insert(nodeScalerInd[iPartition][iNode]);

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
		int arrayId_F = nodeIdToArrayId[iPartition][curNode->label] - nTipNode;
		for(int iChild = 0; iChild < curNode->nChild; iChild ++){
			CuLTreeNode *curChild = curNode->child[iChild];
			int childId = nodeIdToArrayId[iPartition][curChild->label];
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
				pCondlike_F = inIntCondlike[iPartition][arrayId_F] + condlike_cat_offset;
				
				if(curNode->nChild == 2){
					pPMat_L = inPMat[iPartition][PMat_id[0]] + PMat_cat_offset;
					pPMat_R = inPMat[iPartition][PMat_id[1]] + PMat_cat_offset;

					if(whichCase[0] == 1)
						pCondlike_L = inTipState[iPartition][condlike_id[0]];
					else if(whichCase[0] == 2)
						pCondlike_L = inTipCondlike[iPartition][condlike_id[0]];
					else
						pCondlike_L = inIntCondlike[iPartition][condlike_id[0]] + condlike_cat_offset;

					if(whichCase[1] == 1)
						pCondlike_R = inTipState[iPartition][condlike_id[1]];
					else if(whichCase[1] == 2)
						pCondlike_R = inTipCondlike[iPartition][condlike_id[1]];
					else
						pCondlike_R = inIntCondlike[iPartition][condlike_id[1]] + condlike_cat_offset;

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
						pPMat_S = inPMat[iPartition][PMat_id[iChild]] + PMat_cat_offset;
						if(whichCase[iChild] == 1){
							pCondlike_S = inTipState[iPartition][condlike_id[iChild]];
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
								pCondlike_S = inTipCondlike[iPartition][condlike_id[iChild]];
							else
								pCondlike_S = inIntCondlike[iPartition][condlike_id[iChild]] + condlike_cat_offset;

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
			nodeScale_CPU(inIntCondlike[iPartition][arrayId_F], scaleFactor[iPartition] + scalerIndMap[iPartition][curNode->label] * nSitePattern);
		}
	}
}


void compareCondlike(int iPartition, const int nNode, const int *nodeId, const int nEigen, const int *eigenDecompId)
{
	const int nTipNode = nTipState + nTipCondlike;
	const int condlike_size = nSitePattern * nState;
	const int condlike_eigen_size = nRateCategory * condlike_size;
	//const int condlike_node_size = nEigenDecomp * condlike_eigen_size;

	double *pCondlike_CPU, *pCondlike_GPU;
	int cntError = 0;
	for(int iNode = 0; iNode < nNode; iNode ++){
		int arrayId = nodeIdToArrayId[iPartition][nodeId[iNode]] - nTipNode;
		//printf("nodeIdToArrayId[%d] = %d, arrayId = %d\n", iNode, nodeIdToArrayId[iNode], arrayId);

		for(int iEigen = 0; iEigen < nEigen; iEigen ++){
			int offset = eigenDecompId[iEigen] * condlike_eigen_size;

			for(int iRate = 0; iRate < nRateCategory; iRate ++, offset += condlike_size){
				pCondlike_CPU = inIntCondlike[iPartition][arrayId] + offset;
				pCondlike_GPU = outIntCondlike[iPartition][arrayId] + offset;

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


double calculateLikelihood_CPU(int iPartition, double *rootCondlike)
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
					stateSum += pRootCondlike[iState] * inStateFreq[iPartition][iState];
				}
				
				catSum += stateSum * inRateCatWeight[iPartition][iEigen][iRateCat];
			}
		}

		if(catSum <= 0)
			catSum = CUFLT_MIN;
		catSum = log(catSum);

		if(doScale){
			for(int iNode = 0; iNode < nNodeScaler; iNode ++){
				catSum += scaleFactor[iPartition][iPattern + iNode * nSitePattern];
			}
		}

		inSiteLnL[iPartition][iPattern] = catSum;
		lnL += catSum * inSitePatternWeight[iPartition][iPattern];
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


void compareSiteLikelihood(int iPartition)
{
	int cntError = 0;
	for(int iPattern = 0; iPattern < nSitePattern; iPattern ++){
		if(fabs(inSiteLnL[iPartition][iPattern] - outSiteLnL[iPartition][iPattern]) > EPS){
			printf("Error! siteLnL_CPU[%d] = %.12lf, siteLnL_GPU[%d] = %.12lf\n", iPattern, exp(inSiteLnL[iPartition][iPattern]), iPattern, exp(outSiteLnL[iPartition][iPattern]));
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
		printf("Likelihood error! lnL_CPU = %f, lnL_GPU = %f\n", lnL_CPU, lnL_GPU);
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


// Call CuLib for the calculation of likelihood:
void CuLibImpl()
{
	// Initialize instance:
	CuLErrorCode returnState;
	int instanceId;

	returnState = CuLInitializeCuLInstance(nPartition, 
											deviceId, 
											instanceId);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
	//printf("instanceId = %d\n", instanceId);
#ifdef DEBUG
	printf("No error in CuLInitializeCuLInstance()\n");
#endif

	// Init values for each partition:
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		 nEigenDecomp = eigenDecompCountArray[iPartition % nPartition];
		 nRateCategory = rateCatCountArray[iPartition % nPartition];
		 //nRateCategory = 1;

		 nSitePattern = sitePatternCountArray[iPartition % nPartition];
		 
		 nTipState = tipStateCountArray[iPartition % nPartition];
		 nTipCondlike = tipCondlikeCountArray[iPartition % nPartition];
		 nIntCondlike = intCondlikeCountArray[iPartition % nPartition];

	
		// Specify params for the current partition:
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

		// Specify tip state and tip condlike
		if(nTipState > 0){
			returnState = CuLSpecifyPartitionTipState(instanceId,
												iPartition,
												nTipState, 
												tipStateNodeId[iPartition], 
												(const int **)inTipState[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLSpecifyPartitionTipState()\n");
#endif
		}


		// Set tip condlike:
		if(nTipCondlike > 0){
			returnState = CuLSpecifyPartitionTipCondlike(instanceId,
													iPartition,
													nTipCondlike, 
													tipCondlikeNodeId[iPartition], 
													(const double **)inTipCondlike[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLSpecifyPartitionTipCondlike()\n");
#endif
		}
	}	// End for(iPartition ...)


	// Calculate likelihood for nIteration times:
	int nNodeForPMat = nNode - 1, *nodeId = new int[nNodeForPMat];
	double *curBrLen = new double[nNodeForPMat];

	struct timeval tBegin;
	timeBegin(&tBegin);
	for(int itr = 0; itr < nIteration; itr ++){
		for(int iPartition = 0; iPartition < nPartition; iPartition ++){

			nEigenDecomp = eigenDecompCountArray[iPartition % nPartition];
			nRateCategory = rateCatCountArray[iPartition % nPartition];

			nSitePattern = sitePatternCountArray[iPartition % nPartition];
		 
			nTipState = tipStateCountArray[iPartition % nPartition];
			nTipCondlike = tipCondlikeCountArray[iPartition % nPartition];
			nIntCondlike = intCondlikeCountArray[iPartition % nPartition];

			// Set tree topology:
			returnState = CuLSpecifyPartitionTreeTopology(instanceId,
													iPartition,
													partitionRoot[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLSpecifyPartitionTreeTopology()\n");
#endif


			// Set state frequency:
			returnState = CuLSpecifyPartitionStateFrequency(instanceId,
													iPartition,
													inStateFreq[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLSpecifyPartitionStateFrequency()\n");
#endif

			// Set pattern weight:
			returnState = CuLSpecifyPartitionSitePatternWeight(instanceId,
															iPartition,
															inSitePatternWeight[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLSpecifyPartitionSitePatternWeight()\n");
#endif


			// Set rates:
			returnState = CuLSpecifyPartitionRate(instanceId,
												iPartition,
												inRate[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLSpecifyPartitionRate()\n");
#endif


			// Set rate category weights:
			returnState = CuLSpecifyPartitionRateCategoryWeight(instanceId,
															iPartition,
															nEigenDecomp,	
															eigenDecompId[iPartition], 
															(const double**)inRateCatWeight[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLSpecifyPartitionRateCategoryWeight()\n");
#endif

	
			// Set eigen decomposition:
			returnState = CuLSpecifyPartitionEigenDecomposition(instanceId,
															iPartition,
															nEigenDecomp, 
															eigenDecompId[iPartition], 
															(const double**)inU[iPartition], 
															(const double **)inV[iPartition], 
															(const double **)inR[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLSpecifyPartitionEigenDecomposition()\n");
#endif

			// Set node label to array index map:
			returnState = CuLMapPartitionNodeIndToArrayInd(instanceId,
														iPartition,
														nNode, 
														nodeIdToArrayId[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLMapPartitionNodeIndToArrayInd()\n");
#endif


			// Set node label to array index map:
			returnState = CuLSpecifyPartitionNodeScalerIndex(instanceId,
															iPartition,
															nNodeScaler,
															nodeScalerInd[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLSpecifyPartitionNodeScalerIndex()\n");
#endif

			// Calculate transition matrix:
			int curNode = 0;
			for(int iNode = 0; iNode < nNode; iNode ++){
				if(iNode == rootId[iPartition])
					continue;
		
				nodeId[curNode] = iNode;
				curBrLen[curNode] = inBrLen[iPartition][iNode];
				curNode ++;
			}

			//const int nNodeToCalc_PMat = 6;
			//int pNodeId_PMat[nNodeToCalc_PMat] = {4, 5, 6, 9, 10, 13};

	
			returnState = CuLCalculatePartitionTransitionMatrixMulti(instanceId,
														iPartition,
														nNodeForPMat, 
														nodeId, 
														nEigenDecomp, 
														eigenDecompId[iPartition], 
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
																inBrLen[iPartition]);
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
															nEigenDecomp, 
															eigenDecompId[iPartition], 
															outPMat[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLGetPartitionTransitionMatrixMulti()\n");
#endif
		*/
	
			/*
			returnState = CuLGetPartitionTransitionMatrixAll(instanceId,
															iPartition,
															nNodeForPMat, 
															nodeId, 
															outPMat[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLGetPartitionTransitionMatrixAll()\n");
#endif
			
			printValuesToFile(outPMat[iPartition], nNodeForPMat, nEigenDecomp * nRateCategory, nState, nState, "PMat_GPU.txt");
			

			
			// Calculate PMat on CPU:
			calculatePMat_CPU(iPartition, nNodeForPMat, nodeId, nEigenDecomp, nRateCategory, eigenDecompId[iPartition]);
			printValuesToFile(inPMat[iPartition], nNodeForPMat, nEigenDecomp * nRateCategory, nState, nState, "PMat_CPU.txt");
#ifdef DEBUG
			printf("No error in calculatePMat_CPU()\n");
#endif

			comparePMat(iPartition, nNodeForPMat, nEigenDecomp);
#ifdef DEBUG
			printf("No error in comparePMat()\n");
#endif
			*/

	
			// Calculate condlike of all internal nodes:
			//const int nNodeToCalc_condlike = 5;
			//int pNodeId_condlike[nNodeToCalc_condlike] = {6, 8, 10, 12, 14};
	
			/*
			returnState = CuLCalculatePartitionCondlikeMulti(instanceId,
													iPartition,
													nIntCondlike, 
													intCondlikeNodeId[iPartition], 
													nEigenDecomp, 
													eigenDecompId[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLCalculatePartitionCondlikeMulti()\n");
#endif
	*/
	
			returnState = CuLCalculatePartitionCondlikeAll(instanceId,
													iPartition,
													nIntCondlike, 
													intCondlikeNodeId[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLCalculatePartitionCondlikeAll()\n");
#endif
	
		/*
			// Copy internal condlike back:
			returnState = CuLGetPartitionIntCondlikeMulti(instanceId,
													iPartition,
													nIntCondlike, 
													intCondlikeNodeId[iPartition], 
													nEigenDecomp, 
													eigenDecompId[iPartition], 
													outIntCondlike[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLGetPartitionIntCondlikeMulti()\n");
#endif
	*/
		/*
			returnState = CuLGetPartitionIntCondlikeAll(instanceId,
												iPartition,
												nIntCondlike, 
												intCondlikeNodeId[iPartition],
												outIntCondlike[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLGetPartitionIntCondlikeAll()\n");
#endif
			*/
			/*
			printValuesToFile(outIntCondlike[iPartition], nIntCondlike, nEigenDecomp * nRateCategory, nSitePattern, nState, "condlike_GPU.txt");
			
			// Calculate condlike on CPU:
			calculateCondlike_CPU(iPartition, partitionRoot[iPartition], nIntCondlike, intCondlikeNodeId[iPartition], nEigenDecomp, eigenDecompId[iPartition]);
#ifdef DEBUG
			printf("No error in calculateCondlike_CPU()\n");
#endif
	
			printValuesToFile(inIntCondlike[iPartition], nIntCondlike, nEigenDecomp * nRateCategory, nSitePattern, nState, "condlike_CPU.txt");


			compareCondlike(iPartition, nIntCondlike, intCondlikeNodeId[iPartition], nEigenDecomp, eigenDecompId[iPartition]);
#ifdef DEBUG
			printf("No error in compareCondlike()\n");
#endif
			*/
		
			returnState = CuLCalculatePartitionLikelihoodAsync(instanceId,
													iPartition);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
#ifdef DEBUG
			printf("No error in CuLCalculatePartitionLikelihoodAsync()\n");
#endif

			/*
			returnState = CuLGetPartitionLikelihood(instanceId,
											iPartition,
											lnL_GPU[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
			printValuesToFile(&lnL_GPU[iPartition], 1, 1, "lnL_GPU.txt");
#ifdef DEBUG
		printf("No error in CuLGetPartitionLikelihood()\n");
#endif
		*/
			/*
			double *rootCondlike = inIntCondlike[iPartition][nodeIdToArrayId[iPartition][rootId[iPartition]] - nTipState - nTipCondlike];
			lnL_CPU[iPartition] = calculateLikelihood_CPU(iPartition, rootCondlike);
#ifdef DEBUG
			printf("No error in calculateLikelihood_CPU()\n");
#endif

			compareLikelihood(&lnL_CPU[iPartition], lnL_GPU[iPartition]);
			*/

			/*
			// Copy site likelihood back:
			returnState = CuLGetPartitionSiteLikelihood(instanceId,
												iPartition,
												outSiteLnL[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
			printValuesToFile(outSiteLnL[iPartition], nSitePattern, 1, "siteLnL_GPU.txt");
#ifdef DEBUG
			printf("No error in CuLGetPartitionSiteLikelihood()\n");
#endif

			//compareSiteLikelihood(iPartition);
		*/

		}	// End for(iPartition ...)


		// Get the final likelihood for each partition from GPU:
		for(int iPartition = 0; iPartition < nPartition; iPartition ++){
			returnState = CuLGetPartitionLikelihood(instanceId,
											iPartition,
											lnL_GPU[iPartition]);
			if(CUL_SUCCESS != returnState)
				printErrorCode(returnState);
			//printValuesToFile(&lnL_GPU[iPartition], 1, 1, "lnL_GPU.txt");
		}	// End for(iPartition ...)
	}	// End for(itr ...)

	time_lnL_cuLib = timeEnd(tBegin);

	// Finalize the instance:
	returnState = CuLFinalizeInstance(instanceId);
	//printf("instanceId = %d\n", instanceId);
	if(CUL_SUCCESS != returnState)
		printErrorCode(returnState);
#ifdef DEBUG
	printf("No error in CuLFinalizeInstance()\n");
#endif

	if(curBrLen){
		free(curBrLen);
		curBrLen = NULL;
	}

	if(nodeId){
		free(nodeId);
		nodeId = NULL;
	}
}


// Call CPU for the calculation:
void CPUImpl()
{
	// Calculate the likelihood from the begining on CPU:

	int nNodeForPMat = nNode - 1, *nodeId = new int[nNodeForPMat];
	double *curBrLen = new double[nNodeForPMat];

	for(int itr = 0; itr < 1; itr ++){
		for(int iPartition = 0; iPartition < nPartition; iPartition ++){
			nEigenDecomp = eigenDecompCountArray[iPartition % nPartition];
			nRateCategory = rateCatCountArray[iPartition % nPartition];

			nSitePattern = sitePatternCountArray[iPartition % nPartition];
		 
			nTipState = tipStateCountArray[iPartition % nPartition];
			nTipCondlike = tipCondlikeCountArray[iPartition % nPartition];
			nIntCondlike = intCondlikeCountArray[iPartition % nPartition];

			// Calculate transition matrix
			int curNode = 0;
			for(int iNode = 0; iNode < nNode; iNode ++){
				if(iNode == rootId[iPartition])
					continue;
		
				nodeId[curNode] = iNode;
				curBrLen[curNode] = inBrLen[iPartition][iNode];
				curNode ++;
			}


			// Calculate PMat on CPU:
			calculatePMat_CPU(iPartition, nNodeForPMat, nodeId, nEigenDecomp, nRateCategory, eigenDecompId[iPartition]);
			//printValuesToFile(inPMat[iPartition], nNodeForPMat, nEigenDecomp * nRateCategory, nState, nState, "PMat_CPU.txt");
#ifdef DEBUG
			printf("No error in calculatePMat_CPU()\n");
#endif

			// Calculate condlike on CPU:
			calculateCondlike_CPU(iPartition, partitionRoot[iPartition], nIntCondlike, intCondlikeNodeId[iPartition], nEigenDecomp, eigenDecompId[iPartition]);
			//printValuesToFile(inIntCondlike[iPartition], nIntCondlike, nEigenDecomp * nRateCategory, nSitePattern, nState, "condlike_CPU.txt");
#ifdef DEBUG
			printf("No error in calculateCondlike_CPU()\n");
#endif

			// Calculate likelihood on CPU:
			double *rootCondlike = inIntCondlike[iPartition][nodeIdToArrayId[iPartition][rootId[iPartition]] - nTipState - nTipCondlike];
			lnL_CPU[iPartition] = calculateLikelihood_CPU(iPartition, rootCondlike);
			//printValuesToFile(inSiteLnL[iPartition], nSitePattern, 1, "siteLnL_CPU.txt");
			//printValuesToFile(&lnL_CPU[iPartition], 1, 1, "lnL_CPU.txt");
#ifdef DEBUG
			printf("No error in calculateLikelihood_CPU()\n");
#endif
		}
	}

	if(curBrLen){
		free(curBrLen);
		curBrLen = NULL;
	}

	if(nodeId){
		free(nodeId);
		nodeId = NULL;
	}
}



/*
// Use a single beagle instance for all partitions, beagleUpdatePartialsByPartition() is needed, but not available in version 2.1.3(the version in 26);
void BEAGLEImpl_intergrate(){

	// For beagle intergrate implemention, make sure the eigen decomposition count and rate category count is the same for all partitions:
	int nTotalSitePattern = 0;
	
	nEigenDecomp = eigenDecompCountArray[0];
	nRateCategory = rateCatCountArray[0];

	nTipState = tipStateCountArray[0];
	nTipCondlike = tipCondlikeCountArray[0];
	nIntCondlike = intCondlikeCountArray[0];

	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		nTotalSitePattern += sitePatternCountArray[iPartition % nPartition];
		
		assert(eigenDecompCountArray[iPartition % nPartition] == nEigenDecomp);
		assert(rateCatCountArray[iPartition % nPartition] == nRateCategory);
		assert(tipStateCountArray[iPartition % nPartition] == nTipState);
		assert(tipCondlikeCountArray[iPartition % nPartition] == nTipCondlike);
		assert(intCondlikeCountArray[iPartition % nPartition] == nIntCondlike);
	}


	// initialize the instance
    BeagleInstanceDetails instDetails;
   
    // create an instance of the BEAGLE library
	int beagleInstance = beagleCreateInstance(
								  nTipState + nTipCondlike,
                                  (nTipCondlike + nIntCondlike) * nEigenDecomp,  
                                  nTipState,
                                  nState,
                                  nTotalSitePattern,
                                  nEigenDecomp * nPartition,
                                  nNode * nEigenDecomp * nPartition,
								  nRateCategory,		
                                  nNodeScaler * nPartition,  
                                  &deviceId,			 
                                  1,			  
								  BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_SCALING_MANUAL ,   
                                  0 | BEAGLE_FLAG_PRECISION_DOUBLE | BEAGLE_FLAG_FRAMEWORK_CUDA | BEAGLE_FLAG_SCALERS_LOG,  
                                  &instDetails);
    if (beagleInstance < 0) {
	    fprintf(stderr, "Failed to obtain BEAGLE instance\n\n");
	    exit(1);
    }

	int rNumber = instDetails.resourceNumber;
    fprintf(stdout, "Using resource %i:\n", rNumber);
    fprintf(stdout, "\tRsrc Name : %s\n",instDetails.resourceName);
    fprintf(stdout, "\tImpl Name : %s\n", instDetails.implName);
    fprintf(stdout, "\tImpl Desc : %s\n", instDetails.implDescription);
    fprintf(stdout, "\tFlags:");
    printFlags(instDetails.flags);
    fprintf(stdout, "\n\n");


	// Set tip states:
	int *inState = new int[nTotalSitePattern], offset;
	for(int iTipState = 0; iTipState < nTipState; iTipState ++){
		offset = 0;
		for(int iPartition = 0; iPartition < nPartition; iPartition ++){
			int count = sitePatternCountArray[iPartition];
			memcpy(inState + offset, inTipState[iPartition][iTipState], count * sizeof(int));
			offset += count;
		}

		beagleSetTipStates(beagleInstance, tipStateNodeId[0][iTipState], inState);
	}

	if(inState){
		free(inState);
		inState = NULL;
	}


	// Set tip partials:
	double *inPartial = new double[nTotalSitePattern * nState];
	for(int iTipCondlike = 0; iTipCondlike < nTipCondlike; iTipCondlike ++){
		offset = 0;
		for(int iPartition = 0; iPartition < nPartition; iPartition ++){
			int count = sitePatternCountArray[iPartition] * nState;
			memcpy(inPartial + offset, inTipCondlike[iPartition][iTipCondlike], count * sizeof(double));
			offset += count;
		}

		beagleSetTipPartials(beagleInstance, tipCondlikeNodeId[iPartition][iTipCondlike] + nTipState, inPartial);
	}

	if(inPartial){
		free(inPartial);
		inPartial = NULL;
	}


	// Set category rates, assert rates are the same for all partitions:
	beagleSetCategoryRates(beagleInstance, inRate[0]);


	// Set state freq && rate category weights && eigen decomposition:
	int ind = 0;
	for(int iPartition = 0, ; iPartition < nPartition; iPartition ++){
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++, ind ++){
			beagleSetStateFrequencies(beagleInstance, ind, inStateFreq[iPartition]);
			beagleSetCategoryWeights(beagleInstance, ind, inRateCatWeight[iPartition][iEigen]);
			beagleSetEigenDecomposition(beagleInstance, ind, inU[iPartition][iEigen], inV[iPartition][iEigen], inR[iPartition][iEigen]);
		}
	}


	// Set site pattern weights:
	int *inPatternWeight = new int[nTotalSitePattern];
	offset = 0;
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		int count = sitePatternCountArray[iPattern];
		memcpy(inPatternWeight + offset, inSitePatternWeight[iPartition], count * sizeof(int));
	}

	beagleSetPatternWeights(beagleInstance, inPatternWeight);

	if(inPatternWeight){
		free(inPatternWeight);
		inPatternWeight = NULL;
	}



	// Calculate likelihood from the begining with BEAGLE:
	for(int itr = 0; itr < nIteration; itr ++){
		
		// Calculate PMat for each eigen decomposition of each partition:
		int nNodeForPMat = nNode - 1, *nodeId = new int[nNodeForPMat];
		double *curBrLen = new double[nNodeForPMat];

		ind = 0;
		for(int iPartition = 0; iPartition < nPartition; iPartition ++){
			int curNode = 0;
			for(int iNode = 0; iNode < nNode; iNode ++){
				if(iNode == rootId[iPartition])
					continue;
		
				//nodeId[curNode] = iNode;
				curBrLen[curNode] = inBrLen[iPartition][iNode];
				curNode ++;
			}

			int curOffset = iPartition * nEigenDecomp * nNode;
			for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++, ind ++, curOffset += nNode){
				curNode = 0;
				for(int iNode = 0; iNode < nNode; iNode ++){
					if(iNode == rootId[iPartition])
					continue;
		
					nodeId[curNode] = iNode + curOffset;
					curNode ++;
				}

				beagleUpdateTransitionMatrices(beagleInstance,
												ind,
												nodeId,
												NULL,
												NULL,
												curBrLen,
												nNodeForPMat);

			}
		}


		// Set operations to update condlike:
		const int nOp = nIntCondlike * nEigenDecomp;
		BeagleOperation *operations = new BeagleOperation[nOp];
	}
}
*/

// Use a single beagle instance for each partition;
void BEAGLEImpl_seperate(){

	//printf("Going to call BEAGLE for calculation:\n");

	int *beagleInstance = new int [nPartition];
	int gpuId = deviceId + 1;		// For beagle, device id starts from 1.

	// initialize the instance
    BeagleInstanceDetails instDetails;
   
    // create an instance of the BEAGLE library for each partition:
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		
		nSitePattern = sitePatternCountArray[iPartition % nPartition];
		nEigenDecomp = eigenDecompCountArray[iPartition % nPartition];
		nRateCategory = rateCatCountArray[iPartition % nPartition];

		nTipState = tipStateCountArray[iPartition % nPartition];
		nTipCondlike = tipCondlikeCountArray[iPartition % nPartition];
		nIntCondlike = intCondlikeCountArray[iPartition % nPartition];

		beagleInstance[iPartition] = beagleCreateInstance(
								  nTipState + nTipCondlike,	
                                  nTipCondlike + nIntCondlike * nEigenDecomp, 
                                  nTipState,	
                                  nState,	
                                  nSitePattern,	
                                  nEigenDecomp,	
                                  nNode * nEigenDecomp,	
								  nRateCategory,	
                                  nNodeScaler * nEigenDecomp,  
                                  &gpuId,
                                  1,	
								  BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_SCALING_MANUAL ,  
                                  0 | BEAGLE_FLAG_PRECISION_DOUBLE | BEAGLE_FLAG_FRAMEWORK_CUDA | BEAGLE_FLAG_SCALERS_LOG, 
                                  &instDetails);
		if (beagleInstance[iPartition] < 0) {
			fprintf(stderr, "Failed to obtain BEAGLE instance\n\n");
			exit(1);
		}

		int rNumber = instDetails.resourceNumber;
		fprintf(stdout, "Using resource %i:\n", rNumber);
		fprintf(stdout, "\tRsrc Name : %s\n",instDetails.resourceName);
		fprintf(stdout, "\tImpl Name : %s\n", instDetails.implName);
		fprintf(stdout, "\tImpl Desc : %s\n", instDetails.implDescription);
		fprintf(stdout, "\tFlags:");
		//printFlags(instDetails.flags);
		fprintf(stdout, "\n\n");


		// Set tip states:
		for(int iTipState = 0; iTipState < nTipState; iTipState ++)
			beagleSetTipStates(beagleInstance[iPartition], tipStateNodeId[iPartition][iTipState], inTipState[iPartition][iTipState]);


		// Set tip condlikes:
		for(int iTipCondlike = 0; iTipCondlike < nTipCondlike; iTipCondlike ++){
			beagleSetTipPartials(beagleInstance[iPartition], tipCondlikeNodeId[iPartition][iTipCondlike] + nTipState, inTipCondlike[iPartition][iTipCondlike]);
		}
	}	// End for(iPartition ...)


	// Calculate likelihood for nIteration times
	int nNodeForPMat = nNode - 1, *nodeId = new int[nNodeForPMat];
	double *curBrLen = new double[nNodeForPMat];

	int curNode;
	struct timeval tBegin;
	timeBegin(&tBegin);
	for(int itr = 0; itr < nIteration; itr ++){
		for(int iPartition = 0; iPartition < nPartition; iPartition ++){

			nSitePattern = sitePatternCountArray[iPartition % nPartition];
			nEigenDecomp = eigenDecompCountArray[iPartition % nPartition];
			nRateCategory = rateCatCountArray[iPartition % nPartition];

			nTipState = tipStateCountArray[iPartition % nPartition];
			nTipCondlike = tipCondlikeCountArray[iPartition % nPartition];
			nIntCondlike = intCondlikeCountArray[iPartition % nPartition];

			// Set category rates, assert rates are the same for all partitions:
			beagleSetCategoryRates(beagleInstance[iPartition], inRate[iPartition]);

			// Set state freq && rate category weights && eigen decomposition:
			int ind = 0;
			for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++, ind ++){
				beagleSetStateFrequencies(beagleInstance[iPartition], ind, inStateFreq[iPartition]);
				beagleSetCategoryWeights(beagleInstance[iPartition], ind, inRateCatWeight[iPartition][iEigen]);
				beagleSetEigenDecomposition(beagleInstance[iPartition], ind, inU[iPartition][iEigen], inV[iPartition][iEigen], inR[iPartition][iEigen]);
			}

			// Set pattern weights:
			beagleSetPatternWeights(beagleInstance[iPartition], inSitePatternWeight[iPartition]);

			// Set branch length of nodes:
			curNode = 0;
			for(int iNode = 0; iNode < nNode; iNode ++){
				if(iNode == rootId[iPartition])
					continue;

				curBrLen[curNode] = inBrLen[iPartition][iNode];
				curNode ++;
			}

			// Set node index and update PMat for all eigen decomposition:
			// 假设PMat组织方式为[iNode][iEigen]，也即同一个node的不同eigen的PMat相邻；
			for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
				curNode = 0;
				for(int iNode = 0; iNode < nNode; iNode ++){
					if(iNode == rootId[iPartition])
					continue;
		
					nodeId[curNode] = iNode * nEigenDecomp + iEigen;
					curNode ++;
				}

				beagleUpdateTransitionMatrices(beagleInstance[iPartition],
												eigenDecompId[iPartition][iEigen],
												nodeId,
												NULL,
												NULL,
												curBrLen,
												nNodeForPMat);
			}	// End for(iEigen = ...)


			// Set operations and update condlike:
			int *cumulativeScalingIndices = new int[nEigenDecomp];
			int nTipNode = nTipState + nTipCondlike;
			BeagleOperation *operations = new BeagleOperation[nIntCondlike];

			int nOp;
			for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
				cumulativeScalingIndices[iEigen] = BEAGLE_OP_NONE;
				
				nOp = 0;
				for(int iNode = 0; iNode < nIntCondlike; iNode ++, nOp ++){
					curNode = intCondlikeNodeId[iPartition][iNode];
					int child[2], childArrayId[2];
					child[0] = sonId[iPartition][curNode][0];
					child[1] = sonId[iPartition][curNode][1];
					childArrayId[0] = nodeIdToArrayId[iPartition][child[0]];
					childArrayId[1] = nodeIdToArrayId[iPartition][child[1]];

					operations[nOp].destinationPartials = nTipNode + (nodeIdToArrayId[iPartition][curNode] - nTipNode) * nEigenDecomp + iEigen;
					operations[nOp].destinationScaleWrite = BEAGLE_OP_NONE;
					operations[nOp].destinationScaleRead = BEAGLE_OP_NONE;
					
					operations[nOp].child1TransitionMatrix = child[0] * nEigenDecomp + iEigen;
					operations[nOp].child2TransitionMatrix = child[1] * nEigenDecomp + iEigen;

					if(childArrayId[0] < nTipNode)
						operations[nOp].child1Partials = childArrayId[0];
					else
						operations[nOp].child1Partials = nTipNode + (childArrayId[0] - nTipNode) * nEigenDecomp + iEigen;

					if(childArrayId[1] < nTipNode)
						operations[nOp].child2Partials = childArrayId[1];
					else
						operations[nOp].child2Partials = nTipNode + (childArrayId[1] - nTipNode) * nEigenDecomp + iEigen;
				}	// End for(iNode = ...)
				
				beagleUpdatePartials(beagleInstance[iPartition], 
									operations, 
									nOp, 
									cumulativeScalingIndices[iEigen]);

			}	// End for(iEigen = ...)

			if(operations != NULL){
				free(operations);
				operations = NULL;
			}


			// Calculate root likelihood:
			int *rootArrayId = new int[nEigenDecomp];
			for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++){
				rootArrayId[iEigen] = nTipNode + (nodeIdToArrayId[iPartition][partitionRoot[iPartition]->label] - nTipNode) * nEigenDecomp + iEigen;
			}

			beagleCalculateRootLogLikelihoods(beagleInstance[iPartition],
												rootArrayId,
												eigenDecompId[iPartition],
												eigenDecompId[iPartition],
												cumulativeScalingIndices,
												nEigenDecomp,
												&lnL_beagle[iPartition]);

			if(rootArrayId != NULL){
				free(rootArrayId);
				rootArrayId = NULL;
			}

			if(cumulativeScalingIndices){
				free(cumulativeScalingIndices);
				cumulativeScalingIndices = NULL;
			}
		}	// End for(iPartition ...)
	}	// End for(itr ...)

	time_lnL_beagle = timeEnd(tBegin);


	// Finalize the instances:
	for(int iPartition = 0; iPartition < nPartition; iPartition ++)
		beagleFinalizeInstance(beagleInstance[iPartition]);


	if(beagleInstance){
		free(beagleInstance);
		beagleInstance = NULL;
	}

	if(curBrLen){
		free(curBrLen);
		curBrLen = NULL;
	}

	if(nodeId){
		free(nodeId);
		nodeId = NULL;
	}
}




// TODO: 目前暂时只考虑一个partition，一条chain，之后考虑若有多条chain，如何调用???
void compareTime()
{
	// Allocate memory and init values:
	allocateMemory();
	initValues();

	// Init values for each partition:
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		 nEigenDecomp = eigenDecompCountArray[iPartition % nPartition];
		 nRateCategory = rateCatCountArray[iPartition % nPartition];
		 //nRateCategory = 1;

		 nSitePattern = sitePatternCountArray[iPartition % nPartition];
		 
		 nTipState = tipStateCountArray[iPartition % nPartition];
		 nTipCondlike = tipCondlikeCountArray[iPartition % nPartition];
		 nIntCondlike = intCondlikeCountArray[iPartition % nPartition];

		 assert(nTipState + nTipCondlike + nIntCondlike == nNode);

		 printf("\nFor %d-th partition, nState = %d, nTipState = %d, nTipCondlike = %d, nEigenDecomp = %d, nRateCategory = %d, nSitePattern = %d\n\n", iPartition, nState, nTipState, nTipCondlike, nEigenDecomp, nRateCategory, nSitePattern);

		CuLTreeNode *root;
		if(isRooted)
			root = buildTree1(iPartition, nNode);
		else
			root = buildTree1_unrooted(iPartition, nNode);
		printTree(root);

		rootId[iPartition] = root->label;
		partitionRoot[iPartition] = root;

		printNodeRelation(iPartition);

		// Set eigen decomposition id:
		for(int iEigen = 0; iEigen < nEigenDecomp; iEigen ++)
			eigenDecompId[iPartition][iEigen] = iEigen;

		// Set tip state && tip condlike id:
		for(int iNode = 0; iNode < nTipState; iNode ++)
			tipStateNodeId[iPartition][iNode] = iNode;
		for(int iNode = 0; iNode < nTipCondlike; iNode ++)
			tipCondlikeNodeId[iPartition][iNode] = iNode;
	}	// End for(iPartition ...)


	struct timeval tBegin;
	long long time_cuLib, time_CPU, time_beagle;

	// Pre-run the CuLib && beagle to start the CUDA:
	CuLibImpl();
	if(!(nEigenDecomp == 1 && nRateCategory == 3 && nSitePattern == 5000))
		BEAGLEImpl_seperate();


	// Call CuLib for the calculation of likelihood:
	printf("Goint to call CuLib for calculation...\n");
	timeBegin(&tBegin);
	CuLibImpl();
	time_cuLib = timeEnd(tBegin);

	// Call CPU for the calculation of likelihood:
	printf("Goint to call CPU for calculation...\n");
	timeBegin(&tBegin);
	CPUImpl();
	time_CPU = timeEnd(tBegin);
	printf("Time used for CPU: %lld.%06lld\n", time_CPU / multiple, time_CPU % multiple);

	// Call BEAGLE for the calculation:
	if(!(nEigenDecomp == 1 && nRateCategory == 3 && nSitePattern == 5000)){
		printf("Going to call BEAGLE for calculation...\n");
		timeBegin(&tBegin);
		BEAGLEImpl_seperate();
		time_beagle = timeEnd(tBegin);
	}

	FILE *ftime_all = fopen("all_time.txt", "a");
	FILE *ftime_lnL = fopen("lnL_time.txt", "a");
	fprintf(ftime_all, "\nFor nIteration = %d, nPartition = %d, nState = %d, nNode = %d, nEigenDecomp = (", nIteration, nPartition, nState, nNode);
	fprintf(ftime_lnL, "\nFor nIteration = %d, nPartition = %d, nState = %d, nNode = %d, nEigenDecomp = (", nIteration, nPartition, nState, nNode);
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		if(iPartition > 0){
			fprintf(ftime_all, ", ");
			fprintf(ftime_lnL, ", ");
		}
		fprintf(ftime_all, "%d", eigenDecompCountArray[iPartition]);
		fprintf(ftime_lnL, "%d", eigenDecompCountArray[iPartition]);
	}

	fprintf(ftime_all, "), nRateCategory = (");
	fprintf(ftime_lnL, "), nRateCategory = (");
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		if(iPartition > 0){
			fprintf(ftime_all, ", ");
			fprintf(ftime_lnL, ", ");
		}
		fprintf(ftime_all, "%d", rateCatCountArray[iPartition]);
		fprintf(ftime_lnL, "%d", rateCatCountArray[iPartition]);
	}

	fprintf(ftime_all, "), nSitePattern = (");
	fprintf(ftime_lnL, "), nSitePattern = (");
	for(int iPartition = 0; iPartition < nPartition; iPartition ++){
		if(iPartition > 0){
			fprintf(ftime_all, ", ");
			fprintf(ftime_lnL, ", ");
		}
		fprintf(ftime_all, "%d", sitePatternCountArray[iPartition]);
		fprintf(ftime_lnL, "%d", sitePatternCountArray[iPartition]);
	}
	fprintf(ftime_all, ") :\n");
	fprintf(ftime_lnL, ") :\n");

	fprintf(ftime_all, "CuLib: %lld.%06lld\n", time_cuLib / multiple, time_cuLib % multiple);
	fprintf(ftime_all, "beagle: %lld.%06lld\n", time_beagle / multiple, time_beagle % multiple);
	fprintf(ftime_all, "CPU: %lld.%06lld\n", time_CPU / multiple, time_CPU % multiple);
	fclose(ftime_all);



	fprintf(ftime_lnL, "CuLib: %lld.%06lld\n", time_lnL_cuLib / multiple, time_lnL_cuLib % multiple);
	fprintf(ftime_lnL, "beagle: %lld.%06lld\n", time_lnL_beagle / multiple, time_lnL_beagle % multiple);
	fclose(ftime_lnL);

	FILE *ftime = fopen("CuLib_time_all.txt", "a");
	fprintf(ftime, "%lld.%06lld\t", time_cuLib / multiple, time_cuLib % multiple);
	fclose(ftime);

	ftime = fopen("BEAGLE_time_all.txt", "a");
	fprintf(ftime, "%lld.%06lld\t", time_beagle / multiple, time_beagle % multiple);
	fclose(ftime);

	ftime = fopen("CPU_time_all.txt", "a");
	fprintf(ftime, "%lld.%06lld\t", time_CPU / multiple, time_CPU % multiple);
	fclose(ftime);

	ftime = fopen("CuLib_time_lnL.txt", "a");
	fprintf(ftime, "%lld.%06lld\t", time_lnL_cuLib / multiple, time_lnL_cuLib % multiple);
	fclose(ftime);

	ftime = fopen("BEAGLE_time_lnL.txt", "a");
	fprintf(ftime, "%lld.%06lld\t", time_lnL_beagle / multiple, time_lnL_beagle % multiple);
	fclose(ftime);


	// Compare the results of CuLib and CPU:
	for(int iPartition = 0; iPartition < nPartition; iPartition ++)
		compareLikelihood(lnL_CPU[iPartition], lnL_GPU[iPartition]);


	// Compare the results of BEAGLE and CPU:
	for(int iPartition = 0; iPartition < nPartition; iPartition ++)
		compareLikelihood(lnL_CPU[iPartition], lnL_beagle[iPartition]);

	FILE *flnL = fopen("lnL_CuLib.txt", "a");
	fprintf(flnL, "%.8lf\t", lnL_GPU[0]);
	fclose(flnL);

	flnL = fopen("lnL_CPU.txt", "a");
	fprintf(flnL, "%.8lf\t", lnL_CPU[0]);
	fclose(flnL);

	flnL = fopen("lnL_BEAGLE.txt", "a");
	fprintf(flnL, "%.8lf\t", lnL_beagle[0]);
	fclose(flnL);

	// Free the memory allocated before:
	freeMemory();
}


// Wait 18 seconds:
void idleWait(int nSecond)
{
	int nIter = 23250 * sqrt(nSecond);
	for(int i = 0; i < nIter; i ++)
		for(int j = 0; j < nIter; j ++);
}


int main()
{
	nIteration = 3;
	nPartition = 10;
	deviceId = 0;
	nNode = 63;
	nNodeScaler = 0;
	//nNodeScaler = nIntCondlike - 1;

#ifndef TRANSPOSE_PMAT
	transposePMat = false;
#else
	transposePMat = true;
#endif

	if(transposePMat)
		printf("Going to transpose PMat for CPU...\n");
	else
		printf("Not going to transpose PMat for CPU...\n");

	doScale = false;
	isRooted = true;

	for(int iState = 0; iState < 1; iState ++){

		//nState = iState;
		//nState = stateCountArray[iState];
		nState = 61;
		printf("For state = %d:\n", nState);
		
		
		// Run 3 times, and take the averate time:
		for(int itr = 0; itr < 3; itr ++){
			printf("\n-----------------------\nThe %d-th run:\n", itr);

			for(int iCat = 0; iCat < 4; iCat ++){
				if(iCat == 2)
					continue;

				if(iCat % 2 == 0){
					for(int iPartition = 0; iPartition < nPartition; iPartition ++)
						rateCatCountArray[iPartition] = 1;
				}
				else{
					for(int iPartition = 0; iPartition < nPartition; iPartition ++)
						rateCatCountArray[iPartition] = 3;
				}

				if(iCat / 2 == 0){
					for(int iPartition = 0; iPartition < nPartition; iPartition ++)
						eigenDecompCountArray[iPartition] = 1;
				}
				else{
					for(int iPartition = 0; iPartition < nPartition; iPartition ++)
						eigenDecompCountArray[iPartition] = 4;
				}

				FILE *ftime = fopen("all_time.txt", "a");
				fprintf(ftime, "\n\nnEigenDecomp = %d && nRateCat = %d:\n", eigenDecompCountArray[0], rateCatCountArray[0]);
				fclose(ftime);

				ftime = fopen("lnL_time.txt", "a");
				fprintf(ftime, "\n\nnEigenDecomp = %d && nRateCat = %d:\n", eigenDecompCountArray[0], rateCatCountArray[0]);
				fclose(ftime);

				int nPattern = 5;
				if(iCat == 0)
					nPattern = 4;
				else if(iCat == 1)
					nPattern = 3;
				else if(iCat == 3)
					nPattern = 2;

				for(int iPattern = 0; iPattern < nPattern; iPattern ++){
					for(int iPartition = 0; iPartition < nPartition; iPartition ++){
						sitePatternCountArray[iPartition] = patternCount[iPattern];
						//sitePatternCountArray[iPartition] = 5000;
					}
					compareTime();
				}
				ftime = fopen("CuLib_time_all.txt", "a");
				fprintf(ftime, "\t");
				fclose(ftime);

				ftime = fopen("BEAGLE_time_all.txt", "a");
				fprintf(ftime, "\t");
				fclose(ftime);

				ftime = fopen("CPU_time_all.txt", "a");
				fprintf(ftime, "\t");
				fclose(ftime);

				ftime = fopen("CuLib_time_lnL.txt", "a");
				fprintf(ftime, "\t");
				fclose(ftime);

				ftime = fopen("BEAGLE_time_lnL.txt", "a");
				fprintf(ftime, "\t");
				fclose(ftime);

				FILE *flnL = fopen("lnL_CPU.txt", "a");
				fprintf(flnL, "\t");
				fclose(flnL);

				flnL = fopen("lnL_CuLib.txt", "a");
				fprintf(flnL, "\t");
				fclose(flnL);

				flnL = fopen("lnL_BEAGLE.txt", "a");
				fprintf(flnL, "\t");
				fclose(flnL);

				// Wait 5 seconds:
				idleWait(5);
			}

			
			FILE *ftime2 = fopen("CuLib_time_all.txt", "a");
			fprintf(ftime2, "\n");
			fclose(ftime2);

			ftime2 = fopen("BEAGLE_time_all.txt", "a");
			fprintf(ftime2, "\n");
			fclose(ftime2);

			ftime2 = fopen("CPU_time_all.txt", "a");
			fprintf(ftime2, "\n");
			fclose(ftime2);

			ftime2 = fopen("CuLib_time_lnL.txt", "a");
			fprintf(ftime2, "\n");
			fclose(ftime2);

			ftime2 = fopen("BEAGLE_time_lnL.txt", "a");
			fprintf(ftime2, "\n");
			fclose(ftime2);

			FILE *flnL = fopen("lnL_CPU.txt", "a");
			fprintf(flnL, "\n");
			fclose(flnL);

			flnL = fopen("lnL_CuLib.txt", "a");
			fprintf(flnL, "\n");
			fclose(flnL);

			flnL = fopen("lnL_BEAGLE.txt", "a");
			fprintf(flnL, "\n");
			fclose(flnL);

			// Wait 10 seconds:
			struct timeval tBegin;
			timeBegin(&tBegin);
			idleWait(10);
			long long sec = timeEnd(tBegin);
			printf("Time waited: %lld.%06lld\n", sec / multiple, sec % multiple);
		}
	}

	return 0;
}