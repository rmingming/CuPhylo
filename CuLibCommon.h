#ifndef CULIB_COMMON_H
#define CULIB_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <map>

using namespace std;

#define CUFlt double
// min of float: 1e-35, min of double: 1e-300
#define CUFLT_MIN 1e-300

#define MAX_STREAM 32	// Maximum stream count

//#define DEBUG_DESTROY

#define DEBUG_TIME

#define TRANSPOSE_PMAT

enum CuLErrorCode
{
	CUL_SUCCESS						= 0,			// No error;
	CUL_ERROR_COMMON				= 1,			// General error;
	CUL_ERROR_BAD_ALLOC				= 2,
	CUL_ERROR_OUT_OF_DEVICE_MEMORY	= 3,			// memory allocate error;
	CUL_ERROR_DEVICE_NOT_AVALAIABLE = 4,	// The device doesn't exist or is not available
	CUL_ERROR_INDEX_OUT_OF_RANGE	= 5,
	CUL_ERROR_BAD_INSTANCE			= 6,		// Null instance
	CUL_ERROR_NOT_INITIALIZED		= 7,
	CUL_ERROR_BAD_PARAM_VALUE		= 8,
	CUL_ERROR_INTERNAL				= 9		// internal error
};

// MAY: todo: 注意无根树的情况：root有三个child?
struct CuLTreeNode{
	int label;			// the index of the current node
	int nChild;
	CuLTreeNode *child[3];

	CuLTreeNode(int l, int n):label(l), nChild(n){
		child[0] = child[1] = child[2] = NULL;
	}
};

typedef struct
{
	int UV_offset;
	int R_offset;
	int brLen_offset;
	int P_offset;
} CuLPMatOffset;		// Offset for calculation of transition matrix


typedef struct
{
	int nChild;
	int whichCase;		// which case is the current node, range from 1 to 6
	int isChildTip[3];
	int child_case[3];
	int child_P_offset[3];			// 注意：PMat的offset直接对应的是node在tree中的id；
	int child_condlike_offset[3];
	int father_condlike_offset;
} CuLCondlikeOp;


inline void printError(const char* errorStr)
{
	printf("Error: %s\n", errorStr);
	exit(-1);
}

inline void printErrorCode(CuLErrorCode returnState)
{
	printf("Error code: %d\n", returnState);
	exit(-1);
}


#endif