#include <stdio.h>
#include "head_define.h"
#include "cuarmaMex.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Initialize the MathWorks GPU API. */
    if(mxInitGPU()==MX_GPU_FAILURE)  return;

    cuarma::matrix<double,cuarma::column_major> cuarmaMat(3,3);

    for (int i = 0; i < 3; ++i)
    	for (int j = 0; j < 3; ++j)
    		cuarmaMat(i,j) = i+j+5.26;

    plhs[0] = cuarmaSetImagData<double>(cuarmaMat);

	return;
}