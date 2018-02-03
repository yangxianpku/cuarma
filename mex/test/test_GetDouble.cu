#include <stdio.h>
#include "head_define.h"
#include "cuarmaMex.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs != 1)
	{
		mexErrMsgTxt("Incorrect nuymber of input arguments!");  
	}

	if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS )
	{
		mexErrMsgTxt("Input must be of type double!");
	}

	if ((mxIsComplex(prhs[0])))
	{
		mexErrMsgTxt("Input must be real.");
	}

    /* Initialize the MathWorks GPU API. */
    if(mxInitGPU()==MX_GPU_FAILURE)  return;

    cuarma::scalar<double> arma_s = cuarmaGetDouble(prhs[0]);
    double a = arma_s;;

    mexPrintf("Input Scalar is: %0.8f \n",a);

	return;
}