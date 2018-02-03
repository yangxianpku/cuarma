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

    cuarma::matrix<double,cuarma::column_major> cuamraMat = cuarmaGetData<double>(prhs[0]);
    double a;
    for(int i = 0; i < cuamraMat.size1(); i++)
    {
    	   for(int j =0;j<cuamraMat.size2();j++)
    	   {
    	   		a = cuamraMat(i,j);
    	   		mexPrintf("MatData[%d,%d] =%0.8f \n",i,j,a);
    	   }
    }

	return;
}