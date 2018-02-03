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

    /* Initialize the MathWorks GPU API. */
    if(mxInitGPU()==MX_GPU_FAILURE)  return;

    cuarma::matrix<double,cuarma::column_major> cuarmaMat = cuarmaGetComplex<double>(prhs[0]);
    double a,b;
    for(int i =0;i<cuarmaMat.size1();i++)
    {
    	   for(int j =0;j<cuarmaMat.size2();j=j+2)
    	   {
    	   		a = cuarmaMat(i,j);
                b = cuarmaMat(i,j+1);
    	   		mexPrintf("Mat-Real[%d,%d] =%0.8f, ",i,j,a);
                mexPrintf("Mat-Imag[%d,%d] =%0.8f \n",i,j,b);
    	   }
    }

	return;
}