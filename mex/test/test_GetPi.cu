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

    cuarma::matrix<double,cuarma::column_major> cuarmaMatPi = cuarmaGetPi(prhs[0]);
    for(int i =0;i<cuarmaMatPi.size1();i++)
    {
    	   for(int j =0;j<cuarmaMatPi.size2();j++)
    	   {
    	   		a = cuarmaMatPi(i,j);
    	   		mexPrintf("MatPi[%d,%d] =%0.8f \n",i,j,a);
    	   }
    }

	return;
}