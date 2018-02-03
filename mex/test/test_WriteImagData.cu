#include <stdio.h>
#include "head_define.h"
#include "cuarmaMex.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	char const * const filename ="data.mat";

    /* Initialize the MathWorks GPU API. */
    if(mxInitGPU()==MX_GPU_FAILURE)  return;

    cuarma::matrix<double,cuarma::column_major> cuamraMat(3,3);

    for(int i = 0; i < cuamraMat.size1(); i++)
    {
    	   for(int j =0;j<cuamraMat.size2();j++)
    	   {
    	   		cuamraMat(i,j) = i+j+0.12;
    	   }
    }

    int result = cuarmaWriteImagDataToMatFile<double>(filename,cuamraMat,"d");
	return;
}