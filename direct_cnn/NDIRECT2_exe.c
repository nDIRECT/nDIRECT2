#include "NDIRECT2.h"

void NDIRECT2_dnn_conv_fwd_exec(int H, int W, int N, int C, float *input,
                                 int K, int R, int S, float *filter,
                                 int padh, int padw, int stride, float* output)
{

	if(R == 1)
	{

		//Dete_grad_N_threads_nums(N, K, H, W, stride);
		if (R == 1 && S == 1 && stride == 2)
		{
			LIB2_R1_s1(H, W, N, C, input, K, 1, 1, filter, padh, padw, 2, output);
		}
		else if(R == 1 && S == 1 && stride == 1)
		{
			LIB2_R1_s2(H, W, N, C, input, K, 1, 1, filter, padh, padw, 1, output);
		}
	}
	else if(R == 3)
	{
        if(stride==1)
        {
            LIB2_R3_s1(H, W, N , C, input, K, R, S, filter, 
            	padh, padw, stride, output);
        }
    }
	else
	{
		LIB2_R7_s2(H, W, N , C, input, K, R, S, filter, 
			padh, padw, stride, output);
	}
}