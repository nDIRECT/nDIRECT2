//Authors: pengyu wang
#include "./kernel/1x1_n12m32.h"
#include "NDIRECT2.h"

void LIB2_R1_s1(int H, int W, int N, int C, float *input, int K, int R, int S, float *filter, int padh, int padw, int stride, float* output){

    avx512_1x1s1(H, W, N, C, input, K, R, filter, output);
}