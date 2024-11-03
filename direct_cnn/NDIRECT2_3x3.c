//Authors: pengyu wang
#include "./kernel/3x3_n32m12.h"
#include "NDIRECT2.h"

void LIB2_R3_s1(int H, int W, int N, int C, float *input, int K, int R, int S, float *filter, int padh, int padw, int stride, float* output){

    avx512_dircet_cnn_3x3s1(filter, input, output, K, C, H, R);

}
