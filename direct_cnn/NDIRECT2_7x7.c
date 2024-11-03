#include "./kernel/7x7n16m24.h"

#include "NDIRECT2.h"

void LIB2_R7_s2(int H, int W, int N, int C, float *input, int K, int R, int S, float *filter, int padh, int padw, int stride, float* output){
    
    avx512_dircet_cnn_7x7s2(filter, input, output, K, C, H, R);

}