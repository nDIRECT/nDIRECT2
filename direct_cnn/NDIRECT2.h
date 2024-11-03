#ifndef __DIRECT2__ 
#define __DIRECT2__ 

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#include <iostream>
#include <vector>


using namespace std;
#define NUM 48
#define CONV_K_1x1 512
#define CONV_HW_1x1 4096
#define CONV_C_1x1 256

//adjust to fit the interface
#define CONV_K_3x3 64
#define CONV_C_3x3 64

#define CONV_K_7x7 16
#define CONV_C_7x7 3

//extern vector<int>vec;
//extern int Tm, Tn;

void NDIRECT2_flush();

void transform2_filter_1x1(const int outch, const int inch, float* kernel, float* out_kernel);

void transform2_filter_3x3(const int outch, const int inch, float* kernel, float* out_kernel);

void transform2_filter_7x7(const int outch, const int inch, const int k_h, 
                const int k_w, float* kernel, float* out_kernel);

void NDIRECT2_dnn_conv_fwd_exec(int H, int W, int N, int C, float *input,
                                 int K, int R, int S, float *filter,
                                 int padh, int padw, int stride, float* output);
void LIB2_R1_s1(int H, int W, int N, int C, float *input, 
            int K, int R, int S, float *filter, int padh, int padw, int stride, float* output);
void LIB2_R1_s2(int H, int W, int N, int C, float *input, 
    int K, int R, int S, float *filter, int padh, int padw, int stride, float* output);

void LIB2_R3_s1(int H, int W, int N, int C, float *input, 
            int K, int R, int S, float *filter, int padh, int padw, int stride, float* output);
void LIB2_R3_s2(int H, int W, int N, int C, float *input, 
    int K, int R, int S, float *filter, int padh, int padw, int stride, float* output);
void LIB2_R7_s2(int H, int W, int N, int C, float *input, 
    int K, int R, int S, float *filter, int padh, int padw, int stride, float* output);

#endif