### NDIRECT2
----------------------
Contact: Pengyu Wang (pengyu_wang@nudt.edu.cn)

NDIRECT2 is a library for direct convolution on x86-based processors, with a focus on providing high performance, high data reusability, and DL framework compatibility. It preserves the conventional `NCHW/NHWC` data format, which is compatible with mainstream deep learning frameworks, such as Pytorch and MXNet. NDIRECT2 leverages the AVX-512 extension to implement direct convolution.

This work continues to be optimized.
### Software Dependences
------------------------
* [GNU Compiler (GCC)](https://gcc.gnu.org/) (>=v8.2)
* [OpenMP](https://www.openmp.org/) 

### Hardware Platform
-------------------------
Intel Xeon gold 6252N or other x86-based processors with equipped with the AXV-512 SIMD extension.

### Compile and Install
----------------------
```bash
$ cd direct_cnn/
$ make
$ make install PREFIX= specify the installation path
```
### Compile with NDIRECT
----------------------
```cs
#include <stdio.h>
#include <stdlib.h>
#include "NDIRECT2.h"
```
### API
----------------------
We illustrate the convolution interface of NDIRECT2.
```cs
NDIRECT2_dnn_conv_fwd_exec(int H, int W, int N, int C, float *input,
                                 int K, int R, int S, float *filter,
                                 int padh, int padw, int stride, float* output);
```
### Running Benchmark
----------------------
To run the command
```bash
$ cd direct_cnn/test
$ make
$ ./run.sh
```
will evaluate the performance of convolution layers from ResNet-50 and VggNet-16.
```bash
$ cd Fuse/BF16/Conv+DWConv+Conv
$ ./a.out
```
will evaluate the performance of fused Conv+DWConv+Conv.
```bash
$ cd Fuse/BF16/Conv+Conv+Conv
$ ./b.out
```
will evaluate the performance of fused Conv+Conv+Conv.
```bash
$ cd MXNet/
$ export OMP_NUM_THREADS=48
$ python3 mxnet_resnet50_48.py
```
will evaluate the performance of CNN inference of ResNet-50.
```bash
$ cd Ansor/model
$ export TVM_NUM_THREADS=48
$ python3 tune_network_x86.py
```
will autotune the performance of CNN model.
```bash
$ cd Ansor/layer
$ export TVM_NUM_THREADS=48
$ python3 tune_conv2d_layer_cpu.py
```
will autotune the direct convolution performance of CNN layers.

### Getting Started
----------------------
The following source code provides an instance  to evaluate the convolution performance.
```cs
#include <stdio.h>
#include <stdlib.h>
#include "NDIRECT2.h"

static double gtod_ref_time_sec = 0.0;
double dclock()
{
	double the_time, norm_sec;
	struct timeval tv;
	gettimeofday( &tv, NULL );
	if ( gtod_ref_time_sec == 0.0 )
		gtod_ref_time_sec = ( double ) tv.tv_sec;
	norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;
	the_time = norm_sec + tv.tv_usec * 1.0e-6;
	return the_time;
}

void random_matrix( int m, int n, float *a)
{
	double drand48();
	int i, j;
	for ( i = 0; i < m; i++ )
		for ( j = 0; j < n; j++ )
		{
			a[i * n + j] = 2.0 * (float)drand48( ) - 1.0 + 0.000001 * (i + j);
		}
}

int main()
{
	int H = 224, W = 224, N = 64, C = 3;
	int K = 64, R = 7, S = 7;
	int padh = 3, padw = 3, stride = 2;

	int out_H = (H - R + 2 * pad_h)/stride + 1; 
    int out_W = (W - S + 2 * pad_w)/stride + 1;

    filter = ( float * ) malloc( C * K * R *S *sizeof( float ));
    input = ( float * ) malloc(N * C * H * W * sizeof( float ));
    output = ( float * ) malloc(N * K * (out_H * out_W) * sizeof( float ));
    random_matrix(K, C * R * S, filter);
    random_matrix(C, N * H * W, input);
	// warm up
    NDIRECT2_dnn_conv_fwd_exec(H, W, N , C, input, K, R, S, filter,
    padh, padw, stride, output);
    
    // evaluate
    start = dclock();
    for ( i = 0; i < loop ; i++)
    	NDIRECT2_dnn_conv_fwd_exec(H, W, N, C, input, K, R, S, filter, padh, padw, stride, output);
    cost = (dclock() - start) / loop;

    free(filter);
    free(input);
    free(output);
	return 0;
}
```
The corresponding makefile for this program is:
```makefile
NDIRECT2_PREFIX = path to install NDIRECT2
NDIRECT2_INC    = $(NDIRECT_PREFIX)/ND2/include
NDIRECT2_LIB    = $(NDIRECT_PREFIX)/ND2/lib

OTHER_LIBS  = -fopenmp -mavx512f

CC          = g++
CFLAGS      = -g -fopenmp -mavx512f -O2 -I$(NDIRECT2_INC) -L$(NDIRECT2_LIB) -lnd2
LINKER      = $(CC)

OBJS        = test.o

%.o: %.c
	 $(CC) $(CFLAGS) -c -fopenmp -mavx512f $< -o $@

all: $(OBJS)
	$(LINKER) $(OBJS) $(CFLAGS) $(OTHER_LIBS) -o a.out
	
.PHONY:clean
clean:
	rm -f *.o *.a *.so
```
### Integrate with MXNet
We need to change [the lines 307-349 of mxnet/blob/master/src/operator/nn/convolution-inl.h](https://github.com/apache/mxnet/blob/master/src/operator/nn/convolution-inl.h#L307-L349) to `NDIRECT2_dnn_conv_fwd_exec(
	in_data[conv::kData].shape_[2], in_data[conv::kData].shape_[3],
        num_, in_data[conv::kData].shape_[1], in_data[conv::kData].dptr<float>(),
        M, param_.kernel[0], param_.kernel[1], (float*)weight_3d[0].dptr_,
        param_.pad[0], param_.pad[1], param_.stride[0],(float*)output_4d[0].dptr_);`

### Compared with CNN models/layers that autotuned by Ansor
We utilize the ansor sub-module from TVM to autotune the ResNet and VGGNet varients. Also, we use ansor to autotune the direct convolution with different parameters. The implementation code can be accessed in nDIRECT2/Ansor.

### Note
----------------------
NDIRECT2 adopts the traditional `NCHW/NHWC` and `KCRS` data formats to store input and filter tensors. Therefore, you can integrate NDIRECT2 with DL frameworks for CNNs training and inference directly.
