#include <malloc.h>
#include <sys/time.h>
#include <stdio.h>
//#include <cblas.h>
#include <stdbool.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include <mkl_cblas.h>

#include <math.h>

#define NUM 48
#define CONV_K 512
#define CONV_HW 4096
#define CONV_C 256

int Tm, Tn, T;


static double gtod_ref_time_sec = 0.0;

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename DType>
inline void im2col_cpu(const DType* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    DType* data_col) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  // TODO(junwu): we tested adding openmp (w/ & w/o collapse clause) here
  // for testing the performance of convolution operator,
  // but the total runtime increased by 0.8s for images of shape
  // (8, 32, 64, 64) and decreased by 0.2s for images of shape
  // (16, 64, 64, 64). Both kernel shapes are (8, 8). We think the
  // bottleneck of the convolution operator probably lies in dot().
  // Hence, adding more threads to the loops contributes little
  // toward improving the convolution operator's performance.
  // We will revisit this issue in the future.
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}


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
  //double drand48();
  int i,j;

  // #pragma omp parallel for num_threads(num)
  for ( i=0; i< m; i++ )
    for( j =0; j < n; j++)
    {
        a[i*n+j]= 2.0 * (float)drand48() - 1.0;
        //a[i*n+j] = 1.0;
    }
}



void transform_filter(const int outch, const int inch, float* kernel, float* out_kernel)
{
    int mr = 32, cr = CONV_C;

    int i, j , k, ii,jj,kk, h,w;
    int st = 0;

    for(j = 0; j < inch; j = j + cr)
    {
        cr = CONV_C;
        if(inch - j < CONV_C)
            cr = inch - j;
        for(i = 0; i < outch; i = i + mr)
        {   
            for(jj = j; jj < j + cr; jj++)
            {
                for(kk=0; kk<mr; kk++){
                    
                    out_kernel[ st+kk ] = kernel[ (i+kk) * inch + jj];
                }
                st += mr;
            }
        }       
    }
}

void verify_transform_filter(const int outch, const int inch, float* out_kernel)
{
    int M = outch*inch;
    int N = 1;
    int i,j;
    for(i=0; i<M; i++){
        for(j=0; j<N; j++){
            if(fabs(out_kernel[i * N + j] - 1.0) > 0.001 )
            {
                printf("i = %d, j= %d\n",i ,j );
                printf("out_kernel= %lf \n", out_kernel[i*N + j]);
            }
        }
        
    }
}

void direct_1x1_N32M12_AVX512_pack(float *output, float *trans_filter, float *input, int Kb, int LEN_HWb, int Cb, int input_HW_size, float *input_buffer, int cc){
    asm volatile(


        ".macro KERNEL12x32_PACK_K1                         \n"

        "   vbroadcastss    8(%%rax), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm9          \n"
        "   vmovups         (%%rax), %%ymm6                 \n"
        "   vbroadcastss    12(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm11         \n"
        "   vmovups         32(%%rax), %%xmm7                 \n"
        //"   prefetcht0      256(%%rax)                      \n"
        "   prefetcht0      48(%%rax)                      \n"
        "   vbroadcastss    16(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm13         \n"

        //"   prefetcht2      128(%%rbx)                      \n"

        "   vbroadcastss    20(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm15         \n"
        "   vmovups         %%ymm6, (%%r14)                 \n"
        "   vbroadcastss    24(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm17         \n"

        //"   prefetcht2      192(%%rbx)                      \n"
        "   add $128, %%rbx                                 \n"
        "   vbroadcastss    28(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm19         \n"
        "   vmovups         %%xmm7, 32(%%r14)                 \n"
        "   vbroadcastss    32(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm21         \n"


        "   prefetcht0      (%%rbx)                      \n"
        "   vbroadcastss    36(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm23         \n"

        "   vbroadcastss    40(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm25         \n"
        "   prefetcht0      64(%%rbx)                      \n"
        //"   prefetcht0      384(%%rax)                      \n"

        "   vbroadcastss    44(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm26         \n"
        "   vmovups         (%%rbx), %%zmm6                 \n" 
        "   add             %%r8, %%rax                      \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm27         \n"

        "   vbroadcastss    (%%rax), %%zmm0                 \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm28         \n"
        "   vmovups         64(%%rbx), %%zmm7               \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm29         \n"
        
        "   vbroadcastss    4(%%rax), %%zmm1                \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm30         \n"

        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm31         \n"
        //"   prefetcht2      (%%rax)                      \n"
        "   add            $48, %%r14                     \n"

        

        ".endm                                              \n"


        ".macro KERNEL12x32_PACK_K2                         \n"

        "   vbroadcastss    8(%%rax), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        "   vmovups         (%%rax), %%ymm4                 \n"
        //"   prefetcht0      256(%%rax)                      \n"

        "   vbroadcastss    12(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"
        "   vmovups         32(%%rax), %%xmm5                 \n"
        //"   prefetcht2      128(%%rbx)                      \n"
        "   prefetcht0      48(%%rax)                      \n"
        "   vbroadcastss    16(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"
        "   vmovups         %%ymm4, (%%r14)                 \n"
        //"   prefetcht2      192(%%rbx)                      \n"

        "   vbroadcastss    20(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"
        "   vmovups         %%xmm5, 32(%%r14)                 \n"
        "   vbroadcastss    24(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"
        "   add $128, %%rbx                                 \n"

        "   vbroadcastss    28(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"

        "   vbroadcastss    32(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"
        "   prefetcht0      (%%rbx)                      \n"

        "   vbroadcastss    36(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm23         \n"

        "   vbroadcastss    40(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm25         \n"
        "   prefetcht0      64(%%rbx)                      \n"

        "   vbroadcastss    44(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"
        "   vmovups         (%%rbx), %%zmm4                 \n" 
        "   add             %%r8, %%rax                      \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm27         \n"

        "   vbroadcastss    (%%rax), %%zmm0                 \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vmovups         64(%%rbx), %%zmm5               \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm29         \n"

        "   vbroadcastss    4(%%rax), %%zmm1                \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"

        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm31         \n"
        //"   prefetcht2      (%%rax)                      \n"
        "   add            $48, %%r14                     \n"


        ".endm                                              \n"



        ".macro KERNEL12x32_PACK_END_K                      \n"


        "   vbroadcastss    8(%%rax), %%zmm2                \n"

        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        "   vmovups         (%%rax), %%ymm4                 \n"
        
        "   vbroadcastss    12(%%rax), %%zmm3               \n"
        
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"
        "   prefetcht0      48(%%rax)                      \n"
        //"   prefetcht0      256(%%rax)                      \n"

        "   vbroadcastss    16(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"
        "   vmovups         %%ymm4, (%%r14)                 \n"
        "   vbroadcastss    20(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"

        "   vbroadcastss    24(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"
        
        "   vbroadcastss    28(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"

        "   vbroadcastss    32(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"
        "   vmovups         32(%%rax), %%xmm5                 \n"
        
        "   vbroadcastss    36(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm23         \n"

        "   vbroadcastss    40(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm25         \n"
        "   vmovups         %%xmm5, 32(%%r14)                 \n"
        "   vbroadcastss    44(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"

        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm27         \n"


        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm29         \n"
        "   add $128, %%rbx                                 \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm31         \n"

        ".endm                                              \n"



        ".macro KERNEL12x32_K1                              \n"

        "   vbroadcastss    8(%%rax), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm9          \n"

        "   vbroadcastss    12(%%rax), %%zmm3               \n"
        
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm11         \n"

        "   prefetcht0      48(%%rax)                      \n"

        "   vbroadcastss    16(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm13         \n"

        "   vbroadcastss    20(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm15         \n"

        "   vbroadcastss    24(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm17         \n"
        "   add            $128, %%rbx                     \n"
        "   vbroadcastss    28(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm19         \n"

        "   vbroadcastss    32(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm21         \n"

        "   prefetcht0      (%%rbx)                       \n"

        "   vbroadcastss    36(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm23         \n"

        "   prefetcht0      64(%%rbx)                       \n"

        "   vbroadcastss    40(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm25         \n"

        "   vbroadcastss    44(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm26         \n"
        "   vmovups         (%%rbx), %%zmm6                 \n" 
        "   add             $48, %%rax                      \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm27         \n"

        "   vbroadcastss    (%%rax), %%zmm0                 \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm28         \n"
        "   vmovups         64(%%rbx), %%zmm7               \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm29         \n"

        "   vbroadcastss    4(%%rax), %%zmm1                \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm31         \n"
        //"   prefetcht0      64(%%rbx)                       \n"
        

        ".endm                                              \n"


        ".macro KERNEL12x32_K2                              \n"

        "   vbroadcastss    8(%%rax), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"

        "   vbroadcastss    12(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"

        "   prefetcht0      48(%%rax)                      \n"

        "   vbroadcastss    16(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"

        "   vbroadcastss    20(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"

        "   vbroadcastss    24(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"

        "   vbroadcastss    28(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"
        "   add            $128, %%rbx                     \n"
        "   vbroadcastss    32(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"

        "   prefetcht0      (%%rbx)                       \n"

        "   vbroadcastss    36(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm23         \n"

        "   prefetcht0      64(%%rbx)                       \n"

        "   vbroadcastss    40(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm25         \n"

        "   vbroadcastss    44(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"
        "   vmovups         (%%rbx), %%zmm4                 \n" 
        "   add            $48, %%rax                      \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm27         \n"

        "   vbroadcastss    (%%rax), %%zmm0                 \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vmovups         64(%%rbx), %%zmm5               \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm29         \n"

        "   vbroadcastss    4(%%rax), %%zmm1                \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm31         \n"


        ".endm                                              \n"



        ".macro KERNEL12x32_END_K                           \n"


        "   vbroadcastss    8(%%rax), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        "   vbroadcastss    12(%%rax), %%zmm3               \n"
        
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"

        //"   prefetcht0      256(%%rax)                      \n"

        "   vbroadcastss    16(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"

        "   vbroadcastss    20(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"

        "   vbroadcastss    24(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"

        "   vbroadcastss    28(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"

        "   vbroadcastss    32(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"

        "   vbroadcastss    36(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm23         \n"

        "   vbroadcastss    40(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm25         \n"

        "   vbroadcastss    44(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"

        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm27         \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm29         \n"
        "   add             $128, %%rbx                     \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm31         \n"
        

        ".endm                                              \n"


        ".macro ST_12x32   \n"
        
        //  1 5 9 13
        "   vunpcklps %%zmm10, %%zmm8, %%zmm0    \n"
        "   vunpcklps %%zmm18, %%zmm16, %%zmm1    \n"
        
        "   vunpcklps %%zmm14, %%zmm12, %%zmm2    \n"
        "   vunpcklps %%zmm22, %%zmm20, %%zmm3    \n"
        
        
        "   movl    $0xaa, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm4     \n"
        "   vpermq  $0x80, %%zmm3, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r10)         \n"     //  input 1
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm6      \n"
        "   vmovups %%ymm6, (%%r12)            \n"     //input 9
        
        "   vmovups %%ymm5, (%%r11)         \n"     //  input 5
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups %%ymm7, (%%r13)            \n"     //input 13
        //  
        "   vunpcklps %%zmm26, %%zmm24, %%zmm0    \n"
        "   vunpcklps %%zmm30, %%zmm28, %%zmm2    \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r10)         \n"     //  input 1
        "   vextractf32x4  $0x1, %%zmm0, %%xmm4      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm5      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm6      \n"    //  input 13
        
        "   vmovups %%xmm4, 32(%%r11)         \n"     //  input 5
        "   vmovups %%xmm5, 32(%%r12)         \n"     //  input 9
        "   vmovups %%xmm6, 32(%%r13)         \n"     //  input 13

        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        
        // 2 6 10 14
        
        "   vunpcklps %%zmm10, %%zmm8, %%zmm0    \n"
        "   vunpcklps %%zmm18, %%zmm16, %%zmm1    \n"
        
        "   vunpcklps %%zmm14, %%zmm12, %%zmm2    \n"
        "   vunpcklps %%zmm22, %%zmm20, %%zmm3    \n"
        
        "   movl    $0x55, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%zmm2, %%zmm4     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm3%{%%k1%}      \n"
        "   vmovups %%zmm3, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm3, %%zmm2%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        
        "   vmovups %%ymm2, (%%r10)         \n"     //  input 2
        "   vextractf64x4  $0x1,%%zmm2, %%ymm6      \n"
        "   vmovups %%ymm6, (%%r12)            \n"     //input 10
        
        "   vmovups %%ymm5, (%%r11)         \n"     //  input 6
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups %%ymm7, (%%r13)            \n"     //input 14
        
        //  
        "   vunpcklps %%zmm26, %%zmm24, %%zmm0    \n"
        "   vunpcklps %%zmm30, %%zmm28, %%zmm2    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%xmm2, 32(%%r10)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm2, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm2, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm2, %%xmm6      \n"    //  input 14
        
        "   vmovups %%xmm4, 32(%%r11)         \n"     //  input 6
        "   vmovups %%xmm5, 32(%%r12)         \n"     //  input 10
        "   vmovups %%xmm6, 32(%%r13)         \n"     //  input 14
        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        //3 7 11 15
        "   vunpckhps %%zmm10, %%zmm8, %%zmm0    \n"
        "   vunpckhps %%zmm18, %%zmm16, %%zmm1    \n"
        
        "   vunpckhps %%zmm14, %%zmm12, %%zmm2    \n"
        "   vunpckhps %%zmm22, %%zmm20, %%zmm3    \n"
        
        "   movl    $0xaa, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm4     \n"
        "   vpermq   $0x80, %%zmm3, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r10)         \n"     //  input 3
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm6      \n"
        "   vmovups %%ymm6, (%%r12)            \n"     //input 11
        
        "   vmovups %%ymm5, (%%r11)         \n"     //  input 7
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups %%ymm7, (%%r13)            \n"     //input 15
        
        //  
        "   vunpckhps %%zmm26, %%zmm24, %%zmm0    \n"
        "   vunpckhps %%zmm30, %%zmm28, %%zmm2    \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r10)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm0, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm0, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm0, %%xmm6      \n"    //  input 14
        
        "   vmovups %%xmm4, 32(%%r11)         \n"     //  input 6
        "   vmovups %%xmm5, 32(%%r12)         \n"     //  input 10
        "   vmovups %%xmm6, 32(%%r13)         \n"     //  input 14
        
        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        
        
        //4 8 12 16
        "   vunpckhps %%zmm10, %%zmm8, %%zmm0    \n"
        "   vunpckhps %%zmm18, %%zmm16, %%zmm1    \n"
        
        "   vunpckhps %%zmm14, %%zmm12, %%zmm2    \n"
        "   vunpckhps %%zmm22, %%zmm20, %%zmm3    \n"
        
        "   movl    $0x55, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%zmm2, %%zmm4     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm3%{%%k1%}      \n"
        "   vmovups %%zmm3, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm3, %%zmm2%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        
        "   vmovups %%ymm2, (%%r10)         \n"     //  input 4
        "   vextractf64x4  $0x1,%%zmm2, %%ymm6      \n"
        "   vmovups %%ymm6, (%%r12)            \n"     //input 12
        
        "   vmovups %%ymm5, (%%r11)         \n"     //  input 8
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups %%ymm7, (%%r13)            \n"     //input 16
        
        //  
        "   vunpckhps %%zmm26, %%zmm24, %%zmm0    \n"
        "   vunpckhps %%zmm30, %%zmm28, %%zmm2    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%xmm2, 32(%%r10)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm2, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm2, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm2, %%xmm6      \n"    //  input 14
        
        "   vmovups %%xmm4, 32(%%r11)         \n"     //  input 6
        "   vmovups %%xmm5, 32(%%r12)         \n"     //  input 10
        "   vmovups %%xmm6, 32(%%r13)         \n"     //  input 14
        
        
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        
        "   shl $2, %%r8                \n"
        
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   prefetcht2      (%%r11)                         \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   prefetcht2      (%%r12)                         \n"
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   prefetcht2      (%%r13)                         \n"
        
        "   shr $2, %%r8                \n"
        
        
        //17 21 25 29
        "   vunpcklps %%zmm11, %%zmm9, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm17, %%zmm1    \n"
        
        "   vunpcklps %%zmm15, %%zmm13, %%zmm2    \n"
        "   vunpcklps %%zmm23, %%zmm21, %%zmm3    \n"
        
        
        "   movl    $0xaa, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm4     \n"
        "   vpermq   $0x80, %%zmm3, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r10)         \n"     //  input 17
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm6      \n"
        "   vmovups %%ymm6, (%%r12)            \n"     //input 21
        
        "   vmovups %%ymm5, (%%r11)         \n"     //  input 25
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups %%ymm7, (%%r13)            \n"     //input 29
        
        //  
        "   vunpcklps %%zmm27, %%zmm25, %%zmm0    \n"
        "   vunpcklps %%zmm31, %%zmm29, %%zmm2    \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r10)         \n"     //  input 1
        "   vextractf32x4  $0x1, %%zmm0, %%xmm4      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm5      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm6      \n"    //  input 13
        
        "   vmovups %%xmm4, 32(%%r11)         \n"     //  input 5
        "   vmovups %%xmm5, 32(%%r12)         \n"     //  input 9
        "   vmovups %%xmm6, 32(%%r13)         \n"     //  input 13
        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        //18 22 26 30
        "   vunpcklps %%zmm11, %%zmm9, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm17, %%zmm1    \n"
        
        "   vunpcklps %%zmm15, %%zmm13, %%zmm2    \n"
        "   vunpcklps %%zmm23, %%zmm21, %%zmm3    \n"
        
        "   movl    $0x55, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%zmm2, %%zmm4     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm3%{%%k1%}      \n"
        "   vmovups %%zmm3, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm3, %%zmm2%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        
        "   vmovups %%ymm2, (%%r10)         \n"     //  input 18
        "   vextractf64x4  $0x1,%%zmm2, %%ymm6      \n"
        "   vmovups %%ymm6, (%%r12)            \n"     //input 26
        
        "   vmovups %%ymm5, (%%r11)         \n"     //  input 22
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups %%ymm7, (%%r13)            \n"     //input 30
        
        //  
        "   vunpcklps %%zmm27, %%zmm25, %%zmm0    \n"
        "   vunpcklps %%zmm31, %%zmm29, %%zmm2    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%xmm2, 32(%%r10)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm2, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm2, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm2, %%xmm6      \n"    //  input 14
        
        "   vmovups %%xmm4, 32(%%r11)         \n"     //  input 6
        "   vmovups %%xmm5, 32(%%r12)         \n"     //  input 10
        "   vmovups %%xmm6, 32(%%r13)         \n"     //  input 14
        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        
        
        //19 23 27 31
        "   vunpckhps %%zmm11, %%zmm9, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm17, %%zmm1    \n"
        
        "   vunpckhps %%zmm15, %%zmm13, %%zmm2    \n"
        "   vunpckhps %%zmm23, %%zmm21, %%zmm3    \n"
        
        
        
        "   movl    $0xaa, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm4     \n"
        "   vpermq   $0x80, %%zmm3, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r10)         \n"     //  input 19
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm6      \n"
        "   vmovups %%ymm6, (%%r12)            \n"     //input 23
        
        "   vmovups %%ymm5, (%%r11)         \n"     //  input 27
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups %%ymm7, (%%r13)            \n"     //input 31
        
        //  
        "   vunpckhps %%zmm27, %%zmm25, %%zmm0    \n"
        "   vunpckhps %%zmm31, %%zmm29, %%zmm2    \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r10)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm0, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm0, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm0, %%xmm6      \n"    //  input 14
        
        "   vmovups %%xmm4, 32(%%r11)         \n"     //  input 6
        "   vmovups %%xmm5, 32(%%r12)         \n"     //  input 10
        "   vmovups %%xmm6, 32(%%r13)         \n"     //  input 14
        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        //20 24 28 32
        "   vunpckhps %%zmm11, %%zmm9, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm17, %%zmm1    \n"
        
        "   vunpckhps %%zmm15, %%zmm13, %%zmm2    \n"
        "   vunpckhps %%zmm23, %%zmm21, %%zmm3    \n"
        
        
        "   movl    $0x55, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%zmm2, %%zmm4     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm3%{%%k1%}      \n"
        "   vmovups %%zmm3, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm3, %%zmm2%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        
        "   vmovups %%ymm2, (%%r10)         \n"     //  input 20
        "   vextractf64x4  $0x1,%%zmm2, %%ymm6      \n"
        "   vmovups %%ymm6, (%%r12)            \n"     //input 24
        
        "   vmovups %%ymm5, (%%r11)         \n"     //  input 28
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups %%ymm7, (%%r13)            \n"     //input 32
        
        //  
        "   vunpckhps %%zmm27, %%zmm25, %%zmm0    \n"
        "   vunpckhps %%zmm31, %%zmm29, %%zmm2    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%xmm2, 32(%%r10)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm2, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm2, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm2, %%xmm6      \n"    //  input 14
        
        "   vmovups %%xmm4, 32(%%r11)         \n"     //  input 6
        "   vmovups %%xmm5, 32(%%r12)         \n"     //  input 10
        "   vmovups %%xmm6, 32(%%r13)         \n"     //  input 14
        
        ".endm      \n"
        
        //-----------------add
        
        ".macro ADD_C_12x32   \n"

        //  1 5 9 13
        "   vunpcklps %%zmm10, %%zmm8, %%zmm0    \n"
        "   vunpcklps %%zmm18, %%zmm16, %%zmm1    \n"
        
        "   vunpcklps %%zmm14, %%zmm12, %%zmm2    \n"
        "   vunpcklps %%zmm22, %%zmm20, %%zmm3    \n"
        
        
        "   movl    $0xaa, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm4     \n"
        "   vpermq  $0x80, %%zmm3, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm6      \n"
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        
        "   vmovups (%%r10), %%ymm1     \n"
        "   vmovups (%%r11), %%ymm2     \n"
        "   vmovups (%%r12), %%ymm3     \n"
        "   vmovups (%%r13), %%ymm4     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm2, %%ymm5, %%ymm5       \n"
        "   vaddps %%ymm3, %%ymm6, %%ymm6       \n"
        "   vaddps %%ymm4, %%ymm7, %%ymm7       \n"
        
        "   vmovups %%ymm0, (%%r10)         \n"
        "   vmovups %%ymm5, (%%r11)         \n"
        "   vmovups %%ymm6, (%%r12)         \n"
        "   vmovups %%ymm7, (%%r13)         \n"
        //  
        "   vunpcklps %%zmm26, %%zmm24, %%zmm0    \n"
        "   vunpcklps %%zmm30, %%zmm28, %%zmm2    \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm0, %%xmm4      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm5      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm6      \n"    //  input 13
        
        "   vmovups 32(%%r10), %%xmm1     \n"
        "   vmovups 32(%%r11), %%xmm2     \n"
        "   vmovups 32(%%r12), %%xmm3     \n"
        "   vmovups 32(%%r13), %%xmm7     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm2, %%xmm4, %%xmm4       \n"
        "   vaddps %%xmm3, %%xmm5, %%xmm5       \n"
        "   vaddps %%xmm7, %%xmm6, %%xmm6       \n"
        
        "   vmovups %%xmm0, 32(%%r10)         \n"
        "   vmovups %%xmm4, 32(%%r11)         \n"
        "   vmovups %%xmm5, 32(%%r12)         \n"
        "   vmovups %%xmm6, 32(%%r13)         \n"
        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        
        // 2 6 10 14
        
        "   vunpcklps %%zmm10, %%zmm8, %%zmm0    \n"
        "   vunpcklps %%zmm18, %%zmm16, %%zmm1    \n"
        
        "   vunpcklps %%zmm14, %%zmm12, %%zmm2    \n"
        "   vunpcklps %%zmm22, %%zmm20, %%zmm3    \n"
        
        "   movl    $0x55, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%zmm2, %%zmm4     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm3%{%%k1%}      \n"
        "   vmovups %%zmm3, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm3, %%zmm2%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm2, %%ymm6      \n"
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups (%%r10), %%ymm0     \n"
        "   vmovups (%%r11), %%ymm1     \n"
        "   vmovups (%%r12), %%ymm3     \n"
        "   vmovups (%%r13), %%ymm4     \n"
        
        "   vaddps %%ymm0, %%ymm2, %%ymm2       \n"
        "   vaddps %%ymm1, %%ymm5, %%ymm5       \n"
        "   vaddps %%ymm3, %%ymm6, %%ymm6      \n"
        "   vaddps %%ymm4, %%ymm7, %%ymm7       \n"
        
        "   vmovups %%ymm2, (%%r10)         \n"
        "   vmovups %%ymm5, (%%r11)         \n"
        "   vmovups %%ymm6, (%%r12)         \n"
        "   vmovups %%ymm7, (%%r13)         \n"
        
        //  
        "   vunpcklps %%zmm26, %%zmm24, %%zmm0    \n"
        "   vunpcklps %%zmm30, %%zmm28, %%zmm2    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm2, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm2, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm2, %%xmm6      \n"    //  input 14
        
        "   vmovups 32(%%r10), %%xmm0     \n"
        "   vmovups 32(%%r11), %%xmm1     \n"
        "   vmovups 32(%%r12), %%xmm3     \n"
        "   vmovups 32(%%r13), %%xmm7     \n"
        
        "   vaddps %%xmm0, %%xmm2, %%xmm2       \n"
        "   vaddps %%xmm1, %%xmm4, %%xmm4       \n"
        "   vaddps %%xmm3, %%xmm5, %%xmm5       \n"
        "   vaddps %%xmm7, %%xmm6, %%xmm6       \n"
        
        "   vmovups %%xmm2, 32(%%r10)         \n"
        "   vmovups %%xmm4, 32(%%r11)         \n"
        "   vmovups %%xmm5, 32(%%r12)         \n"
        "   vmovups %%xmm6, 32(%%r13)         \n"
        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        //3 7 11 15
        "   vunpckhps %%zmm10, %%zmm8, %%zmm0    \n"
        "   vunpckhps %%zmm18, %%zmm16, %%zmm1    \n"
        
        "   vunpckhps %%zmm14, %%zmm12, %%zmm2    \n"
        "   vunpckhps %%zmm22, %%zmm20, %%zmm3    \n"
        
        "   movl    $0xaa, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm4     \n"
        "   vpermq   $0x80, %%zmm3, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm6      \n"
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"

        
        "   vmovups (%%r10), %%ymm1     \n"
        "   vmovups (%%r11), %%ymm2     \n"
        "   vmovups (%%r12), %%ymm3     \n"
        "   vmovups (%%r13), %%ymm4     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm2, %%ymm5, %%ymm5       \n"
        "   vaddps %%ymm3, %%ymm6, %%ymm6       \n"
        "   vaddps %%ymm4, %%ymm7, %%ymm7       \n"
        
        "   vmovups %%ymm0, (%%r10)         \n"
        "   vmovups %%ymm5, (%%r11)         \n"
        "   vmovups %%ymm6, (%%r12)         \n"
        "   vmovups %%ymm7, (%%r13)         \n"
        //  
        "   vunpckhps %%zmm26, %%zmm24, %%zmm0    \n"
        "   vunpckhps %%zmm30, %%zmm28, %%zmm2    \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm0, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm0, %%xmm6      \n"    //  input 14
        
        "   vmovups 32(%%r10), %%xmm1     \n"
        "   vmovups 32(%%r11), %%xmm2     \n"
        "   vmovups 32(%%r12), %%xmm3     \n"
        "   vmovups 32(%%r13), %%xmm7     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm2, %%xmm4, %%xmm4       \n"
        "   vaddps %%xmm3, %%xmm5, %%xmm5       \n"
        "   vaddps %%xmm7, %%xmm6, %%xmm6       \n"
        
        "   vmovups %%xmm0, 32(%%r10)         \n"
        "   vmovups %%xmm4, 32(%%r11)         \n"
        "   vmovups %%xmm5, 32(%%r12)         \n"
        "   vmovups %%xmm6, 32(%%r13)         \n"
        
        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        
        
        //4 8 12 16
        "   vunpckhps %%zmm10, %%zmm8, %%zmm0    \n"
        "   vunpckhps %%zmm18, %%zmm16, %%zmm1    \n"
        
        "   vunpckhps %%zmm14, %%zmm12, %%zmm2    \n"
        "   vunpckhps %%zmm22, %%zmm20, %%zmm3    \n"
        
        "   movl    $0x55, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%zmm2, %%zmm4     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm3%{%%k1%}      \n"
        "   vmovups %%zmm3, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm3, %%zmm2%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        
        "   vextractf64x4  $0x1,%%zmm2, %%ymm6      \n"
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups (%%r10), %%ymm0     \n"
        "   vmovups (%%r11), %%ymm1     \n"
        "   vmovups (%%r12), %%ymm3     \n"
        "   vmovups (%%r13), %%ymm4     \n"
        
        "   vaddps %%ymm0, %%ymm2, %%ymm2       \n"
        "   vaddps %%ymm1, %%ymm5, %%ymm5       \n"
        "   vaddps %%ymm3, %%ymm6, %%ymm6       \n"
        "   vaddps %%ymm4, %%ymm7, %%ymm7       \n"
        
        "   vmovups %%ymm2, (%%r10)         \n"
        "   vmovups %%ymm5, (%%r11)         \n"
        "   vmovups %%ymm6, (%%r12)         \n"
        "   vmovups %%ymm7, (%%r13)         \n"
        
        //  
        "   vunpckhps %%zmm26, %%zmm24, %%zmm0    \n"
        "   vunpckhps %%zmm30, %%zmm28, %%zmm2    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm2, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm2, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm2, %%xmm6      \n"    //  input 14
        
        "   vmovups 32(%%r10), %%xmm0     \n"
        "   vmovups 32(%%r11), %%xmm1     \n"
        "   vmovups 32(%%r12), %%xmm3     \n"
        "   vmovups 32(%%r13), %%xmm7     \n"
        
        "   vaddps %%xmm0, %%xmm2, %%xmm2       \n"
        "   vaddps %%xmm1, %%xmm4, %%xmm4       \n"
        "   vaddps %%xmm3, %%xmm5, %%xmm5       \n"
        "   vaddps %%xmm7, %%xmm6, %%xmm6       \n"
        
        "   vmovups %%xmm2, 32(%%r10)         \n"
        "   vmovups %%xmm4, 32(%%r11)         \n"
        "   vmovups %%xmm5, 32(%%r12)         \n"
        "   vmovups %%xmm6, 32(%%r13)         \n"
        
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        
        "   shl $2, %%r8                \n"
        
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        //"   prefetcht2      (%%r11)                         \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        //"   prefetcht2      (%%r12)                         \n"
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        //"   prefetcht2      (%%r13)                         \n"
        
        "   shr $2, %%r8                \n"
        
        //17 21 25 29
        "   vunpcklps %%zmm11, %%zmm9, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm17, %%zmm1    \n"
        
        "   vunpcklps %%zmm15, %%zmm13, %%zmm2    \n"
        "   vunpcklps %%zmm23, %%zmm21, %%zmm3    \n"
        
        
        "   movl    $0xaa, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm4     \n"
        "   vpermq   $0x80, %%zmm3, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm6      \n"
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups (%%r10), %%ymm1     \n"
        "   vmovups (%%r11), %%ymm2     \n"
        "   vmovups (%%r12), %%ymm3     \n"
        "   vmovups (%%r13), %%ymm4     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm2, %%ymm5, %%ymm5       \n"
        "   vaddps %%ymm3, %%ymm6, %%ymm6       \n"
        "   vaddps %%ymm4, %%ymm7, %%ymm7       \n"
        
        "   vmovups %%ymm0, (%%r10)         \n"
        "   vmovups %%ymm5, (%%r11)         \n"
        "   vmovups %%ymm6, (%%r12)         \n"
        "   vmovups %%ymm7, (%%r13)         \n"
        
        //  
        "   vunpcklps %%zmm27, %%zmm25, %%zmm0    \n"
        "   vunpcklps %%zmm31, %%zmm29, %%zmm2    \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm0, %%xmm4      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm5      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm6      \n"    //  input 13
        
        
        "   vmovups 32(%%r10), %%xmm1     \n"
        "   vmovups 32(%%r11), %%xmm2     \n"
        "   vmovups 32(%%r12), %%xmm3     \n"
        "   vmovups 32(%%r13), %%xmm7     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm2, %%xmm4, %%xmm4       \n"
        "   vaddps %%xmm3, %%xmm5, %%xmm5       \n"
        "   vaddps %%xmm7, %%xmm6, %%xmm6       \n"
        
        "   vmovups %%xmm0, 32(%%r10)         \n"
        "   vmovups %%xmm4, 32(%%r11)         \n"
        "   vmovups %%xmm5, 32(%%r12)         \n"
        "   vmovups %%xmm6, 32(%%r13)         \n"
        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        
        
        //18 22 26 30
        "   vunpcklps %%zmm11, %%zmm9, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm17, %%zmm1    \n"
        
        "   vunpcklps %%zmm15, %%zmm13, %%zmm2    \n"
        "   vunpcklps %%zmm23, %%zmm21, %%zmm3    \n"
        
        
        "   movl    $0x55, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%zmm2, %%zmm4     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm3%{%%k1%}      \n"
        "   vmovups %%zmm3, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm3, %%zmm2%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm2, %%ymm6      \n"
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"
        "   vmovups (%%r10), %%ymm0     \n"
        "   vmovups (%%r11), %%ymm1     \n"
        "   vmovups (%%r12), %%ymm3     \n"
        "   vmovups (%%r13), %%ymm4     \n"
        
        "   vaddps %%ymm0, %%ymm2, %%ymm2       \n"
        "   vaddps %%ymm1, %%ymm5, %%ymm5       \n"
        "   vaddps %%ymm3, %%ymm6, %%ymm6      \n"
        "   vaddps %%ymm4, %%ymm7, %%ymm7       \n"
        
        "   vmovups %%ymm2, (%%r10)         \n"
        "   vmovups %%ymm5, (%%r11)         \n"
        "   vmovups %%ymm6, (%%r12)         \n"
        "   vmovups %%ymm7, (%%r13)         \n"
        
        //  
        "   vunpcklps %%zmm27, %%zmm25, %%zmm0    \n"
        "   vunpcklps %%zmm31, %%zmm29, %%zmm2    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm2, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm2, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm2, %%xmm6      \n"    //  input 14
        
        "   vmovups 32(%%r10), %%xmm0     \n"
        "   vmovups 32(%%r11), %%xmm1     \n"
        "   vmovups 32(%%r12), %%xmm3     \n"
        "   vmovups 32(%%r13), %%xmm7     \n"
        
        "   vaddps %%xmm0, %%xmm2, %%xmm2       \n"
        "   vaddps %%xmm1, %%xmm4, %%xmm4       \n"
        "   vaddps %%xmm3, %%xmm5, %%xmm5       \n"
        "   vaddps %%xmm7, %%xmm6, %%xmm6       \n"
        
        "   vmovups %%xmm2, 32(%%r10)         \n"
        "   vmovups %%xmm4, 32(%%r11)         \n"
        "   vmovups %%xmm5, 32(%%r12)         \n"
        "   vmovups %%xmm6, 32(%%r13)         \n"
        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        
        
        //19 23 27 31
        "   vunpckhps %%zmm11, %%zmm9, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm17, %%zmm1    \n"
        
        "   vunpckhps %%zmm15, %%zmm13, %%zmm2    \n"
        "   vunpckhps %%zmm23, %%zmm21, %%zmm3    \n"
        
        
        "   movl    $0xaa, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm4     \n"
        "   vpermq   $0x80, %%zmm3, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm6      \n"
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"

        "   vmovups (%%r10), %%ymm1     \n"
        "   vmovups (%%r11), %%ymm2     \n"
        "   vmovups (%%r12), %%ymm3     \n"
        "   vmovups (%%r13), %%ymm4     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm2, %%ymm5, %%ymm5       \n"
        "   vaddps %%ymm3, %%ymm6, %%ymm6       \n"
        "   vaddps %%ymm4, %%ymm7, %%ymm7       \n"
        
        "   vmovups %%ymm0, (%%r10)         \n"
        "   vmovups %%ymm5, (%%r11)         \n"
        "   vmovups %%ymm6, (%%r12)         \n"
        "   vmovups %%ymm7, (%%r13)         \n"
        //  
        "   vunpckhps %%zmm27, %%zmm25, %%zmm0    \n"
        "   vunpckhps %%zmm31, %%zmm29, %%zmm2    \n"
        
        "   vpermq   $0x80, %%zmm2, %%zmm0%{%%k1%}      \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm0, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm0, %%xmm6      \n"    //  input 14
        "   vmovups 32(%%r10), %%xmm1     \n"
        "   vmovups 32(%%r11), %%xmm2     \n"
        "   vmovups 32(%%r12), %%xmm3     \n"
        "   vmovups 32(%%r13), %%xmm7     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm2, %%xmm4, %%xmm4       \n"
        "   vaddps %%xmm3, %%xmm5, %%xmm5       \n"
        "   vaddps %%xmm7, %%xmm6, %%xmm6       \n"
        
        "   vmovups %%xmm0, 32(%%r10)         \n"
        "   vmovups %%xmm4, 32(%%r11)         \n"
        "   vmovups %%xmm5, 32(%%r12)         \n"
        "   vmovups %%xmm6, 32(%%r13)         \n"
        
        "   add %%r8, %%r10    \n"
        "   add %%r8, %%r11    \n"
        "   add %%r8, %%r12    \n"
        "   add %%r8, %%r13    \n"
        //20 24 28 32
        "   vunpckhps %%zmm11, %%zmm9, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm17, %%zmm1    \n"
        
        "   vunpckhps %%zmm15, %%zmm13, %%zmm2    \n"
        "   vunpckhps %%zmm23, %%zmm21, %%zmm3    \n"
        
        "   movl    $0x55, %%r15d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r15d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"
        "   vmovups %%zmm2, %%zmm4     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm3%{%%k1%}      \n"
        "   vmovups %%zmm3, %%zmm5     \n"
        
        "   vpermq  $0x40, %%zmm3, %%zmm2%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm4, %%zmm5%{%%k3%}        \n"
        
        "   vextractf64x4  $0x1,%%zmm2, %%ymm6      \n"
        "   vextractf64x4 $0x1, %%zmm5,  %%ymm7      \n"

        "   vmovups (%%r10), %%ymm0     \n"
        "   vmovups (%%r11), %%ymm1     \n"
        "   vmovups (%%r12), %%ymm3     \n"
        "   vmovups (%%r13), %%ymm4     \n"
        
        "   vaddps %%ymm0, %%ymm2, %%ymm2       \n"
        "   vaddps %%ymm1, %%ymm5, %%ymm5       \n"
        "   vaddps %%ymm3, %%ymm6, %%ymm6       \n"
        "   vaddps %%ymm4, %%ymm7, %%ymm7       \n"
        
        "   vmovups %%ymm2, (%%r10)         \n"
        "   vmovups %%ymm5, (%%r11)         \n"
        "   vmovups %%ymm6, (%%r12)         \n"
        "   vmovups %%ymm7, (%%r13)         \n"
        //  
        "   vunpckhps %%zmm27, %%zmm25, %%zmm0    \n"
        "   vunpckhps %%zmm31, %%zmm29, %%zmm2    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm2%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm2, %%xmm4      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm2, %%xmm5      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm2, %%xmm6      \n"    //  input 14
        
        "   vmovups 32(%%r10), %%xmm0     \n"
        "   vmovups 32(%%r11), %%xmm1     \n"
        "   vmovups 32(%%r12), %%xmm3     \n"
        "   vmovups 32(%%r13), %%xmm7     \n"
        
        "   vaddps %%xmm0, %%xmm2, %%xmm2       \n"
        "   vaddps %%xmm1, %%xmm4, %%xmm4       \n"
        "   vaddps %%xmm3, %%xmm5, %%xmm5       \n"
        "   vaddps %%xmm7, %%xmm6, %%xmm6       \n"
        
        "   vmovups %%xmm2, 32(%%r10)         \n"
        "   vmovups %%xmm4, 32(%%r11)         \n"
        "   vmovups %%xmm5, 32(%%r12)         \n"
        "   vmovups %%xmm6, 32(%%r13)         \n"
        
        ".endm      \n"


        "CONV_KERNEL12x32:                                \n"

        "   mov     %[output], %%rcx                             \n"
        "   mov     %[trans_filter], %%rbx                             \n"
        "   mov     %[input], %%r9                             \n"

        "   prefetcht0      (%%rax)                         \n"
        "   movl     %[LEN_HWb], %%esi                   \n"
        "   movl     %[Cb], %%edx                             \n"  // Cb
        "   movl     %[input_HW_size], %%r8d                             \n" 
        "   mov         %[input_buffer], %%r14                            \n"   //input_b
        "   movl        %[Kb], %%edi                             \n"


        "BEGIN_PACK:                                        \n"

        "   prefetcht0      (%%rbx)                         \n" 
        "   shl $2, %%r8                \n"
        "   mov     %%rcx, %%r10                            \n"  // C0
        "   mov     %%r9, %%rax                         \n"
        //"   prefetcht2      (%%r10)                         \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        //"   prefetcht2      (%%r11)                         \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        //"   prefetcht2      (%%r12)                         \n"
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        //"   prefetcht2      (%%r13)                         \n"
        
        "   shr $2, %%r8                \n"
        

        "   vmovups     (%%rbx), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  
        "   vmovups     64(%%rbx), %%zmm5                   \n"

        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"
        "   vpxorq      %%zmm13, %%zmm13, %%zmm13           \n" 
        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 
        "   vpxorq      %%zmm15, %%zmm15, %%zmm15           \n"
        "   vpxorq      %%zmm16, %%zmm16, %%zmm16           \n"
        "   vpxorq      %%zmm17, %%zmm17, %%zmm17           \n" 
        "   vpxorq      %%zmm18, %%zmm18, %%zmm18           \n" 
        "   vpxorq      %%zmm19, %%zmm19, %%zmm19           \n"

        "   vbroadcastss    (%%rax), %%zmm0                 \n"
        "   vbroadcastss    4(%%rax), %%zmm1                \n"

        "   vpxorq      %%zmm20, %%zmm20, %%zmm20           \n"
        "   vpxorq      %%zmm21, %%zmm21, %%zmm21           \n"
        //"   prefetcht2      64(%%r10)                       \n"
        "   vpxorq      %%zmm22, %%zmm22, %%zmm22           \n" 
        "   vpxorq      %%zmm23, %%zmm23, %%zmm23           \n"
        //"   prefetcht2      64(%%r11)                       \n"
        "   vpxorq      %%zmm24, %%zmm24, %%zmm24           \n"
        "   vpxorq      %%zmm25, %%zmm25, %%zmm25           \n"
        //"   prefetcht2      64(%%r12)                       \n"
        "   vpxorq      %%zmm26, %%zmm26, %%zmm26           \n" 
        "   vpxorq      %%zmm27, %%zmm27, %%zmm27           \n"
        //"   prefetcht2      64(%%r13)                       \n"
        "   vpxorq      %%zmm28, %%zmm28, %%zmm28           \n"
        "   vpxorq      %%zmm29, %%zmm29, %%zmm29           \n" 
        "   vpxorq      %%zmm30, %%zmm30, %%zmm30           \n" 
        "   vpxorq      %%zmm31, %%zmm31, %%zmm31           \n"   

        "   sub    $8, %%rdx                               \n"

        
        "MAIN_PACK_K:                                       \n"


        "   KERNEL12x32_PACK_K1                             \n"
        "   KERNEL12x32_PACK_K2                             \n"
        "   KERNEL12x32_PACK_K1                             \n"
        "   KERNEL12x32_PACK_K2                             \n"
        "   KERNEL12x32_PACK_K1                             \n"
        "   KERNEL12x32_PACK_K2                             \n"
        "   KERNEL12x32_PACK_K1                             \n"
        "   cmp     $0, %%rdx                               \n"
        
        "   je      PACK_SAVE                             \n"

        "   KERNEL12x32_PACK_K2                             \n"

        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_PACK_K                             \n"
        

        "PACK_SAVE:                                       \n"
        "   movl %[cc], %%r15d              \n"
        "   KERNEL12x32_PACK_END_K                          \n"
        "   mov     %[input_buffer], %%r14                            \n"
        "   cmp     $0, %%r15                               \n"

        "   je      PACK_ST_C                                  \n"
        //"   ADD_C_12x32                                     \n"
        "   jmp PACK_Kb_END                 \n"
        "PACK_ST_C:                                            \n"
        //"   ST_12x32                                      \n"
        
        "   movl     %[cc], %%r15d                         \n"
        "   movl     %[Cb], %%edx                             \n"  // Cb
        "PACK_Kb_END:               \n"
        "   sub    $1, %%rdi       \n"


        
        "   je     END_M                              \n"
        
        
        "BEGIN_M:                                           \n"

        
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        
        "   shl $2, %%r8                \n"
        "   movl %[cc], %%r15d          \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        //"   prefetcht2      (%%r11)                         \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        //"   prefetcht2      (%%r12)                         \n"
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        //"   prefetcht2      (%%r13)                         \n"
        
        "   shr $2, %%r8                \n"
        "   movl %[Cb], %%edx           \n"
        
        "BEGIN_K:                                           \n"

        "   vmovups     (%%rbx), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  
        "   vmovups     64(%%rbx), %%zmm5                   \n"

        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"
        "   vpxorq      %%zmm13, %%zmm13, %%zmm13           \n" 
        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 
        "   vpxorq      %%zmm15, %%zmm15, %%zmm15           \n"
        "   vpxorq      %%zmm16, %%zmm16, %%zmm16           \n"
        "   vpxorq      %%zmm17, %%zmm17, %%zmm17           \n" 
        "   vpxorq      %%zmm18, %%zmm18, %%zmm18           \n" 
        "   vpxorq      %%zmm19, %%zmm19, %%zmm19           \n"

        "   vbroadcastss    (%%rax), %%zmm0                 \n"
        "   vbroadcastss    4(%%rax), %%zmm1                \n"

        "   vpxorq      %%zmm20, %%zmm20, %%zmm20           \n"
        "   vpxorq      %%zmm21, %%zmm21, %%zmm21           \n" 
        "   vpxorq      %%zmm22, %%zmm22, %%zmm22           \n" 
        "   vpxorq      %%zmm23, %%zmm23, %%zmm23           \n"
        "   vpxorq      %%zmm24, %%zmm24, %%zmm24           \n"
        "   vpxorq      %%zmm25, %%zmm25, %%zmm25           \n" 
        "   vpxorq      %%zmm26, %%zmm26, %%zmm26           \n" 
        "   vpxorq      %%zmm27, %%zmm27, %%zmm27           \n"
        "   vpxorq      %%zmm28, %%zmm28, %%zmm28           \n"
        "   vpxorq      %%zmm29, %%zmm29, %%zmm29           \n" 
        "   vpxorq      %%zmm30, %%zmm30, %%zmm30           \n" 
        "   vpxorq      %%zmm31, %%zmm31, %%zmm31           \n"   
        
        "   sub $8, %%rdx                               \n"

        "MAIN_K:                                            \n"


        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        
        "   je      EDGE_K                                  \n"
        
        "   KERNEL12x32_K2                                  \n"

        "   sub    $8, %%rdx                               \n"
        
        "   jmp     MAIN_K                                  \n"
        
        "EDGE_K:                                            \n"
        
        "   KERNEL12x32_END_K                               \n"
        
        
        /*
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"

        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"

        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"

        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_K2                                  \n"
        "   KERNEL12x32_K1                                  \n"
        "   KERNEL12x32_END_K                               \n"
        */
        "BEGIN_SAVE:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C                                  \n"
        //"   ADD_C_12x32                                     \n"
        "   jmp Kb_END                  \n"
        "ST_C:                                            \n"
        //"   ST_12x32                                      \n"
        
        //"   movl     %[cc], %%r15d                         \n"
        //"   movl     %[Cb], %%edx                             \n"  // Cb
        "Kb_END:                    \n"
        "   sub     $1, %%rdi       \n"
        "   jne     BEGIN_M                              \n"
        
        
        "END_M:                                             \n"
        
        "   add $48, %%rcx               \n"
        "   add $48, %%r9               \n"
        "   movl     %[Cb], %%edx                             \n"  // Cb
        //"   sub $1, %%rsi      \n"
        "   movl     %[Kb], %%edi                             \n"
        "   mov     %[trans_filter], %%rbx                             \n"
        "   sub $1, %%rsi      \n"
        "   jne BEGIN_PACK      \n"
        
        

        :    
        :
         [output]               "m" (output),
         [trans_filter]         "m" (trans_filter),
         [input]                "m" (input),
         [Kb]                   "m" (Kb),
         [LEN_HWb]              "m" (LEN_HWb),
         [Cb]                   "m" (Cb),
         [input_HW_size]        "m" (input_HW_size),
         [input_buffer]         "m" (input_buffer),
         [cc]                   "m" (cc)
        :
         "rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
         "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
         "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
         "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
         "zmm30", "zmm31", "memory"
    
    );


}

void avx512_dircet_cnn_1x1s1(int H, int W, int N, int C, float *input, int K, int R, float *filter, float *output)
{
    int S = R;

    int out_W = W;
    int out_H = H;
    
    int stride_in = H * W;
    int stride_out = out_W * out_H;
    
    void *ptr, *ptr1;
    int Tn = NUM;               // N
    int Tm = NUM / Tn;          // 1
    posix_memalign(&ptr, 64, NUM * CONV_C * 12 * sizeof( float ));
    //posix_memalign(&ptr1, 64, CONV_C * Tm * CONV_K * sizeof( float ));
    float *NDIRECT_input = (float *)ptr;
    //float *NDIRECT_filter = (float *)ptr1;

    int input_HW_size = H * W;


    // -----------note------------------
    //int Num_K_block = (K / Tm + CONV_K - 1) / CONV_K;
    //int Num_C_block = (C + CONV_C - 1) / CONV_C;
    int Num_K_block = K /CONV_K;
    int Num_C_block = C / CONV_C;
    int dim_size = Num_K_block * Num_C_block * Tn;


    int thread_num_inputs = N / Tn;     // input channels s per thread  1
    int img_Tn = Tn / N;                // threads per input channel    1
    if(thread_num_inputs > 1)
        img_Tn = 1;
    else
        thread_num_inputs = 1;

    int gride_HW_size = input_HW_size / img_Tn;     // input_HW_size
    int gride_K_size = K / Tm;                      // K
    int gride_HW_e = input_HW_size % img_Tn;        // 0
    int gride_K_e = K % Tm;                         // 0
    
    //printf("input: %p, output: %p, filter: %p \n", input, output, NDIRECT_filter);
    
    #pragma omp parallel num_threads(NUM)
    {
        int i, j, k, C_index, K_index, pre_index;
        int id = omp_get_thread_num();
        int ii, jj, cc, nn, iis, jjs, ccs, HW_to, K_to, C_to;
        int HWb, Kb, Cb;
        int N_id = id / Tn;         // 0
        
        
        if (id  % img_Tn < gride_HW_e)
        {
            jjs = (id % img_Tn) * (gride_HW_size + 1);
            HW_to = jjs + (gride_HW_size + 1);
        }
        else
        {
            jjs = (id % img_Tn) * gride_HW_size + gride_HW_e;   // 0
            HW_to = jjs + gride_HW_size;        // gride_HW_size
        }

        iis = N_id * gride_K_size;          // 0
        K_to = iis + gride_K_size;          // K

        ccs = 0;
        C_to = C;

        #pragma omp barrier
        
        for ( jj = jjs; jj < HW_to; jj = jj + HWb)          // 0 gride_HW_size HWb
        {
            HWb = CONV_HW;
            if (HW_to - jj < CONV_HW)
            {
                HWb = HW_to - jj;
            }
            //vk * CONV_C * (K/vk) * (C/CONV_c)
            for (cc = ccs; cc < C_to; cc = cc + Cb)         // 0 C Cb
            {

                C_index = cc / CONV_C;
                Cb = CONV_C;
                if (C_to - cc < CONV_C)
                    Cb = C_to - cc;

                float *buffer_input = input + ((id % Tn) / img_Tn)  * thread_num_inputs * C * input_HW_size + cc * input_HW_size + jj;      //input + id * CHW + cc* HW + 0(jj)
                float *buffer_filter = filter + cc * K;
                for ( ii = iis ; ii < K_to; ii = ii + Kb)   // 0 K Kb
                {
                    
                    pre_index = C_index * Num_K_block * Tn + (ii - iis) / CONV_K * Tn;
                    Kb = CONV_K;
                    if (K_to - ii < CONV_K)
                    {
                        Kb = K_to - ii;
                    }

                    float *buffer_filter1 = buffer_filter + ii * Cb; //filter + kC + 0(cc)
                    float *buffer_output = output + ( (id % Tn) / img_Tn) *thread_num_inputs * K * input_HW_size + ii * input_HW_size + jj ;    //output + id * KHW + ii* HW + 0(jj)

                    //int Num_K_tiles = (Kb + 31) / 32;
                    //int Num_K_tiles = Kb / 32;
                    //float *temp_NDIRECT_filter = NDIRECT_filter + (id / Tn) * CONV_C * CONV_K;  // NDIRECT_filter
                    //int EDGE_Kb = Kb % 32;



                    
                    
                    for(nn = 0; nn < thread_num_inputs; nn++)       // 0 1 1
                    {
                        float *buffer_output1 = buffer_output + nn * K * input_HW_size; //output + id * KHW + ii* HW + 0(jj)
                        float *buffer_input1 = buffer_input + nn * C * input_HW_size; //input + id * CHW + cc* HW + 0(jj)
                        int EDGE_HWb = HWb % 12;
                        int LEN_HWb = HWb - EDGE_HWb;
                        if (LEN_HWb > 0)
                            //printf("here    \n");
                            //printf("buffer_output1: %p, temp_NDIRECT_filter: %p \n", buffer_output1, temp_NDIRECT_filter);
                            //printf("buffer_input1: %p, NDIRECT_input: %p \n", buffer_input1, NDIRECT_input);
                            
                            //printf("Kb/32: %d, LEN_HWb/12: %d, Cb: %d, input_HW_size<<2: %d, cc: %d \n", Kb/32, LEN_HWb/12, Cb, input_HW_size<<2, cc);
                            
                            direct_1x1_N32M12_AVX512_pack(buffer_output1, buffer_filter1, buffer_input1, Kb/32, LEN_HWb/12, Cb, input_HW_size<<2, &NDIRECT_input[id * CONV_C * 12], cc);

                            //printf("here end   \n");
                        /*
                        if (EDGE_HWb != 0)      // 4 1
                        {
                            float *temp_buffer_input = buffer_input1 + LEN_HWb;
                            float *temp_buffer_output = buffer_output1 + LEN_HWb;


                            if (EDGE_HWb >= 4)
                            {

                                direct_1x1_N32M4_AVX512_pack(temp_buffer_output, temp_NDIRECT_filter, temp_buffer_input, Kb, 4, Cb, input_HW_size, C, &NDIRECT_input[id * CONV_C * 12], cc);
                                EDGE_HWb = EDGE_HWb - 4;
                                temp_buffer_input = temp_buffer_input + 4;
                                temp_buffer_output = temp_buffer_output + 4;
                            }

                            if (EDGE_HWb >= 1)
                            {
                                direct_1x1_N32M7_AVX512_pack(temp_buffer_output, temp_NDIRECT_filter, temp_buffer_input, Kb, 1, Cb, input_HW_size, C, &NDIRECT_input[id * CONV_C * 12], cc);
                            }
                        }
                        */
                    }


                }
            }

        }
        
        
    }
    
    free(NDIRECT_input);
}


int main(int argc, char *argv[]){
    //openblas_set_num_threads(1);
    int j, loop = 1;
    int pc, lda, ldb, ldc;
    double start, cost;
    
    int H = 12, W = 1, N = NUM, C = 32;
    int K = 32, R = 1, S = 1;
    int padh = 0, padw = 0, stride = 1;
    
    int i=1;
    
    if (argc > 1) loop         = atoi(argv[i++]);
    if (argc > 2) C         = atoi(argv[i++]);
    if (argc > 3) K         = atoi(argv[i++]);
    if (argc > 4) H         = atoi(argv[i++]);
    if (argc > 5) W         = atoi(argv[i++]);
    if (argc > 6) R         = atoi(argv[i++]);
    if (argc > 7) S         = atoi(argv[i++]);
    if (argc > 8) stride    = atoi(argv[i++]);
    if (argc > 9) padh      = atoi(argv[i++]);
    if (argc > 10) padw      = atoi(argv[i++]);
    if (argc > 11) padw      = atoi(argv[i++]);
    
    double pes = 128;
    
    int out_H = H;
    int out_W = W;
    int flag = 0;
    double ops = (double)2.0 * N * C * out_H * out_W * R * S * K * 1.0e-9;
    
    int stride_out_size = out_H*out_W*sizeof(float);
    int stride_in_size = H*W*sizeof(float);
    
    double result = R*S*C*1.0;
    
    float *filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 64);
    
    if(filter == NULL ){
        printf("filter aligned error!\n");
        return 0;
        
    }
    
    float *trans_filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 64);
    //float *trans_filter = (float*)_mm_malloc(K*C*3*3*sizeof(float), 32);
    
    if(filter == NULL ){
        printf("trans_filter aligned error!\n");
        return 0;
        
    }
    
    float *input =  (float*)_mm_malloc((size_t)N*C*H*W*sizeof(float), 64);
    
    if(input == NULL ){
        printf("input aligned error!\n");
        return 0;
        
    }
    
    float *output = (float*)_mm_malloc((size_t)N*K*out_H*out_W*sizeof(float), 64);
    
    if(output == NULL ){
        printf("output aligned error!\n");
        return 0;
        
    }

    float *output1 = (float*)malloc(N*K*out_H*out_W*sizeof(float));
    float *data_col = (float*)malloc(N*C*(H/stride*W/stride)*R*S*sizeof(float));
    //float *output1 = (float*)_mm_malloc(N*K*H*W*sizeof(float), 32);
    printf("N = %d, K = %d, C = %d, H = %d, W = %d\n", N, K, C, H, W);
    for(int pc = 0; pc < 1; pc++)
    {
        random_matrix(K, C * R *S, filter);
        random_matrix(N, C * H * W, input);
        
        //printf("1\n");
        transform_filter(K, C, filter, trans_filter);
        //verify_transform_filter(K, C, trans_filter);
        /*
        for(i = 0; i < 5; i++)
        {
            avx512_dircet_cnn_1x1s1(H,W,N,C,input,K,R,trans_filter, output);
        }
        */
        //printf("3\n");
        start = dclock();
        for(i = 0; i < loop; i++)
        {
            avx512_dircet_cnn_1x1s1(H,W,N,C,input,K,R,trans_filter, output);
        }
        cost = (dclock() - start) / loop;

        
        printf("Gflops = %.3f, effic = %.3f %\n", ops/cost, ops/cost/NUM/pes * 100);
    
    }
    
    
    lda = C * R * S;
    ldb = H * W;
    ldc = H  * W;

    int input_HW_size = H * W;
    int output_HW_size =  H * W;
    
    for(int i = 0; i < N; i++)
    {

      im2col_cpu(input + i * input_HW_size * C , C, H, W, R, S, padh, padh, stride, stride, 1, 1, data_col + i * C * output_HW_size);


      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, output_HW_size, C * R *S, 1.0, filter, lda, 
                  data_col + i * C * output_HW_size, ldb, 0.0, output1 + i * K * output_HW_size, ldc);
      
    }
    
    int M;
    
    M = N * K;
    N = out_H * out_W;
    //printf("i = %d, j= %d\n",0 ,0 );
    //printf("out0= %lf \n", output[0]);
    //printf("out1= %lf \n", output[1]);
    //printf("out2= %lf \n", output[2]);
    // 比对im2col + gemm与 直接卷积的结果是否一致，来确保自己写的代码是否正确
    /*
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            //if((fabs(output[i * N + j] - result) > 0.001))
            //if((fabs(output[i * N + j] - output1[i * N + j]) > 0.001) && (fabs(output[i * N + j]) < 0.001))
            if((fabs(output[i * N + j] - output1[i * N + j]) > 0.001))
            {
                printf("i = %d, j= %d\n",i+1 ,j+1 );
                printf("out= %lf , out1= %lf\n", output[i*N + j], output1[i*N + j]);
                flag =1;
            }
        }
    }
    

    if(flag == 0)
        printf("结果正确\n");
    */

    free(filter);
    free(input);
    free(output);
    free(output1);
    free(data_col);
    return 0;
    
}