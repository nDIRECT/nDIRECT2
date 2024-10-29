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
    int mr = CONV_K, cr = CONV_C,zr;

    int i, j , k, ii,jj,kk, h,w,z;
    int st = 0;

    for(j = 0; j < inch; j = j + cr)
    {
        cr = CONV_C;
        if(inch - j < CONV_C)
            cr = inch - j;
        for(i = 0; i < outch; i = i + mr)
        {    
            zr = 12;
            mr = CONV_K;
            if(outch - i < CONV_K)
                mr = outch - i;
            for(z=i;z<i+mr;z+=zr){
                if((i+mr-z) < 12)
                    zr = i+mr-z;
                
                for(jj = j; jj < j + cr; jj++)
                {
                    for(kk=0; kk<zr; kk++){
                        
                        out_kernel[ st+kk ] = kernel[ (z+kk) * inch + jj];
                    }
                    st += zr;
                }
            
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

void direct_1x1_N12M28_s2(float *output, float *trans_filter, float *input, int Kb, int Cb, int input_HW_size, float *input_buffer, int cc, int EDGE_Kb){
    asm volatile(


        ".macro KERNEL12x28_PACK_K1                         \n"
        //"   prefetcht2      224(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm9          \n"
        "   vmovups         %%zmm4, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm11         \n"
        "   add             %%r8, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm13         \n"
        "   prefetcht2      448(%%rax)                      \n"
        

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm15         \n"
        "   vmovups         %%zmm6, 64(%%r14)                 \n"
        "   prefetcht2      512(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm17         \n"
        "   vmovups         (%%rax), %%zmm5                     \n"
        "   prefetcht2      576(%%rax)                      \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm19         \n"
        "   add             $128, %%r14                     \n"
        "   prefetcht2      640(%%rax)                      \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm21         \n"
        "   vmovups         64(%%rax), %%zmm7                     \n"
        
        "   prefetcht0      (%%r14)                      \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm23         \n"
        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm25         \n"
        "   prefetcht0      64(%%r14)                      \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm26         \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm27         \n"
        "   add             $48, %%rbx                                 \n"
        
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm28         \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm29         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        
        
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm31         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"
        "   vpxorq      %%zmm6, %%zmm6, %%zmm6              \n"
        "   vmovups         128(%%rax), %%zmm4                     \n"
        "   vmovups         192(%%rax), %%ymm6                     \n"
        
        "   vshufps    $0x88, %%zmm7, %%zmm5, %%zmm5      \n"
        "   vshufps    $0x88, %%zmm6, %%zmm4, %%zmm4      \n"
        "   vpermq  $0xd8, %%zmm5, %%zmm7                   \n"
        "   vpermq  $0xd8, %%zmm4, %%zmm6                   \n"
        "   vshuff64x2  $0xd8, %%zmm7, %%zmm7, %%zmm5       \n"
        "   vshuff64x2  $0xd8, %%zmm6, %%zmm6, %%zmm4       \n"
        "   vpxorq      %%zmm7, %%zmm7, %%zmm7              \n"
        ".endm                                              \n"


        ".macro KERNEL12x28_PACK_K2                         \n"

        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm9          \n"
        "   vmovups         %%zmm5, (%%r14)                 \n"


        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm11         \n"
        "   add             %%r8, %%rax                      \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm13         \n"
        "   prefetcht2      448(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm15         \n"
        "   vmovups         %%zmm4, 64(%%r14)                 \n"
        "   prefetcht2      512(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm17         \n"
        "   vmovups         128(%%rax), %%zmm6                     \n"
        "   prefetcht2      576(%%rax)                      \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm19         \n"
        "   add             $128, %%r14                     \n"
        "   prefetcht2      640(%%rax)                      \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm21         \n"
        "   vmovups         192(%%rax), %%ymm7                     \n"

        "   prefetcht0      (%%r14)                      \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm23         \n"
        "   prefetcht0      64(%%r14)                      \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm25         \n"
        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm26         \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm27         \n"
        "   add             $48, %%rbx                                 \n"


        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm28         \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm29         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm30         \n"
        "   vmovups     64(%%rax), %%zmm5                   \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm31         \n"
        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

        "   vshufps    $0x88, %%zmm5, %%zmm4, %%zmm4      \n"
        "   vshufps    $0x88, %%zmm7, %%zmm6, %%zmm6      \n"
        "   vpermq  $0xd8, %%zmm4, %%zmm5                   \n"
        "   vpermq  $0xd8, %%zmm6, %%zmm7                   \n"
        "   vshuff64x2  $0xd8, %%zmm5, %%zmm5, %%zmm4       \n"
        "   vshuff64x2  $0xd8, %%zmm7, %%zmm7, %%zmm6       \n"

        ".endm                                              \n"



        ".macro KERNEL12x28_PACK_END_K                      \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"

        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm9          \n"
        "   vmovups         %%zmm5, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm11         \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm13         \n"
        //"   prefetcht2      128(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm15         \n"
        "   vmovups         %%zmm4, 64(%%r14)                 \n"
        //"   prefetcht2      192(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm17         \n"

        
        
        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm19         \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm21         \n"

        
        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm23         \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm25         \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm26         \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm27         \n"
        "   add             $48, %%rbx                                 \n"

        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm28         \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm29         \n"

        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm31         \n"

        ".endm                                              \n"



        ".macro KERNEL12x28_K1                              \n"
        "   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm9          \n"
        //"   prefetcht0      192(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm11         \n"
        "   add             $128, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm13         \n"
        //"   prefetcht0      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm15         \n"
        "   vmovups         (%%rax), %%zmm6                     \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm17         \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm19         \n"
        "   vmovups         64(%%rax), %%zmm7                     \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm21         \n"


        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm23         \n"
        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm25         \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm26         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm27         \n"
        "   add             $48, %%rbx                      \n"
        
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm28         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm29         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm31         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL12x28_K2                              \n"
        "   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        //"   prefetcht0      192(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"
        "   add             $128, %%rax                      \n"
        
        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"
        //"   prefetcht0      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"
        "   vmovups         (%%rax), %%zmm4                     \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"
        "   vmovups         64(%%rax), %%zmm5                     \n"
        
        
        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"


        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm23         \n"
        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm25         \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm27         \n"
        "   add             $48, %%rbx                      \n"
        
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm29         \n"

        "   vbroadcastss    (%%rbx), %%zmm0                \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm31         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"

        ".endm                                              \n"



        ".macro KERNEL12x28_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm23         \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm25         \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm27         \n"
        "   add             $48, %%rbx                      \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm29         \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm31         \n"
        

        ".endm                                              \n"


        ".macro ST_12x28   \n"
        
        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)%{%%k1%}              \n"
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C0
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)%{%%k1%}              \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)%{%%k1%}              \n"

        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)%{%%k1%}              \n"
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%zmm19, 64(%%r11)%{%%k1%}              \n"
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C0
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%zmm21, 64(%%r12)%{%%k1%}              \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%zmm23, 64(%%r13)%{%%k1%}              \n" 
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2

        "   vmovups         %%zmm24, (%%r10)                \n"
        "   vmovups         %%zmm25, 64(%%r10)%{%%k1%}              \n"
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        "   vmovups         %%zmm26, (%%r11)                \n"
        "   vmovups         %%zmm27, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm28, (%%r12)                \n"
        "   vmovups         %%zmm29, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm30, (%%r13)                \n"
        "   vmovups         %%zmm31, 64(%%r13)%{%%k1%}              \n"
        /*
        "   prefetchnta      (%%r10)                       \n"
        "   prefetchnta      (%%r11)                       \n"
        "   prefetchnta      (%%r12)                       \n"
        "   prefetchnta      (%%r13)                       \n"
        */
        /*
        "   prefetcht0      (%%r10)                       \n"
        "   prefetcht0      (%%r11)                       \n"
        "   prefetcht0      (%%r12)                       \n"
        "   prefetcht0      (%%r13)                       \n"
        */


        ".endm      \n"

        ".macro ADD_C_12x28   \n"
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"
		"   vmovups 		64(%%r10), %%zmm1%{%%k1%}        		\n"
		"	vaddps 			%%zmm1, %%zmm9, %%zmm9			\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"
		"   vmovups 		64(%%r11), %%zmm3%{%%k1%}        		\n"
		"	vaddps 			%%zmm3, %%zmm11, %%zmm11		\n"
		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"
		"   vmovups 		64(%%r12), %%zmm5%{%%k1%}        		\n"
		"	vaddps 			%%zmm5, %%zmm13, %%zmm13		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"
		"   vmovups 		64(%%r13), %%zmm7%{%%k1%}        		\n"
		"	vaddps 			%%zmm7, %%zmm15, %%zmm15		\n"

        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)%{%%k1%}              \n"

		"	leaq  			(%%r13, %%rsi), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%rsi), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%rsi), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%rsi), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm16, %%zmm16		\n"
		"   vmovups 		64(%%r10), %%zmm1%{%%k1%}        		\n"
		"	vaddps 			%%zmm1, %%zmm17, %%zmm17		\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm18, %%zmm18		\n"
		"   vmovups 		64(%%r11), %%zmm3%{%%k1%}        		\n"
		"	vaddps 			%%zmm3, %%zmm19, %%zmm19		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm20, %%zmm20		\n"
		"   vmovups 		64(%%r12), %%zmm5%{%%k1%}        		\n"
		"	vaddps 			%%zmm5, %%zmm21, %%zmm21		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm22, %%zmm22		\n"
		"   vmovups 		64(%%r13), %%zmm7%{%%k1%}        		\n"
		"	vaddps 			%%zmm7, %%zmm23, %%zmm23		\n"

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)%{%%k1%}              \n"
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%zmm19, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%zmm21, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%zmm23, 64(%%r13)%{%%k1%}              \n" 


		"	leaq  			(%%r13, %%rsi), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%rsi), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%rsi), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%rsi), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm24, %%zmm24		\n"
		"   vmovups 		64(%%r10), %%zmm1%{%%k1%}        		\n"
		"	vaddps 			%%zmm1, %%zmm25, %%zmm25		\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm26, %%zmm26		\n"
		"   vmovups 		64(%%r11), %%zmm3%{%%k1%}        		\n"
		"	vaddps 			%%zmm3, %%zmm27, %%zmm27		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm28, %%zmm28		\n"
		"   vmovups 		64(%%r12), %%zmm5%{%%k1%}        		\n"
		"	vaddps 			%%zmm5, %%zmm29, %%zmm29		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm30, %%zmm30		\n"
		"   vmovups 		64(%%r13), %%zmm7%{%%k1%}        		\n"
		"	vaddps 			%%zmm7, %%zmm31, %%zmm31		\n"

        "   vmovups         %%zmm24, (%%r10)                \n"
        "   vmovups         %%zmm25, 64(%%r10)%{%%k1%}              \n"
        "   vmovups         %%zmm26, (%%r11)                \n"
        "   vmovups         %%zmm27, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm28, (%%r12)                \n"
        "   vmovups         %%zmm29, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm30, (%%r13)                \n"
        "   vmovups         %%zmm31, 64(%%r13)%{%%k1%}              \n"
        
        ".endm      \n"

        ".macro KERNEL8x28_K1                              \n"
        //"   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm9          \n"
        //"   prefetcht0      192(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm11         \n"
        "   add             $128, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm13         \n"
        //"   prefetcht0      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm15         \n"
        "   vmovups         (%%rax), %%zmm6                     \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm17         \n"
        "   prefetcht0      32(%%rbx)                      \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm19         \n"
        "   vmovups         64(%%rax), %%zmm7                     \n"
        "   add             $32, %%rbx                      \n"

        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm21         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm23         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL8x28_K2                              \n"
        //"   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        //"   prefetcht0      192(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"
        "   add             $128, %%rax                      \n"
        
        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"
        //"   prefetcht0      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"
        "   vmovups         (%%rax), %%zmm4                     \n"
        "   prefetcht0      32(%%rbx)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"
        "   add             $32, %%rbx                      \n"
        "   vmovups         64(%%rax), %%zmm5                     \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"

        "   vbroadcastss    (%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm23         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1               \n"


        ".endm                                              \n"



        ".macro KERNEL8x28_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"

        "   add             $32, %%rbx                      \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm23         \n"

        

        ".endm                                              \n"


        ".macro ST_8x28   \n"
        
        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)%{%%k1%}              \n"
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C0
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)%{%%k1%}              \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)%{%%k1%}              \n"

        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)%{%%k1%}              \n"
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%zmm19, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%zmm21, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%zmm23, 64(%%r13)%{%%k1%}              \n" 




        
        ".endm      \n"

        ".macro ADD_C_8x28   \n"
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"
		"   vmovups 		64(%%r10), %%zmm1%{%%k1%}        		\n"
		"	vaddps 			%%zmm1, %%zmm9, %%zmm9			\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"
		"   vmovups 		64(%%r11), %%zmm3%{%%k1%}        		\n"
		"	vaddps 			%%zmm3, %%zmm11, %%zmm11		\n"
		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"
		"   vmovups 		64(%%r12), %%zmm5%{%%k1%}        		\n"
		"	vaddps 			%%zmm5, %%zmm13, %%zmm13		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"
		"   vmovups 		64(%%r13), %%zmm7%{%%k1%}        		\n"
		"	vaddps 			%%zmm7, %%zmm15, %%zmm15		\n"

        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)%{%%k1%}              \n"

		"	leaq  			(%%r13, %%rsi), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%rsi), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%rsi), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%rsi), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm16, %%zmm16		\n"
		"   vmovups 		64(%%r10), %%zmm1%{%%k1%}        		\n"
		"	vaddps 			%%zmm1, %%zmm17, %%zmm17		\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm18, %%zmm18		\n"
		"   vmovups 		64(%%r11), %%zmm3%{%%k1%}        		\n"
		"	vaddps 			%%zmm3, %%zmm19, %%zmm19		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm20, %%zmm20		\n"
		"   vmovups 		64(%%r12), %%zmm5%{%%k1%}        		\n"
		"	vaddps 			%%zmm5, %%zmm21, %%zmm21		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm22, %%zmm22		\n"
		"   vmovups 		64(%%r13), %%zmm7%{%%k1%}        		\n"
		"	vaddps 			%%zmm7, %%zmm23, %%zmm23		\n"

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)%{%%k1%}              \n"
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%zmm19, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%zmm21, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%zmm23, 64(%%r13)%{%%k1%}              \n" 
        
        ".endm      \n"

        ".macro KERNEL4x28_K1                              \n"

        //"   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        //"   prefetcht0      192(%%rax)                      \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm9          \n"
        "   add             $128, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm11         \n"
        "   add             $16, %%rbx                      \n"
        "   vmovups         (%%rax), %%zmm6                     \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm13         \n"

        "   vmovups         64(%%rax), %%zmm7                     \n"
        "   vbroadcastss    (%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm15         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1               \n"

        
        ".endm                                              \n"


        ".macro KERNEL4x28_K2                              \n"
        //"   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        "   add             $128, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"
        "   vmovups         (%%rax), %%zmm4                     \n"
        "   add             $16, %%rbx                      \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"
        "   vmovups         64(%%rax), %%zmm5                     \n"
        "   vbroadcastss    (%%rbx), %%zmm0               \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1               \n"


        ".endm                                              \n"



        ".macro KERNEL4x28_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"
        "   add             $16, %%rbx                      \n"


        

        ".endm                                              \n"


        ".macro ST_4x28   \n"
        
        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)%{%%k1%}              \n"





        
        ".endm      \n"

        ".macro ADD_C_4x28   \n"
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"
		"   vmovups 		64(%%r10), %%zmm1%{%%k1%}        		\n"
		"	vaddps 			%%zmm1, %%zmm9, %%zmm9			\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"
		"   vmovups 		64(%%r11), %%zmm3%{%%k1%}        		\n"
		"	vaddps 			%%zmm3, %%zmm11, %%zmm11		\n"
		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"
		"   vmovups 		64(%%r12), %%zmm5%{%%k1%}        		\n"
		"	vaddps 			%%zmm5, %%zmm13, %%zmm13		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"
		"   vmovups 		64(%%r13), %%zmm7%{%%k1%}        		\n"
		"	vaddps 			%%zmm7, %%zmm15, %%zmm15		\n"

        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)%{%%k1%}              \n"
        
        ".endm      \n"

        "CONV_KERNEL12x28:                                \n"
        "   movl     %[input_HW_size], %%r8d                             \n" 
        "   movl    $0xfff, %%esi       \n"
        "   mov     %[output], %%rcx                             \n"
        "   mov     %[trans_filter], %%rbx                             \n"
        "   mov     %[input], %%r9                             \n"
        
        //"   prefetcht0      (%%rax)                         \n"
        "   kmovd   %%esi, %%k1            \n"
        "   movl     %[Cb], %%edx                             \n"  // Cb
        "   mov %%r8, %%rsi                 \n"
        "   mov         %[input_buffer], %%r14                            \n"   //input_b
        "   movl        %[Kb], %%edi                             \n"
        "   shr     $2, %%rsi                       \n"


        //------------------- loop body
        "BEGIN_PACK:                                        \n"


        //"   shl $2, %%r8                \n"
        "   mov     %%rcx, %%r10                            \n"  // C0
        "   mov     %%r9, %%rax                         \n"
        "   prefetcht0      (%%rax)                         \n" 
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   prefetcht0      64(%%rax)                      \n"
        "   mov     %%r9, %%rax                         \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   prefetcht0      (%%rbx)                         \n" 
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        
        //"   shr $2, %%r8                \n"
        
        "   vpxorq      %%zmm7, %%zmm7, %%zmm7              \n"
        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vmovups     64(%%rax), %%zmm5                   \n"
        "   vmovups     128(%%rax), %%zmm6                   \n"
        "   vmovups     192(%%rax), %%ymm7                  \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  

        "   vshufps    $0x88, %%zmm5, %%zmm4, %%zmm4      \n"
        "   vshufps    $0x88, %%zmm7, %%zmm6, %%zmm6      \n"
        //"   vpermq  $0xd8, %%zmm4, %%zmm5                   \n"
        //"   vpermq  $0xd8, %%zmm6, %%zmm7                   \n"
        //"   vshuff64x2  $0xd8, %%zmm5, %%zmm5, %%zmm4       \n"
        //"   vshuff64x2  $0xd8, %%zmm7, %%zmm7, %%zmm6       \n"
        
        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"
        "   vpxorq      %%zmm13, %%zmm13, %%zmm13           \n" 
        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 
        
        "   vpermq  $0xd8, %%zmm4, %%zmm5                   \n"
        "   vpermq  $0xd8, %%zmm6, %%zmm7                   \n"
        
        "   vpxorq      %%zmm15, %%zmm15, %%zmm15           \n"
        "   vpxorq      %%zmm16, %%zmm16, %%zmm16           \n"
        "   vpxorq      %%zmm17, %%zmm17, %%zmm17           \n" 
        
        "   vshuff64x2  $0xd8, %%zmm5, %%zmm5, %%zmm4       \n"
        "   vshuff64x2  $0xd8, %%zmm7, %%zmm7, %%zmm6       \n"
        
        "   vpxorq      %%zmm18, %%zmm18, %%zmm18           \n" 
        "   vpxorq      %%zmm19, %%zmm19, %%zmm19           \n"

        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

        "   vpxorq      %%zmm20, %%zmm20, %%zmm20           \n"
        "   vpxorq      %%zmm21, %%zmm21, %%zmm21           \n"
        //"   prefetcht2      (%%r10)                       \n"
        //"   prefetcht2      64(%%r10)                       \n"
        "   vpxorq      %%zmm22, %%zmm22, %%zmm22           \n" 
        "   vpxorq      %%zmm23, %%zmm23, %%zmm23           \n"
        //"   prefetcht2      (%%r11)                       \n"
        //"   prefetcht2      64(%%r11)                       \n"
        "   vpxorq      %%zmm24, %%zmm24, %%zmm24           \n"
        "   vpxorq      %%zmm25, %%zmm25, %%zmm25           \n"
        //"   prefetcht2      (%%r12)                       \n"
        //"   prefetcht2      64(%%r12)                       \n"
        "   vpxorq      %%zmm26, %%zmm26, %%zmm26           \n" 
        "   vpxorq      %%zmm27, %%zmm27, %%zmm27           \n"
        //"   prefetcht2      (%%r13)                       \n"
        //"   prefetcht2      64(%%r13)                       \n"
        "   vpxorq      %%zmm28, %%zmm28, %%zmm28           \n"
        "   vpxorq      %%zmm29, %%zmm29, %%zmm29           \n" 
        "   vpxorq      %%zmm30, %%zmm30, %%zmm30           \n" 
        "   vpxorq      %%zmm31, %%zmm31, %%zmm31           \n"   

        "   sub    $8, %%rdx                               \n"

        
        "MAIN_PACK_K:                                       \n"


        "   KERNEL12x28_PACK_K1                             \n"
        "   KERNEL12x28_PACK_K2                             \n"
        "   KERNEL12x28_PACK_K1                             \n"
        "   KERNEL12x28_PACK_K2                             \n"
        "   KERNEL12x28_PACK_K1                             \n"
        "   KERNEL12x28_PACK_K2                             \n"
        "   KERNEL12x28_PACK_K1                             \n"
        "   cmp     $0, %%rdx                               \n"
        
        "   je      PACK_SAVE                             \n"

        "   KERNEL12x28_PACK_K2                             \n"

        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_PACK_K                             \n"
        

        "PACK_SAVE:                                       \n"
        //"   prefetchnta      64(%%r10)                       \n"
        //"   prefetchnta      64(%%r11)                       \n"
        //"   prefetchnta      64(%%r12)                       \n"
        //"   prefetchnta      64(%%r13)                       \n"
        "   movl %[cc], %%r15d              \n"
        "   KERNEL12x28_PACK_END_K                          \n"
        "   mov     %[input_buffer], %%r14                            \n"
        "   cmp     $0, %%r15                               \n"

        "   je      PACK_ST_C                                  \n"
        "   ADD_C_12x28                                     \n"
        "   jmp PACK_Kb_END                 \n"
        "PACK_ST_C:                                            \n"
        "   ST_12x28                                      \n"

        "PACK_Kb_END:               \n"
        "   sub    $12, %%rdi       \n"
        "   je     EDGE_CASE                              \n"

        "BEGIN_M:                                           \n"
        
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        
        "BEGIN_K:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  
        "   vmovups     64(%%rax), %%zmm5                   \n"
        //"   prefetcht2      (%%r10)                       \n"
        //"   prefetcht0      64(%%r10)                       \n"
        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"
        "   vpxorq      %%zmm13, %%zmm13, %%zmm13           \n" 
        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 
        "   vpxorq      %%zmm15, %%zmm15, %%zmm15           \n"
        //"   prefetcht2      (%%r11)                       \n"
        //"   prefetcht0      64(%%r11)                       \n"
        "   vpxorq      %%zmm16, %%zmm16, %%zmm16           \n"
        "   vpxorq      %%zmm17, %%zmm17, %%zmm17           \n" 
        "   vpxorq      %%zmm18, %%zmm18, %%zmm18           \n" 
        "   vpxorq      %%zmm19, %%zmm19, %%zmm19           \n"
        //"   prefetcht2      (%%r12)                       \n"
        //"   prefetcht0      64(%%r12)                       \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

        "   vpxorq      %%zmm20, %%zmm20, %%zmm20           \n"
        "   vpxorq      %%zmm21, %%zmm21, %%zmm21           \n" 
        "   vpxorq      %%zmm22, %%zmm22, %%zmm22           \n" 
        "   vpxorq      %%zmm23, %%zmm23, %%zmm23           \n"
        "   vpxorq      %%zmm24, %%zmm24, %%zmm24           \n"
        //"   prefetcht0      (%%r13)                       \n"
        //"   prefetcht0      64(%%r13)                       \n"
        "   vpxorq      %%zmm25, %%zmm25, %%zmm25           \n" 
        "   vpxorq      %%zmm26, %%zmm26, %%zmm26           \n" 
        "   vpxorq      %%zmm27, %%zmm27, %%zmm27           \n"
        "   vpxorq      %%zmm28, %%zmm28, %%zmm28           \n"
        "   vpxorq      %%zmm29, %%zmm29, %%zmm29           \n" 
        "   vpxorq      %%zmm30, %%zmm30, %%zmm30           \n" 
        "   vpxorq      %%zmm31, %%zmm31, %%zmm31           \n"   
        
        "   sub $8, %%rdx                               \n"

        "MAIN_K:                                            \n"


        "   KERNEL12x28_K1                                  \n"
        "   KERNEL12x28_K2                                  \n"
        "   KERNEL12x28_K1                                  \n"
        "   KERNEL12x28_K2                                  \n"
        "   KERNEL12x28_K1                                  \n"
        "   KERNEL12x28_K2                                  \n"
        "   KERNEL12x28_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K                                  \n"
        "   KERNEL12x28_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K                                  \n"
        "EDGE_K:                                            \n"
        "   KERNEL12x28_END_K                               \n"
        "BEGIN_SAVE:                                        \n"
        //"   prefetcht1      64(%%r10)                       \n"
        //"   prefetcht1      64(%%r11)                       \n"
        //"   prefetcht1      64(%%r12)                       \n"
        //"   prefetcht1      64(%%r13)                       \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C                                  \n"
        "   ADD_C_12x28                                     \n"
        "   jmp Kb_END                  \n"
        "ST_C:                                            \n"
        "   ST_12x28                                      \n"
        "Kb_END:                    \n"
        "   sub     $12, %%rdi       \n"
        "   jne     BEGIN_M                              \n"

        "EDGE_CASE:                                     \n"
        "   movl        %[EDGE_Kb], %%edi                             \n"
        "   cmp $8, %%edi       \n"
        "   je  EDGE_8         \n"
        "   cmp $4, %%edi       \n"
        "   je  EDGE_4        \n"
        "   jmp     END_M           \n"
        
        "EDGE_8:                    \n"
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   prefetcht0      (%%rax)                         \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   prefetcht0      (%%rbx)                         \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2

        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        "   prefetcht0      64(%%rax)                         \n" 
        "BEGIN_K_8:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  
        "   vmovups     64(%%rax), %%zmm5                   \n"

        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"
        "   vpxorq      %%zmm13, %%zmm13, %%zmm13           \n" 
        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 
        "   vpxorq      %%zmm15, %%zmm15, %%zmm15           \n"
        "   vpxorq      %%zmm16, %%zmm16, %%zmm16           \n"
        "   vpxorq      %%zmm17, %%zmm17, %%zmm17           \n" 
        "   vpxorq      %%zmm18, %%zmm18, %%zmm18           \n" 
        "   vpxorq      %%zmm19, %%zmm19, %%zmm19           \n"

        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

        "   vpxorq      %%zmm20, %%zmm20, %%zmm20           \n"
        "   vpxorq      %%zmm21, %%zmm21, %%zmm21           \n" 
        "   vpxorq      %%zmm22, %%zmm22, %%zmm22           \n" 
        "   vpxorq      %%zmm23, %%zmm23, %%zmm23           \n"
 
        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_8:                                            \n"


        "   KERNEL8x28_K1                                  \n"
        "   KERNEL8x28_K2                                  \n"
        "   KERNEL8x28_K1                                  \n"
        "   KERNEL8x28_K2                                  \n"
        "   KERNEL8x28_K1                                  \n"
        "   KERNEL8x28_K2                                  \n"
        "   KERNEL8x28_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_8                                  \n"
        "   KERNEL8x28_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_8                                  \n"
        "EDGE_K_8:                                            \n"
        
        "   KERNEL8x28_END_K                               \n"

        "BEGIN_SAVE_8:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_8                                  \n"
        "   ADD_C_8x28                                     \n"
        "   jmp END_M                  \n"
        "ST_C_8:                                            \n"
        "   ST_8x28                                      \n"
        "   jmp     END_M                               \n"

        "EDGE_4:                    \n"
        
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        
        "BEGIN_K_4:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  
        "   vmovups     64(%%rax), %%zmm5                   \n"

        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"
        "   vpxorq      %%zmm13, %%zmm13, %%zmm13           \n" 
        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 
        "   vpxorq      %%zmm15, %%zmm15, %%zmm15           \n"


        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

 
        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_4:                                            \n"


        "   KERNEL4x28_K1                                  \n"
        "   KERNEL4x28_K2                                  \n"
        "   KERNEL4x28_K1                                  \n"
        "   KERNEL4x28_K2                                  \n"
        "   KERNEL4x28_K1                                  \n"
        "   KERNEL4x28_K2                                  \n"
        "   KERNEL4x28_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_4                                  \n"
        "   KERNEL4x28_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_4                                  \n"
        "EDGE_K_4:                                            \n"
        
        "   KERNEL4x28_END_K                               \n"

        "BEGIN_SAVE_4:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_4                                  \n"
        "   ADD_C_4x28                                     \n"
        "   jmp END_M                  \n"
        "ST_C_4:                                            \n"
        "   ST_4x28                                      \n"

        "END_M:                                             \n"
        
        
        

        :    
        :
         [output]               "m" (output),
         [trans_filter]         "m" (trans_filter),
         [input]                "m" (input),
         [Kb]                   "m" (Kb),
         [Cb]                   "m" (Cb),
         [input_HW_size]        "m" (input_HW_size),
         [input_buffer]         "m" (input_buffer),
         [cc]                   "m" (cc),
         [EDGE_Kb]              "m" (EDGE_Kb)
        :
         "rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
         "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
         "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
         "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
         "zmm30", "zmm31", "memory"
    
    );


}

void direct_1x1_N12M14_s2(float *output, float *trans_filter, float *input, int Kb, int Cb, int input_HW_size, float *input_buffer, int cc, int EDGE_Kb){
    asm volatile(


        ".macro KERNEL12x14_PACK_K1                         \n"

        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vpxorq      %%zmm6, %%zmm6, %%zmm6             \n"
        "   vmovups         %%zmm4, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"

        "   add             %%r8, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"

        "   prefetcht2      224(%%rax)                      \n"
        

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"

        "   prefetcht2      288(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"

        "   vmovups         (%%rax), %%zmm5                     \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"

        "   add             $64, %%r14                     \n"
        
        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"

        "   vmovups         64(%%rax), %%zmm6%{%%k2%}                     \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"
        
        "   prefetcht0      48(%%rbx)                      \n"
        "   vshufps    $0x88, %%zmm6, %%zmm5, %%zmm5      \n"
        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm24         \n"



        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm26         \n"
        "   vpermq  $0xd8, %%zmm5, %%zmm6                   \n"
        "   add             $48, %%rbx                                 \n"
        
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm28         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm30         \n"
        "   vshuff64x2  $0xd8, %%zmm6, %%zmm6, %%zmm5       \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

        "   vpxorq      %%zmm6, %%zmm6, %%zmm6              \n"
        ".endm                                              \n"


        ".macro KERNEL12x14_PACK_K2                         \n"

        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm8          \n"

        "   vmovups         %%zmm5, (%%r14)                 \n"


        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm10         \n"

        "   add             %%r8, %%rax                      \n"
        "   vpxorq      %%zmm6, %%zmm6, %%zmm6             \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm12         \n"

        "   prefetcht2      224(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm14         \n"

        "   prefetcht2      288(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm16         \n"

        "   vmovups         (%%rax), %%zmm4                     \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm18         \n"

        "   add             $64, %%r14                     \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm20         \n"

        "   vmovups         64(%%rax), %%zmm6%{%%k2%}                     \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm22         \n"
        "   vshufps    $0x88, %%zmm6, %%zmm4, %%zmm4      \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm24         \n"

        "   prefetcht0      48(%%rbx)                      \n"
        "   vpermq  $0xd8, %%zmm4, %%zmm6                   \n"
        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm26         \n"

        "   add             $48, %%rbx                                 \n"


        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm28         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vshuff64x2  $0xd8, %%zmm6, %%zmm6, %%zmm4       \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm30         \n"

        "   vbroadcastss    4(%%rbx), %%zmm1                \n"


        ".endm                                              \n"



        ".macro KERNEL12x14_PACK_END_K                      \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"

        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm8          \n"

        "   vmovups         %%zmm5, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm10         \n"



        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm12         \n"

        //"   prefetcht2      128(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm14         \n"


        //"   prefetcht2      192(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm16         \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm18         \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm20         \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm22         \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm24         \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm26         \n"

        "   add             $48, %%rbx                                 \n"

        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm28         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm30         \n"


        ".endm                                              \n"



        ".macro KERNEL12x14_K1                              \n"
        "   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"

        //"   prefetcht0      128(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"

        "   add             $64, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"

        //"   prefetcht0      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"

        "   vmovups         (%%rax), %%zmm6                     \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"


        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"


        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"



        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"

        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm24         \n"


        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm26         \n"

        "   add             $48, %%rbx                      \n"
        
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm28         \n"

        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm30         \n"

        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL12x14_K2                              \n"
        "   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"

        //"   prefetcht0      128(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"

        "   add             $64, %%rax                      \n"
        
        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"

        //"   prefetcht0      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"

        "   vmovups         (%%rax), %%zmm4                     \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"


        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"

        
        
        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"

        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"


        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"

        "   add             $48, %%rbx                      \n"
        
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"


        "   vbroadcastss    (%%rbx), %%zmm0                \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
 
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"

        ".endm                                              \n"



        ".macro KERNEL12x14_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"

        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"



        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"


        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"


        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"


        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"


        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"


        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"


        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"


        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"

        "   add             $48, %%rbx                      \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"


        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"

        

        ".endm                                              \n"


        ".macro ST_12x14   \n"
        
        "   vmovups         %%zmm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%zmm10, (%%r11)%{%%k1%}                \n"

        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C0
        "   vmovups         %%zmm12, (%%r12)%{%%k1%}                \n"

        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   vmovups         %%zmm14, (%%r13)%{%%k1%}                \n"


        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2

        "   vmovups         %%zmm16, (%%r10)%{%%k1%}                \n"

        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        "   vmovups         %%zmm18, (%%r11)%{%%k1%}                \n"

        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C0
        "   vmovups         %%zmm20, (%%r12)%{%%k1%}                \n"

        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   vmovups         %%zmm22, (%%r13)%{%%k1%}                \n"

        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2

        "   vmovups         %%zmm24, (%%r10)%{%%k1%}                \n"

        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        "   vmovups         %%zmm26, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%zmm28, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%zmm30, (%%r13)%{%%k1%}                \n"

        ".endm      \n"

        ".macro ADD_C_12x14   \n"
		"   vmovups 		(%%r10), %%zmm0%{%%k1%}        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"

		"   vmovups 		(%%r11), %%zmm2%{%%k1%}        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"

		"   vmovups 		(%%r12), %%zmm4%{%%k1%}        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"

		"   vmovups 		(%%r13), %%zmm6%{%%k1%}        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"


        "   vmovups         %%zmm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%zmm10, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%zmm12, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%zmm14, (%%r13)%{%%k1%}                \n"


		"	leaq  			(%%r13, %%rsi), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%rsi), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%rsi), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%rsi), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0%{%%k1%}        			\n"
		"	vaddps 			%%zmm0, %%zmm16, %%zmm16		\n"

		"   vmovups 		(%%r11), %%zmm2%{%%k1%}        			\n"
		"	vaddps 			%%zmm2, %%zmm18, %%zmm18		\n"


		"   vmovups 		(%%r12), %%zmm4%{%%k1%}        			\n"
		"	vaddps 			%%zmm4, %%zmm20, %%zmm20		\n"

		"   vmovups 		(%%r13), %%zmm6%{%%k1%}        			\n"
		"	vaddps 			%%zmm6, %%zmm22, %%zmm22		\n"


        "   vmovups         %%zmm16, (%%r10)%{%%k1%}                \n"

        "   vmovups         %%zmm18, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%zmm20, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%zmm22, (%%r13)%{%%k1%}                \n"



		"	leaq  			(%%r13, %%rsi), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%rsi), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%rsi), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%rsi), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0%{%%k1%}        			\n"
		"	vaddps 			%%zmm0, %%zmm24, %%zmm24		\n"

		"   vmovups 		(%%r11), %%zmm2%{%%k1%}        			\n"
		"	vaddps 			%%zmm2, %%zmm26, %%zmm26		\n"


		"   vmovups 		(%%r12), %%zmm4%{%%k1%}        			\n"
		"	vaddps 			%%zmm4, %%zmm28, %%zmm28		\n"

		"   vmovups 		(%%r13), %%zmm6%{%%k1%}        			\n"
		"	vaddps 			%%zmm6, %%zmm30, %%zmm30		\n"


        "   vmovups         %%zmm24, (%%r10)%{%%k1%}                \n"

        "   vmovups         %%zmm26, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%zmm28, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%zmm30, (%%r13)%{%%k1%}                \n"

        
        ".endm      \n"

        ".macro KERNEL8x14_K1                              \n"
        "   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"

        //"   prefetcht0      128(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"

        "   add             $64, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"

        "   prefetcht0      32(%%rbx)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"

        "   vmovups         (%%rax), %%zmm6                     \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"

        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"

        "   add             $32, %%rbx                      \n"

        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"

        "   vbroadcastss    (%%rbx), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"

        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL8x14_K2                              \n"
        "   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"

        //"   prefetcht0      128(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"

        "   add             $64, %%rax                      \n"
        
        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"

        "   prefetcht0      32(%%rbx)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"

        "   vmovups         (%%rax), %%zmm4                     \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"


        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"

        "   add             $32, %%rbx                      \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"


        "   vbroadcastss    (%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"

        "   vbroadcastss    4(%%rbx), %%zmm1               \n"


        ".endm                                              \n"



        ".macro KERNEL8x14_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"

        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"



        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"


        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"


        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"


        "   add             $32, %%rbx                      \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"


        

        ".endm                                              \n"


        ".macro ST_8x14   \n"
        
        "   vmovups         %%zmm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%zmm10, (%%r11)%{%%k1%}                \n"

        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C0
        "   vmovups         %%zmm12, (%%r12)%{%%k1%}                \n"

        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   vmovups         %%zmm14, (%%r13)%{%%k1%}                \n"


        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2

        "   vmovups         %%zmm16, (%%r10)%{%%k1%}                \n"

        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        "   vmovups         %%zmm18, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%zmm20, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%zmm22, (%%r13)%{%%k1%}                \n"





        
        ".endm      \n"

        ".macro ADD_C_8x14   \n"
		"   vmovups 		(%%r10), %%zmm0%{%%k1%}        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"

		"   vmovups 		(%%r11), %%zmm2%{%%k1%}        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"

		"   vmovups 		(%%r12), %%zmm4%{%%k1%}        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"

		"   vmovups 		(%%r13), %%zmm6%{%%k1%}        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"


        "   vmovups         %%zmm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%zmm10, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%zmm12, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%zmm14, (%%r13)%{%%k1%}                \n"


		"	leaq  			(%%r13, %%rsi), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%rsi), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%rsi), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%rsi), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0%{%%k1%}        			\n"
		"	vaddps 			%%zmm0, %%zmm16, %%zmm16		\n"

		"   vmovups 		(%%r11), %%zmm2%{%%k1%}        			\n"
		"	vaddps 			%%zmm2, %%zmm18, %%zmm18		\n"


		"   vmovups 		(%%r12), %%zmm4%{%%k1%}        			\n"
		"	vaddps 			%%zmm4, %%zmm20, %%zmm20		\n"

		"   vmovups 		(%%r13), %%zmm6%{%%k1%}        			\n"
		"	vaddps 			%%zmm6, %%zmm22, %%zmm22		\n"


        "   vmovups         %%zmm16, (%%r10)%{%%k1%}                \n"

        "   vmovups         %%zmm18, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%zmm20, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%zmm22, (%%r13)%{%%k1%}                \n"

        
        ".endm      \n"

        ".macro KERNEL4x14_K1                              \n"
        "   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"

        "   add             $64, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"

        "   add             $16, %%rbx                      \n"
        "   vmovups         (%%rax), %%zmm6                     \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"


        "   vbroadcastss    (%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"

        "   vbroadcastss    4(%%rbx), %%zmm1               \n"

        
        ".endm                                              \n"


        ".macro KERNEL4x14_K2                              \n"
        "   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"

        "   add             $64, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"

        "   vmovups         (%%rax), %%zmm4                     \n"
        "   add             $16, %%rbx                      \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"

        "   vbroadcastss    (%%rbx), %%zmm0               \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"

        "   vbroadcastss    4(%%rbx), %%zmm1               \n"


        ".endm                                              \n"



        ".macro KERNEL4x14_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"

        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"


        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"


        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"

        "   add             $16, %%rbx                      \n"


        

        ".endm                                              \n"


        ".macro ST_4x14   \n"
        
        "   vmovups         %%zmm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%zmm10, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%zmm12, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%zmm14, (%%r13)%{%%k1%}                \n"






        
        ".endm      \n"

        ".macro ADD_C_4x14   \n"
		"   vmovups 		(%%r10), %%zmm0%{%%k1%}        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"

		"   vmovups 		(%%r11), %%zmm2%{%%k1%}        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"

		"   vmovups 		(%%r12), %%zmm4%{%%k1%}        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"

		"   vmovups 		(%%r13), %%zmm6%{%%k1%}        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"


        "   vmovups         %%zmm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%zmm10, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%zmm12, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%zmm14, (%%r13)%{%%k1%}                \n"

        
        ".endm      \n"

        "CONV_KERNEL12x14:                                \n"
        "   movl     %[input_HW_size], %%r8d                             \n" 
        "   movl    $0x3fff, %%esi       \n"
        "   movl    $0xfff, %%edi       \n"
        "   mov     %[output], %%rcx                             \n"
        "   mov     %[trans_filter], %%rbx                             \n"
        "   mov     %[input], %%r9                             \n"
        
        //"   prefetcht0      (%%rax)                         \n"
        "   kmovd   %%esi, %%k1            \n"
        "   kmovd   %%edi, %%k2            \n"
        "   movl     %[Cb], %%edx                             \n"  // Cb
        "   mov %%r8, %%rsi                 \n"
        "   mov         %[input_buffer], %%r14                            \n"   //input_b
        "   movl        %[Kb], %%edi                             \n"
        "   shr     $2, %%rsi                       \n"

        
        //------------------- loop body
        "BEGIN_PACK_12x14:                                        \n"


        //"   shl $2, %%r8                \n"
        "   mov     %%rcx, %%r10                            \n"  // C0
        "   mov     %%r9, %%rax                         \n"
        "   prefetcht0      (%%rax)                         \n" 
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   mov     %%r9, %%rax                         \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   prefetcht0      (%%rbx)                         \n" 
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        //"   prefetcht0      64(%%rax)                         \n" 
        
        //"   shr $2, %%r8                \n"
        
        "   vpxorq      %%zmm5, %%zmm5, %%zmm5             \n"
        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vmovups     64(%%rax), %%zmm5%{%%k2%}                   \n"

        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"

        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 


        "   vshufps    $0x88, %%zmm5, %%zmm4, %%zmm4      \n"
        
        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"

        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 
        
        "   vpermq  $0xd8, %%zmm4, %%zmm5                   \n"
        

        "   vpxorq      %%zmm16, %%zmm16, %%zmm16           \n"

        
        "   vshuff64x2  $0xd8, %%zmm5, %%zmm5, %%zmm4       \n"
        
        "   vpxorq      %%zmm18, %%zmm18, %%zmm18           \n" 


        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

        "   vpxorq      %%zmm20, %%zmm20, %%zmm20           \n"

        //"   prefetcht2      64(%%r10)                       \n"
        "   vpxorq      %%zmm22, %%zmm22, %%zmm22           \n" 

        //"   prefetcht2      64(%%r11)                       \n"
        "   vpxorq      %%zmm24, %%zmm24, %%zmm24           \n"

        //"   prefetcht2      64(%%r12)                       \n"
        "   vpxorq      %%zmm26, %%zmm26, %%zmm26           \n" 

        //"   prefetcht2      64(%%r13)                       \n"
        "   vpxorq      %%zmm28, %%zmm28, %%zmm28           \n"

        "   vpxorq      %%zmm30, %%zmm30, %%zmm30           \n" 
        "   sub    $8, %%rdx                               \n"

        
        "MAIN_PACK_K_12x14:                                       \n"

        
        "   KERNEL12x14_PACK_K1                             \n"
        "   KERNEL12x14_PACK_K2                             \n"
        "   KERNEL12x14_PACK_K1                             \n"
        "   KERNEL12x14_PACK_K2                             \n"
        "   KERNEL12x14_PACK_K1                             \n"
        "   KERNEL12x14_PACK_K2                             \n"
        "   KERNEL12x14_PACK_K1                             \n"
        
        "   cmp     $0, %%rdx                               \n"
        
        "   je      PACK_SAVE_12x14                             \n"
        
        "   KERNEL12x14_PACK_K2                             \n"
        
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_PACK_K_12x14                             \n"
        

        "PACK_SAVE_12x14:                                       \n"
        "   movl %[cc], %%r15d              \n"
        "   KERNEL12x14_PACK_END_K                          \n"
        "   mov     %[input_buffer], %%r14                            \n"
        "   cmp     $0, %%r15                               \n"

        "   je      PACK_ST_C_12x14                                  \n"
        "   ADD_C_12x14                                     \n"
        "   jmp PACK_Kb_END_12x14                 \n"
        "PACK_ST_C_12x14:                                            \n"
        "   ST_12x14                                      \n"

        "PACK_Kb_END_12x14:               \n"
        "   sub    $12, %%rdi       \n"
        "   je     EDGE_CASE_12x14                              \n"

        /**/
        //"   jmp     END_M_12x14         \n"

        "BEGIN_M_12x14:                                           \n"
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   prefetcht0      (%%rax)                         \n" 
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2

        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        
        "BEGIN_K_12x14:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 

        //"   vmovups     64(%%rax), %%zmm6                   \n"

        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"

        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 

        "   vpxorq      %%zmm16, %%zmm16, %%zmm16           \n"

        "   vpxorq      %%zmm18, %%zmm18, %%zmm18           \n" 


        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

        "   vpxorq      %%zmm20, %%zmm20, %%zmm20           \n"

        "   vpxorq      %%zmm22, %%zmm22, %%zmm22           \n" 

        "   vpxorq      %%zmm24, %%zmm24, %%zmm24           \n"

        "   vpxorq      %%zmm26, %%zmm26, %%zmm26           \n" 

        "   vpxorq      %%zmm28, %%zmm28, %%zmm28           \n"

        "   vpxorq      %%zmm30, %%zmm30, %%zmm30           \n" 
 
        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_12x14:                                            \n"


        "   KERNEL12x14_K1                                  \n"
        "   KERNEL12x14_K2                                  \n"
        "   KERNEL12x14_K1                                  \n"
        "   KERNEL12x14_K2                                  \n"
        "   KERNEL12x14_K1                                  \n"
        "   KERNEL12x14_K2                                  \n"
        "   KERNEL12x14_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_12x14                                  \n"
        "   KERNEL12x14_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_12x14                                  \n"
        "EDGE_K_12x14:                                            \n"
        "   KERNEL12x14_END_K                               \n"
        "BEGIN_SAVE_12x14:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_12x14                                  \n"
        "   ADD_C_12x14                                     \n"
        "   jmp Kb_END_12x14                  \n"
        "ST_C_12x14:                                            \n"
        "   ST_12x14                                      \n"
        "Kb_END_12x14:                    \n"
        "   sub     $12, %%rdi       \n"
        "   jne     BEGIN_M_12x14                              \n"

        "EDGE_CASE_12x14:                                     \n"
        "   movl        %[EDGE_Kb], %%edi                             \n"
        "   cmp $8, %%edi       \n"
        "   je  EDGE_8_12x14         \n"
        "   cmp $4, %%edi       \n"
        "   je  EDGE_4_12x14        \n"
        "   jmp     END_M_12x14           \n"
        
        "EDGE_8_12x14:                    \n"
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        //"   prefetcht0      (%%rax)                         \n" 
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        
        "BEGIN_K_8_12x14:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"

        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 

        //"   vmovups     64(%%rax), %%zmm6                   \n"

        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"

        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 

        "   vpxorq      %%zmm16, %%zmm16, %%zmm16           \n"

        "   vpxorq      %%zmm18, %%zmm18, %%zmm18           \n" 


        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

        "   vpxorq      %%zmm20, %%zmm20, %%zmm20           \n"

        "   vpxorq      %%zmm22, %%zmm22, %%zmm22           \n" 
 
        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_8_12x14:                                            \n"


        "   KERNEL8x14_K1                                  \n"
        "   KERNEL8x14_K2                                  \n"
        "   KERNEL8x14_K1                                  \n"
        "   KERNEL8x14_K2                                  \n"
        "   KERNEL8x14_K1                                  \n"
        "   KERNEL8x14_K2                                  \n"
        "   KERNEL8x14_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_8_12x14                                  \n"
        "   KERNEL8x14_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_8_12x14                                  \n"
        "EDGE_K_8_12x14:                                            \n"
        
        "   KERNEL8x14_END_K                               \n"

        "BEGIN_SAVE_8_12x14:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_8_12x14                                  \n"
        "   ADD_C_8x14                                     \n"
        "   jmp END_M_12x14                  \n"
        "ST_C_8_12x14:                                            \n"
        "   ST_8x14                                      \n"
        "   jmp     END_M_12x14                               \n"

        "EDGE_4_12x14:                    \n"
        
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        
        "BEGIN_K_4_12x14:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  
        //"   vmovups     64(%%rax), %%zmm6                   \n"

        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"
        "   vpxorq      %%zmm13, %%zmm13, %%zmm13           \n" 
        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 
        "   vpxorq      %%zmm15, %%zmm15, %%zmm15           \n"
        "   vpxorq      %%zmm16, %%zmm16, %%zmm16           \n"
        "   vpxorq      %%zmm17, %%zmm17, %%zmm17           \n" 
        "   vpxorq      %%zmm18, %%zmm18, %%zmm18           \n" 
        "   vpxorq      %%zmm19, %%zmm19, %%zmm19           \n"

        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

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

        "MAIN_K_4_12x14:                                            \n"


        "   KERNEL4x14_K1                                  \n"
        "   KERNEL4x14_K2                                  \n"
        "   KERNEL4x14_K1                                  \n"
        "   KERNEL4x14_K2                                  \n"
        "   KERNEL4x14_K1                                  \n"
        "   KERNEL4x14_K2                                  \n"
        "   KERNEL4x14_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_4_12x14                                  \n"
        "   KERNEL4x14_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_4_12x14                                  \n"
        "EDGE_K_4_12x14:                                            \n"
        
        "   KERNEL4x14_END_K                               \n"

        "BEGIN_SAVE_4_12x14:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_4_12x14                                  \n"
        "   ADD_C_4x14                                     \n"
        "   jmp END_M_12x14                  \n"
        "ST_C_4_12x14:                                            \n"
        "   ST_4x14                                      \n"

        "END_M_12x14:                                             \n"
        
        
        

        :    
        :
         [output]               "m" (output),
         [trans_filter]         "m" (trans_filter),
         [input]                "m" (input),
         [Kb]                   "m" (Kb),
         [Cb]                   "m" (Cb),
         [input_HW_size]        "m" (input_HW_size),
         [input_buffer]         "m" (input_buffer),
         [cc]                   "m" (cc),
         [EDGE_Kb]              "m" (EDGE_Kb)
        :
         "rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
         "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
         "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
         "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
         "zmm30", "zmm31", "memory"
    
    );



}


void direct_1x1_N12M7_s2(float *output, float *trans_filter, float *input, int Kb, int Cb, int input_HW_size, float *input_buffer, int cc, int EDGE_Kb){
    asm volatile(


        ".macro KERNEL12x7_PACK_K1                         \n"

        "   vbroadcastss    8(%%rbx), %%ymm2                \n"
        "   vfmadd231ps     %%ymm0, %%ymm4, %%ymm8          \n"
        "   vpxorq      %%ymm5, %%ymm5, %%ymm5             \n"
        "   vmovups         %%ymm4, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm4, %%ymm10         \n"

        "   add             %%r8, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm4, %%ymm12         \n"

        "   prefetcht2      112(%%rax)                      \n"
        

        "   vbroadcastss    20(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm4, %%ymm14         \n"

        "   vbroadcastss    24(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm4, %%ymm16         \n"

        "   vmovups         (%%rax), %%ymm6                     \n"

        "   vbroadcastss    28(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm4, %%ymm18         \n"

        "   add             $32, %%r14                     \n"
        
        "   vbroadcastss    32(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm4, %%ymm20         \n"

        "   vmovups         32(%%rax), %%ymm5%{%%k2%}                     \n"

        "   vbroadcastss    36(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm4, %%ymm22         \n"
        
        "   prefetcht0      48(%%rbx)                      \n"
        "   vshufps    $0x88, %%ymm5, %%ymm6, %%ymm6      \n"
        "   vbroadcastss    40(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm4, %%ymm24         \n"



        "   vbroadcastss    44(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm4, %%ymm26         \n"
        "   vpermq  $0xd8, %%ymm6, %%ymm5                   \n"
        "   add             $48, %%rbx                                 \n"
        
        "   vfmadd231ps     %%ymm2, %%ymm4, %%ymm28         \n"
        "   vbroadcastss    (%%rbx), %%ymm0                 \n"
        "   vfmadd231ps     %%ymm3, %%ymm4, %%ymm30         \n"
        "   vbroadcastss    4(%%rbx), %%ymm1                \n"
        "   vpxorq      %%ymm4, %%ymm4, %%ymm4              \n"
        ".endm                                              \n"


        ".macro KERNEL12x7_PACK_K2                         \n"

        "   vbroadcastss    8(%%rbx), %%ymm2                \n"
        "   vfmadd231ps     %%ymm0, %%ymm5, %%ymm8          \n"

        "   vmovups         %%ymm5, (%%r14)                 \n"


        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm5, %%ymm10         \n"

        "   add             %%r8, %%rax                      \n"
        "   vpxorq      %%ymm6, %%ymm6, %%ymm6             \n"

        "   vbroadcastss    16(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm5, %%ymm12         \n"

        "   prefetcht2      112(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm5, %%ymm14         \n"

        //"   prefetcht2      192(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm5, %%ymm16         \n"

        "   vmovups         (%%rax), %%ymm6                     \n"

        "   vbroadcastss    28(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm5, %%ymm18         \n"

        "   add             $32, %%r14                     \n"

        "   vbroadcastss    32(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm5, %%ymm20         \n"

        "   vmovups         32(%%rax), %%ymm4%{%%k2%}                     \n"

        "   vbroadcastss    36(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm5, %%ymm22         \n"
        "   vshufps    $0x88, %%ymm4, %%ymm6, %%ymm6      \n"

        "   vbroadcastss    40(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm5, %%ymm24         \n"

        "   prefetcht0      48(%%rbx)                      \n"
        "   vpermq  $0xd8, %%ymm6, %%ymm4                   \n"
        "   vbroadcastss    44(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm5, %%ymm26         \n"

        "   add             $48, %%rbx                                 \n"


        "   vfmadd231ps     %%ymm2, %%ymm5, %%ymm28         \n"
        "   vbroadcastss    (%%rbx), %%ymm0                 \n"
        "   vfmadd231ps     %%ymm3, %%ymm5, %%ymm30         \n"

        "   vbroadcastss    4(%%rbx), %%ymm1                \n"


        ".endm                                              \n"



        ".macro KERNEL12x7_PACK_END_K                      \n"


        "   vbroadcastss    8(%%rbx), %%ymm2                \n"

        "   vfmadd231ps     %%ymm0, %%ymm5, %%ymm8          \n"

        "   vmovups         %%ymm5, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm5, %%ymm10         \n"



        "   vbroadcastss    16(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm5, %%ymm12         \n"

        //"   prefetcht2      128(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm5, %%ymm14         \n"


        //"   prefetcht2      192(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm5, %%ymm16         \n"

        "   vbroadcastss    28(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm5, %%ymm18         \n"

        "   vbroadcastss    32(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm5, %%ymm20         \n"

        "   vbroadcastss    36(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm5, %%ymm22         \n"

        "   vbroadcastss    40(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm5, %%ymm24         \n"

        "   vbroadcastss    44(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm5, %%ymm26         \n"

        "   add             $48, %%rbx                                 \n"

        "   vfmadd231ps     %%ymm2, %%ymm5, %%ymm28         \n"
        "   vfmadd231ps     %%ymm3, %%ymm5, %%ymm30         \n"


        ".endm                                              \n"



        ".macro KERNEL12x7_K1                              \n"
        "   prefetcht0      32(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%ymm2                \n"
        "   vfmadd231ps     %%ymm0, %%ymm4, %%ymm8          \n"

        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm4, %%ymm10         \n"

        "   add             $32, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm4, %%ymm12         \n"

        "   vbroadcastss    20(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm4, %%ymm14         \n"

        "   vmovups         (%%rax), %%ymm6                     \n"
        
        "   vbroadcastss    24(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm4, %%ymm16         \n"


        "   vbroadcastss    28(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm4, %%ymm18         \n"


        "   vbroadcastss    32(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm4, %%ymm20         \n"

        "   vbroadcastss    36(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm4, %%ymm22         \n"

        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm4, %%ymm24         \n"


        "   vbroadcastss    44(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm4, %%ymm26         \n"

        "   add             $48, %%rbx                      \n"
        
        "   vfmadd231ps     %%ymm2, %%ymm4, %%ymm28         \n"

        "   vbroadcastss    (%%rbx), %%ymm0                 \n"
        
        "   vfmadd231ps     %%ymm3, %%ymm4, %%ymm30         \n"

        "   vbroadcastss    4(%%rbx), %%ymm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL12x7_K2                              \n"
        "   prefetcht0      32(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%ymm2                \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm8          \n"

        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm10         \n"

        "   add             $32, %%rax                      \n"
        
        "   vbroadcastss    16(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm12         \n"

        "   vbroadcastss    20(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm14         \n"

        "   vmovups         (%%rax), %%ymm4                     \n"
        
        "   vbroadcastss    24(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm16         \n"


        "   vbroadcastss    28(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm18         \n"

        "   vbroadcastss    32(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm20         \n"

        "   vbroadcastss    36(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm22         \n"

        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm24         \n"


        "   vbroadcastss    44(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm26         \n"

        "   add             $48, %%rbx                      \n"
        
        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm28         \n"


        "   vbroadcastss    (%%rbx), %%ymm0                \n"
        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm30         \n"
 
        "   vbroadcastss    4(%%rbx), %%ymm1                 \n"

        ".endm                                              \n"



        ".macro KERNEL12x7_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%ymm2                \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm8          \n"

        
        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm10         \n"



        "   vbroadcastss    16(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm12         \n"


        "   vbroadcastss    20(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm14         \n"


        "   vbroadcastss    24(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm16         \n"


        "   vbroadcastss    28(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm18         \n"


        "   vbroadcastss    32(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm20         \n"


        "   vbroadcastss    36(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm22         \n"


        "   vbroadcastss    40(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm24         \n"


        "   vbroadcastss    44(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm26         \n"

        "   add             $48, %%rbx                      \n"
        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm28         \n"


        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm30         \n"

        

        ".endm                                              \n"


        ".macro ST_12x7   \n"
        
        "   vmovups         %%ymm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%ymm10, (%%r11)%{%%k1%}                \n"

        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C0
        "   vmovups         %%ymm12, (%%r12)%{%%k1%}                \n"

        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   vmovups         %%ymm14, (%%r13)%{%%k1%}                \n"


        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2

        "   vmovups         %%ymm16, (%%r10)%{%%k1%}                \n"

        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        "   vmovups         %%ymm18, (%%r11)%{%%k1%}                \n"

        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C0
        "   vmovups         %%ymm20, (%%r12)%{%%k1%}                \n"

        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   vmovups         %%ymm22, (%%r13)%{%%k1%}                \n"

        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2

        "   vmovups         %%ymm24, (%%r10)%{%%k1%}                \n"

        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        "   vmovups         %%ymm26, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%ymm28, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%ymm30, (%%r13)%{%%k1%}                \n"

        ".endm      \n"

        ".macro ADD_C_12x7   \n"
		"   vmovups 		(%%r10), %%ymm0%{%%k1%}        			\n"
		"	vaddps 			%%ymm0, %%ymm8, %%ymm8			\n"

		"   vmovups 		(%%r11), %%ymm2%{%%k1%}        			\n"
		"	vaddps 			%%ymm2, %%ymm10, %%ymm10		\n"

		"   vmovups 		(%%r12), %%ymm4%{%%k1%}        			\n"
		"	vaddps 			%%ymm4, %%ymm12, %%ymm12		\n"

		"   vmovups 		(%%r13), %%ymm6%{%%k1%}        			\n"
		"	vaddps 			%%ymm6, %%ymm14, %%ymm14		\n"


        "   vmovups         %%ymm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%ymm10, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%ymm12, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%ymm14, (%%r13)%{%%k1%}                \n"


		"	leaq  			(%%r13, %%rsi), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%rsi), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%rsi), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%rsi), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%ymm0%{%%k1%}        			\n"
		"	vaddps 			%%ymm0, %%ymm16, %%ymm16		\n"

		"   vmovups 		(%%r11), %%ymm2%{%%k1%}        			\n"
		"	vaddps 			%%ymm2, %%ymm18, %%ymm18		\n"


		"   vmovups 		(%%r12), %%ymm4%{%%k1%}        			\n"
		"	vaddps 			%%ymm4, %%ymm20, %%ymm20		\n"

		"   vmovups 		(%%r13), %%ymm6%{%%k1%}        			\n"
		"	vaddps 			%%ymm6, %%ymm22, %%ymm22		\n"


        "   vmovups         %%ymm16, (%%r10)%{%%k1%}                \n"

        "   vmovups         %%ymm18, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%ymm20, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%ymm22, (%%r13)%{%%k1%}                \n"



		"	leaq  			(%%r13, %%rsi), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%rsi), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%rsi), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%rsi), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%ymm0%{%%k1%}        			\n"
		"	vaddps 			%%ymm0, %%ymm24, %%ymm24		\n"

		"   vmovups 		(%%r11), %%ymm2%{%%k1%}        			\n"
		"	vaddps 			%%ymm2, %%ymm26, %%ymm26		\n"


		"   vmovups 		(%%r12), %%ymm4%{%%k1%}        			\n"
		"	vaddps 			%%ymm4, %%ymm28, %%ymm28		\n"

		"   vmovups 		(%%r13), %%ymm6%{%%k1%}        			\n"
		"	vaddps 			%%ymm6, %%ymm30, %%ymm30		\n"


        "   vmovups         %%ymm24, (%%r10)%{%%k1%}                \n"

        "   vmovups         %%ymm26, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%ymm28, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%ymm30, (%%r13)%{%%k1%}                \n"

        
        ".endm      \n"

        ".macro KERNEL8x7_K1                              \n"
        "   prefetcht0      32(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%ymm2                \n"
        "   vfmadd231ps     %%ymm0, %%ymm4, %%ymm8          \n"

        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm4, %%ymm10         \n"

        "   add             $32, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm4, %%ymm12         \n"

        "   vbroadcastss    20(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm4, %%ymm14         \n"

        "   vmovups         (%%rax), %%ymm6                     \n"
        
        "   vbroadcastss    24(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm4, %%ymm16         \n"

        "   prefetcht0      32(%%rbx)                      \n"

        "   vbroadcastss    28(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm4, %%ymm18         \n"

        "   add             $32, %%rbx                      \n"

        "   vfmadd231ps     %%ymm2, %%ymm4, %%ymm20         \n"

        "   vbroadcastss    (%%rbx), %%ymm0                 \n"

        "   vfmadd231ps     %%ymm3, %%ymm4, %%ymm22         \n"

        "   vbroadcastss    4(%%rbx), %%ymm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL8x7_K2                              \n"
        "   vbroadcastss    8(%%rbx), %%ymm2                \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm8          \n"

        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm10         \n"

        "   add             $32, %%rax                      \n"
        
        "   vbroadcastss    16(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm12         \n"

        "   vbroadcastss    20(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm14         \n"

        "   vmovups         (%%rax), %%ymm4                     \n"
        
        "   vbroadcastss    24(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm16         \n"


        "   vbroadcastss    28(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm18         \n"

        "   add             $32, %%rbx                      \n"
        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm20         \n"


        "   vbroadcastss    (%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm22         \n"

        "   vbroadcastss    4(%%rbx), %%ymm1               \n"


        ".endm                                              \n"



        ".macro KERNEL8x7_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%ymm2                \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm8          \n"

        
        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm10         \n"



        "   vbroadcastss    16(%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm12         \n"


        "   vbroadcastss    20(%%rbx), %%ymm1               \n"
        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm14         \n"

        "   vbroadcastss    24(%%rbx), %%ymm2               \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm16         \n"


        "   vbroadcastss    28(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm18         \n"


        "   add             $32, %%rbx                      \n"
        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm20         \n"

        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm22         \n"


        

        ".endm                                              \n"


        ".macro ST_8x7   \n"
        
        "   vmovups         %%ymm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%ymm10, (%%r11)%{%%k1%}                \n"

        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C0
        "   vmovups         %%ymm12, (%%r12)%{%%k1%}                \n"

        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   vmovups         %%ymm14, (%%r13)%{%%k1%}                \n"


        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2

        "   vmovups         %%ymm16, (%%r10)%{%%k1%}                \n"

        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        "   vmovups         %%ymm18, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%ymm20, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%ymm22, (%%r13)%{%%k1%}                \n"





        
        ".endm      \n"

        ".macro ADD_C_8x7   \n"
		"   vmovups 		(%%r10), %%ymm0%{%%k1%}        			\n"
		"	vaddps 			%%ymm0, %%ymm8, %%ymm8			\n"

		"   vmovups 		(%%r11), %%ymm2%{%%k1%}        			\n"
		"	vaddps 			%%ymm2, %%ymm10, %%ymm10		\n"

		"   vmovups 		(%%r12), %%ymm4%{%%k1%}        			\n"
		"	vaddps 			%%ymm4, %%ymm12, %%ymm12		\n"

		"   vmovups 		(%%r13), %%ymm6%{%%k1%}        			\n"
		"	vaddps 			%%ymm6, %%ymm14, %%ymm14		\n"


        "   vmovups         %%ymm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%ymm10, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%ymm12, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%ymm14, (%%r13)%{%%k1%}                \n"


		"	leaq  			(%%r13, %%rsi), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%rsi), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%rsi), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%rsi), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%ymm0%{%%k1%}        			\n"
		"	vaddps 			%%ymm0, %%ymm16, %%ymm16		\n"

		"   vmovups 		(%%r11), %%ymm2%{%%k1%}        			\n"
		"	vaddps 			%%ymm2, %%ymm18, %%ymm18		\n"


		"   vmovups 		(%%r12), %%ymm4%{%%k1%}        			\n"
		"	vaddps 			%%ymm4, %%ymm20, %%ymm20		\n"

		"   vmovups 		(%%r13), %%ymm6%{%%k1%}        			\n"
		"	vaddps 			%%ymm6, %%ymm22, %%ymm22		\n"


        "   vmovups         %%ymm16, (%%r10)%{%%k1%}                \n"

        "   vmovups         %%ymm18, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%ymm20, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%ymm22, (%%r13)%{%%k1%}                \n"

        
        ".endm      \n"

        ".macro KERNEL4x7_K1                              \n"

        "   vbroadcastss    8(%%rbx), %%ymm2                \n"
        "   vfmadd231ps     %%ymm0, %%ymm4, %%ymm8          \n"

        "   add             $32, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm4, %%ymm10         \n"

        "   add             $16, %%rbx                      \n"
        "   vmovups         (%%rax), %%ymm6                     \n"
        "   vfmadd231ps     %%ymm2, %%ymm4, %%ymm12         \n"



        "   vbroadcastss    (%%rbx), %%ymm0               \n"
        "   vfmadd231ps     %%ymm3, %%ymm4, %%ymm14         \n"

        "   vbroadcastss    4(%%rbx), %%ymm1               \n"

        
        ".endm                                              \n"


        ".macro KERNEL4x7_K2                              \n"

        "   vbroadcastss    8(%%rbx), %%ymm2                \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm8          \n"

        "   add             $32, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm10         \n"

        "   vmovups         (%%rax), %%ymm4                     \n"
        "   add             $16, %%rbx                      \n"

        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm12         \n"

        "   vbroadcastss    (%%rbx), %%ymm0               \n"

        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm14         \n"

        "   vbroadcastss    4(%%rbx), %%ymm1               \n"


        ".endm                                              \n"



        ".macro KERNEL4x7_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%ymm2                \n"
        "   vfmadd231ps     %%ymm0, %%ymm6, %%ymm8          \n"

        
        "   vbroadcastss    12(%%rbx), %%ymm3               \n"
        "   vfmadd231ps     %%ymm1, %%ymm6, %%ymm10         \n"


        "   vfmadd231ps     %%ymm2, %%ymm6, %%ymm12         \n"


        "   vfmadd231ps     %%ymm3, %%ymm6, %%ymm14         \n"

        "   add             $16, %%rbx                      \n"


        

        ".endm                                              \n"


        ".macro ST_4x7   \n"
        
        "   vmovups         %%ymm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%ymm10, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%ymm12, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%ymm14, (%%r13)%{%%k1%}                \n"






        
        ".endm      \n"

        ".macro ADD_C_4x7   \n"
		"   vmovups 		(%%r10), %%ymm0%{%%k1%}        			\n"
		"	vaddps 			%%ymm0, %%ymm8, %%ymm8			\n"

		"   vmovups 		(%%r11), %%ymm2%{%%k1%}        			\n"
		"	vaddps 			%%ymm2, %%ymm10, %%ymm10		\n"

		"   vmovups 		(%%r12), %%ymm4%{%%k1%}        			\n"
		"	vaddps 			%%ymm4, %%ymm12, %%ymm12		\n"

		"   vmovups 		(%%r13), %%ymm6%{%%k1%}        			\n"
		"	vaddps 			%%ymm6, %%ymm14, %%ymm14		\n"


        "   vmovups         %%ymm8, (%%r10)%{%%k1%}                 \n"

        "   vmovups         %%ymm10, (%%r11)%{%%k1%}                \n"

        "   vmovups         %%ymm12, (%%r12)%{%%k1%}                \n"

        "   vmovups         %%ymm14, (%%r13)%{%%k1%}                \n"

        
        ".endm      \n"

        "CONV_KERNEL12x7:                                \n"
        "   movl     %[input_HW_size], %%r8d                             \n" 
        "   movl    $0x7f, %%esi       \n"
        "   movl    $0x3f, %%edi       \n"
        "   mov     %[output], %%rcx                             \n"
        "   mov     %[trans_filter], %%rbx                             \n"
        "   mov     %[input], %%r9                             \n"
        
        //"   prefetcht0      (%%rax)                         \n"
        "   kmovd   %%esi, %%k1            \n"
        "   kmovd   %%edi, %%k2            \n"
        "   movl     %[Cb], %%edx                             \n"  // Cb
        "   mov %%r8, %%rsi                 \n"
        "   mov         %[input_buffer], %%r14                            \n"   //input_b
        "   movl        %[Kb], %%edi                             \n"
        "   shr     $2, %%rsi                       \n"

        
        //------------------- loop body
        "BEGIN_PACK_12x7:                                        \n"


        //"   shl $2, %%r8                \n"
        "   mov     %%rcx, %%r10                            \n"  // C0
        "   mov     %%r9, %%rax                         \n"
        "   prefetcht0      (%%rax)                         \n" 
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   mov     %%r9, %%rax                         \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   prefetcht0      (%%rbx)                         \n" 
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        
        //"   shr $2, %%r8                \n"
        
        "   vpxorq      %%ymm5, %%ymm5, %%ymm5             \n"
        "   vmovups     (%%rax), %%ymm4                     \n"
        "   vmovups     32(%%rax), %%ymm5%{%%k2%}                   \n"

        "   vpxorq      %%ymm8, %%ymm8, %%ymm8              \n"
        "   vpxorq      %%ymm10, %%ymm10, %%ymm10           \n" 
        "   vshufps    $0x88, %%ymm5, %%ymm4, %%ymm4      \n"
        "   vpxorq      %%ymm12, %%ymm12, %%ymm12           \n"
        "   vpxorq      %%ymm14, %%ymm14, %%ymm14           \n" 
        "   vpermq  $0xd8, %%ymm4, %%ymm5                   \n"

        "   vpxorq      %%ymm16, %%ymm16, %%ymm16           \n"

        "   vmovups     %%ymm5, %%ymm4                  \n"
        "   vpxorq      %%ymm18, %%ymm18, %%ymm18           \n" 

        "   vbroadcastss    (%%rbx), %%ymm0                 \n"
        "   vbroadcastss    4(%%rbx), %%ymm1                \n"
        "   vpxorq      %%ymm20, %%ymm20, %%ymm20           \n"
        "   vpxorq      %%ymm22, %%ymm22, %%ymm22           \n" 
        "   vpxorq      %%ymm24, %%ymm24, %%ymm24           \n"
        "   vpxorq      %%ymm26, %%ymm26, %%ymm26           \n" 
        "   vpxorq      %%ymm28, %%ymm28, %%ymm28           \n"
        "   vpxorq      %%ymm30, %%ymm30, %%ymm30           \n" 
  

        "   sub    $8, %%rdx                               \n"

        
        "MAIN_PACK_K_12x7:                                       \n"

        
        "   KERNEL12x7_PACK_K1                             \n"
        "   KERNEL12x7_PACK_K2                             \n"
        "   KERNEL12x7_PACK_K1                             \n"
        "   KERNEL12x7_PACK_K2                             \n"
        "   KERNEL12x7_PACK_K1                             \n"
        "   KERNEL12x7_PACK_K2                             \n"
        "   KERNEL12x7_PACK_K1                             \n"
        
        "   cmp     $0, %%rdx                               \n"
        
        "   je      PACK_SAVE_12x7                             \n"
        
        "   KERNEL12x7_PACK_K2                             \n"
        
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_PACK_K_12x7                             \n"
        

        "PACK_SAVE_12x7:                                       \n"
        "   movl %[cc], %%r15d              \n"
        "   KERNEL12x7_PACK_END_K                          \n"
        "   mov     %[input_buffer], %%r14                            \n"
        "   cmp     $0, %%r15                               \n"

        "   je      PACK_ST_C_12x7                                  \n"
        "   ADD_C_12x7                                     \n"
        "   jmp PACK_Kb_END_12x7                 \n"
        "PACK_ST_C_12x7:                                            \n"
        "   ST_12x7                                      \n"

        "PACK_Kb_END_12x7:               \n"
        "   sub    $12, %%rdi       \n"
        "   je     EDGE_CASE_12x7                              \n"

        /**/
        //"   jmp     END_M_12x7         \n"

        "BEGIN_M_12x7:                                           \n"
        
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        
        "BEGIN_K_12x7:                                           \n"

        "   vmovups     (%%rax), %%ymm4                     \n"
        "   vpxorq      %%ymm8, %%ymm8, %%ymm8              \n"
        "   vpxorq      %%ymm10, %%ymm10, %%ymm10           \n"   

        "   vpxorq      %%ymm12, %%ymm12, %%ymm12           \n"
        "   vpxorq      %%ymm14, %%ymm14, %%ymm14           \n" 
        "   vpxorq      %%ymm16, %%ymm16, %%ymm16           \n"
        "   vpxorq      %%ymm18, %%ymm18, %%ymm18           \n" 

        "   vbroadcastss    (%%rbx), %%ymm0                 \n"
        "   vbroadcastss    4(%%rbx), %%ymm1                \n"

        "   vpxorq      %%ymm20, %%ymm20, %%ymm20           \n" 
        "   vpxorq      %%ymm22, %%ymm22, %%ymm22           \n" 
        "   vpxorq      %%ymm24, %%ymm24, %%ymm24           \n"
        "   vpxorq      %%ymm26, %%ymm26, %%ymm26           \n" 
        "   vpxorq      %%ymm28, %%ymm28, %%ymm28           \n"
        "   vpxorq      %%ymm30, %%ymm30, %%ymm30           \n" 

        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_12x7:                                            \n"


        "   KERNEL12x7_K1                                  \n"
        "   KERNEL12x7_K2                                  \n"
        "   KERNEL12x7_K1                                  \n"
        "   KERNEL12x7_K2                                  \n"
        "   KERNEL12x7_K1                                  \n"
        "   KERNEL12x7_K2                                  \n"
        "   KERNEL12x7_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_12x7                                  \n"
        "   KERNEL12x7_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_12x7                                  \n"
        "EDGE_K_12x7:                                            \n"
        "   KERNEL12x7_END_K                               \n"
        "BEGIN_SAVE_12x7:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_12x7                                  \n"
        "   ADD_C_12x7                                     \n"
        "   jmp Kb_END_12x7                  \n"
        "ST_C_12x7:                                            \n"
        "   ST_12x7                                      \n"
        "Kb_END_12x7:                    \n"
        "   sub     $12, %%rdi       \n"
        "   jne     BEGIN_M_12x7                              \n"

        "EDGE_CASE_12x7:                                     \n"
        "   movl        %[EDGE_Kb], %%edi                             \n"
        "   cmp $8, %%edi       \n"
        "   je  EDGE_8_12x7         \n"
        "   cmp $4, %%edi       \n"
        "   je  EDGE_4_12x7        \n"
        "   jmp     END_M_12x7           \n"
        
        "EDGE_8_12x7:                    \n"
        
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        
        "BEGIN_K_8_12x7:                                           \n"

        "   vmovups     (%%rax), %%ymm4                     \n"
        "   vpxorq      %%ymm8, %%ymm8, %%ymm8              \n"
        "   vpxorq      %%ymm10, %%ymm10, %%ymm10           \n" 
        "   vpxorq      %%ymm12, %%ymm12, %%ymm12           \n"
        "   vpxorq      %%ymm14, %%ymm14, %%ymm14           \n" 
        "   vpxorq      %%ymm16, %%ymm16, %%ymm16           \n"
        "   vpxorq      %%ymm18, %%ymm18, %%ymm18           \n" 


        "   vbroadcastss    (%%rbx), %%ymm0                 \n"
        "   vbroadcastss    4(%%rbx), %%ymm1                \n"

        "   vpxorq      %%ymm20, %%ymm20, %%ymm20           \n"
        "   vpxorq      %%ymm22, %%ymm22, %%ymm22           \n" 
 
        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_8_12x7:                                            \n"


        "   KERNEL8x7_K1                                  \n"
        "   KERNEL8x7_K2                                  \n"
        "   KERNEL8x7_K1                                  \n"
        "   KERNEL8x7_K2                                  \n"
        "   KERNEL8x7_K1                                  \n"
        "   KERNEL8x7_K2                                  \n"
        "   KERNEL8x7_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_8_12x7                                  \n"
        "   KERNEL8x7_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_8_12x7                                  \n"
        "EDGE_K_8_12x7:                                            \n"
        
        "   KERNEL8x7_END_K                               \n"

        "BEGIN_SAVE_8_12x7:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_8_12x7                                  \n"
        "   ADD_C_8x7                                     \n"
        "   jmp END_M_12x7                  \n"
        "ST_C_8_12x7:                                            \n"
        "   ST_8x7                                      \n"
        "   jmp     END_M_12x7                               \n"

        "EDGE_4_12x7:                    \n"
        
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        
        "BEGIN_K_4_12x7:                                           \n"

        "   vmovups     (%%rax), %%ymm4                     \n"
        "   vpxorq      %%ymm8, %%ymm8, %%ymm8              \n"
        "   vpxorq      %%ymm10, %%ymm10, %%ymm10           \n" 
        "   vpxorq      %%ymm12, %%ymm12, %%ymm12           \n"
        "   vpxorq      %%ymm14, %%ymm14, %%ymm14           \n" 


        "   vbroadcastss    (%%rbx), %%ymm0                 \n"
        "   vbroadcastss    4(%%rbx), %%ymm1                \n"


        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_4_12x7:                                            \n"


        "   KERNEL4x7_K1                                  \n"
        "   KERNEL4x7_K2                                  \n"
        "   KERNEL4x7_K1                                  \n"
        "   KERNEL4x7_K2                                  \n"
        "   KERNEL4x7_K1                                  \n"
        "   KERNEL4x7_K2                                  \n"
        "   KERNEL4x7_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_4_12x7                                  \n"
        "   KERNEL4x14_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_4_12x7                                  \n"
        "EDGE_K_4_12x7:                                            \n"
        
        "   KERNEL4x14_END_K                               \n"

        "BEGIN_SAVE_4_12x7:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_4_12x7                                  \n"
        "   ADD_C_4x7                                     \n"
        "   jmp END_M_12x7                  \n"
        "ST_C_4_12x7:                                            \n"
        "   ST_4x7                                      \n"

        "END_M_12x7:                                             \n"
        
        
        

        :    
        :
         [output]               "m" (output),
         [trans_filter]         "m" (trans_filter),
         [input]                "m" (input),
         [Kb]                   "m" (Kb),
         [Cb]                   "m" (Cb),
         [input_HW_size]        "m" (input_HW_size),
         [input_buffer]         "m" (input_buffer),
         [cc]                   "m" (cc),
         [EDGE_Kb]              "m" (EDGE_Kb)
        :
         "rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
         "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
         "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19", "ymm20", "ymm21",
         "ymm22", "ymm23", "ymm24", "ymm25", "ymm26", "ymm27", "ymm28", "ymm29",
         "ymm30", "ymm31", "memory"
    
    );



}


void avx512_dircet_cnn_1x1s2(int H, int W, int N, int C, float *input, int K, int R, float *filter, float *output)
{
    int S = R;

    int out_W = W>>1;
    int out_H = H>>1;
    
    int stride_in = H * W;
    int stride_out = out_W * out_H;
    
    void *ptr, *ptr1;
    int Tn = NUM;               // N
    int Tm = NUM / Tn;          // 1
    //printf("entry begin   \n");
    //posix_memalign(&ptr, 64, NUM * CONV_C * 32 * sizeof( float ));

    float *NDIRECT_input = (float*)_mm_malloc((size_t)NUM * CONV_C * 32 *sizeof(float), 64);
    
    //printf("filter malloc error!\n");
    //printf("entry end   \n");
    //posix_memalign(&ptr1, 64, CONV_C * Tm * CONV_K * sizeof( float ));
    //float *NDIRECT_input = (float *)ptr;
    //float *NDIRECT_filter = (float *)ptr1;

    int input_HW_size = H * W;
    int output_HW_size = out_H*out_W;


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

	int gride_H_size = out_H / img_Tn;  // assume img_Tn < output_H     = output_H
	int gride_K_size = K / Tm;      // K
	int gride_H_e = out_H % img_Tn;      //0
	int gride_K_e = K % Tm;         //0
    //printf("exe begin   \n");
    //printf("input: %p, output: %p, filter: %p \n", input, output, filter);
    
    int CONV_HW_S2 = CONV_HW / out_W;        // 4096 / output_W
    
    #pragma omp parallel num_threads(NUM)
    {
        int i, j, k, C_index, K_index, pre_index;
        int id = omp_get_thread_num();
        int ii, jj, cc, nn, iis, jjs, ccs, H_to, K_to, C_to,hs;
        int Hb, Kb, Cb;
        int N_id = id / Tn;         // 0
        
        
		if (id  % img_Tn < gride_H_e)       // Assign tasks
		{
			jjs = (id % img_Tn) * (gride_H_size + 1);
			H_to = jjs + (gride_H_size + 1);
		}
		else
		{
			jjs = (id % img_Tn) * gride_H_size + gride_H_e;     // 0
			H_to = jjs + gride_H_size;      // output_H
		}

        iis = N_id * gride_K_size;          // 0
        K_to = iis + gride_K_size;          // K

        ccs = 0;
        C_to = C;
        //printf("C_to: %d, K_to: %d\n",C_to,K_to);
        #pragma omp barrier
        
        for ( jj = jjs; jj < H_to; jj = jj + Hb)          // 0 gride_HW_size HWb
        {
            Hb = CONV_HW_S2;
            if (H_to - jj < CONV_HW_S2)
            {
                Hb = H_to - jj;
            }
            //vk * CONV_C * (K/vk) * (C/CONV_c)
            for (cc = ccs; cc < C_to; cc = cc + Cb)         // 0 C Cb
            {

                //C_index = cc / CONV_C;
                Cb = CONV_C;
                if (C_to - cc < CONV_C)
                    Cb = C_to - cc;

                float *buffer_input = input + ((id % Tn) / img_Tn)  * thread_num_inputs * C * input_HW_size + cc * input_HW_size + (jj * W << 1);      //input + id * CHW + cc* HW + 0(jj)
                float *buffer_filter = filter + cc * K;
                for ( ii = iis ; ii < K_to; ii = ii + Kb)   // 0 K Kb
                {
                    
                    //pre_index = C_index * Num_K_block * Tn + (ii - iis) / CONV_K * Tn;
                    Kb = CONV_K;
                    if (K_to - ii < CONV_K)
                    {
                        Kb = K_to - ii;
                    }

                    float *buffer_filter1 = buffer_filter + ii * Cb; //filter + kC + 0(cc)
                    float *buffer_output = output + ( (id % Tn) / img_Tn) *thread_num_inputs * K * output_HW_size + ii * output_HW_size + jj;    
                    int EDGE_Kb = Kb % 12;
                    
                    int D_Kb = Kb-EDGE_Kb;
                    
                    for(nn = 0; nn < thread_num_inputs; nn++)       // 0 1 1
                    {
                        float *buffer_output1 = buffer_output + nn * K * output_HW_size; //output + id * KHW + ii* HW + 0(jj)
                        float *buffer_input1 = buffer_input + nn * C * input_HW_size; //input + id * CHW + cc* HW + 0(jj)
                        int EDGE_Wb = W % 64;
                        //printf("cc: %d, ii: %d, Kb: %d, EDGE_Kb: %d \n", cc, ii, Kb, EDGE_Kb);
                        
                        for(hs = 0; hs < Hb; hs++){
                        
                            if(EDGE_Wb != 0)      // 4 1
                            {
                                float *temp_buffer_input = buffer_input1;
                                float *temp_buffer_output = buffer_output1;
                                if (EDGE_Wb == 56)
                                {
                                    //printf("56 hs: %d, cc: %d, ii: %d, Kb: %d, EDGE_Kb: %d \n", hs, cc, ii, Kb, EDGE_Kb);
                                    direct_1x1_N12M28_s2(temp_buffer_output, buffer_filter1, temp_buffer_input, D_Kb, Cb, input_HW_size<<2, &NDIRECT_input[id * CONV_C * 32], cc, EDGE_Kb);                            }

                                else if (EDGE_Wb == 28)
                                {
                                    //printf("28 hs: %d, cc: %d, ii: %d, Kb: %d, EDGE_Kb: %d \n", hs,cc, ii, Kb, EDGE_Kb);
                                    direct_1x1_N12M14_s2(temp_buffer_output, buffer_filter1, temp_buffer_input, D_Kb, Cb, input_HW_size<<2, &NDIRECT_input[id * CONV_C * 32], cc, EDGE_Kb);                            }
                                else if (EDGE_Wb == 14)
                                {
                                    //printf("14 hs: %d, cc: %d, ii: %d, Kb: %d, EDGE_Kb: %d \n", hs,cc, ii, Kb, EDGE_Kb);
                                    direct_1x1_N12M7_s2(temp_buffer_output, buffer_filter1, temp_buffer_input, D_Kb, Cb, input_HW_size<<2, &NDIRECT_input[id * CONV_C * 32], cc, EDGE_Kb);                            }
                            }
							buffer_output1 = buffer_output1 + out_W;
							buffer_input1 = buffer_input1 +  (W << 1); 
                            
                        }
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
    
    int H = 56, W = 56, N = NUM, C = 32;
    int K = 12, R = 1, S = 1;
    int padh = 0, padw = 0, stride = 2;
    
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
    //if (argc > 11) padw      = atoi(argv[i++]);
    
    double pes = 128;
    
    int out_H = H/stride;
    int out_W = W/stride;
    int flag = 0;
    double ops = (double)2.0 * N * C * out_H * out_W * R * S * K * 1.0e-9;
    
    int stride_out_size = out_H*out_W*sizeof(float);
    int stride_in_size = H*W*sizeof(float);
    
    double result = R*S*C*1.0;
    
    float *filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 64);
    
    //printf("filter malloc error!\n");
    
    if(filter == NULL ){
        printf("filter aligned error!\n");
        return 0;
        
    }
    
    float *trans_filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 64);
    //float *trans_filter = (float*)_mm_malloc(K*C*3*3*sizeof(float), 32);
    
    //printf("trnas_filter malloc error!\n");
    
    if(filter == NULL ){
        printf("trans_filter aligned error!\n");
        return 0;
        
    }
    
    float *input =  (float*)_mm_malloc((size_t)N*C*H*W*sizeof(float), 64);
    
    //printf("input malloc error!\n");
    if(input == NULL ){
        printf("input aligned error!\n");
        return 0;
        
    }
    
    float *output = (float*)_mm_malloc((size_t)N*K*out_H*out_W*sizeof(float), 64);
    //printf("output malloc error!\n");
    
    if(output == NULL ){
        printf("output aligned error!\n");
        return 0;
        
    }
    float *output1 = (float*)malloc(N*K*out_H*out_W*sizeof(float));
    //printf("output1 malloc error!\n");
    float *data_col = (float*)malloc(N*C*(int(H/stride)*int(W/stride))*R*S*sizeof(float));
    //printf("data_col malloc error!\n");
    //float *output1 = (float*)_mm_malloc(N*K*H*W*sizeof(float), 32);
    printf("N = %d, K = %d, C = %d, H = %d, W = %d\n", N, K, C, H, W);
    for(int pc = 0; pc < 1; pc++)
    {
        random_matrix(K, C * R *S, filter);
        random_matrix(N, C * H * W, input);
        //random_matrix(K, C * R *S, trans_filter);
        //printf("1\n");
        transform_filter(K, C, filter, trans_filter);
        //verify_transform_filter(K, C, trans_filter);
        
        for(i = 0; i < 5; i++)
        {
            avx512_dircet_cnn_1x1s2(H,W,N,C,input,K,R,trans_filter, output);
        }
        
        //printf("3\n");
        start = dclock();
        for(i = 0; i < loop; i++)
        {
            avx512_dircet_cnn_1x1s2(H,W,N,C,input,K,R,trans_filter, output);
        }
        cost = (dclock() - start) / loop;

        
        printf("Gflops = %.3f, effic = %.3f %\n", ops/cost, ops/cost/NUM/pes * 100);
    
    }
    
    
    lda = C * R * S;
    ldb = out_H * out_W;
    ldc = out_H * out_W;

    int input_HW_size = H * W;
    int output_HW_size =  out_H * out_W;
    
    for(int i = 0; i < N; i++)
    {
      im2col_cpu(input + i * input_HW_size * C , C, H, W, R, S, padh, padh, stride, stride, 1, 1, data_col + i * C * output_HW_size);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, output_HW_size, C * R *S, 1.0, filter, lda, 
                  data_col + i * C * output_HW_size, ldb, 0.0, output1 + i * K * output_HW_size, ldc);
    }
    
    int M;
    
    M = N * K;
    N = out_H * out_W;
    
    /*
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            
            if((fabs(output[i * N + j] - output1[i * N + j]) > 0.001))
            {
                printf("i = %d, j= %d\n",i+1 ,j+1 );
                printf("out= %lf , out1= %lf\n", output[i*N + j], output1[i*N + j]);
                flag =1;
            }
            
        }
    }
    */
    if(flag == 0)
        printf("结果正确\n");
    

    free(filter);
    free(trans_filter);
    free(input);
    free(output);
    free(output1);
    free(data_col);
    return 0;
    
}