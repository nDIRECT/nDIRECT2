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

#define NUM 1
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

void random_matrix_input( int m, int n, float *a)
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


void random_matrix_filter( int m, int n, float *a)
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


void transform_filter(const int outch, const int inch, const int k_h, 
                const int k_w, float* kernel, float* out_kernel)
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

            //for(jj = j; jj < j + cr; jj++)
            for(h = 0; h < k_h; h ++)
            {
                //for(h = 0; h < k_h; h ++)
                for(jj = j; jj < j + cr; jj++)
                {
                    for(w = 0; w < k_w; w++)
                    {
                        
                        for(kk=0; kk<mr; kk++){
                            
                            out_kernel[ st+kk ] = kernel[ (i+kk) * inch * k_h * k_w+ 
                                                  jj * k_h * k_w + 
                                                   h * k_w + w ];
                            
                        }
                        st += mr;
                    }
                
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


void direct_1x1_N32M7_s2(float *output, float *trans_filter, float *input, int Kb, int Cb, int input_HW_size, float *input_buffer, int cc){
    asm volatile(


        ".macro KERNEL32x7_PACK_K1                         \n"

        "   vbroadcastss    16(%%rax), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm9          \n"
        "   add $128, %%rbx                                 \n"
        "   vmovups         %%ymm31, (%%r14)                \n"

        "   vbroadcastss    24(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm11         \n"
        "   vmovups         (%%rbx), %%zmm6                 \n"
        "   add            $32, %%r14                     \n"

        "   vbroadcastss    32(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm13         \n"
         "   vmovups         64(%%rbx), %%zmm7               \n"

        "   vbroadcastss    40(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm15         \n"

        "   vbroadcastss    48(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm17         \n"
        "   add             %%r8, %%rax                      \n"

        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm19         \n"
        "   vbroadcastss    (%%rax), %%zmm0                \n"
        "   vmovups     (%%rax), %%ymm30                     \n"
        "   vmovups     32(%%rax), %%ymm31                   \n"

        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vshufps    $0x88, %%ymm31, %%ymm30, %%ymm30      \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm21         \n"
        "   vbroadcastss    8(%%rax), %%zmm1                \n"
        "   vpermq  $0xd8, %%ymm30, %%ymm31                   \n"
        

        ".endm                                              \n"


        ".macro KERNEL32x7_PACK_K2                         \n"

        "   vbroadcastss    16(%%rax), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        "   add $128, %%rbx                                 \n"
        "   vmovups         %%ymm31, (%%r14)                \n"

        "   vbroadcastss    24(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"
        "   vmovups         (%%rbx), %%zmm4                 \n"
        "   add            $32, %%r14                     \n"


        "   vbroadcastss    32(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"
         "   vmovups         64(%%rbx), %%zmm5               \n"

        "   vbroadcastss    40(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"

        "   vbroadcastss    48(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"
        "   add             %%r8, %%rax                      \n"

        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"
        "   vbroadcastss    (%%rax), %%zmm0                \n"
        "   vmovups     (%%rax), %%ymm30                     \n"
        "   vmovups     32(%%rax), %%ymm31                   \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vshufps    $0x88, %%ymm31, %%ymm30, %%ymm30      \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"
        "   vbroadcastss    8(%%rax), %%zmm1                \n"
        "   vpermq  $0xd8, %%ymm30, %%ymm31                   \n"


        ".endm                                              \n"



        ".macro KERNEL32x7_PACK_END_K                      \n"


        "   vbroadcastss    16(%%rax), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        "   add $128, %%rbx                                 \n"
        "   vmovups         %%ymm31, (%%r14)                \n"
        
        "   vbroadcastss    24(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"


        "   vbroadcastss    32(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"

        "   vbroadcastss    40(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"

        "   vbroadcastss    48(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"
        
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"

        ".endm                                              \n"



        ".macro KERNEL32x7_K1                              \n"

        "   vbroadcastss    8(%%rax), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm9          \n"
        "   add             $128, %%rbx                     \n"
        
        "   vbroadcastss    12(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm11         \n"
        "   vmovups         (%%rbx), %%zmm6                 \n" 

        "   vbroadcastss    16(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm13         \n"
        "   vmovups         64(%%rbx), %%zmm7               \n"

        "   vbroadcastss    20(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm15         \n"

        "   vbroadcastss    24(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm17         \n"
        "   add             $32, %%rax                      \n"

        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm19         \n"
        "   vbroadcastss    (%%rax), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm21         \n"
        "   vbroadcastss    4(%%rax), %%zmm1                \n"

        

        ".endm                                              \n"


        ".macro KERNEL32x7_K2                              \n"

        "   vbroadcastss    8(%%rax), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        "   add            $128, %%rbx                     \n"

        "   vbroadcastss    12(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"
        "   vmovups         (%%rbx), %%zmm4                 \n" 

        "   vbroadcastss    16(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"
        "   vmovups         64(%%rbx), %%zmm5               \n"

        "   vbroadcastss    20(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"

        "   vbroadcastss    24(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"
        "   add             $32, %%rax                      \n"

        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"
        "   vbroadcastss    (%%rax), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"
        "   vbroadcastss    4(%%rax), %%zmm1                \n"





        ".endm                                              \n"



        ".macro KERNEL32x7_END_K                           \n"


        "   vbroadcastss    8(%%rax), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        "   add             $128, %%rbx                     \n"
        
        "   vbroadcastss    12(%%rax), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"

        "   vbroadcastss    16(%%rax), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"

        "   vbroadcastss    20(%%rax), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"

        "   vbroadcastss    24(%%rax), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"

        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"


        

        ".endm                                              \n"


        ".macro ST_32x7   \n"
        
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

        "   vmovups %%ymm0, (%%r10)%{%%k4%}         \n"     //  input 1
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"     //  input 5
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"     //  input 9
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"     //  input 13
        
        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
        
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
        
        "   vmovups %%ymm2, (%%r10)%{%%k4%}         \n"     //  input 1
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"     //  input 5
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"     //  input 9
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"     //  input 13
        
        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
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

        
        "   vmovups %%ymm0, (%%r10)%{%%k4%}         \n"     //  input 1
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"     //  input 5
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"     //  input 9
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"     //  input 13
        
        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
        
        
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


        "   vmovups %%ymm2, (%%r10)%{%%k4%}         \n"     //  input 1
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"     //  input 5
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"     //  input 9
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"     //  input 13

        
        
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        
        "   shl $2, %%rsi                \n"
        
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   prefetcht2      (%%r11)                         \n"
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   prefetcht2      (%%r12)                         \n"
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        "   prefetcht2      (%%r13)                         \n"
        
        "   shr $2, %%rsi                \n"
        
        
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

        

        "   vmovups %%ymm0, (%%r10)%{%%k4%}         \n"     //  input 1
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"     //  input 5
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"     //  input 9
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"     //  input 13

        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
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
        
        "   vmovups %%ymm2, (%%r10)%{%%k4%}         \n"     //  input 1
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"     //  input 5
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"     //  input 9
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"     //  input 13
        
        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
        
        
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

        "   vmovups %%ymm0, (%%r10)%{%%k4%}         \n"     //  input 1
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"     //  input 5
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"     //  input 9
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"     //  input 13
        
        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
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

        
        "   vmovups %%ymm2, (%%r10)%{%%k4%}         \n"     //  input 1
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"     //  input 5
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"     //  input 9
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"     //  input 13
        
        ".endm      \n"
        
        //-----------------add
        
        ".macro ADD_C_32x7   \n"

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
        
        "   vmovups %%ymm0, (%%r10)%{%%k4%}         \n"
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"

        
        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
        
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
        
        "   vmovups %%ymm2, (%%r10)%{%%k4%}         \n"
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"
        
        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
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
        
        "   vmovups %%ymm0, (%%r10)%{%%k4%}         \n"
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"

        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
        
        
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
        
        "   vmovups %%ymm2, (%%r10)%{%%k4%}         \n"
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"

        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        
        "   shl $2, %%rsi                \n"
        
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        
        "   shr $2, %%rsi                \n"
        
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
        
        "   vmovups %%ymm0, (%%r10)%{%%k4%}         \n"
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"
        
        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
        
        
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
        
        "   vmovups %%ymm2, (%%r10)%{%%k4%}         \n"
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"
        
        
        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
        
        
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
        
        "   vmovups %%ymm0, (%%r10)%{%%k4%}         \n"
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"
        
        "   add %%rsi, %%r10    \n"
        "   add %%rsi, %%r11    \n"
        "   add %%rsi, %%r12    \n"
        "   add %%rsi, %%r13    \n"
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
        "   vmovups (%%r13), %%ymm4%{%%k4%}     \n"
        
        "   vaddps %%ymm0, %%ymm2, %%ymm2       \n"
        "   vaddps %%ymm1, %%ymm5, %%ymm5       \n"
        "   vaddps %%ymm3, %%ymm6, %%ymm6       \n"
        "   vaddps %%ymm4, %%ymm7, %%ymm7       \n"
        
        "   vmovups %%ymm2, (%%r10)%{%%k4%}         \n"
        "   vmovups %%ymm5, (%%r11)%{%%k4%}         \n"
        "   vmovups %%ymm6, (%%r12)%{%%k4%}         \n"
        "   vmovups %%ymm7, (%%r13)%{%%k4%}         \n"

        
        ".endm      \n"


        "CONV_KERNEL32x7:                                \n"

        "   mov     %[output], %%rcx                             \n"
        "   mov     %[trans_filter], %%rbx                             \n"
        "   mov     %[input], %%r9                             \n"
        "   movl    $0x7f, %%edi       \n"


        "   movl     %[Cb], %%edx                             \n"  // Cb
        "   movl     %[input_HW_size], %%r8d                             \n" 
        
        "   kmovd   %%edi, %%k4            \n"
        
        "   mov         %[input_buffer], %%r14                            \n"
        "   movl        %[Kb], %%edi                             \n"
        "   mov %%r8, %%rsi                 \n"
        "   shr     $2, %%rsi                       \n"
        
        "BEGIN_PACK:                                        \n"

        "   prefetcht0      (%%rbx)                         \n" 
        "   shl $2, %%rsi                \n"
        "   mov     %%rcx, %%r10                            \n"  // C0
        "   mov     %%r9, %%rax                         \n"

        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1

        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3
        
        
        "   shr $2, %%rsi                \n"
        
        "   vmovups     (%%rax), %%ymm30                     \n"
        "   vmovups     32(%%rax), %%ymm31                   \n"

        "   vmovups     (%%rbx), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  
        "   vmovups     64(%%rbx), %%zmm5                   \n"
        "   vshufps    $0x88, %%ymm31, %%ymm30, %%ymm30      \n"
        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"
        "   vpxorq      %%zmm13, %%zmm13, %%zmm13           \n" 
        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 
        
        "   vbroadcastss    (%%rax), %%zmm0                 \n"
        "   vbroadcastss    8(%%rax), %%zmm1                \n"
        "   vpxorq      %%zmm15, %%zmm15, %%zmm15           \n"
        "   vpxorq      %%zmm16, %%zmm16, %%zmm16           \n"
        "   vpxorq      %%zmm17, %%zmm17, %%zmm17           \n" 
        "   vpxorq      %%zmm18, %%zmm18, %%zmm18           \n" 
        "   vpxorq      %%zmm19, %%zmm19, %%zmm19           \n"
        "   vpermq  $0xd8, %%ymm30, %%ymm31                   \n"

        "   vpxorq      %%zmm20, %%zmm20, %%zmm20           \n"
        "   vpxorq      %%zmm21, %%zmm21, %%zmm21           \n"

  

        "   sub    $8, %%rdx                               \n"

        
        "MAIN_PACK_K:                                       \n"


        "   KERNEL32x7_PACK_K1                             \n"
        "   KERNEL32x7_PACK_K2                             \n"
        "   KERNEL32x7_PACK_K1                             \n"
        "   KERNEL32x7_PACK_K2                             \n"
        "   KERNEL32x7_PACK_K1                             \n"
        "   KERNEL32x7_PACK_K2                             \n"
        "   KERNEL32x7_PACK_K1                             \n"
        "   cmp     $0, %%rdx                               \n"
        
        "   je      PACK_SAVE                             \n"

        "   KERNEL32x7_PACK_K2                             \n"

        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_PACK_K                             \n"
        

        "PACK_SAVE:                                       \n"
        "   movl %[cc], %%r15d              \n"
        "   KERNEL32x7_PACK_END_K                          \n"
        "   mov     %[input_buffer], %%r14                            \n"
        "   cmp     $0, %%r15                               \n"

        "   je      PACK_ST_C                                  \n"
        "   ADD_C_32x7                                     \n"
        "   jmp PACK_Kb_END                 \n"
        "PACK_ST_C:                                            \n"
        "   ST_32x7                                      \n"
        
        "   movl     %[cc], %%r15d                         \n"
        "   movl     %[Cb], %%edx                             \n"  // Cb
        "PACK_Kb_END:               \n"
        "   sub    $1, %%rdi       \n"


        
        "   je     END_M                              \n"
        
        
        "BEGIN_M:                                           \n"

        
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r13, %%rsi), %%r10                 \n"  // C3
        
        "   shl $2, %%rsi                \n"
        "   movl %[cc], %%r15d          \n"
        "   leaq    (%%r10, %%rsi), %%r11                 \n"  // C1
        "   leaq    (%%r11, %%rsi), %%r12                 \n"  // C2
        "   leaq    (%%r12, %%rsi), %%r13                 \n"  // C3

        "   shr $2, %%rsi                \n"
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


        "   KERNEL32x7_K1                                  \n"
        "   KERNEL32x7_K2                                  \n"
        "   KERNEL32x7_K1                                  \n"
        "   KERNEL32x7_K2                                  \n"
        "   KERNEL32x7_K1                                  \n"
        "   KERNEL32x7_K2                                  \n"
        "   KERNEL32x7_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        
        "   je      EDGE_K                                  \n"
        
        "   KERNEL32x7_K2                                  \n"

        "   sub    $8, %%rdx                               \n"
        
        "   jmp     MAIN_K                                  \n"
        
        "EDGE_K:                                            \n"
        
        "   KERNEL32x7_END_K                               \n"
        
        "BEGIN_SAVE:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C                                  \n"
        "   ADD_C_32x7                                     \n"
        "   jmp Kb_END                  \n"
        "ST_C:                                            \n"
        "   ST_32x7                                      \n"
        
        //"   movl     %[cc], %%r15d                         \n"
        //"   movl     %[Cb], %%edx                             \n"  // Cb
        "Kb_END:                    \n"
        "   sub     $1, %%rdi       \n"
        "   jne     BEGIN_M                              \n"
        
        
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
         [cc]                   "m" (cc)
        :
         "rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
         "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
         "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
         "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
         "zmm30", "zmm31", "memory"
    
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
    float *NDIRECT_input = (float*)_mm_malloc((size_t)NUM * CONV_C * 8 *sizeof(float), 64);
    int input_HW_size = H * W;
    int output_HW_size = out_H*out_W;

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
        printf("H_to: %d, C_to: %d, K_to: %d\n",H_to, C_to,K_to);
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

                    
                    for(nn = 0; nn < thread_num_inputs; nn++)       // 0 1 1
                    {
                        float *buffer_output1 = buffer_output + nn * K * output_HW_size; //output + id * KHW + ii* HW + 0(jj)
                        float *buffer_input1 = buffer_input + nn * C * input_HW_size; //input + id * CHW + cc* HW + 0(jj)
                        int EDGE_Wb = W % 24;
                        //printf("cc: %d, ii: %d, Kb: %d, EDGE_Kb: %d \n", cc, ii, Kb, EDGE_Kb);
                        
                        for(hs = 0; hs < Hb; hs++){
                        
                            if(EDGE_Wb != 0)      // 4 1
                            {
                                float *temp_buffer_input = buffer_input1;
                                float *temp_buffer_output = buffer_output1;
                                if (EDGE_Wb == 14)
                                {
                                    printf("14 Hb: %d, hs: %d, cc: %d, ii: %d, Kb: %d\n", Hb, hs, cc, ii, Kb);
                                    direct_1x1_N32M7_s2(temp_buffer_output, buffer_filter1, temp_buffer_input, Kb>>5, Cb, input_HW_size<<2, &NDIRECT_input[id * CONV_C * 8], cc);
                                }
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
    
    int H = 14, W = 14, N = NUM, C = 32;
    int K = 32, R = 1, S = 1;
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
        random_matrix_filter(K, C * R *S, filter);
        random_matrix_input(N, C * H * W, input);
        //random_matrix(K, C * R *S, trans_filter);
        //printf("1\n");
        transform_filter(K, C, R,S,filter, trans_filter);
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