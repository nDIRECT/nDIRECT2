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
#define CONV_K 256
#define CONV_C 64

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
      //


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

void verify_transform_filter(const int outch, const int inch, const int k_h, 
                const int k_w, float* out_kernel)
{
    int M = outch*inch;
    int N = k_h * k_w;
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

void direct_3x3_N32M12_AVX512_pack(float *img_start_0, float *img_pack, float *kernal, float *out, int ic_count, int ic, int stride_in_size, int hh_tag, int h, int kk, int ww, int W_size){
    long stride_filter = 0;
    if(hh_tag == 0){
        
        stride_filter = ic_count * 3 * 32 * sizeof(float);
        
    }
    asm volatile(
        ".macro PACK_KERNEL32x11_K1     \n"
        "   vbroadcastss (%%r10), %%zmm27    \n "
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vbroadcastss 8(%%r10), %%zmm30    \n "
        "   movl    $0, %%r8d      \n"
        "   vbroadcastss 12(%%r10), %%zmm31    \n "
        "   vmovups (%%r10), %%ymm28     \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   vmovups 32(%%r10), %%xmm29     \n"
        
        
        "   vbroadcastss 16(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        "   movl %%r8d, (%%rdx)     \n"
        
        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm17 \n"
        "   movl 48(%%r10), %%r8d     \n"
        
        "   vbroadcastss 24(%%r10), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm18 \n"
        "   vmovups %%ymm28, 4(%%rdx)     \n"
        
        "   vbroadcastss 28(%%r10), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        "   vmovups %%xmm29, 36(%%rdx)     \n"
        
        "   vbroadcastss 32(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        "   movl %%r8d, 52(%%rdx)     \n"
        
        "   vbroadcastss 36(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm21 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm22 \n"
        "   addq $56, %%rdx      \n"
        

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm25 \n"
        ".endm                          \n"
    
        ".macro PACK_KERNEL32x11_K2     \n"
        "   vbroadcastss (%%r10), %%zmm1    \n "
        "   vbroadcastss 4(%%r10), %%zmm27    \n "
        "   vbroadcastss 8(%%r10), %%zmm30    \n "
        "   vbroadcastss 12(%%r10), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 20(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"
        
        "   vbroadcastss 28(%%r10), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 32(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 36(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm20 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm21 \n"
        
        "   vbroadcastss 44(%%r10), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm24 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm25 \n"
        
        
        ".endm                          \n"
    
        ".macro PACK_KERNEL32x11_K3     \n"
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vbroadcastss 8(%%r10), %%zmm27    \n "
        "   vbroadcastss 12(%%r10), %%zmm30    \n "
        "   vbroadcastss 16(%%r10), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"
        
        "   vbroadcastss 28(%%r10), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"
        
        "   vbroadcastss 32(%%r10), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 36(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm20 \n"
        
        "   vbroadcastss 44(%%r10), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm21 \n"
        
        "   vbroadcastss 48(%%r10), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        

        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm24 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm25 \n"
        
        
        ".endm                          \n"
        
        ".macro PACK_KERNEL32x12_K1     \n"
        "   vbroadcastss (%%r10), %%zmm1    \n "
        "   vbroadcastss 4(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   vmovups 32(%%r10), %%xmm29     \n"
        
        "   vbroadcastss 12(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        "   movl 48(%%r10), %%r8d     \n"
        
        "   vbroadcastss 20(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        "   vmovups %%ymm28, (%%rdx)     \n"
        
        "   vbroadcastss 24(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        "   vmovups %%xmm29, 32(%%rdx)     \n"
        
        "   vbroadcastss 28(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        "   movl %%r8d, 48(%%rdx)     \n"
        
        "   vbroadcastss 32(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"
        "   movl 52(%%r10), %%r8d     \n"
        
        "   vbroadcastss 36(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        "   movl %%r8d, 52(%%rdx)     \n"
        
        "   vbroadcastss 40(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        "   addq $56, %%rdx      \n"
        
        "   vbroadcastss 44(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        ".endm                              \n"
        
        ".macro PACK_KERNEL32x12_K2     \n"
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vbroadcastss 8(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        
        "   vbroadcastss 12(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"

        
        "   vbroadcastss 16(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"

        
        "   vbroadcastss 24(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"

        
        "   vbroadcastss 28(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"

        
        "   vbroadcastss 32(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        
        "   vbroadcastss 36(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"

        
        "   vbroadcastss 40(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"

        
        "   vbroadcastss 44(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        
        "   vbroadcastss 48(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        ".endm                              \n"
        
        ".macro PACK_KERNEL32x12_K3     \n"
        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        "   vbroadcastss 12(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        
        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"

        
        "   vbroadcastss 20(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"

        
        "   vbroadcastss 28(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"

        
        "   vbroadcastss 32(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"

        
        "   vbroadcastss 36(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        
        "   vbroadcastss 40(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"

        
        "   vbroadcastss 44(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"

        
        "   vbroadcastss 48(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        
        "   vbroadcastss 52(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        ".endm                              \n"
        
        
        //--------------------- main
        
        ".macro KERNEL32x12_K1  \n"
        
        "   vbroadcastss (%%rdx), %%zmm1    \n "
        "   vbroadcastss 4(%%rdx), %%zmm27    \n "
        "   vbroadcastss 8(%%rdx), %%zmm30    \n "
        "   vbroadcastss 12(%%rdx), %%zmm31    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 16(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 20(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"
        
        "   vbroadcastss 24(%%rdx), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"
        
        "   vbroadcastss 28(%%rdx), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 32(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 36(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm20 \n"
        
        "   vbroadcastss 40(%%rdx), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm21 \n"
        
        "   vbroadcastss 44(%%rdx), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm24 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm25 \n"
        
        ".endm      \n"
        
        ".macro KERNEL32x12_K2  \n"
        
        "   vbroadcastss 4(%%rdx), %%zmm1    \n "
        "   vbroadcastss 8(%%rdx), %%zmm27    \n "
        "   vbroadcastss 12(%%rdx), %%zmm30    \n "
        "   vbroadcastss 16(%%rdx), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 20(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 24(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"
        
        "   vbroadcastss 28(%%rdx), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"
        
        "   vbroadcastss 32(%%rdx), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 36(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 40(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm20 \n"
        
        "   vbroadcastss 44(%%rdx), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm21 \n"
        
        "   vbroadcastss 48(%%rdx), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm24 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm25 \n"
        
        ".endm      \n"
        
        ".macro KERNEL32x12_K3  \n"
        
        "   vbroadcastss 8(%%rdx), %%zmm1    \n "
        "   vbroadcastss 12(%%rdx), %%zmm27    \n "
        "   vbroadcastss 16(%%rdx), %%zmm30    \n "
        "   vbroadcastss 20(%%rdx), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 24(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 28(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"
        
        "   vbroadcastss 32(%%rdx), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"
        
        "   vbroadcastss 36(%%rdx), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 40(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 44(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm20 \n"
        
        "   vbroadcastss 48(%%rdx), %%zmm30    \n "
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm21 \n"
        
        "   vbroadcastss 52(%%rdx), %%zmm31    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm24 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm25 \n"
        
        ".endm      \n"
        
        //----------------
        
        ".macro AVX512_N32M12_ST_END_PACK   \n"
        
        "   shl $2, %%r11     \n"
        "   shl $3, %%r14       \n"
        "   add %%r11, %%r8    \n"      // output 5
        "   add %%r14, %%r9    \n"      // output 9
        "   add %%r11, %%r14    \n"
        
        "   shl $1, %%rbx       \n"     //inc 2
        "   add %%r14, %%r10    \n"     //output 13
        
        "   sub %%r15, %%r11    \n"     //inc 3
        "   add %%r15, %%r14    \n"
        //  1 5 9 13
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq  $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"     //  input 1
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 9
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 5
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 13
        //  
        "   vunpcklps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpcklps %%zmm13, %%zmm12, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r12)         \n"     //  input 1
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 13
        
        "   vmovups %%xmm28, 32(%%r8)         \n"     //  input 5
        "   vmovups %%xmm29, 32(%%r9)         \n"     //  input 9
        "   vmovups %%xmm30, 32(%%r10)         \n"     //  input 13
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        // 2 6 10 14
        
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"     //  input 2
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 10
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 6
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 14
        
        //  
        "   vunpcklps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpcklps %%zmm13, %%zmm12, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 32(%%r12)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups %%xmm28, 32(%%r8)         \n"     //  input 6
        "   vmovups %%xmm29, 32(%%r9)         \n"     //  input 10
        "   vmovups %%xmm30, 32(%%r10)         \n"     //  input 14
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //3 7 11 15
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"     //  input 3
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 11
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 7
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 15
        
        //  
        "   vunpckhps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpckhps %%zmm13, %%zmm12, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r12)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 14
        
        "   vmovups %%xmm28, 32(%%r8)         \n"     //  input 6
        "   vmovups %%xmm29, 32(%%r9)         \n"     //  input 10
        "   vmovups %%xmm30, 32(%%r10)         \n"     //  input 14
        
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //4 8 12 16
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"     //  input 4
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 12
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 8
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 16
        
        //  
        "   vunpckhps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpckhps %%zmm13, %%zmm12, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 32(%%r12)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups %%xmm28, 32(%%r8)         \n"     //  input 6
        "   vmovups %%xmm29, 32(%%r9)         \n"     //  input 10
        "   vmovups %%xmm30, 32(%%r10)         \n"     //  input 14
        
        "   add %%r14, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        //17 21 25 29
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"     //  input 17
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 21
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 25
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 29
        
        //  
        "   vunpcklps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpcklps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r12)         \n"     //  input 1
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 13
        
        "   vmovups %%xmm28, 32(%%r8)         \n"     //  input 5
        "   vmovups %%xmm29, 32(%%r9)         \n"     //  input 9
        "   vmovups %%xmm30, 32(%%r10)         \n"     //  input 13
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //18 22 26 30
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"     //  input 18
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 26
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 22
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 30
        
        //  
        "   vunpcklps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpcklps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 32(%%r12)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups %%xmm28, 32(%%r8)         \n"     //  input 6
        "   vmovups %%xmm29, 32(%%r9)         \n"     //  input 10
        "   vmovups %%xmm30, 32(%%r10)         \n"     //  input 14
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //19 23 27 31
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"     //  input 19
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 23
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 27
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 31
        
        //  
        "   vunpckhps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpckhps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r12)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 14
        
        "   vmovups %%xmm28, 32(%%r8)         \n"     //  input 6
        "   vmovups %%xmm29, 32(%%r9)         \n"     //  input 10
        "   vmovups %%xmm30, 32(%%r10)         \n"     //  input 14
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //20 24 28 32
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"     //  input 20
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 24
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 28
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 32
        
        //  
        "   vunpckhps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpckhps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 32(%%r12)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups %%xmm28, 32(%%r8)         \n"     //  input 6
        "   vmovups %%xmm29, 32(%%r9)         \n"     //  input 10
        "   vmovups %%xmm30, 32(%%r10)         \n"     //  input 14
        
        ".endm      \n"
        
        //-----------------add
        
        ".macro AVX512_N32M12_ADD_END_PACK   \n"
        "   shl $2, %%r11     \n"
        "   shl $3, %%r14       \n"
        "   add %%r11, %%r8    \n"      // output 5
        "   add %%r14, %%r9    \n"      // output 9
        "   add %%r11, %%r14    \n"
        
        "   shl $1, %%rbx       \n"     //inc 2
        "   add %%r14, %%r10    \n"     //output 13
        
        "   sub %%r15, %%r11    \n"     //inc 3
        //"   mov %[stride_out_size], %%r14   \n"
        "   add %%r15, %%r14    \n"
        //  1 5 9 13
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq  $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        
        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        //  
        "   vunpcklps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpcklps %%zmm13, %%zmm12, %%zmm26    \n"

        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 5
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 9
        
        "   vmovups 32(%%r12), %%xmm1     \n"
        "   vmovups 32(%%r8), %%xmm26     \n"
        "   vmovups 32(%%r9), %%xmm27     \n"
        "   vmovups 32(%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, 32(%%r12)         \n"
        "   vmovups %%xmm28, 32(%%r8)         \n"
        "   vmovups %%xmm29, 32(%%r9)         \n"
        "   vmovups %%xmm30, 32(%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        // 2 6 10 14
        
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30      \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        
        //  
        "   vunpcklps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpcklps %%zmm13, %%zmm12, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups 32(%%r12), %%xmm0     \n"
        "   vmovups 32(%%r8), %%xmm1     \n"
        "   vmovups 32(%%r9), %%xmm27     \n"
        "   vmovups 32(%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, 32(%%r12)         \n"
        "   vmovups %%xmm28, 32(%%r8)         \n"
        "   vmovups %%xmm29, 32(%%r9)         \n"
        "   vmovups %%xmm30, 32(%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //3 7 11 15
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        
        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        //  
        "   vunpckhps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpckhps %%zmm13, %%zmm12, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 14
        

        
        "   vmovups 32(%%r12), %%xmm1     \n"
        "   vmovups 32(%%r8), %%xmm26     \n"
        "   vmovups 32(%%r9), %%xmm27     \n"
        "   vmovups 32(%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, 32(%%r12)         \n"
        "   vmovups %%xmm28, 32(%%r8)         \n"
        "   vmovups %%xmm29, 32(%%r9)         \n"
        "   vmovups %%xmm30, 32(%%r10)         \n"
        
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //4 8 12 16
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        
        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        
        //  
        "   vunpckhps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpckhps %%zmm13, %%zmm12, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14

        
        "   vmovups 32(%%r12), %%xmm0     \n"
        "   vmovups 32(%%r8), %%xmm1     \n"
        "   vmovups 32(%%r9), %%xmm27     \n"
        "   vmovups 32(%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, 32(%%r12)         \n"
        "   vmovups %%xmm28, 32(%%r8)         \n"
        "   vmovups %%xmm29, 32(%%r9)         \n"
        "   vmovups %%xmm30, 32(%%r10)         \n"
        
        
        "   add %%r14, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        //17 21 25 29
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        
        //  
        "   vunpcklps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpcklps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 13
        
        "   vmovups 32(%%r12), %%xmm1     \n"
        "   vmovups 32(%%r8), %%xmm26     \n"
        "   vmovups 32(%%r9), %%xmm27     \n"
        "   vmovups 32(%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, 32(%%r12)         \n"
        "   vmovups %%xmm28, 32(%%r8)         \n"
        "   vmovups %%xmm29, 32(%%r9)         \n"
        "   vmovups %%xmm30, 32(%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //18 22 26 30
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29      \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        
        //  
        "   vunpcklps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpcklps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups 32(%%r12), %%xmm0     \n"
        "   vmovups 32(%%r8), %%xmm1     \n"
        "   vmovups 32(%%r9), %%xmm27     \n"
        "   vmovups 32(%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, 32(%%r12)         \n"
        "   vmovups %%xmm28, 32(%%r8)         \n"
        "   vmovups %%xmm29, 32(%%r9)         \n"
        "   vmovups %%xmm30, 32(%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //19 23 27 31
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        //  
        "   vunpckhps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpckhps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 14
        "   vmovups 32(%%r12), %%xmm1     \n"
        "   vmovups 32(%%r8), %%xmm26     \n"
        "   vmovups 32(%%r9), %%xmm27     \n"
        "   vmovups 32(%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, 32(%%r12)         \n"
        "   vmovups %%xmm28, 32(%%r8)         \n"
        "   vmovups %%xmm29, 32(%%r9)         \n"
        "   vmovups %%xmm30, 32(%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //20 24 28 32
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        //  
        "   vunpckhps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpckhps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups 32(%%r12), %%xmm0     \n"
        "   vmovups 32(%%r8), %%xmm1     \n"
        "   vmovups 32(%%r9), %%xmm27     \n"
        "   vmovups 32(%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28      \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, 32(%%r12)         \n"
        "   vmovups %%xmm28, 32(%%r8)         \n"
        "   vmovups %%xmm29, 32(%%r9)         \n"
        "   vmovups %%xmm30, 32(%%r10)         \n"
        
        ".endm      \n"
        
        "AVX512_N32M12_ENTRANCE_PACK:            \n"
        "   mov %[img_start_0], %%r10    \n"
        "   mov %[img_start_0], %%rdi    \n"
        "   mov %[img_pack],    %%rdx      \n"
        "   mov %[kernal], %%r11 \n"
        "   movl %[h], %%r12d    \n"

        "   movl %[ic_count], %%r13d  \n"           //
        "   mov %[stride_in_size], %%r14    \n"
        "   mov %[W_size], %%r15    \n"
        "   movl %[kk], %%ebx    \n"
        "   movl %[hh_tag], %%eax   \n"
        "   movl %[ww], %%ecx        \n"
        "   mov %[stride_filter], %%rsi    \n"

        "   vpxorq  %%zmm2, %%zmm2, %%zmm2  \n"
        "   vpxorq  %%zmm3, %%zmm3, %%zmm3  \n"
        "   vpxorq  %%zmm4, %%zmm4, %%zmm4  \n"
        "   vpxorq  %%zmm5, %%zmm5, %%zmm5  \n"
        "   vpxorq  %%zmm6, %%zmm6, %%zmm6  \n"
        "   vpxorq  %%zmm7, %%zmm7, %%zmm7  \n"
        "   vpxorq  %%zmm8, %%zmm8, %%zmm8  \n"
        "   vpxorq  %%zmm9, %%zmm9, %%zmm9  \n"
        "   vpxorq  %%zmm10, %%zmm10, %%zmm10  \n"
        "   vpxorq  %%zmm11, %%zmm11, %%zmm11  \n"
        "   vpxorq  %%zmm12, %%zmm12, %%zmm12  \n"
        "   vpxorq  %%zmm13, %%zmm13, %%zmm13  \n"
        "   vpxorq  %%zmm14, %%zmm14, %%zmm14  \n"
        "   vpxorq  %%zmm15, %%zmm15, %%zmm15  \n"
        "   vpxorq  %%zmm16, %%zmm16, %%zmm16  \n"
        "   vpxorq  %%zmm17, %%zmm17, %%zmm17  \n"
        "   vpxorq  %%zmm18, %%zmm18, %%zmm18  \n"
        "   vpxorq  %%zmm19, %%zmm19, %%zmm19  \n"
        "   vpxorq  %%zmm20, %%zmm20, %%zmm20  \n"
        "   vpxorq  %%zmm21, %%zmm21, %%zmm21  \n"
        "   vpxorq  %%zmm22, %%zmm22, %%zmm22  \n"
        "   vpxorq  %%zmm23, %%zmm23, %%zmm23  \n"
        "   vpxorq  %%zmm24, %%zmm24, %%zmm24  \n"
        "   vpxorq  %%zmm25, %%zmm25, %%zmm25  \n"
        //"   prefetcht0 (%%r11)                 \n"
        
        "K_Branch:                  \n"
        "   mov $2, %%r9d       \n"
        "   cmp $0, %%ebx       \n"
        "   jne BEGIN_Kb_Pre                 \n"
        
        //kk=0
        "   cmp $0, %%eax          \n"
        "   je PACK_Htag0                     \n"
        "   cmp $1, %%eax           \n"
        "   je PACK_Htag1                     \n"
        
        //2
        "PACK_Htag2:                \n"
        "   cmp $0, %%r12d           \n"
        "   jne PACK_Htag2_Fetch2                   \n"
        
        "PACK_Htag2_Pack2:                \n"       //h=0
        
        "   cmp $0, %%ecx           \n"
        "   je PACK_Pad_Zero       \n"
        
        "PACK_Htag2_Main:           \n"     //kk=0 hh_tag=2 h=0 ww>0
        "   jmp PACK_Main_C         \n"
        
        //"PACK_Htag2_Pad:           \n"      //kk=0 hh_tag=2 h=0 ww=0
        //"   jmp PACK_Pad_Zero       \n"
        
        "PACK_Htag2_Fetch2:                \n"       //kk=0 hh_tag=2 h>0
        
        "   jmp BEGIN_Kb_Fetch2                \n"
        
        
        //1
        "PACK_Htag1:                \n"
        "   cmp $0, %%r12d          \n"
        "   jne BEGIN_Kb_Fetch2      \n"            
        "   mov $3, %%r9d       \n"             //kk=0 hh_tag=1 h=0
        
        "   cmp $0, %%ecx           \n"
        "   je PACK_Pad_Zero       \n"
        
        "PACK_Htag1_Main:           \n"     //kk=0 hh_tag=1 h=0 ww=0
        "   jmp PACK_Main_C         \n"
        
        //"PACK_Htag1_Pad:           \n"      //kk=0 hh_tag=1 h=0 ww>0
        //"   jmp PACK_Pad_Zero       \n"
        
        //"PACK_Htag1_Fetch2:                \n"       //kk=0 hh_tag=1 h>0
        
        //"   jmp BEGIN_Kb_Fetch2                \n"
        
        
        //0
        "PACK_Htag0:                \n"
        "   addq %%rsi, %%r11        \n"
        "   cmp $0, %%ecx           \n"
        "   je PACK_Pad_Zero       \n"
        
        "PACK_Htag0_Main:           \n"     //kk=0 hh_tag=0 h=0 ww>0
        "   jmp PACK_Main_C         \n"
        
        //"PACK_Htag0_Pad:           \n"      //kk=0 hh_tag=0 h=0 ww=0
        //"   jmp PACK_Pad_Zero       \n"
        
        "PACK_Pad_Zero_Pre:             \n"
        "   addq $128,%%r11       \n"
        "   addq %%r14, %%r10   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%r10)                 \n"
        
        "PACK_Pad_Zero:             \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x11_K1        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x11_K2        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"         
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x11_K3        \n"
        
        "   subl $1, %%r13d       \n"
        "   jne     PACK_Pad_Zero_Pre       \n"
        "   subl $1, %%r9d          \n"
        "   je      AVX512_N32M12_MAIN_END_PACK     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     PACK_Pad_Zero            \n"

        
        
        "PACK_Main_C_Pre:                       \n"
        "   addq $128,%%r11       \n"
        "   addq %%r14, %%r10   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%r10)                 \n"
        
        "PACK_Main_C:                       \n"

        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x12_K1        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x12_K2        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x12_K3        \n"
        
        "   subl $1, %%r13d       \n"
        "   jne     PACK_Main_C_Pre       \n"
        "   subl $1, %%r9d          \n"
        "   je      AVX512_N32M12_MAIN_END_PACK     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     PACK_Main_C            \n"

        
        "BEGIN_Kb_Mid_Fetch2:                  \n"
        "   addq $128, %%r11       \n"
        "   addq $56, %%rdx   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb_Fetch2:                  \n"
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x12_K1    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"

        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x12_K2    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x12_K3    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid_Fetch2                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid_Fetch2                      \n"
        "   cmp $2, %%eax           \n"
        "   je AVX512_N32M12_MAIN_END_PACK          \n"
        "   addq %%r15, %%rdi        \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128, %%r11       \n"
        "   addq $56, %%rdx   \n"
        "   movl $1, %%r9d          \n"
        "   mov %%rdi, %%r10       \n"
        "   cmp $0, %%ecx           \n"
        "   je PACK_Pad_Zero            \n"
        "   jmp PACK_Main_C             \n"

        


        "BEGIN_Kb_Pre:                  \n"
        "   addq %%rsi, %%r11       \n"
        "   cmp $1, %%eax               \n"
        "   jne  BEGIN_Kb                \n"
        "   mov $3, %%r9d               \n"
        "   jmp  BEGIN_Kb                   \n"
        
        "BEGIN_Kb_Pre_Filter:                  \n"
        "   addq %%rsi, %%r11       \n"
        "   addq $56, %%rdx   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        "   jmp  BEGIN_Kb                   \n"
        
        "BEGIN_Kb_Mid:                  \n"
        "   addq $128, %%r11       \n"
        "   addq $56, %%rdx   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb:                  \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x12_K1    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"

        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x12_K2    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x12_K3    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid                      \n"
        
        
        //  ADD or ST
        "AVX512_N32M12_MAIN_END_PACK:       \n"
        "   movl %[ic], %%eax   \n"
        "   mov %[out], %%r12    \n"
        "   mov %[out], %%r8    \n"
        "   mov %[out], %%r9    \n"
        "   mov %[out], %%r10    \n"
        "   mov %[stride_in_size], %%rbx   \n"
        "   mov %[stride_in_size], %%r15   \n"
        "   mov %[stride_in_size], %%r14   \n"
        "   mov %[stride_in_size], %%r11   \n"
        "   cmp $0, %%eax    \n"
        "   je AVX512_N32M12_ST_PACK    \n"
        
        // ADD
        "   AVX512_N32M12_ADD_END_PACK   \n"
        "   jmp AVX512_N32M12_END_PACK          \n"
        
        //  ST
        "AVX512_N32M12_ST_PACK:     \n"
        "   AVX512_N32M12_ST_END_PACK     \n"
        
        "AVX512_N32M12_END_PACK:     \n"
        
    :
    
    
    :
        [img_start_0]                 "m"     (img_start_0),
        [img_pack]                    "m"     (img_pack),
        [kernal]                      "m"     (kernal),
        [out]                       "m"     (out),
        [ic_count]                  "m"     (ic_count),
        [ic]                        "m"     (ic),
        [stride_in_size]            "m"     (stride_in_size),
        [hh_tag]                         "m"     (hh_tag),
        [h]                         "m"     (h),
        [kk]                         "m"     (kk),
        [ww]                         "m"     (ww),
        [W_size]                         "m"     (W_size),
        [stride_filter]             "m" (stride_filter)
    
    
    :
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi","r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory","k0","k1","k2","k3","k4"
    
    );
    
    

}

void direct_3x3_N32M8_AVX512_pack(float *img_start_0, float *img_pack, float *kernal, float *out, int ic_count, int ic, int stride_in_size, int hh_tag, int h, int kk, int ww, int W_size){
    long stride_filter = 0;
    if(hh_tag == 0){
        
        stride_filter = ic_count * 3 * 32 * sizeof(float);
        
    }
    asm volatile(
        
        ".macro PACK_KERNEL32x8_K1     \n"
        "   vbroadcastss (%%r10), %%zmm1    \n "
        "   vbroadcastss 4(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   movl 32(%%r10), %%r8d     \n"
        
        "   vbroadcastss 12(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        "   vmovups %%ymm28, (%%rdx)     \n"
        
        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        "   movl %%r8d, 32(%%rdx)     \n"
        
        "   vbroadcastss 20(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"

        
        "   vbroadcastss 24(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 28(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        "   addq $36, %%rdx      \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"



        ".endm                              \n"
        
        ".macro PACK_KERNEL32x8_K2     \n"
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vbroadcastss 8(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        
        "   vbroadcastss 12(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"

        
        "   vbroadcastss 16(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"

        
        "   vbroadcastss 24(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"

        
        "   vbroadcastss 28(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"

        
        "   vbroadcastss 32(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"

        ".endm                              \n"
        
        ".macro PACK_KERNEL32x8_K3     \n"
        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        "   vbroadcastss 12(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        
        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"

        
        "   vbroadcastss 20(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"

        
        "   vbroadcastss 28(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"

        
        "   vbroadcastss 32(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"


        ".endm                              \n"
        
        
        //--------------------- main
        
        ".macro KERNEL32x8_K1  \n"
        
        "   vbroadcastss (%%rdx), %%zmm1    \n "
        "   vbroadcastss 4(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 8(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 12(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 16(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 20(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 24(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 28(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"

        
        ".endm      \n"
        
        ".macro KERNEL32x8_K2  \n"
        
        "   vbroadcastss 4(%%rdx), %%zmm1    \n "
        "   vbroadcastss 8(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 12(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 16(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 20(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 24(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 28(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 32(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"

        
        ".endm      \n"
        
        ".macro KERNEL32x8_K3  \n"
        
        "   vbroadcastss 8(%%rdx), %%zmm1    \n "
        "   vbroadcastss 12(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 16(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 20(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 24(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 28(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 32(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        
        ".endm      \n"
        
        //----------------
        
        ".macro AVX512_N32M8_ST_END_PACK   \n"
        
        "   shl $2, %%r11     \n"
        "   shl $3, %%r14       \n"
        "   add %%r11, %%r8    \n"      // output 5
        "   add %%r14, %%r9    \n"      // output 9
        "   add %%r11, %%r14    \n"
        
        "   shl $1, %%rbx       \n"     //inc 2
        "   add %%r14, %%r10    \n"     //output 13
        
        "   sub %%r15, %%r11    \n"     //inc 3
        "   add %%r15, %%r14    \n"
        //  1 5 9 13
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq  $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"     //  input 1
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 9
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 5
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 13
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        // 2 6 10 14
        
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"     //  input 2
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 10
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 6
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 14
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //3 7 11 15
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"     //  input 3
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 11
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 7
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 15
        
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //4 8 12 16
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"     //  input 4
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 12
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 8
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 16
        
        "   add %%r14, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        //17 21 25 29
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"     //  input 17
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 21
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 25
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 29

        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //18 22 26 30
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"     //  input 18
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 26
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 22
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 30
        
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //19 23 27 31
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"     //  input 19
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 23
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 27
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 31

        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //20 24 28 32
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"     //  input 20
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, (%%r9)            \n"     //input 24
        
        "   vmovups %%ymm29, (%%r8)         \n"     //  input 28
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, (%%r10)            \n"     //input 32
        
        ".endm      \n"
        
        //-----------------add
        
        ".macro AVX512_N32M8_ADD_END_PACK   \n"
        "   shl $2, %%r11     \n"
        "   shl $3, %%r14       \n"
        "   add %%r11, %%r8    \n"      // output 5
        "   add %%r14, %%r9    \n"      // output 9
        "   add %%r11, %%r14    \n"
        
        "   shl $1, %%rbx       \n"     //inc 2
        "   add %%r14, %%r10    \n"     //output 13
        
        "   sub %%r15, %%r11    \n"     //inc 3
        //"   mov %[stride_out_size], %%r14   \n"
        "   add %%r15, %%r14    \n"
        //  1 5 9 13
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq  $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        
        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        // 2 6 10 14
        
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30      \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //3 7 11 15
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        
        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"

        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //4 8 12 16
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        
        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        
        
        "   add %%r14, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        //17 21 25 29
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //18 22 26 30
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29      \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //19 23 27 31
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //20 24 28 32
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)         \n"
        "   vmovups %%ymm29, (%%r8)         \n"
        "   vmovups %%ymm30, (%%r9)         \n"
        "   vmovups %%ymm31, (%%r10)         \n"

        
        ".endm      \n"
        
        "AVX512_N32M8_ENTRANCE_PACK:            \n"
        "   mov %[img_start_0], %%r10    \n"
        "   mov %[img_start_0], %%rdi    \n"
        "   mov %[img_pack],    %%rdx      \n"
        "   mov %[kernal], %%r11 \n"
        "   movl %[h], %%r12d    \n"

        "   movl %[ic_count], %%r13d  \n"           //
        "   mov %[stride_in_size], %%r14    \n"
        "   mov %[W_size], %%r15    \n"
        "   movl %[kk], %%ebx    \n"
        "   movl %[hh_tag], %%eax   \n"
        "   movl %[ww], %%ecx        \n"
        "   mov %[stride_filter], %%rsi    \n"

        "   vpxorq  %%zmm2, %%zmm2, %%zmm2  \n"
        "   vpxorq  %%zmm3, %%zmm3, %%zmm3  \n"
        "   vpxorq  %%zmm4, %%zmm4, %%zmm4  \n"
        "   vpxorq  %%zmm5, %%zmm5, %%zmm5  \n"
        "   vpxorq  %%zmm6, %%zmm6, %%zmm6  \n"
        "   vpxorq  %%zmm7, %%zmm7, %%zmm7  \n"
        "   vpxorq  %%zmm8, %%zmm8, %%zmm8  \n"
        "   vpxorq  %%zmm9, %%zmm9, %%zmm9  \n"

        "   vpxorq  %%zmm14, %%zmm14, %%zmm14  \n"
        "   vpxorq  %%zmm15, %%zmm15, %%zmm15  \n"
        "   vpxorq  %%zmm16, %%zmm16, %%zmm16  \n"
        "   vpxorq  %%zmm17, %%zmm17, %%zmm17  \n"
        "   vpxorq  %%zmm18, %%zmm18, %%zmm18  \n"
        "   vpxorq  %%zmm19, %%zmm19, %%zmm19  \n"
        "   vpxorq  %%zmm20, %%zmm20, %%zmm20  \n"
        "   vpxorq  %%zmm21, %%zmm21, %%zmm21  \n"
        
        "K_Branch_8:                  \n"
        "   mov $2, %%r9d       \n"
        "   cmp $0, %%ebx       \n"
        "   jne BEGIN_Kb_Pre_8                 \n"
        
        //kk=0
        "   cmp $0, %%eax          \n"
        "   je PACK_Htag0_8                     \n"
        "   cmp $1, %%eax           \n"
        "   je PACK_Htag1_8                     \n"
        
        //2
        "PACK_Htag2_8:                \n"
        "   cmp $0, %%r12d           \n"
        "   jne BEGIN_Kb_Fetch2_8                   \n"
        "   jmp PACK_Main_C_8         \n"
        
        
        //1
        "PACK_Htag1_8:                \n"
        "   cmp $0, %%r12d          \n"
        "   jne BEGIN_Kb_Fetch2_8      \n"            
        "   mov $3, %%r9d       \n"             //kk=0 hh_tag=1 h=0
        "   jmp PACK_Main_C_8         \n"
        
        
        //0
        "PACK_Htag0_8:                \n"
        "   addq %%rsi, %%r11        \n"
        "   jmp PACK_Main_C_8         \n"

        
        
        "PACK_Main_C_Pre_8:                       \n"
        "   addq $128,%%r11       \n"
        "   addq %%r14, %%r10   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%r10)                 \n"
        
        "PACK_Main_C_8:                       \n"

        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x8_K1        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x8_K2        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x8_K3        \n"
        
        "   subl $1, %%r13d       \n"
        "   jne     PACK_Main_C_Pre_8       \n"
        "   subl $1, %%r9d          \n"
        "   je      AVX512_N32M8_MAIN_END_PACK     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     PACK_Main_C_8            \n"

        
        "BEGIN_Kb_Mid_Fetch2_8:                  \n"
        "   addq $128, %%r11       \n"
        "   addq $36, %%rdx   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb_Fetch2_8:                  \n"
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x8_K1    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"

        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x8_K2    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x8_K3    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid_Fetch2_8                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid_Fetch2_8                      \n"
        "   cmp $2, %%eax           \n"
        "   je AVX512_N32M8_MAIN_END_PACK          \n"
        "   addq %%r15, %%rdi        \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128, %%r11       \n"
        "   addq $36, %%rdx   \n"
        "   movl $1, %%r9d          \n"
        "   mov %%rdi, %%r10       \n"

        "   jmp PACK_Main_C_8             \n"

        


        "BEGIN_Kb_Pre_8:                  \n"
        "   addq %%rsi, %%r11       \n"
        "   cmp $1, %%eax               \n"
        "   jne  BEGIN_Kb_8                \n"
        "   mov $3, %%r9d               \n"
        "   jmp  BEGIN_Kb_8                   \n"
        
        "BEGIN_Kb_Mid_8:                  \n"
        "   addq $128, %%r11       \n"
        "   addq $36, %%rdx   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb_8:                  \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x8_K1    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"

        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x8_K2    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x8_K3    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid_8                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid_8                      \n"
        
        
        //  ADD or ST
        "AVX512_N32M8_MAIN_END_PACK:       \n"
        "   movl %[ic], %%eax   \n"
        "   mov %[out], %%r12    \n"
        "   mov %[out], %%r8    \n"
        "   mov %[out], %%r9    \n"
        "   mov %[out], %%r10    \n"
        "   mov %[stride_in_size], %%rbx   \n"
        "   mov %[stride_in_size], %%r15   \n"
        "   mov %[stride_in_size], %%r14   \n"
        "   mov %[stride_in_size], %%r11   \n"
        "   cmp $0, %%eax    \n"
        "   je AVX512_N32M8_ST_PACK    \n"
        
        // ADD
        "   AVX512_N32M8_ADD_END_PACK   \n"
        "   jmp AVX512_N32M8_END_PACK          \n"
        
        //  ST
        "AVX512_N32M8_ST_PACK:     \n"
        "   AVX512_N32M8_ST_END_PACK     \n"
        
        "AVX512_N32M8_END_PACK:     \n"
        
    :
    
    
    :
        [img_start_0]                 "m"     (img_start_0),
        [img_pack]                    "m"     (img_pack),
        [kernal]                      "m"     (kernal),
        [out]                       "m"     (out),
        [ic_count]                  "m"     (ic_count),
        [ic]                        "m"     (ic),
        [stride_in_size]            "m"     (stride_in_size),
        [hh_tag]                         "m"     (hh_tag),
        [h]                         "m"     (h),
        [kk]                         "m"     (kk),
        [ww]                         "m"     (ww),
        [W_size]                         "m"     (W_size),
        [stride_filter]             "m" (stride_filter)
    
    
    :
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi","r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory","k0","k1","k2","k3","k4"
    
    );
    
    

}

void direct_3x3_N32M7_AVX512_pack(float *img_start_0, float *img_pack, float *kernal, float *out, int ic_count, int ic, int stride_in_size, int hh_tag, int h, int kk, int ww, int W_size){
    long stride_filter = 0;
    if(hh_tag == 0){
        
        stride_filter = ic_count * 3 * 32 * sizeof(float);
        
    }
    asm volatile(
        
        ".macro PACK_KERNEL32x7_K1     \n"
        "   vbroadcastss (%%r10), %%zmm27    \n "
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vmovups (%%r10), %%xmm28     \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   vmovups %%xmm28, (%%rdx)     \n"
        
        "   vbroadcastss 8(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        "   movq 16(%%r10), %%r8     \n"
        
        "   vbroadcastss 12(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        "   movq %%r8, 16(%%rdx)     \n"
        
        "   vbroadcastss 16(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        "   movl 24(%%r10), %%r8d     \n"
        
        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        "   movl %%r8d, 24(%%rdx)     \n"

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        "   addq $32, %%rdx      \n"
        

        ".endm                              \n"
        
        ".macro PACK_KERNEL32x7_K2     \n"
        "   vbroadcastss (%%r10), %%zmm1    \n "
        "   vbroadcastss 4(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 12(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 20(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        



        ".endm                              \n"
        
        ".macro PACK_KERNEL32x7_K3     \n"
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vbroadcastss 8(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 12(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 16(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"




        ".endm                              \n"
        
        
        //--------------------- main
        
        ".macro KERNEL32x7_K1  \n"
        
        "   vbroadcastss (%%rdx), %%zmm27    \n "
        "   vbroadcastss 4(%%rdx), %%zmm1    \n "


        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"

        
        "   vbroadcastss 8(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"

        
        "   vbroadcastss 12(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"

        
        "   vbroadcastss 16(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"

        
        "   vbroadcastss 20(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"


        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"


        
        ".endm      \n"
        
        ".macro KERNEL32x7_K2  \n"
        
        "   vbroadcastss (%%rdx), %%zmm1    \n "
        "   vbroadcastss 4(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 8(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 12(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 16(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 20(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 24(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        
        ".endm      \n"
        
        ".macro KERNEL32x7_K3  \n"
        "   vbroadcastss 4(%%rdx), %%zmm1    \n "
        "   vbroadcastss 8(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 12(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 16(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 20(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 24(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"

        
        ".endm      \n"
        
        //----------------
        
        ".macro AVX512_N32M7_ST_END_PACK   \n"
        
        "   shl $2, %%r11     \n"
        "   shl $3, %%r14       \n"
        "   add %%r11, %%r8    \n"      // output 5
        "   add %%r14, %%r9    \n"      // output 9
        "   add %%r11, %%r14    \n"
        
        "   shl $1, %%rbx       \n"     //inc 2
        "   add %%r14, %%r10    \n"     //output 13
        
        "   sub %%r15, %%r11    \n"     //inc 3
        "   add %%r15, %%r14    \n"
        //  1 5 9 13
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq  $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vmovups %%ymm0, (%%r12)%{%%k1%}         \n"     //  input 1
        "   vmovups %%ymm30, (%%r9)%{%%k1%}            \n"     //input 9
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"     //  input 5
        "   vmovups %%ymm31, (%%r10)%{%%k1%}            \n"     //input 13
        
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        // 2 6 10 14
        
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"

        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        
        
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vmovups %%ymm26, (%%r12)%{%%k1%}         \n"     //  input 1
        "   vmovups %%ymm30, (%%r9)%{%%k1%}            \n"     //input 9
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"     //  input 5
        "   vmovups %%ymm31, (%%r10)%{%%k1%}            \n"     //input 13
        
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //3 7 11 15
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vmovups %%ymm0, (%%r12)%{%%k1%}         \n"     //  input 1
        "   vmovups %%ymm30, (%%r9)%{%%k1%}            \n"     //input 9
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"     //  input 5
        "   vmovups %%ymm31, (%%r10)%{%%k1%}            \n"     //input 13
        
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //4 8 12 16
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vmovups %%ymm26, (%%r12)%{%%k1%}         \n"     //  input 1
        "   vmovups %%ymm30, (%%r9)%{%%k1%}            \n"     //input 9
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"     //  input 5
        "   vmovups %%ymm31, (%%r10)%{%%k1%}            \n"     //input 13
        
        
        "   add %%r14, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        //17 21 25 29
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"


        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vmovups %%ymm0, (%%r12)%{%%k1%}         \n"     //  input 1
        "   vmovups %%ymm30, (%%r9)%{%%k1%}            \n"     //input 9
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"     //  input 5
        "   vmovups %%ymm31, (%%r10)%{%%k1%}            \n"     //input 13

        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //18 22 26 30
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vmovups %%ymm26, (%%r12)%{%%k1%}         \n"     //  input 1
        "   vmovups %%ymm30, (%%r9)%{%%k1%}            \n"     //input 9
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"     //  input 5
        "   vmovups %%ymm31, (%%r10)%{%%k1%}            \n"     //input 13
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //19 23 27 31
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"

        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"


        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vmovups %%ymm0, (%%r12)%{%%k1%}         \n"     //  input 1
        "   vmovups %%ymm30, (%%r9)%{%%k1%}            \n"     //input 9
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"     //  input 5
        "   vmovups %%ymm31, (%%r10)%{%%k1%}            \n"     //input 13

        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //20 24 28 32
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vmovups %%ymm26, (%%r12)%{%%k1%}         \n"     //  input 1
        "   vmovups %%ymm30, (%%r9)%{%%k1%}            \n"     //input 9
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"     //  input 5
        "   vmovups %%ymm31, (%%r10)%{%%k1%}            \n"     //input 13
        
        ".endm      \n"
        
        //-----------------add
        
        ".macro AVX512_N32M7_ADD_END_PACK   \n"
        "   shl $2, %%r11     \n"
        "   shl $3, %%r14       \n"
        "   add %%r11, %%r8    \n"      // output 5
        "   add %%r14, %%r9    \n"      // output 9
        "   add %%r11, %%r14    \n"
        
        "   shl $1, %%rbx       \n"     //inc 2
        "   add %%r14, %%r10    \n"     //output 13
        
        "   sub %%r15, %%r11    \n"     //inc 3
        //"   mov %[stride_out_size], %%r14   \n"
        "   add %%r15, %%r14    \n"
        //  1 5 9 13
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq  $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        
        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)%{%%k1%}         \n"
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"
        "   vmovups %%ymm30, (%%r9)%{%%k1%}         \n"
        "   vmovups %%ymm31, (%%r10)%{%%k1%}         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        // 2 6 10 14
        
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        
        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30      \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)%{%%k1%}         \n"
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"
        "   vmovups %%ymm30, (%%r9)%{%%k1%}         \n"
        "   vmovups %%ymm31, (%%r10)%{%%k1%}         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //3 7 11 15
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"

        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)%{%%k1%}         \n"
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"
        "   vmovups %%ymm30, (%%r9)%{%%k1%}         \n"
        "   vmovups %%ymm31, (%%r10)%{%%k1%}         \n"

        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //4 8 12 16
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"
        
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)%{%%k1%}         \n"
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"
        "   vmovups %%ymm30, (%%r9)%{%%k1%}         \n"
        "   vmovups %%ymm31, (%%r10)%{%%k1%}         \n"
        
        
        "   add %%r14, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        //17 21 25 29
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)%{%%k1%}         \n"
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"
        "   vmovups %%ymm30, (%%r9)%{%%k1%}         \n"
        "   vmovups %%ymm31, (%%r10)%{%%k1%}         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //18 22 26 30
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29      \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)%{%%k1%}         \n"
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"
        "   vmovups %%ymm30, (%%r9)%{%%k1%}         \n"
        "   vmovups %%ymm31, (%%r10)%{%%k1%}         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //19 23 27 31
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vmovups (%%r12), %%ymm1     \n"
        "   vmovups (%%r8), %%ymm26     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm1, %%ymm0, %%ymm0       \n"
        "   vaddps %%ymm26, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm0, (%%r12)%{%%k1%}         \n"
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"
        "   vmovups %%ymm30, (%%r9)%{%%k1%}         \n"
        "   vmovups %%ymm31, (%%r10)%{%%k1%}         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //20 24 28 32
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm19, %%zmm18, %%zmm1    \n"
        
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpckhps %%zmm21, %%zmm20, %%zmm27    \n"
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"
        
        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        

        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   movl $0x7f, %%r13d              \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vmovups (%%r12), %%ymm0     \n"
        "   vmovups (%%r8), %%ymm1     \n"
        "   vmovups (%%r9), %%ymm27     \n"
        "   vmovups (%%r10), %%ymm28     \n"
        
        "   vaddps %%ymm0, %%ymm26, %%ymm26       \n"
        "   vaddps %%ymm1, %%ymm29, %%ymm29       \n"
        "   vaddps %%ymm27, %%ymm30, %%ymm30       \n"
        "   vaddps %%ymm28, %%ymm31, %%ymm31       \n"
        
        "   vmovups %%ymm26, (%%r12)%{%%k1%}         \n"
        "   vmovups %%ymm29, (%%r8)%{%%k1%}         \n"
        "   vmovups %%ymm30, (%%r9)%{%%k1%}         \n"
        "   vmovups %%ymm31, (%%r10)%{%%k1%}         \n"

        
        ".endm      \n"
        
        "AVX512_N32M7_ENTRANCE_PACK:            \n"
        "   mov %[img_start_0], %%r10    \n"
        "   mov %[img_start_0], %%rdi    \n"
        "   mov %[img_pack],    %%rdx      \n"
        "   mov %[kernal], %%r11 \n"
        "   movl %[h], %%r12d    \n"

        "   movl %[ic_count], %%r13d  \n"           //
        "   mov %[stride_in_size], %%r14    \n"
        "   mov %[W_size], %%r15    \n"
        "   movl %[kk], %%ebx    \n"
        "   movl %[hh_tag], %%eax   \n"
        "   movl %[ww], %%ecx        \n"
        "   mov %[stride_filter], %%rsi    \n"

        "   vpxorq  %%zmm2, %%zmm2, %%zmm2  \n"
        "   vpxorq  %%zmm3, %%zmm3, %%zmm3  \n"
        "   vpxorq  %%zmm4, %%zmm4, %%zmm4  \n"
        "   vpxorq  %%zmm5, %%zmm5, %%zmm5  \n"
        "   vpxorq  %%zmm6, %%zmm6, %%zmm6  \n"
        "   vpxorq  %%zmm7, %%zmm7, %%zmm7  \n"
        "   vpxorq  %%zmm8, %%zmm8, %%zmm8  \n"
        "   vpxorq  %%zmm9, %%zmm9, %%zmm9  \n"

        "   vpxorq  %%zmm14, %%zmm14, %%zmm14  \n"
        "   vpxorq  %%zmm15, %%zmm15, %%zmm15  \n"
        "   vpxorq  %%zmm16, %%zmm16, %%zmm16  \n"
        "   vpxorq  %%zmm17, %%zmm17, %%zmm17  \n"
        "   vpxorq  %%zmm18, %%zmm18, %%zmm18  \n"
        "   vpxorq  %%zmm19, %%zmm19, %%zmm19  \n"
        "   vpxorq  %%zmm20, %%zmm20, %%zmm20  \n"
        "   vpxorq  %%zmm21, %%zmm21, %%zmm21  \n"
        
        "K_Branch_7:                  \n"
        "   mov $2, %%r9d       \n"
        "   cmp $0, %%ebx       \n"
        "   jne BEGIN_Kb_Pre_7                 \n"
        
        //kk=0
        "   cmp $0, %%eax          \n"
        "   je PACK_Htag0_7                     \n"
        "   cmp $1, %%eax           \n"
        "   je PACK_Htag1_7                     \n"
        
        //2
        "PACK_Htag2_7:                \n"
        "   cmp $0, %%r12d           \n"
        "   jne BEGIN_Kb_Fetch2_7                   \n"
        "   jmp PACK_Main_C_7         \n"
        
        
        //1
        "PACK_Htag1_7:                \n"
        "   cmp $0, %%r12d          \n"
        "   jne BEGIN_Kb_Fetch2_7      \n"            
        "   mov $3, %%r9d       \n"             //kk=0 hh_tag=1 h=0
        "   jmp PACK_Main_C_7         \n"
        
        
        //0
        "PACK_Htag0_7:                \n"
        "   addq %%rsi, %%r11        \n"
        "   jmp PACK_Main_C_7         \n"

        
        
        "PACK_Main_C_Pre_7:                       \n"
        "   addq $128,%%r11       \n"
        "   addq %%r14, %%r10   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%r10)                 \n"
        
        "PACK_Main_C_7:                       \n"

        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x7_K1        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x7_K2        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x7_K3        \n"
        
        "   subl $1, %%r13d       \n"
        "   jne     PACK_Main_C_Pre_7       \n"
        "   subl $1, %%r9d          \n"
        "   je      AVX512_N32M7_MAIN_END_PACK     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     PACK_Main_C_7            \n"

        
        "BEGIN_Kb_Mid_Fetch2_7:                  \n"
        "   addq $128, %%r11       \n"
        "   addq $32, %%rdx   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb_Fetch2_7:                  \n"
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x7_K1    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"

        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x7_K2    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x7_K3    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid_Fetch2_7                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid_Fetch2_7                      \n"
        "   cmp $2, %%eax           \n"
        "   je AVX512_N32M7_MAIN_END_PACK          \n"
        "   addq %%r15, %%rdi        \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128, %%r11       \n"
        "   addq $32, %%rdx   \n"
        "   movl $1, %%r9d          \n"
        "   mov %%rdi, %%r10       \n"

        "   jmp PACK_Main_C_7             \n"

        


        "BEGIN_Kb_Pre_7:                  \n"
        "   addq %%rsi, %%r11       \n"
        "   cmp $1, %%eax               \n"
        "   jne  BEGIN_Kb_7                \n"
        "   mov $3, %%r9d               \n"
        "   jmp  BEGIN_Kb_7                   \n"
        
        "BEGIN_Kb_Mid_7:                  \n"
        "   addq $128, %%r11       \n"
        "   addq $32, %%rdx   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb_7:                  \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x7_K1    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"

        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x7_K2    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x7_K3    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid_7                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid_7                      \n"
        
        
        //  ADD or ST
        "AVX512_N32M7_MAIN_END_PACK:       \n"
        "   movl %[ic], %%eax   \n"
        "   mov %[out], %%r12    \n"
        "   mov %[out], %%r8    \n"
        "   mov %[out], %%r9    \n"
        "   mov %[out], %%r10    \n"
        "   mov %[stride_in_size], %%rbx   \n"
        "   mov %[stride_in_size], %%r15   \n"
        "   mov %[stride_in_size], %%r14   \n"
        "   mov %[stride_in_size], %%r11   \n"
        "   cmp $0, %%eax    \n"
        "   je AVX512_N32M7_ST_PACK    \n"
        
        // ADD
        "   AVX512_N32M7_ADD_END_PACK   \n"
        "   jmp AVX512_N32M7_END_PACK          \n"
        
        //  ST
        "AVX512_N32M7_ST_PACK:     \n"
        "   AVX512_N32M7_ST_END_PACK     \n"
        
        "AVX512_N32M7_END_PACK:     \n"
        
    :
    
    
    :
        [img_start_0]                 "m"     (img_start_0),
        [img_pack]                    "m"     (img_pack),
        [kernal]                      "m"     (kernal),
        [out]                       "m"     (out),
        [ic_count]                  "m"     (ic_count),
        [ic]                        "m"     (ic),
        [stride_in_size]            "m"     (stride_in_size),
        [hh_tag]                         "m"     (hh_tag),
        [h]                         "m"     (h),
        [kk]                         "m"     (kk),
        [ww]                         "m"     (ww),
        [W_size]                         "m"     (W_size),
        [stride_filter]             "m" (stride_filter)
    
    
    :
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi","r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory","k0","k1","k2","k3","k4"
    
    );
    
    

}

void direct_3x3_N32M4_AVX512_pack(float *img_start_0, float *img_pack, float *kernal, float *out, int ic_count, int ic, int stride_in_size, int hh_tag, int h, int kk, int ww, int W_size){
    long stride_filter = 0;
    if(hh_tag == 0){
        
        stride_filter = ic_count * 3 * 32 * sizeof(float);
        
    }
    asm volatile(
        
        ".macro PACK_KERNEL32x4_K1     \n"
        "   vbroadcastss (%%r10), %%zmm1    \n "
        "   vbroadcastss 4(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        "   vmovups (%%r10), %%xmm28     \n"
        
        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   movl 16(%%r10), %%r8d     \n"
        
        "   vbroadcastss 12(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        "   vmovups %%xmm28, (%%rdx)     \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        "   movl %%r8d, 16(%%rdx)     \n"
        "   addq $20, %%rdx      \n"




        ".endm                              \n"
        
        ".macro PACK_KERNEL32x4_K2     \n"
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vbroadcastss 8(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        
        "   vbroadcastss 12(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"

        
        "   vbroadcastss 16(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"



        ".endm                              \n"
        
        ".macro PACK_KERNEL32x4_K3     \n"
        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        "   vbroadcastss 12(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        
        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"





        ".endm                              \n"
        
        
        //--------------------- main
        
        ".macro KERNEL32x4_K1  \n"
        
        "   vbroadcastss (%%rdx), %%zmm1    \n "
        "   vbroadcastss 4(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 8(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 12(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        


        
        ".endm      \n"
        
        ".macro KERNEL32x4_K2  \n"
        
        "   vbroadcastss 4(%%rdx), %%zmm1    \n "
        "   vbroadcastss 8(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 12(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 16(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"


        
        ".endm      \n"
        
        ".macro KERNEL32x4_K3  \n"
        
        "   vbroadcastss 8(%%rdx), %%zmm1    \n "
        "   vbroadcastss 12(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vbroadcastss 16(%%rdx), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        

        
        ".endm      \n"
        
        //----------------
        
        ".macro AVX512_N32M4_ST_END_PACK   \n"
        
        "   shl $2, %%r11     \n"
        "   shl $3, %%r14       \n"
        "   add %%r11, %%r8    \n"      // output 5
        "   add %%r14, %%r9    \n"      // output 9
        "   add %%r11, %%r14    \n"
        
        "   shl $1, %%rbx       \n"     //inc 2
        "   add %%r14, %%r10    \n"     //output 13
        
        "   sub %%r15, %%r11    \n"     //inc 3
        "   add %%r15, %%r14    \n"
        //  1 5 9 13
        
        "   movl    $0xaa, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, (%%r12)         \n"     //  input 1
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 13
        
        "   vmovups %%xmm28, (%%r8)         \n"     //  input 5
        "   vmovups %%xmm29, (%%r9)         \n"     //  input 9
        "   vmovups %%xmm30, (%%r10)         \n"     //  input 13
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        // 2 6 10 14
        
        "   movl    $0x55, %%r13d       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, (%%r12)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups %%xmm28, (%%r8)         \n"     //  input 6
        "   vmovups %%xmm29, (%%r9)         \n"     //  input 10
        "   vmovups %%xmm30, (%%r10)         \n"     //  input 14
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //3 7 11 15
        "   movl    $0xaa, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, (%%r12)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 14
        
        "   vmovups %%xmm28, (%%r8)         \n"     //  input 6
        "   vmovups %%xmm29, (%%r9)         \n"     //  input 10
        "   vmovups %%xmm30, (%%r10)         \n"     //  input 14
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //4 8 12 16
        "   movl    $0x55, %%r13d       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, (%%r12)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups %%xmm28, (%%r8)         \n"     //  input 6
        "   vmovups %%xmm29, (%%r9)         \n"     //  input 10
        "   vmovups %%xmm30, (%%r10)         \n"     //  input 14
        
        "   add %%r14, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        //17 21 25 29
        "   movl    $0xaa, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, (%%r12)         \n"     //  input 1
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 13
        
        "   vmovups %%xmm28, (%%r8)         \n"     //  input 5
        "   vmovups %%xmm29, (%%r9)         \n"     //  input 9
        "   vmovups %%xmm30, (%%r10)         \n"     //  input 13

        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //18 22 26 30
        "   movl    $0x55, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, (%%r12)         \n"     //  input 1
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 13
        
        "   vmovups %%xmm28, (%%r8)         \n"     //  input 5
        "   vmovups %%xmm29, (%%r9)         \n"     //  input 9
        "   vmovups %%xmm30, (%%r10)         \n"     //  input 13
        
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //19 23 27 31
        "   movl    $0xaa, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, (%%r12)         \n"     //  input 1
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 13
        
        "   vmovups %%xmm28, (%%r8)         \n"     //  input 5
        "   vmovups %%xmm29, (%%r9)         \n"     //  input 9
        "   vmovups %%xmm30, (%%r10)         \n"     //  input 13

        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //20 24 28 32
        "   movl    $0x55, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, (%%r12)         \n"     //  input 1
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 13
        
        "   vmovups %%xmm28, (%%r8)         \n"     //  input 5
        "   vmovups %%xmm29, (%%r9)         \n"     //  input 9
        "   vmovups %%xmm30, (%%r10)         \n"     //  input 13
        
        ".endm      \n"
        
        //-----------------add
        
        ".macro AVX512_N32M4_ADD_END_PACK   \n"
        "   shl $2, %%r11     \n"
        "   shl $3, %%r14       \n"
        "   add %%r11, %%r8    \n"      // output 5
        "   add %%r14, %%r9    \n"      // output 9
        "   add %%r11, %%r14    \n"
        
        "   shl $1, %%rbx       \n"     //inc 2
        "   add %%r14, %%r10    \n"     //output 13
        
        "   sub %%r15, %%r11    \n"     //inc 3
        //"   mov %[stride_out_size], %%r14   \n"
        "   add %%r15, %%r14    \n"
        //  1 5 9 13
        "   movl    $0xaa, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"

        
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"

        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 5
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 9
        
        "   vmovups (%%r12), %%xmm1     \n"
        "   vmovups (%%r8), %%xmm26     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, (%%r12)         \n"
        "   vmovups %%xmm28, (%%r8)         \n"
        "   vmovups %%xmm29, (%%r9)         \n"
        "   vmovups %%xmm30, (%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        // 2 6 10 14

        "   movl    $0x55, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups (%%r12), %%xmm0     \n"
        "   vmovups (%%r8), %%xmm1     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, (%%r12)         \n"
        "   vmovups %%xmm28, (%%r8)         \n"
        "   vmovups %%xmm29, (%%r9)         \n"
        "   vmovups %%xmm30, (%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //3 7 11 15
        "   movl    $0xaa, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"

        
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"

        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 5
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 9
        
        "   vmovups (%%r12), %%xmm1     \n"
        "   vmovups (%%r8), %%xmm26     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, (%%r12)         \n"
        "   vmovups %%xmm28, (%%r8)         \n"
        "   vmovups %%xmm29, (%%r9)         \n"
        "   vmovups %%xmm30, (%%r10)         \n"

        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //4 8 12 16
        "   movl    $0x55, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups (%%r12), %%xmm0     \n"
        "   vmovups (%%r8), %%xmm1     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, (%%r12)         \n"
        "   vmovups %%xmm28, (%%r8)         \n"
        "   vmovups %%xmm29, (%%r9)         \n"
        "   vmovups %%xmm30, (%%r10)         \n"
        
        
        "   add %%r14, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        //17 21 25 29
        "   movl    $0xaa, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"

        
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"

        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 5
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 9
        
        "   vmovups (%%r12), %%xmm1     \n"
        "   vmovups (%%r8), %%xmm26     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, (%%r12)         \n"
        "   vmovups %%xmm28, (%%r8)         \n"
        "   vmovups %%xmm29, (%%r9)         \n"
        "   vmovups %%xmm30, (%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //18 22 26 30
        "   movl    $0x55, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"

        
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"

        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 5
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 9
        
        "   vmovups (%%r12), %%xmm0     \n"
        "   vmovups (%%r8), %%xmm1     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28      \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, (%%r12)         \n"
        "   vmovups %%xmm28, (%%r8)         \n"
        "   vmovups %%xmm29, (%%r9)         \n"
        "   vmovups %%xmm30, (%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //19 23 27 31
        "   movl    $0xaa, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"

        
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"

        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 5
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 9
        
        "   vmovups (%%r12), %%xmm1     \n"
        "   vmovups (%%r8), %%xmm26     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, (%%r12)         \n"
        "   vmovups %%xmm28, (%%r8)         \n"
        "   vmovups %%xmm29, (%%r9)         \n"
        "   vmovups %%xmm30, (%%r10)         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //20 24 28 32
        "   movl    $0x55, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"

        
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"

        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 5
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 9
        
        "   vmovups (%%r12), %%xmm0     \n"
        "   vmovups (%%r8), %%xmm1     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28      \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, (%%r12)         \n"
        "   vmovups %%xmm28, (%%r8)         \n"
        "   vmovups %%xmm29, (%%r9)         \n"
        "   vmovups %%xmm30, (%%r10)         \n"

        
        ".endm      \n"
        
        "AVX512_N32M4_ENTRANCE_PACK:            \n"
        "   mov %[img_start_0], %%r10    \n"
        "   mov %[img_start_0], %%rdi    \n"
        "   mov %[img_pack],    %%rdx      \n"
        "   mov %[kernal], %%r11 \n"
        "   movl %[h], %%r12d    \n"

        "   movl %[ic_count], %%r13d  \n"           //
        "   mov %[stride_in_size], %%r14    \n"
        "   mov %[W_size], %%r15    \n"
        "   movl %[kk], %%ebx    \n"
        "   movl %[hh_tag], %%eax   \n"
        "   movl %[ww], %%ecx        \n"
        "   mov %[stride_filter], %%rsi    \n"

        "   vpxorq  %%zmm2, %%zmm2, %%zmm2  \n"
        "   vpxorq  %%zmm3, %%zmm3, %%zmm3  \n"
        "   vpxorq  %%zmm4, %%zmm4, %%zmm4  \n"
        "   vpxorq  %%zmm5, %%zmm5, %%zmm5  \n"


        "   vpxorq  %%zmm14, %%zmm14, %%zmm14  \n"
        "   vpxorq  %%zmm15, %%zmm15, %%zmm15  \n"
        "   vpxorq  %%zmm16, %%zmm16, %%zmm16  \n"
        "   vpxorq  %%zmm17, %%zmm17, %%zmm17  \n"

        
        "K_Branch_4:                  \n"
        "   mov $2, %%r9d       \n"
        "   cmp $0, %%ebx       \n"
        "   jne BEGIN_Kb_Pre_4                 \n"
        
        //kk=0
        "   cmp $0, %%eax          \n"
        "   je PACK_Htag0_4                     \n"
        "   cmp $1, %%eax           \n"
        "   je PACK_Htag1_4                     \n"
        
        //2
        "PACK_Htag2_4:                \n"
        "   cmp $0, %%r12d           \n"
        "   jne BEGIN_Kb_Fetch2_4                   \n"
        "   jmp PACK_Main_C_4         \n"
        
        
        //1
        "PACK_Htag1_4:                \n"
        "   cmp $0, %%r12d          \n"
        "   jne BEGIN_Kb_Fetch2_4      \n"            
        "   mov $3, %%r9d       \n"             //kk=0 hh_tag=1 h=0
        "   jmp PACK_Main_C_4         \n"
        
        
        //0
        "PACK_Htag0_4:                \n"
        "   addq %%rsi, %%r11        \n"
        "   jmp PACK_Main_C_4         \n"

        
        
        "PACK_Main_C_Pre_4:                       \n"
        "   addq $128,%%r11       \n"
        "   addq %%r14, %%r10   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%r10)                 \n"
        
        "PACK_Main_C_4:                       \n"

        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x4_K1        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x4_K2        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x4_K3        \n"
        
        "   subl $1, %%r13d       \n"
        "   jne     PACK_Main_C_Pre_4       \n"
        "   subl $1, %%r9d          \n"
        "   je      AVX512_N32M4_MAIN_END_PACK     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     PACK_Main_C_4            \n"

        
        "BEGIN_Kb_Mid_Fetch2_4:                  \n"
        "   addq $128, %%r11       \n"
        "   addq $20, %%rdx   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb_Fetch2_4:                  \n"
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x4_K1    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"

        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x4_K2    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x4_K3    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid_Fetch2_4                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid_Fetch2_4                      \n"
        "   cmp $2, %%eax           \n"
        "   je AVX512_N32M4_MAIN_END_PACK          \n"
        "   addq %%r15, %%rdi        \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128, %%r11       \n"
        "   addq $20, %%rdx   \n"
        "   movl $1, %%r9d          \n"
        "   mov %%rdi, %%r10       \n"

        "   jmp PACK_Main_C_4             \n"

        


        "BEGIN_Kb_Pre_4:                  \n"
        "   addq %%rsi, %%r11       \n"
        "   cmp $1, %%eax               \n"
        "   jne  BEGIN_Kb_4                \n"
        "   mov $3, %%r9d               \n"
        "   jmp  BEGIN_Kb_4                   \n"
        
        "BEGIN_Kb_Mid_4:                  \n"
        "   addq $128, %%r11       \n"
        "   addq $20, %%rdx   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb_4:                  \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x4_K1    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"

        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x4_K2    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x4_K3    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid_4                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid_4                      \n"
        
        
        //  ADD or ST
        "AVX512_N32M4_MAIN_END_PACK:       \n"
        "   movl %[ic], %%eax   \n"
        "   mov %[out], %%r12    \n"
        "   mov %[out], %%r8    \n"
        "   mov %[out], %%r9    \n"
        "   mov %[out], %%r10    \n"
        "   mov %[stride_in_size], %%rbx   \n"
        "   mov %[stride_in_size], %%r15   \n"
        "   mov %[stride_in_size], %%r14   \n"
        "   mov %[stride_in_size], %%r11   \n"
        "   cmp $0, %%eax    \n"
        "   je AVX512_N32M4_ST_PACK    \n"
        
        // ADD
        "   AVX512_N32M4_ADD_END_PACK   \n"
        "   jmp AVX512_N32M4_END_PACK          \n"
        
        //  ST
        "AVX512_N32M4_ST_PACK:     \n"
        "   AVX512_N32M4_ST_END_PACK     \n"
        
        "AVX512_N32M4_END_PACK:     \n"
        
    :
    
    
    :
        [img_start_0]                 "m"     (img_start_0),
        [img_pack]                    "m"     (img_pack),
        [kernal]                      "m"     (kernal),
        [out]                       "m"     (out),
        [ic_count]                  "m"     (ic_count),
        [ic]                        "m"     (ic),
        [stride_in_size]            "m"     (stride_in_size),
        [hh_tag]                         "m"     (hh_tag),
        [h]                         "m"     (h),
        [kk]                         "m"     (kk),
        [ww]                         "m"     (ww),
        [W_size]                         "m"     (W_size),
        [stride_filter]             "m" (stride_filter)
    
    
    :
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi","r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory","k0","k1","k2","k3","k4"
    
    );
    
    

}


void direct_3x3_N32M2_AVX512_pack(float *img_start_0, float *img_pack, float *kernal, float *out, int ic_count, int ic, int stride_in_size, int hh_tag, int h, int kk, int ww, int W_size){
    long stride_filter = 0;
    if(hh_tag == 0){
        
        stride_filter = ic_count * 3 * 32 * sizeof(float);
        
    }
    asm volatile(
        
        ".macro PACK_KERNEL32x2_K1     \n"
        "   vbroadcastss (%%r10), %%zmm1    \n "
        "   vbroadcastss 4(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   movq (%%r10), %%r8     \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        "   movq %%r8, (%%rdx)     \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   movl 8(%%r10), %%r8d     \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   movl %%r8d, 8(%%rdx)     \n"

        "   addq $12, %%rdx      \n"




        ".endm                              \n"
        
        ".macro PACK_KERNEL32x2_K2     \n"
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vbroadcastss 8(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"


        ".endm                              \n"
        
        ".macro PACK_KERNEL32x2_K3     \n"
        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        ".endm                              \n"
        
        
        //--------------------- main
        
        ".macro KERNEL32x2_K1  \n"
        
        "   vbroadcastss (%%rdx), %%zmm1    \n "
        "   vbroadcastss 4(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"

        


        
        ".endm      \n"
        
        ".macro KERNEL32x2_K2  \n"
        
        "   vbroadcastss 4(%%rdx), %%zmm1    \n "
        "   vbroadcastss 8(%%rdx), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"



        
        ".endm      \n"
        
        ".macro KERNEL32x2_K3  \n"
        
        "   vbroadcastss 8(%%rdx), %%zmm1    \n "

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"

        

        
        ".endm      \n"
        
        //----------------
        
        ".macro AVX512_N32M2_ST_END_PACK   \n"
        
        "   shl $2, %%r11     \n"
        "   shl $3, %%r14       \n"
        "   add %%r11, %%r8    \n"      // output 5
        "   add %%r14, %%r9    \n"      // output 9
        "   add %%r11, %%r14    \n"
        
        "   shl $1, %%rbx       \n"     //inc 2
        "   add %%r14, %%r10    \n"     //output 13
        
        "   sub %%r15, %%r11    \n"     //inc 3
        "   add %%r15, %%r14    \n"
        //  1-2
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0x3, %%eax     \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2             \n"
        
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 13
        
        
        "   vmovups %%xmm0, (%%r12)%{%%k2%}         \n"     //  input 1
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"     //  input 5
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"     //  input 9
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"     //  input 13
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        
        
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups %%xmm26, (%%r12)%{%%k2%}         \n"     //  input 2
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"     //  input 6
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"     //  input 10
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"     //  input 14
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        //3-4
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0x3, %%eax     \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2             \n"
        
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 7
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 11
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 15
        
        
        "   vmovups %%xmm0, (%%r12)%{%%k2%}         \n"     //  input 3
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"     //  input 7
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"     //  input 11
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"     //  input 15
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        
        

        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 8
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 12
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 16
        
        "   vmovups %%xmm26, (%%r12)%{%%k2%}         \n"     //  input 4
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"     //  input 8
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"     //  input 12
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"     //  input 16
        
        "   add %%r14, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        //17-18
        
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0x3, %%eax     \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2             \n"
        
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 21
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 25
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 29
        
        
        "   vmovups %%xmm0, (%%r12)%{%%k2%}         \n"     //  input 17
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"     //  input 21
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"     //  input 25
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"     //  input 29
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        
        
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    
        
        "   vmovups %%xmm26, (%%r12)%{%%k2%}         \n"     //  input 18
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"     //  input 22
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"     //  input 26
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"     //  input 30
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        
        
        //19-20
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0x3, %%eax     \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2             \n"
        
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    
        
        
        "   vmovups %%xmm0, (%%r12)%{%%k2%}         \n"     //  input 19
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"     //  input 23
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"     //  input 27
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"     //  input 31
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"  
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    
        
        "   vmovups %%xmm26, (%%r12)%{%%k2%}         \n"     //  input 20
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"     //  input 24
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"     //  input 28
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"     //  input 32
        
        ".endm      \n"
        
        //-----------------add
        
        ".macro AVX512_N32M2_ADD_END_PACK   \n"
        "   shl $2, %%r11     \n"
        "   shl $3, %%r14       \n"
        "   add %%r11, %%r8    \n"      // output 5
        "   add %%r14, %%r9    \n"      // output 9
        "   add %%r11, %%r14    \n"
        
        "   shl $1, %%rbx       \n"     //inc 2
        "   add %%r14, %%r10    \n"     //output 13
        
        "   sub %%r15, %%r11    \n"     //inc 3
        //"   mov %[stride_out_size], %%r14   \n"
        "   add %%r15, %%r14    \n"
        //  1-2
        "   movl    $0x55, %%r13d       \n"
        "   movl    $0x3, %%eax     \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2             \n"
        
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 13
        
        "   vmovups (%%r12), %%xmm1     \n"
        "   vmovups (%%r8), %%xmm26     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, (%%r12)%{%%k2%}         \n"
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"
        
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups (%%r12), %%xmm0     \n"
        "   vmovups (%%r8), %%xmm1     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, (%%r12)%{%%k2%}         \n"
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        //3-4
        "   movl    $0x55, %%r13d       \n"
        //"   movl    $0x3, %%eax     \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        //"   kmovd   %%eax, %%k2             \n"
        
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 7
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 11
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 15
        
        "   vmovups (%%r12), %%xmm1     \n"
        "   vmovups (%%r8), %%xmm26     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, (%%r12)%{%%k2%}         \n"
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"


        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        
        

        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 8
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 12
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 16
        
        "   vmovups (%%r12), %%xmm0     \n"
        "   vmovups (%%r8), %%xmm1     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, (%%r12)%{%%k2%}         \n"
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"
        
        
        "   add %%r14, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        //17-18
        "   movl    $0x55, %%r13d       \n"
        //"   movl    $0x3, %%eax     \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        //"   kmovd   %%eax, %%k2             \n"
        
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 21
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 25
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 29
        
        "   vmovups (%%r12), %%xmm1     \n"
        "   vmovups (%%r8), %%xmm26     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, (%%r12)%{%%k2%}         \n"
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"
        
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        
        
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"   
        
        "   vmovups (%%r12), %%xmm0     \n"
        "   vmovups (%%r8), %%xmm1     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28      \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, (%%r12)%{%%k2%}         \n"
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"
        
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        
        
        //19-20
        "   movl    $0x55, %%r13d       \n"
        //"   movl    $0x3, %%eax     \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        //"   kmovd   %%eax, %%k2             \n"
        
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    
        
        "   vmovups (%%r12), %%xmm1     \n"
        "   vmovups (%%r8), %%xmm26     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm1, %%xmm0, %%xmm0       \n"
        "   vaddps %%xmm26, %%xmm28, %%xmm28       \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm0, (%%r12)%{%%k2%}         \n"
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"
        
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   add %%r15, %%r12    \n"
        "   add %%r15, %%r8    \n"
        "   add %%r15, %%r9    \n"
        "   add %%r15, %%r10    \n"
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"

        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"  
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    
        
        "   vmovups (%%r12), %%xmm0     \n"
        "   vmovups (%%r8), %%xmm1     \n"
        "   vmovups (%%r9), %%xmm27     \n"
        "   vmovups (%%r10), %%xmm31     \n"
        
        "   vaddps %%xmm0, %%xmm26, %%xmm26       \n"
        "   vaddps %%xmm1, %%xmm28, %%xmm28      \n"
        "   vaddps %%xmm27, %%xmm29, %%xmm29       \n"
        "   vaddps %%xmm31, %%xmm30, %%xmm30       \n"
        
        "   vmovups %%xmm26, (%%r12)%{%%k2%}         \n"
        "   vmovups %%xmm28, (%%r8)%{%%k2%}         \n"
        "   vmovups %%xmm29, (%%r9)%{%%k2%}         \n"
        "   vmovups %%xmm30, (%%r10)%{%%k2%}         \n"

        
        ".endm      \n"
        
        "AVX512_N32M2_ENTRANCE_PACK:            \n"
        "   mov %[img_start_0], %%r10    \n"
        "   mov %[img_start_0], %%rdi    \n"
        "   mov %[img_pack],    %%rdx      \n"
        "   mov %[kernal], %%r11 \n"
        "   movl %[h], %%r12d    \n"

        "   movl %[ic_count], %%r13d  \n"           //
        "   mov %[stride_in_size], %%r14    \n"
        "   mov %[W_size], %%r15    \n"
        "   movl %[kk], %%ebx    \n"
        "   movl %[hh_tag], %%eax   \n"
        "   movl %[ww], %%ecx        \n"
        "   mov %[stride_filter], %%rsi    \n"

        "   vpxorq  %%zmm2, %%zmm2, %%zmm2  \n"
        "   vpxorq  %%zmm3, %%zmm3, %%zmm3  \n"
        "   vpxorq  %%zmm4, %%zmm4, %%zmm4  \n"
        "   vpxorq  %%zmm5, %%zmm5, %%zmm5  \n"


        "   vpxorq  %%zmm14, %%zmm14, %%zmm14  \n"
        "   vpxorq  %%zmm15, %%zmm15, %%zmm15  \n"
        "   vpxorq  %%zmm16, %%zmm16, %%zmm16  \n"
        "   vpxorq  %%zmm17, %%zmm17, %%zmm17  \n"

        
        "K_Branch_2:                  \n"
        "   mov $2, %%r9d       \n"
        "   cmp $0, %%ebx       \n"
        "   jne BEGIN_Kb_Pre_2                 \n"
        
        //kk=0
        "   cmp $0, %%eax          \n"
        "   je PACK_Htag0_2                     \n"
        "   cmp $1, %%eax           \n"
        "   je PACK_Htag1_2                     \n"
        
        //2
        "PACK_Htag2_2:                \n"
        "   cmp $0, %%r12d           \n"
        "   jne BEGIN_Kb_Fetch2_2                   \n"
        "   jmp PACK_Main_C_2         \n"
        
        
        //1
        "PACK_Htag1_2:                \n"
        "   cmp $0, %%r12d          \n"
        "   jne BEGIN_Kb_Fetch2_2      \n"            
        "   mov $3, %%r9d       \n"             //kk=0 hh_tag=1 h=0
        "   jmp PACK_Main_C_2         \n"
        
        
        //0
        "PACK_Htag0_2:                \n"
        "   addq %%rsi, %%r11        \n"
        "   jmp PACK_Main_C_2         \n"

        
        
        "PACK_Main_C_Pre_2:                       \n"
        "   addq $128,%%r11       \n"
        "   addq %%r14, %%r10   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%r10)                 \n"
        
        "PACK_Main_C_2:                       \n"

        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x2_K1        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x2_K2        \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   PACK_KERNEL32x2_K3        \n"
        
        "   subl $1, %%r13d       \n"
        "   jne     PACK_Main_C_Pre_2       \n"
        "   subl $1, %%r9d          \n"
        "   je      AVX512_N32M2_MAIN_END_PACK     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     PACK_Main_C_2            \n"

        
        "BEGIN_Kb_Mid_Fetch2_2:                  \n"
        "   addq $128, %%r11       \n"
        "   addq $12, %%rdx   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb_Fetch2_2:                  \n"
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x2_K1    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"

        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x2_K2    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x2_K3    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid_Fetch2_2                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid_Fetch2_2                      \n"
        "   cmp $2, %%eax           \n"
        "   je AVX512_N32M2_MAIN_END_PACK          \n"
        "   addq %%r15, %%rdi        \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128, %%r11       \n"
        "   addq $12, %%rdx   \n"
        "   movl $1, %%r9d          \n"
        "   mov %%rdi, %%r10       \n"

        "   jmp PACK_Main_C_2             \n"

        


        "BEGIN_Kb_Pre_2:                  \n"
        "   addq %%rsi, %%r11       \n"
        "   cmp $1, %%eax               \n"
        "   jne  BEGIN_Kb_2                \n"
        "   mov $3, %%r9d               \n"
        "   jmp  BEGIN_Kb_2                   \n"
        
        "BEGIN_Kb_Mid_2:                  \n"
        "   addq $128, %%r11       \n"
        "   addq $12, %%rdx   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb_2:                  \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x2_K1    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"

        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x2_K2    \n"
        "   addq $128,%%r11       \n"
        "   prefetcht0 (%%r11)                 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x2_K3    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid_2                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid_2                      \n"
        
        
        //  ADD or ST
        "AVX512_N32M2_MAIN_END_PACK:       \n"
        "   movl %[ic], %%eax   \n"
        "   mov %[out], %%r12    \n"
        "   mov %[out], %%r8    \n"
        "   mov %[out], %%r9    \n"
        "   mov %[out], %%r10    \n"
        "   mov %[stride_in_size], %%rbx   \n"
        "   mov %[stride_in_size], %%r15   \n"
        "   mov %[stride_in_size], %%r14   \n"
        "   mov %[stride_in_size], %%r11   \n"
        "   cmp $0, %%eax    \n"
        "   je AVX512_N32M2_ST_PACK    \n"
        
        // ADD
        "   AVX512_N32M2_ADD_END_PACK   \n"
        "   jmp AVX512_N32M2_END_PACK          \n"
        
        //  ST
        "AVX512_N32M2_ST_PACK:     \n"
        "   AVX512_N32M2_ST_END_PACK     \n"
        
        "AVX512_N32M2_END_PACK:     \n"
        
    :
    
    
    :
        [img_start_0]                 "m"     (img_start_0),
        [img_pack]                    "m"     (img_pack),
        [kernal]                      "m"     (kernal),
        [out]                       "m"     (out),
        [ic_count]                  "m"     (ic_count),
        [ic]                        "m"     (ic),
        [stride_in_size]            "m"     (stride_in_size),
        [hh_tag]                         "m"     (hh_tag),
        [h]                         "m"     (h),
        [kk]                         "m"     (kk),
        [ww]                         "m"     (ww),
        [W_size]                         "m"     (W_size),
        [stride_filter]             "m" (stride_filter)
    
    
    :
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi","r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory","k0","k1","k2","k3","k4"
    
    );
    
    

}




void avx512_Kernel_conv_nchw_3x3s1(float *filter, float *input, float *output, float *input_c, int K, int Kb, int H, int W, int Cb,int c_tag, bool C_end)
{

    int ww, hh, Hb, Wb, h, hh_tag, kk;
    int out_W = W;
    int out_H = H;
    int stride_in = H * W;
    int stride_out = out_H * out_W;
    int Edge;

    int Edge_stride;
    float *input0, *input1, *input2, *output0, *input_c0, *filter0, *output1;
    
    int stride_in_size = stride_in * sizeof(float);
    
    int stride_out_size = stride_out * sizeof(float);

    //Wb = 12;
    
    //Edge = 14;
    for(hh = 0; hh < out_H; hh = hh + Hb)
    {
        Hb = 4;                 
        if (out_H - hh < 4)
        {
            Hb = out_H - hh;
        }
        for(ww = 0; ww < out_W; ww = ww + Wb)
        {
            
            //printf("hh: %d,  ww: %d \n",hh, ww);
            Wb = 12;
            Edge = 14;
            //hh_tag = hh;
            
            if (W - ww < 12)
            {
                Wb = W - ww;
                Edge = Wb + 1;
            }
            
            input0 = input + hh * W  + ww;
            
            Edge_stride = Edge * Cb;
            
            if(ww != 0)
                input0 = input0 - 1;
            
            //input01 = input0 + W;
            //input02 = input01 + W;
            output0 = output + hh * out_W + ww;
            input_c0 = input_c;
            
            
            for(h=0; h<Hb; h++){
                //printf("hh: %d,  h: %d, input1 address1: %x\n",hh, h, input0);
                //printf("hh: %d,  h: %d, input2 address1: %x\n",hh, h, input01);
                //printf("hh: %d,  h: %d, input3 address1: %x\n",hh, h, input02);
                //printf("--------------\n");
                //printf("Wb: %d\n", Wb);
                input1 = input0 + h*out_W - out_W;
                
                hh_tag = 1;
                
                if((hh == 0)&&(h==0)){
                    hh_tag = 0;
                    input1 = input0;
                }
                if((hh + h) == (out_W - 1))     // out_W = out_H 
                {
                    hh_tag = 2;
                }
                
                //printf("hh: %d,  ww: %d, hh_tag: %d\n",hh, ww, hh_tag);
                
                if(Wb == 12){
                    
                    
                    for(kk=0; kk<Kb; kk+=32){
                    
                        //printf("hh: %d, h: %d, ww: %d, hh_tag: %d, kk: %d\n", hh, h, ww, hh_tag, kk);
                        
                        //printf("input1: %lf, input_c00: %lf, input_c01: %lf\n", input1[0], input_c0[0],input_c0[1]);
                        
                        filter0 = filter + kk*9*Cb;
                        
                        output1 = output0 + kk * stride_out + h*out_W;

                        direct_3x3_N32M12_AVX512_pack(input1, input_c0, filter0, output1, Cb, c_tag, stride_in_size, hh_tag, h, kk, ww, out_W<<2);
                    }
                }
                
                else if(Wb == 8)
                {
                    
                    
                    for(kk=0; kk<Kb; kk+=32){
                    

                        
                        filter0 = filter + kk*9*Cb;
                        
                        output1 = output0 + kk * stride_out + h*out_W;

                        direct_3x3_N32M8_AVX512_pack(input1, input_c0, filter0, output1, Cb, c_tag, stride_in_size, hh_tag, h, kk, ww, out_W<<2);
                    }
                    
                }
                
                else if(Wb == 7){
                    
                    
                    for(kk=0; kk<Kb; kk+=32){
                    

                        
                        filter0 = filter + kk*9*Cb;
                        
                        output1 = output0 + kk * stride_out + h*out_W;

                        direct_3x3_N32M7_AVX512_pack(input1, input_c0, filter0, output1, Cb, c_tag, stride_in_size, hh_tag, h, kk, ww, out_W<<2);
                    }
                }
                
                else if(Wb == 4){
                    
                    
                    for(kk=0; kk<Kb; kk+=32){
                    

                        
                        filter0 = filter + kk*9*Cb;
                        
                        output1 = output0 + kk * stride_out + h*out_W;

                        direct_3x3_N32M4_AVX512_pack(input1, input_c0, filter0, output1, Cb, c_tag, stride_in_size, hh_tag, h, kk, ww, out_W<<2);
                    }
                }
                
                else if(Wb == 2){
                    for(kk=0; kk<Kb; kk+=32){
                    

                        
                        filter0 = filter + kk*9*Cb;
                        
                        output1 = output0 + kk * stride_out + h*out_W;

                        direct_3x3_N32M2_AVX512_pack(input1, input_c0, filter0, output1, Cb, c_tag, stride_in_size, hh_tag, h, kk, ww, out_W<<2);
                    }
                }
                if(hh_tag == 0 ){
                    input_c0 -= Edge_stride;
                }
                input_c0 += Edge_stride;
            }
        }
    }
}



void avx512_dircet_cnn_3x3s1(float* filter, float* input, float* output, int K, int C, int H, int R)
{
    int W = H;
    int S = R;

    int out_W = W;
    int out_H = H;
    
    int stride_in = H * W;
    int stride_out = out_W * out_H;
    //float *ptr = (float *)_mm_malloc();
    //posix_memalign(&ptr, 64, NUM * CONV_C * 14 * 14 * sizeof( float ));
    //float *input_c = (float *)ptr;
    float *input_c = ( float * ) malloc( NUM * CONV_C * 14 * 14 * sizeof( float ));


    #pragma omp parallel num_threads(NUM)
    {

        int i, ii, jj, kk, cc, hh, h, ww;
        int Cb, Kb, Wb, Hb;
        bool C_end = 0;
        //float *temp_MVWB = MVWB;


        int id = omp_get_thread_num();

        float *input_c0 = input_c + id * CONV_C * 14 * 14;

        for(cc = 0; cc < C; cc = cc + Cb)
        {
            Cb = CONV_C;
            C_end = 0;
            if (C - cc <= CONV_C)
            {
                Cb = C - cc;
                C_end = 1;
            }

            for(ii = 0; ii < K; ii = ii + Kb)
            {
                Kb = CONV_K;
                //temp_MVWB = MVWB + ii;
                if (K - ii < CONV_K)
                {
                    Kb = K - ii;
                }
                
                float *filter0 = filter + ii * 9 * Cb + 9 * cc * K;
                float *input0 = input + id * C * stride_in + cc * stride_in;
                float *output0 = output + id * K * stride_out + ii * stride_out;
                //printf("address1: %x\n",input0);
                avx512_Kernel_conv_nchw_3x3s1(filter0, input0, output0, input_c0, K, Kb, H, W, Cb, cc, C_end);
            }
        }
    
        
    }
    free(input_c);
}


int main(int argc, char *argv[]){
    int array_C[25] = {64,
                        128,256,
                        512,
                        64,128,256,512,512};
    int array_K[25] = {64,
                        128,256,
                        512,
                        64,128,256,512,512};
    int array_H[25] = {56,
                        28,14,
                        7,224,112,56,28,14};
    int array_W[25] = {56,
                        28,14,
                        7,224,112,56,28,14};
    int array_R[25] = {3,3,3,3,
                        3,3,3,3,3};
    int array_S[25] = {3,3,3,3,
                        3,3,3,3,3};
    int array_pad[25] = {1,1,1,1,
                        1,1,1,1,1};
    int array_str[25] = {1,1,1,1,
                        1,1,1,1,1};
    int z,k;
    FILE *fp;
    if ( (fp = fopen("ndirect3x3.txt", "a+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
    for(k=0;k<5;k++){
        for(z=0;z<9;z++){
            //openblas_set_num_threads(1);
            int j, loop = 20;
            int pc, lda, ldb, ldc;
            double start, cost;

            /*
            int H = 14, W = 14, N = NUM, C = 32;
            int K = 32, R = 3, S = 3;
            int padh = 1, padw = 1, stride = 1;
            */
            int i=1;

            int H = array_H[z];
            int W = array_W[z];
            int N = 48;
            int C = array_C[z];
            int K = array_K[z]; 
            int R = array_R[z];
            int S = array_S[z];
            int padh = array_pad[z]; 
            int padw = array_pad[z];
            int stride = array_str[z];
            
            int out_H = (H - R + padh + padw) / stride + 1;;
            int out_W = out_H;
            printf("%d %d %d %d %d %d %d %d %d",C, K, H, W, R, S, stride, padh, padw);
            fprintf(fp, "%d %d %d %d %d %d %d %d %d",C, K, H, W, R, S, stride, padh, padw);
            /*
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
            */
            double pes = 128;
            int flag = 0;
            double ops = (double)2.0 * N * C * out_H * out_W * R * S * K * 1.0e-9;
            
            int stride_out_size = out_H*out_W*sizeof(float);
            int stride_in_size = H*W*sizeof(float);
            
            //double result = R*S*C*1.0;
            
            // alignment
            float *filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 64);
            //float *filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 128);
            if(filter == NULL ){
                printf("filter aligned error!\n");
                return 0;
                
            }
            
            float *trans_filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 64);
            //float *trans_filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 128);
            //float *trans_filter = (float*)_mm_malloc(K*C*3*3*sizeof(float), 32);
            
            if(filter == NULL ){
                printf("trans_filter aligned error!\n");
                return 0;
                
            }
            
            float *input =  (float*)_mm_malloc((size_t)N*C*H*W*sizeof(float), 64);
            //float *input =  (float*)_mm_malloc((size_t)N*C*H*W*sizeof(float), 128);
            if(input == NULL ){
                printf("input aligned error!\n");
                return 0;
                
            }
            
            float *output = (float*)_mm_malloc((size_t)N*K*out_H*out_W*sizeof(float), 64);
            //float *output = (float*)_mm_malloc((size_t)N*K*out_H*out_W*sizeof(float), 128);
            if(output == NULL ){
                printf("output aligned error!\n");
                return 0;
                
            }
            
            float *output1 = (float*)malloc(N*K*out_H*out_W*sizeof(float));
            float *data_col = (float*)malloc(N*C*(H/stride*W/stride)*R*S*sizeof(float));
            //float *output1 = (float*)_mm_malloc(N*K*H*W*sizeof(float), 32);
            //printf("N = %d, K = %d, C = %d, H = %d, W = %d\n", N, K, C, H, W);
            for(int pc = 0; pc < 1; pc++)
            {
                random_matrix(K, C * R *S, filter);
                random_matrix(N, C * H * W, input);
                
                //printf("1\n");
                transform_filter(K, C, R, S, filter, trans_filter);

                
                for(i = 0; i < 5; i++)
                {
                    avx512_dircet_cnn_3x3s1(trans_filter, input, output, K, C, H, R);
                }
                
                //printf("3\n");
                start = dclock();
                for(i = 0; i < loop; i++)
                {
                    avx512_dircet_cnn_3x3s1(trans_filter, input, output, K, C, H, R);
                }
                cost = (dclock() - start) / loop;
                fprintf(fp, " %.5f", cost);
                fprintf(fp, " %.5f\n", ops/cost);

                //printf(" %.5f", im2col_cost);
                printf(" %.5f", cost);
                printf(" %.5f\n", ops/cost);
                
                //printf("Gflops = %.3f, effic = %.3f %\n", ops/cost, ops/cost/NUM/pes * 100);
            
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
            

            if(flag != 0)
                printf("结果错误\n");
            

            free(filter);
            free(input);
            free(output);
            free(output1);
            free(data_col);
        }
        printf("\n");
        fprintf(fp,"\n");
    }
    return 0;
    
}