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
#define CONV_K 64
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

void random_matrix_1( int m, int n, float *a)
{
  //double drand48();
  int i,j;

  // #pragma omp parallel for num_threads(num)
  for ( i=0; i< m; i++ )
    for( j =0; j < n; j++)
    {
        //a[i*n+j]= 2.0 * (float)drand48() - 1.0;
        a[i*n+j] = 1.0;
    }
      //


}


void transform_filter(const int outch, const int inch, const int k_h, 
                const int k_w, float* kernel, float* out_kernel)
{
    int mr = 12, cr = CONV_C;

    int i, j , k, ii,jj,kk, h,w;
    int st = 0;

    for(j = 0; j < inch; j = j + cr)
    {

        cr = CONV_C;
        if(inch - j < CONV_C)
            cr = inch - j;

        for(i = 0; i < outch; i = i + mr)
        {   
            mr = 12;
            if(outch - i < 12)
                mr = outch - i;
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


void direct_3x3_N12M32_AVX512_pack(float *img_start_0, float *img_pack, float *kernal, float *out, int ic_count, int ic, int stride_in_size, int hh_tag, int h, int kk, int ww, int W_size){
    long stride_filter = 0;
    if(hh_tag == 0){
        
        stride_filter = ic_count * 3 * 12 * sizeof(float);
        
    }
    asm volatile(
        ".macro PACK_KERNEL12x32_K1     \n"
        "   vbroadcastss (%%r11), %%zmm1    \n "
        "   vbroadcastss 4(%%r11), %%zmm27    \n "
        "   vbroadcastss 8(%%r11), %%zmm30    \n "
        "   vbroadcastss 12(%%r11), %%zmm31    \n "

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        "   vmovups %%zmm0, (%%rdx)             \n"
        "   movl 128(%%r10), %%r8d     \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   vbroadcastss 16(%%r11), %%zmm1    \n "
        "   vmovups %%zmm26, 64(%%rdx)             \n"
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"
        "   vbroadcastss 20(%%r11), %%zmm27    \n "
        "   prefetcht1 136(%%r10)                 \n"

        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"
        "   vbroadcastss 24(%%r11), %%zmm30    \n "
        "   movl %%r8d, 128(%%rdx)     \n"

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        "   vbroadcastss 28(%%r11), %%zmm31    \n "
        "   movl 132(%%r10), %%r8d     \n"
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        "   vbroadcastss 32(%%r11), %%zmm1    \n "
        "   prefetcht1 200(%%r10)                 \n"

        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm20 \n"
        "   vbroadcastss 36(%%r11), %%zmm27    \n "
        "   movl %%r8d, 132(%%rdx)     \n"
        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm21 \n"
        "   vbroadcastss 40(%%r11), %%zmm30    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        "   vbroadcastss 44(%%r11), %%zmm31    \n "
        "   addq $136, %%rdx                    \n"
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        "   movl $0xfffe, %%r8d                   \n"


        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm24 \n"
        "   kmovd %%r8d, %%k1           \n"

        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm25 \n"
        
        ".endm                          \n"
    
        ".macro PACK_KERNEL12x32_K2     \n"
        "   vbroadcastss (%%r11), %%zmm1    \n "
        "   vbroadcastss 4(%%r11), %%zmm27    \n "
        "   vbroadcastss 8(%%r11), %%zmm30    \n "
        "   vbroadcastss 12(%%r11), %%zmm31    \n "

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"


        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   vbroadcastss 16(%%r11), %%zmm1    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"
        "   vbroadcastss 20(%%r11), %%zmm27    \n "
        
        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"
        "   vbroadcastss 24(%%r11), %%zmm30    \n "

        
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        "   vbroadcastss 28(%%r11), %%zmm31    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        "   vbroadcastss 32(%%r11), %%zmm1    \n "
        
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm20 \n"
        "   vbroadcastss 36(%%r11), %%zmm27    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm21 \n"
        "   vbroadcastss 40(%%r11), %%zmm30    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        "   vbroadcastss 44(%%r11), %%zmm31    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm24 \n"

        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm25 \n"
        ".endm                          \n"

        
        //--------------------- main
        
        ".macro KERNEL12x32  \n"
        
        "   vbroadcastss (%%r11), %%zmm1    \n "
        "   vbroadcastss 4(%%r11), %%zmm27    \n "
        "   vbroadcastss 8(%%r11), %%zmm30    \n "
        "   vbroadcastss 12(%%r11), %%zmm31    \n "

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"


        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   vbroadcastss 16(%%r11), %%zmm1    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"
        "   vbroadcastss 20(%%r11), %%zmm27    \n "
        
        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"
        "   vbroadcastss 24(%%r11), %%zmm30    \n "

        
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        "   vbroadcastss 28(%%r11), %%zmm31    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        "   vbroadcastss 32(%%r11), %%zmm1    \n "
        
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm20 \n"
        "   vbroadcastss 36(%%r11), %%zmm27    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm21 \n"
        "   vbroadcastss 40(%%r11), %%zmm30    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        "   vbroadcastss 44(%%r11), %%zmm31    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm24 \n"

        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm25 \n"
        
        ".endm      \n"
        
        //----------------
        
        ".macro AVX512_N12M32_ST_END_PACK   \n"
        
        "   vmovups         %%zmm2, (%%r12)                 \n"
        "   vmovups         %%zmm14, 64(%%r12)               \n"
        "   vmovups         %%zmm3, (%%r8)                \n"
        "   vmovups         %%zmm15, 64(%%r8)              \n"
        "   addq %%r14, %%r12                   \n"
        "   vmovups         %%zmm4, (%%r9)                \n"
        "   vmovups         %%zmm16, 64(%%r9)              \n"
        "   addq %%r14, %%r8                   \n"
        "   vmovups         %%zmm5, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"
        "   addq %%r14, %%r9                   \n"


        "   vmovups         %%zmm6, (%%r12)                \n"
        "   vmovups         %%zmm18, 64(%%r12)              \n"
        "   addq %%r14, %%r10                   \n"
        "   vmovups         %%zmm7, (%%r8)                \n"
        "   vmovups         %%zmm19, 64(%%r8)              \n"
        "   addq %%r14, %%r12                   \n"
        "   vmovups         %%zmm8, (%%r9)                \n"
        "   vmovups         %%zmm20, 64(%%r9)              \n"
        "   addq %%r14, %%r8                   \n"
        "   vmovups         %%zmm9, (%%r10)                \n"
        "   vmovups         %%zmm21, 64(%%r10)              \n" 
        "   addq %%r14, %%r9                   \n"

        "   vmovups         %%zmm10, (%%r12)                \n"
        "   vmovups         %%zmm22, 64(%%r12)              \n"
        "   addq %%r14, %%r10                   \n"
        "   vmovups         %%zmm11, (%%r8)                \n"
        "   vmovups         %%zmm23, 64(%%r8)              \n"
        "   vmovups         %%zmm12, (%%r9)                \n"
        "   vmovups         %%zmm24, 64(%%r9)              \n"
        "   vmovups         %%zmm13, (%%r10)                \n"
        "   vmovups         %%zmm25, 64(%%r10)              \n"
        
        ".endm      \n"
        
        //-----------------add
        
        ".macro AVX512_N12M32_ADD_END_PACK   \n"
        "   vmovups 		(%%r12), %%zmm0        			\n"
        "   vmovups 		64(%%r12), %%zmm1        			\n"
        "   vmovups 		(%%r8), %%zmm26        			\n"
        "   vmovups 		64(%%r8), %%zmm27        			\n"
        
        "	vaddps 			%%zmm0, %%zmm2, %%zmm2			\n"
        "   vmovups 		(%%r9), %%zmm28        			\n"
        "	vaddps 			%%zmm1, %%zmm14, %%zmm14			\n"
        "   vmovups 		64(%%r9), %%zmm29        			\n"
        "	vaddps 			%%zmm26, %%zmm3, %%zmm3			\n"
        "   vmovups 		(%%r10), %%zmm30        			\n"
        "	vaddps 			%%zmm27, %%zmm15, %%zmm15			\n"
        "   vmovups 		64(%%r10), %%zmm31        			\n"
        
        "	vaddps 			%%zmm28, %%zmm4, %%zmm4			\n"
        "   vmovups         %%zmm2, (%%r12)                 \n"
        "	vaddps 			%%zmm29, %%zmm16, %%zmm16			\n"
        "   vmovups         %%zmm14, 64(%%r12)                 \n"
        "	vaddps 			%%zmm30, %%zmm5, %%zmm5			\n"
        "   vmovups         %%zmm3, (%%r8)                 \n"
        "	vaddps 			%%zmm31, %%zmm17, %%zmm17			\n"
        "   vmovups         %%zmm15, 64(%%r8)                 \n"
        
        "   addq %%r14, %%r12                   \n"
        "   prefetcht1 (%%r12)                 \n"
        "   addq %%r14, %%r8                   \n"
        "   prefetcht1 (%%r8)                 \n"
        "   vmovups         %%zmm4, (%%r9)                \n"
        "   vmovups         %%zmm16, 64(%%r9)              \n"


        "   vmovups         %%zmm5, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"
        "   addq %%r14, %%r9                   \n"
        //"   prefetcht1 (%%r9)                 \n"

        "   addq %%r14, %%r10                   \n"
        //"   prefetcht1 (%%r10)                 \n"

        "   vmovups 		(%%r12), %%zmm0        			\n"
        "   vmovups 		64(%%r12), %%zmm1        			\n"
        "   vmovups 		(%%r8), %%zmm26        			\n"
        "   vmovups 		64(%%r8), %%zmm27        			\n"
        
        "	vaddps 			%%zmm0, %%zmm6, %%zmm6			\n"
        "   vmovups 		(%%r9), %%zmm28        			\n"
        "	vaddps 			%%zmm1, %%zmm18, %%zmm18			\n"
        "   vmovups 		64(%%r9), %%zmm29        			\n"
        "	vaddps 			%%zmm26, %%zmm7, %%zmm7			\n"
        "   vmovups 		(%%r10), %%zmm30        			\n"
        "	vaddps 			%%zmm27, %%zmm19, %%zmm19			\n"
        "   vmovups 		64(%%r10), %%zmm31        			\n"
        
        "	vaddps 			%%zmm28, %%zmm8, %%zmm8			\n"
        "   vmovups         %%zmm6, (%%r12)                 \n"
        "	vaddps 			%%zmm29, %%zmm20, %%zmm20			\n"
        "   vmovups         %%zmm18, 64(%%r12)                 \n"
        "	vaddps 			%%zmm30, %%zmm9, %%zmm9			\n"
        "   vmovups         %%zmm7, (%%r8)                 \n"
        "	vaddps 			%%zmm31, %%zmm21, %%zmm21			\n"
        "   vmovups         %%zmm19, 64(%%r8)                 \n"
        
        "   addq %%r14, %%r12                   \n"
        "   vmovups         %%zmm8, (%%r9)                \n"
        "   vmovups         %%zmm20, 64(%%r9)              \n"
        "   addq %%r14, %%r8                   \n"
        "   vmovups         %%zmm9, (%%r10)                \n"
        "   vmovups         %%zmm21, 64(%%r10)              \n"
        "   addq %%r14, %%r9                   \n"
        "   addq %%r14, %%r10                   \n"
        
        "   vmovups 		(%%r12), %%zmm0        			\n"
        "   vmovups 		64(%%r12), %%zmm1        			\n"
        "   vmovups 		(%%r8), %%zmm26        			\n"
        "   vmovups 		64(%%r8), %%zmm27        			\n"
        
        "	vaddps 			%%zmm0, %%zmm10, %%zmm10			\n"
        "   vmovups 		(%%r9), %%zmm28        			\n"
        "	vaddps 			%%zmm1, %%zmm22, %%zmm22			\n"
        "   vmovups 		64(%%r9), %%zmm29        			\n"
        "	vaddps 			%%zmm26, %%zmm11, %%zmm11			\n"
        "   vmovups 		(%%r10), %%zmm30        			\n"
        "	vaddps 			%%zmm27, %%zmm23, %%zmm23			\n"
        "   vmovups 		64(%%r10), %%zmm31        			\n"
        
        "	vaddps 			%%zmm28, %%zmm12, %%zmm12			\n"
        "   vmovups         %%zmm10, (%%r12)                 \n"
        "	vaddps 			%%zmm29, %%zmm24, %%zmm24			\n"
        "   vmovups         %%zmm22, 64(%%r12)                 \n"
        "	vaddps 			%%zmm30, %%zmm13, %%zmm13			\n"
        "   vmovups         %%zmm11, (%%r8)                 \n"
        "	vaddps 			%%zmm31, %%zmm25, %%zmm25			\n"
        "   vmovups         %%zmm23, 64(%%r8)                 \n"
        
        "   vmovups         %%zmm12, (%%r9)                \n"
        "   vmovups         %%zmm24, 64(%%r9)              \n"
        "   vmovups         %%zmm13, (%%r10)                \n"
        "   vmovups         %%zmm25, 64(%%r10)              \n"

        ".endm      \n"
        
        "AVX512_N12M32_ENTRANCE_PACK:            \n"
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
        "   movl $0xfffe, %%r8d                   \n"
        "   vpxorq  %%zmm0, %%zmm0, %%zmm0  \n"
        "   vpxorq  %%zmm26, %%zmm26, %%zmm26  \n"
        "   vpxorq  %%zmm2, %%zmm2, %%zmm2  \n"
        "   vpxorq  %%zmm3, %%zmm3, %%zmm3  \n"
        "   vpxorq  %%zmm4, %%zmm4, %%zmm4  \n"
        "   vpxorq  %%zmm5, %%zmm5, %%zmm5  \n"
        "   vpxorq  %%zmm6, %%zmm6, %%zmm6  \n"
        "   kmovd %%r8d, %%k1           \n"
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
        
        //0
        "PACK_Htag0:                \n"
        "   addq %%rsi, %%r11        \n"
        "   cmp $0, %%ecx           \n"
        "   je PACK_Pad_Zero       \n"
        
        "PACK_Htag0_Main:           \n"     //kk=0 hh_tag=0 h=0 ww>0
        "   jmp PACK_Main_C         \n"

        
        "PACK_Pad_Zero_Pre:             \n"
        "   addq $48,%%r11       \n"
        "   addq %%r14, %%r10   \n"
        //"   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%r10)                 \n"
        
        "PACK_Pad_Zero:             \n"
        "   vpxorq  %%zmm0, %%zmm0, %%zmm0  \n"
        "   vmovups (%%r10), %%zmm0%{%%k1%}        \n"          
        "   vmovups 64(%%r10), %%zmm26        \n"
        "   PACK_KERNEL12x32_K1        \n"
        "   addq $48,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"
        
        "   vmovups 4(%%r10), %%zmm0        \n"          
        "   vmovups 68(%%r10), %%zmm26        \n"
        "   PACK_KERNEL12x32_K2        \n"
        "   addq $48,%%r11       \n"

        
        "   vmovups 8(%%r10), %%zmm0        \n"          
        "   vmovups 72(%%r10), %%zmm26        \n"
        "   PACK_KERNEL12x32_K2        \n"
        //"   vpxorq  %%zmm0, %%zmm0, %%zmm0  \n"
        "   subl $1, %%r13d       \n"
        "   jne     PACK_Pad_Zero_Pre       \n"
        "   subl $1, %%r9d          \n"
        "   je      AVX512_N12M32_MAIN_END_PACK     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $48,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     PACK_Pad_Zero            \n"

        
        
        "PACK_Main_C_Pre:                       \n"
        "   addq $48,%%r11       \n"
        "   addq %%r14, %%r10   \n"

        "   prefetcht0 (%%r10)                 \n"
        
        "PACK_Main_C:                       \n"

        "   vmovups (%%r10), %%zmm0        \n"          
        "   vmovups 64(%%r10), %%zmm26        \n"
        "   PACK_KERNEL12x32_K1        \n"
        "   addq $48,%%r11       \n"

        
        "   vmovups 4(%%r10), %%zmm0        \n"          
        "   vmovups 68(%%r10), %%zmm26        \n"
        "   PACK_KERNEL12x32_K2        \n"
        "   addq $48,%%r11       \n"

        
        "   vmovups 8(%%r10), %%zmm0        \n"          
        "   vmovups 72(%%r10), %%zmm26        \n"
        "   PACK_KERNEL12x32_K2        \n"
        
        "   subl $1, %%r13d       \n"
        "   jne     PACK_Main_C_Pre       \n"
        "   subl $1, %%r9d          \n"
        "   je      AVX512_N12M32_MAIN_END_PACK     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $48,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     PACK_Main_C            \n"

        
        "BEGIN_Kb_Mid_Fetch2:                  \n"
        "   addq $48, %%r11       \n"
        "   addq $136, %%rdx   \n"
        //"   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb_Fetch2:                  \n"
        "   vmovups (%%rdx), %%zmm0        \n"
        "   vmovups 64(%%rdx), %%zmm26        \n"
        "   KERNEL12x32    \n"
        "   addq $48,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"

        
        "   vmovups 4(%%rdx), %%zmm0        \n"
        "   vmovups 68(%%rdx), %%zmm26        \n"
        "   KERNEL12x32    \n"
        "   addq $48,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"
        
        "   vmovups 8(%%rdx), %%zmm0        \n"
        "   vmovups 72(%%rdx), %%zmm26        \n"
        "   KERNEL12x32    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid_Fetch2                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid_Fetch2                      \n"
        "   cmp $2, %%eax           \n"
        "   je AVX512_N12M32_MAIN_END_PACK          \n"
        //"   addq %%r15, %%rdi        \n"
        //"   addq %%r15, %%rdi        \n"
        "   leaq (%%rdi, %%r15,2), %%rdi            \n"
        "   addq $48, %%r11       \n"
        "   addq $136, %%rdx   \n"
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
        "   addq $136, %%rdx   \n"
        //"   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        "   jmp  BEGIN_Kb                   \n"
        
        "BEGIN_Kb_Mid:                  \n"
        "   addq $48, %%r11       \n"
        "   addq $136, %%rdx   \n"
        //"   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb:                  \n"
        
        "   vmovups (%%rdx), %%zmm0        \n"
        "   vmovups 64(%%rdx), %%zmm26        \n"
        "   KERNEL12x32    \n"
        "   addq $48,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"

        
        "   vmovups 4(%%rdx), %%zmm0        \n"
        "   vmovups 68(%%rdx), %%zmm26        \n"
        "   KERNEL12x32    \n"
        "   addq $48,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"
        
        "   vmovups 8(%%rdx), %%zmm0        \n"
        "   vmovups 72(%%rdx), %%zmm26        \n"
        "   KERNEL12x32    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid                      \n"
        
        
        //  ADD or ST
        "AVX512_N12M32_MAIN_END_PACK:       \n"
        "   mov %[out], %%r12    \n"
        "   mov %[stride_in_size], %%rbx   \n"
        "   mov %[stride_in_size], %%r14   \n"
        "   movl %[ic], %%eax   \n"
        "   leaq (%%r12, %%rbx), %%r8                 \n"  // C1
        "   leaq (%%r12, %%rbx, 2), %%r9                 \n"  // C1
        "   leaq (%%r8, %%rbx, 2), %%r10                 \n"  // C1
        "   shl $2, %%r14                               \n"
        "   prefetcht1 (%%r12)                 \n"
        "   prefetcht1 (%%r8)                 \n"
        "   prefetcht1 (%%r9)                 \n"
        "   prefetcht1 (%%r10)                 \n"

        "   cmp $0, %%eax    \n"
        "   je AVX512_N12M32_ST_PACK    \n"
        
        // ADD
        "   AVX512_N12M32_ADD_END_PACK   \n"
        "   jmp AVX512_N12M32_END_PACK          \n"
        
        //  ST
        "AVX512_N12M32_ST_PACK:     \n"
        "   AVX512_N12M32_ST_END_PACK     \n"
        
        "AVX512_N12M32_END_PACK:     \n"
        
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

void direct_3x3_N4M32_AVX512_pack(float *img_start_0, float *img_pack, float *kernal, float *out, int ic_count, int ic, int stride_in_size, int hh_tag, int h, int kk, int ww, int W_size){
    long stride_filter = 0;
    if(hh_tag == 0){
        
        stride_filter = ic_count * 3 * 4 * sizeof(float);
        
    }
    asm volatile(

        
        //--------------------- main
        
        ".macro KERNEL4x32  \n"
        
        "   vbroadcastss (%%r11), %%zmm1    \n "
        "   vbroadcastss 4(%%r11), %%zmm27    \n "

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        "   vbroadcastss 8(%%r11), %%zmm30    \n "

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   vbroadcastss 12(%%r11), %%zmm31    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"

        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"


        ".endm      \n"
        
        //----------------
        
        ".macro AVX512_N4M32_ST_END_PACK   \n"
        
        "   vmovups         %%zmm2, (%%r12)                 \n"
        "   vmovups         %%zmm14, 64(%%r12)               \n"
        "   vmovups         %%zmm3, (%%r8)                \n"
        "   vmovups         %%zmm15, 64(%%r8)              \n"
        "   vmovups         %%zmm4, (%%r9)                \n"
        "   vmovups         %%zmm16, 64(%%r9)              \n"
        "   vmovups         %%zmm5, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"
        ".endm      \n"
        
        //-----------------add
        
        ".macro AVX512_N4M32_ADD_END_PACK   \n"
        "   vmovups 		(%%r12), %%zmm0        			\n"
        "   vmovups 		64(%%r12), %%zmm1        			\n"
        "   vmovups 		(%%r8), %%zmm26        			\n"
        "   vmovups 		64(%%r8), %%zmm27        			\n"
        
        "	vaddps 			%%zmm0, %%zmm2, %%zmm2			\n"
        "   vmovups 		(%%r9), %%zmm28        			\n"
        "	vaddps 			%%zmm1, %%zmm14, %%zmm14			\n"
        "   vmovups 		64(%%r9), %%zmm29        			\n"
        "	vaddps 			%%zmm26, %%zmm3, %%zmm3			\n"
        "   vmovups 		(%%r10), %%zmm30        			\n"
        "	vaddps 			%%zmm27, %%zmm15, %%zmm15			\n"
        "   vmovups 		64(%%r10), %%zmm31        			\n"
        
        "	vaddps 			%%zmm28, %%zmm4, %%zmm4			\n"
        "   vmovups         %%zmm2, (%%r12)                 \n"
        "	vaddps 			%%zmm29, %%zmm16, %%zmm16			\n"
        "   vmovups         %%zmm14, 64(%%r12)                 \n"
        "	vaddps 			%%zmm30, %%zmm5, %%zmm5			\n"
        "   vmovups         %%zmm3, (%%r8)                 \n"
        "	vaddps 			%%zmm31, %%zmm17, %%zmm17			\n"
        "   vmovups         %%zmm15, 64(%%r8)                 \n"
        
        "   vmovups         %%zmm4, (%%r9)                \n"
        "   vmovups         %%zmm16, 64(%%r9)              \n"
        "   vmovups         %%zmm5, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"


        ".endm      \n"
        
        "AVX512_N4M32_ENTRANCE_PACK:            \n"
        "   mov %[img_pack],    %%rdx      \n"
        "   mov %[kernal], %%r11 \n"
        "   movl %[h], %%r12d    \n"

        "   movl %[ic_count], %%r13d  \n"           //
        "   mov %[stride_in_size], %%r14    \n"

        "   movl %[hh_tag], %%eax   \n"
        "   movl %[ww], %%ecx        \n"
        "   mov %[stride_filter], %%rsi    \n"

        "   vpxorq  %%zmm0, %%zmm0, %%zmm0  \n"
        "   vpxorq  %%zmm26, %%zmm26, %%zmm26  \n"
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
        
        "   mov $2, %%r9d       \n"
        
        "BEGIN_Kb_Pre_4:                  \n"
        "   addq %%rsi, %%r11       \n"
        "   cmp $1, %%eax               \n"
        "   jne  BEGIN_Kb_4                \n"
        "   mov $3, %%r9d               \n"
        "   jmp  BEGIN_Kb_4                   \n"
        
        "BEGIN_Kb_Pre_Filter_4:                  \n"
        "   addq %%rsi, %%r11       \n"
        "   addq $136, %%rdx   \n"
        //"   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        "   jmp  BEGIN_Kb_4                   \n"
        
        "BEGIN_Kb_Mid_4:                  \n"
        "   addq $16, %%r11       \n"
        "   addq $136, %%rdx   \n"
        //"   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "BEGIN_Kb_4:                  \n"
        
        "   vmovups (%%rdx), %%zmm0        \n"
        "   vmovups 64(%%rdx), %%zmm26        \n"
        "   KERNEL4x32    \n"
        "   addq $16,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"

        
        "   vmovups 4(%%rdx), %%zmm0        \n"
        "   vmovups 68(%%rdx), %%zmm26        \n"
        "   KERNEL4x32    \n"
        "   addq $16,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"
        
        "   vmovups 8(%%rdx), %%zmm0        \n"
        "   vmovups 72(%%rdx), %%zmm26        \n"
        "   KERNEL4x32    \n"

        "   subl $1, %%r13d       \n"
        "   jne BEGIN_Kb_Mid_4                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne BEGIN_Kb_Mid_4                      \n"
        
        
        //  ADD or ST
        "AVX512_N4M32_MAIN_END_PACK:       \n"
        "   mov %[out], %%r12    \n"
        "   mov %[stride_in_size], %%rbx   \n"

        "   movl %[ic], %%eax   \n"
        "   leaq (%%r12, %%rbx), %%r8                 \n"  // C1
        "   leaq (%%r12, %%rbx, 2), %%r9                 \n"  // C1
        "   leaq (%%r8, %%rbx, 2), %%r10                 \n"  // C1
        "   prefetcht1 (%%r12)                 \n"
        "   prefetcht1 (%%r8)                 \n"
        "   prefetcht1 (%%r9)                 \n"
        "   prefetcht1 (%%r10)                 \n"

        "   cmp $0, %%eax    \n"
        "   je AVX512_N4M32_ST_PACK    \n"
        
        // ADD
        "   AVX512_N4M32_ADD_END_PACK   \n"
        "   jmp AVX512_N4M32_END_PACK          \n"
        
        //  ST
        "AVX512_N4M32_ST_PACK:     \n"
        "   AVX512_N4M32_ST_END_PACK     \n"
        
        "AVX512_N4M32_END_PACK:     \n"
        
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


void direct_3x3_N12M32_AVX512_pack_end(float *img_start_0, float *img_pack, float *kernal, float *out, int ic_count, int ic, int stride_in_size, int hh_tag, int h, int kk, int ww, int W_size){
    long stride_filter = 0;
    if(hh_tag == 0){
        
        stride_filter = ic_count * 3 * 12 * sizeof(float);
        
    }
    asm volatile(
        ".macro END_PACK_KERNEL12x32_K1     \n"
        "   vbroadcastss (%%r11), %%zmm1    \n "
        "   vbroadcastss 4(%%r11), %%zmm27    \n "
        "   vbroadcastss 8(%%r11), %%zmm30    \n "
        "   vbroadcastss 12(%%r11), %%zmm31    \n "

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        "   vmovups %%zmm0, (%%rdx)             \n"
        "   movl 128(%%r10), %%r8d     \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   vbroadcastss 16(%%r11), %%zmm1    \n "
        "   vmovups %%zmm26, 64(%%rdx)             \n"
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"
        "   vbroadcastss 20(%%r11), %%zmm27    \n "
        "   prefetcht1 136(%%r10)                 \n"

        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"
        "   vbroadcastss 24(%%r11), %%zmm30    \n "
        "   movl %%r8d, 128(%%rdx)     \n"

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        "   vbroadcastss 28(%%r11), %%zmm31    \n "
        "   movl $0, %%r8d     \n"
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        "   vbroadcastss 32(%%r11), %%zmm1    \n "
        "   prefetcht1 200(%%r10)                 \n"

        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm20 \n"
        "   vbroadcastss 36(%%r11), %%zmm27    \n "
        "   movl %%r8d, 132(%%rdx)     \n"
        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm21 \n"
        "   vbroadcastss 40(%%r11), %%zmm30    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        //"   vfmadd231ps %%zmm26, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        "   vbroadcastss 44(%%r11), %%zmm31    \n "
        "   addq $136, %%rdx                    \n"
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        //"   vfmadd231ps %%zmm26, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm12 \n"
        //"   vfmadd231ps %%zmm26, %%zmm30, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm24 \n"

        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm13 \n"
        //"   vfmadd231ps %%zmm26, %%zmm31, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm25 \n"
        
        ".endm                          \n"
    
        ".macro END_PACK_KERNEL12x32_K2     \n"
        "   vbroadcastss (%%r11), %%zmm1    \n "
        "   vbroadcastss 4(%%r11), %%zmm27    \n "
        "   vbroadcastss 8(%%r11), %%zmm30    \n "
        "   vbroadcastss 12(%%r11), %%zmm31    \n "

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"


        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   vbroadcastss 16(%%r11), %%zmm1    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"
        "   vbroadcastss 20(%%r11), %%zmm27    \n "
        
        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"
        "   vbroadcastss 24(%%r11), %%zmm30    \n "

        
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        "   vbroadcastss 28(%%r11), %%zmm31    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        "   vbroadcastss 32(%%r11), %%zmm1    \n "
        
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm20 \n"
        "   vbroadcastss 36(%%r11), %%zmm27    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm21 \n"
        "   vbroadcastss 40(%%r11), %%zmm30    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        //"   vfmadd231ps %%zmm26, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        "   vbroadcastss 44(%%r11), %%zmm31    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        //"   vfmadd231ps %%zmm26, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm12 \n"
        //"   vfmadd231ps %%zmm26, %%zmm30, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm24 \n"

        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm13 \n"
        //"   vfmadd231ps %%zmm26, %%zmm31, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm25 \n"
        ".endm                          \n"

        
        //--------------------- main
        
        ".macro END_KERNEL12x32  \n"
        
        "   vbroadcastss (%%r11), %%zmm1    \n "
        "   vbroadcastss 4(%%r11), %%zmm27    \n "
        "   vbroadcastss 8(%%r11), %%zmm30    \n "
        "   vbroadcastss 12(%%r11), %%zmm31    \n "

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"


        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   vbroadcastss 16(%%r11), %%zmm1    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"
        "   vbroadcastss 20(%%r11), %%zmm27    \n "
        
        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"
        "   vbroadcastss 24(%%r11), %%zmm30    \n "

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        "   vbroadcastss 28(%%r11), %%zmm31    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        "   vbroadcastss 32(%%r11), %%zmm1    \n "
        
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm20 \n"
        "   vbroadcastss 36(%%r11), %%zmm27    \n "

        
        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm21 \n"
        "   vbroadcastss 40(%%r11), %%zmm30    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        "   vbroadcastss 44(%%r11), %%zmm31    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm24 \n"

        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm25 \n"
        
        ".endm      \n"
        
        //----------------
        
        ".macro END_AVX512_N12M32_ST_END_PACK   \n"
        
        "   vmovups         %%zmm2, (%%r12)                 \n"
        "   vmovups         %%zmm14, 64(%%r12)               \n"
        "   vmovups         %%zmm3, (%%r8)                \n"
        "   vmovups         %%zmm15, 64(%%r8)              \n"
        "   addq %%r14, %%r12                   \n"
        "   vmovups         %%zmm4, (%%r9)                \n"
        "   vmovups         %%zmm16, 64(%%r9)              \n"
        "   addq %%r14, %%r8                   \n"
        "   vmovups         %%zmm5, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"
        "   addq %%r14, %%r9                   \n"


        "   vmovups         %%zmm6, (%%r12)                \n"
        "   vmovups         %%zmm18, 64(%%r12)              \n"
        "   addq %%r14, %%r10                   \n"
        "   vmovups         %%zmm7, (%%r8)                \n"
        "   vmovups         %%zmm19, 64(%%r8)              \n"
        "   addq %%r14, %%r12                   \n"
        "   vmovups         %%zmm8, (%%r9)                \n"
        "   vmovups         %%zmm20, 64(%%r9)              \n"
        "   addq %%r14, %%r8                   \n"
        "   vmovups         %%zmm9, (%%r10)                \n"
        "   vmovups         %%zmm21, 64(%%r10)              \n" 
        "   addq %%r14, %%r9                   \n"

        "   vmovups         %%zmm10, (%%r12)                \n"
        "   vmovups         %%zmm22, 64(%%r12)              \n"
        "   addq %%r14, %%r10                   \n"
        "   vmovups         %%zmm11, (%%r8)              \n"
        "   vmovups         %%zmm23, 64(%%r8)              \n"
        "   vmovups         %%zmm12, (%%r9)                \n"
        "   vmovups         %%zmm24, 64(%%r9)              \n"
        
        "   vmovups         %%zmm13, (%%r10)                \n"
        "   vmovups         %%zmm25, 64(%%r10)              \n"
        
        ".endm      \n"
        
        //-----------------add
        
        ".macro END_AVX512_N12M32_ADD_END_PACK   \n"
        "   vmovups 		(%%r12), %%zmm0        			\n"
        "   vmovups 		64(%%r12), %%zmm1        			\n"
        "   vmovups 		(%%r8), %%zmm26        			\n"
        "   vmovups 		64(%%r8), %%zmm27        			\n"
        
        "	vaddps 			%%zmm0, %%zmm2, %%zmm2			\n"
        "   vmovups 		(%%r9), %%zmm28        			\n"
        "	vaddps 			%%zmm1, %%zmm14, %%zmm14			\n"
        "   vmovups 		64(%%r9), %%zmm29        			\n"
        "	vaddps 			%%zmm26, %%zmm3, %%zmm3			\n"
        "   vmovups 		(%%r10), %%zmm30        			\n"
        "	vaddps 			%%zmm27, %%zmm15, %%zmm15			\n"
        "   vmovups 		64(%%r10), %%zmm31        			\n"
        
        "	vaddps 			%%zmm28, %%zmm4, %%zmm4			\n"
        "   vmovups         %%zmm2, (%%r12)                 \n"
        "	vaddps 			%%zmm29, %%zmm16, %%zmm16			\n"
        "   vmovups         %%zmm14, 64(%%r12)                 \n"
        "	vaddps 			%%zmm30, %%zmm5, %%zmm5			\n"
        "   vmovups         %%zmm3, (%%r8)                 \n"
        "	vaddps 			%%zmm31, %%zmm17, %%zmm17			\n"
        "   vmovups         %%zmm15, 64(%%r8)                 \n"
        
        "   addq %%r14, %%r12                   \n"
        "   vmovups         %%zmm4, (%%r9)                \n"
        "   vmovups         %%zmm16, 64(%%r9)              \n"
        "   addq %%r14, %%r8                   \n"
        "   vmovups         %%zmm5, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"
        "   addq %%r14, %%r9                   \n"
        "   addq %%r14, %%r10                   \n"
        
        "   vmovups 		(%%r12), %%zmm0        			\n"
        "   vmovups 		64(%%r12), %%zmm1        			\n"
        "   vmovups 		(%%r8), %%zmm26        			\n"
        "   vmovups 		64(%%r8), %%zmm27        			\n"
        
        "	vaddps 			%%zmm0, %%zmm6, %%zmm6			\n"
        "   vmovups 		(%%r9), %%zmm28        			\n"
        "	vaddps 			%%zmm1, %%zmm18, %%zmm18			\n"
        "   vmovups 		64(%%r9), %%zmm29        			\n"
        "	vaddps 			%%zmm26, %%zmm7, %%zmm7			\n"
        "   vmovups 		(%%r10), %%zmm30        			\n"
        "	vaddps 			%%zmm27, %%zmm19, %%zmm19			\n"
        "   vmovups 		64(%%r10), %%zmm31        			\n"
        
        "	vaddps 			%%zmm28, %%zmm8, %%zmm8			\n"
        "   vmovups         %%zmm6, (%%r12)                 \n"
        "	vaddps 			%%zmm29, %%zmm20, %%zmm20			\n"
        "   vmovups         %%zmm18, 64(%%r12)                 \n"
        "	vaddps 			%%zmm30, %%zmm9, %%zmm9			\n"
        "   vmovups         %%zmm7, (%%r8)                 \n"
        "	vaddps 			%%zmm31, %%zmm21, %%zmm21			\n"
        "   vmovups         %%zmm19, 64(%%r8)                 \n"
        
        "   addq %%r14, %%r12                   \n"
        "   vmovups         %%zmm8, (%%r9)                \n"
        "   vmovups         %%zmm20, 64(%%r9)              \n"
        "   addq %%r14, %%r8                   \n"
        "   vmovups         %%zmm9, (%%r10)                \n"
        "   vmovups         %%zmm21, 64(%%r10)              \n"
        "   addq %%r14, %%r9                   \n"
        "   addq %%r14, %%r10                   \n"
        
        "   vmovups 		(%%r12), %%zmm0        			\n"
        "   vmovups 		64(%%r12), %%zmm1        			\n"
        "   vmovups 		(%%r8), %%zmm26        			\n"
        "   vmovups 		64(%%r8), %%zmm27        			\n"
        
        "	vaddps 			%%zmm0, %%zmm10, %%zmm10			\n"
        "   vmovups 		(%%r9), %%zmm28        			\n"
        "	vaddps 			%%zmm1, %%zmm22, %%zmm22			\n"
        "   vmovups 		64(%%r9), %%zmm29        			\n"
        "	vaddps 			%%zmm26, %%zmm11, %%zmm11			\n"
        "   vmovups 		(%%r10), %%zmm30        			\n"
        "	vaddps 			%%zmm27, %%zmm23, %%zmm23			\n"
        "   vmovups 		64(%%r10), %%zmm31        			\n"
        
        "	vaddps 			%%zmm28, %%zmm12, %%zmm12			\n"
        "   vmovups         %%zmm10, (%%r12)                 \n"
        "	vaddps 			%%zmm29, %%zmm24, %%zmm24			\n"
        "   vmovups         %%zmm22, 64(%%r12)                 \n"
        "	vaddps 			%%zmm30, %%zmm13, %%zmm13			\n"
        "   vmovups         %%zmm11, (%%r8)                 \n"
        "	vaddps 			%%zmm31, %%zmm25, %%zmm25			\n"
        "   vmovups         %%zmm23, 64(%%r8)                 \n"
        
        "   vmovups         %%zmm12, (%%r9)                \n"
        "   vmovups         %%zmm24, 64(%%r9)              \n"
        "   vmovups         %%zmm13, (%%r10)                \n"
        "   vmovups         %%zmm25, 64(%%r10)              \n"

        ".endm      \n"
        
        "END_AVX512_N12M32_ENTRANCE_PACK:            \n"
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
        "   vpxorq  %%zmm26, %%zmm26, %%zmm26  \n"
        "END_K_Branch_12:                  \n"
        "   mov $2, %%r9d       \n"
        "   cmp $0, %%ebx       \n"
        "   jne END_BEGIN_Kb_Pre_12                 \n"
        
        //kk=0
        "   cmp $0, %%eax          \n"
        "   je END_PACK_Htag0_12                     \n"
        "   cmp $1, %%eax           \n"
        "   je END_PACK_Htag1_12                     \n"
        
        //2
        "END_PACK_Htag2_12:                \n"
        "   cmp $0, %%r12d           \n"
        "   jne END_BEGIN_Kb_Fetch2_12                   \n"
        "   jmp END_PACK_Main_C_12         \n"
        
        
        //1
        "END_PACK_Htag1_12:                \n"
        "   cmp $0, %%r12d          \n"
        "   jne END_BEGIN_Kb_Fetch2_12      \n"            
        "   mov $3, %%r9d       \n"             //kk=0 hh_tag=1 h=0
        "   jmp END_PACK_Main_C_12         \n"
        
        
        //0
        "END_PACK_Htag0_12:                \n"
        "   addq %%rsi, %%r11        \n"
        "   jmp END_PACK_Main_C_12         \n"

        
        
        "END_PACK_Main_C_Pre_12:                       \n"
        "   addq $48,%%r11       \n"
        "   addq %%r14, %%r10   \n"
        //"   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%r10)                 \n"
        
        "END_PACK_Main_C_12:                       \n"

        "   vmovups (%%r10), %%zmm0        \n"          
        "   vmovups 64(%%r10), %%zmm26        \n"
        "   END_PACK_KERNEL12x32_K1        \n"
        "   addq $48,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"
        
        "   vmovups 4(%%r10), %%zmm0        \n"          
        "   vmovups 68(%%r10), %%zmm26        \n"
        "   END_PACK_KERNEL12x32_K2        \n"
        "   addq $48,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"


        "   movl $0x7fff, %%r8d                   \n"
        "   kmovd %%r8d, %%k1           \n"
        "   vpxorq  %%zmm26, %%zmm26, %%zmm26  \n"
        //"   vpxorq  %%zmm0, %%zmm0, %%zmm0  \n"
        "   vmovups 8(%%r10), %%zmm0        \n"          
        "   vmovups 72(%%r10), %%zmm26%{%%k1%}        \n"
        //"   vmovups 72(%%r10), %%zmm26        \n"
        "   END_PACK_KERNEL12x32_K2        \n"
        
        "   subl $1, %%r13d       \n"
        "   jne     END_PACK_Main_C_Pre_12       \n"
        "   subl $1, %%r9d          \n"
        "   je      END_AVX512_N12M32_MAIN_END_PACK     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $48,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     END_PACK_Main_C_12            \n"

        
        "END_BEGIN_Kb_Mid_Fetch2_12:                  \n"
        "   addq $48, %%r11       \n"
        "   addq $136, %%rdx   \n"
        //"   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "END_BEGIN_Kb_Fetch2_12:                  \n"
        "   vmovups (%%rdx), %%zmm0        \n"
        "   vmovups 64(%%rdx), %%zmm26        \n"
        "   END_KERNEL12x32    \n"
        "   addq $48,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"

        
        "   vmovups 4(%%rdx), %%zmm0        \n"
        "   vmovups 68(%%rdx), %%zmm26        \n"
        "   END_KERNEL12x32    \n"
        "   addq $48,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"
        
        "   vmovups 8(%%rdx), %%zmm0        \n"
        "   vmovups 72(%%rdx), %%zmm26        \n"
        "   END_KERNEL12x32    \n"

        "   subl $1, %%r13d       \n"
        "   jne END_BEGIN_Kb_Mid_Fetch2_12                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne END_BEGIN_Kb_Mid_Fetch2_12                      \n"
        "   cmp $2, %%eax           \n"
        "   je END_AVX512_N12M32_MAIN_END_PACK          \n"
        "   leaq (%%rdi, %%r15,2), %%rdi            \n"
        "   addq $48, %%r11       \n"
        "   addq $136, %%rdx   \n"
        "   movl $1, %%r9d          \n"
        "   mov %%rdi, %%r10       \n"

        "   jmp END_PACK_Main_C_12             \n"

        


        "END_BEGIN_Kb_Pre_12:                  \n"
        "   addq %%rsi, %%r11       \n"
        "   cmp $1, %%eax               \n"
        "   jne  END_BEGIN_Kb_12                \n"
        "   mov $3, %%r9d               \n"
        "   jmp  END_BEGIN_Kb_12                   \n"
        
        "END_BEGIN_Kb_Mid_12:                  \n"
        "   addq $48, %%r11       \n"
        "   addq $136, %%rdx   \n"
        //"   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "END_BEGIN_Kb_12:                  \n"
        
        "   vmovups (%%rdx), %%zmm0        \n"
        "   vmovups 64(%%rdx), %%zmm26        \n"
        "   END_KERNEL12x32    \n"
        "   addq $48,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"

        
        "   vmovups 4(%%rdx), %%zmm0        \n"
        "   vmovups 68(%%rdx), %%zmm26        \n"
        "   END_KERNEL12x32    \n"
        "   addq $48,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"
        
        "   vmovups 8(%%rdx), %%zmm0        \n"
        "   vmovups 72(%%rdx), %%zmm26        \n"
        "   END_KERNEL12x32    \n"

        "   subl $1, %%r13d       \n"
        "   jne END_BEGIN_Kb_Mid_12                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne END_BEGIN_Kb_Mid_12                      \n"
        
        
        //  ADD or ST
        "END_AVX512_N12M32_MAIN_END_PACK:       \n"
        "   mov %[out], %%r12    \n"
        "   mov %[stride_in_size], %%rbx   \n"
        "   mov %[stride_in_size], %%r14   \n"
        "   movl %[ic], %%eax   \n"
        "   leaq (%%r12, %%rbx), %%r8                 \n"  // C1
        "   leaq (%%r12, %%rbx, 2), %%r9                 \n"  // C1
        "   leaq (%%r8, %%rbx,2), %%r10                 \n"  // C1
        "   shl $2, %%r14                               \n"
        //"   prefetcht1 (%%r12)                 \n"
        //"   prefetcht1 (%%r8)                 \n"
        //"   prefetcht1 (%%r9)                 \n"
        //"   prefetcht1 (%%r10)                 \n"
        "   cmp $0, %%eax    \n"
        "   je END_AVX512_N12M32_ST_PACK    \n"
        
        // ADD
        "   END_AVX512_N12M32_ADD_END_PACK   \n"
        "   jmp END_AVX512_N12M32_END_PACK          \n"
        
        //  ST
        "END_AVX512_N12M32_ST_PACK:     \n"
        "   END_AVX512_N12M32_ST_END_PACK     \n"
        
        "END_AVX512_N12M32_END_PACK:     \n"
        
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


void direct_3x3_N4M32_AVX512_pack_end(float *img_start_0, float *img_pack, float *kernal, float *out, int ic_count, int ic, int stride_in_size, int hh_tag, int h, int kk, int ww, int W_size){
    long stride_filter = 0;
    if(hh_tag == 0){
        
        stride_filter = ic_count * 3 * 4 * sizeof(float);
        
    }
    asm volatile(
        
        //--------------------- main
        
        ".macro END_KERNEL4x32  \n"
        
        "   vbroadcastss (%%r11), %%zmm1    \n "
        "   vbroadcastss 4(%%r11), %%zmm27    \n "
        "   vbroadcastss 8(%%r11), %%zmm30    \n "
        "   vbroadcastss 12(%%r11), %%zmm31    \n "

        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"


        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm30, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm30, %%zmm16 \n"

        "   vfmadd231ps %%zmm0, %%zmm31, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm31, %%zmm17 \n"

        
        ".endm      \n"
        
        //----------------
        
        ".macro END_AVX512_N4M32_ST_END_PACK   \n"
        
        "   vmovups         %%zmm2, (%%r12)                 \n"
        "   vmovups         %%zmm14, 64(%%r12)               \n"
        "   vmovups         %%zmm3, (%%r8)                \n"
        "   vmovups         %%zmm15, 64(%%r8)              \n"
        "   vmovups         %%zmm4, (%%r9)                \n"
        "   vmovups         %%zmm16, 64(%%r9)              \n"
        "   vmovups         %%zmm5, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"

        ".endm      \n"
        
        //-----------------add
        
        ".macro END_AVX512_N4M32_ADD_END_PACK   \n"
        "   vmovups 		(%%r12), %%zmm0        			\n"
        "   vmovups 		64(%%r12), %%zmm1        			\n"
        "   vmovups 		(%%r8), %%zmm26        			\n"
        "   vmovups 		64(%%r8), %%zmm27        			\n"
        
        "	vaddps 			%%zmm0, %%zmm2, %%zmm2			\n"
        "   vmovups 		(%%r9), %%zmm28        			\n"
        "	vaddps 			%%zmm1, %%zmm14, %%zmm14			\n"
        "   vmovups 		64(%%r9), %%zmm29        			\n"
        "	vaddps 			%%zmm26, %%zmm3, %%zmm3			\n"
        "   vmovups 		(%%r10), %%zmm30        			\n"
        "	vaddps 			%%zmm27, %%zmm15, %%zmm15			\n"
        "   vmovups 		64(%%r10), %%zmm31        			\n"
        
        "	vaddps 			%%zmm28, %%zmm4, %%zmm4			\n"
        "   vmovups         %%zmm2, (%%r12)                 \n"
        "	vaddps 			%%zmm29, %%zmm16, %%zmm16			\n"
        "   vmovups         %%zmm14, 64(%%r12)                 \n"
        "	vaddps 			%%zmm30, %%zmm5, %%zmm5			\n"
        "   vmovups         %%zmm3, (%%r8)                 \n"
        "	vaddps 			%%zmm31, %%zmm17, %%zmm17			\n"
        "   vmovups         %%zmm15, 64(%%r8)                 \n"
        "   vmovups         %%zmm4, (%%r9)                \n"
        "   vmovups         %%zmm16, 64(%%r9)              \n"
        "   vmovups         %%zmm5, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"
        
        ".endm      \n"
        
        "END_AVX512_N4M32_ENTRANCE_PACK:            \n"

        "   mov %[img_pack],    %%rdx      \n"
        "   mov %[kernal], %%r11 \n"
        "   movl %[h], %%r12d    \n"

        "   movl %[ic_count], %%r13d  \n"           //
        "   mov %[stride_in_size], %%r14    \n"
        "   mov %[W_size], %%r15    \n"

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
        "   vpxorq  %%zmm26, %%zmm26, %%zmm26  \n"

        "   mov $2, %%r9d       \n"
        
        "END_BEGIN_Kb_Pre_4:                  \n"
        "   addq %%rsi, %%r11       \n"
        "   cmp $1, %%eax               \n"
        "   jne  END_BEGIN_Kb_4                \n"
        "   mov $3, %%r9d               \n"
        "   jmp  END_BEGIN_Kb_4                   \n"
        
        "END_BEGIN_Kb_Mid_4:                  \n"
        "   addq $16, %%r11       \n"
        "   addq $136, %%rdx   \n"
        //"   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%rdx)                 \n"
        
        "END_BEGIN_Kb_4:                  \n"
        
        "   vmovups (%%rdx), %%zmm0        \n"
        "   vmovups 64(%%rdx), %%zmm26        \n"
        "   END_KERNEL4x32    \n"
        "   addq $16,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"

        
        "   vmovups 4(%%rdx), %%zmm0        \n"
        "   vmovups 68(%%rdx), %%zmm26        \n"
        "   END_KERNEL4x32    \n"
        "   addq $16,%%r11       \n"
        //"   prefetcht0 (%%r11)                 \n"
        
        "   vmovups 8(%%rdx), %%zmm0        \n"
        "   vmovups 72(%%rdx), %%zmm26        \n"
        "   END_KERNEL4x32    \n"

        "   subl $1, %%r13d       \n"
        "   jne END_BEGIN_Kb_Mid_4                    \n"
        "   movl %[ic_count], %%r13d      \n"
        "   subl $1, %%r9d          \n"
        "   jne END_BEGIN_Kb_Mid_4                      \n"
        
        
        //  ADD or ST
        "END_AVX512_N4M32_MAIN_END_PACK:       \n"
        "   mov %[out], %%r12    \n"
        "   mov %[stride_in_size], %%rbx   \n"
        "   mov %[stride_in_size], %%r14   \n"
        "   movl %[ic], %%eax   \n"
        "   leaq (%%r12, %%rbx), %%r8                 \n"  // C1
        "   leaq (%%r12, %%rbx, 2), %%r9                 \n"  // C1
        "   leaq (%%r8, %%rbx,2), %%r10                 \n"  // C1
        //"   prefetcht1 (%%r12)                 \n"
        //"   prefetcht1 (%%r8)                 \n"
        //"   prefetcht1 (%%r9)                 \n"
        //"   prefetcht1 (%%r10)                 \n"
        "   cmp $0, %%eax    \n"
        "   je END_AVX512_N4M32_ST_PACK    \n"
        
        // ADD
        "   END_AVX512_N4M32_ADD_END_PACK   \n"
        "   jmp END_AVX512_N4M32_END_PACK          \n"
        
        //  ST
        "END_AVX512_N4M32_ST_PACK:     \n"
        "   END_AVX512_N4M32_ST_END_PACK     \n"
        
        "END_AVX512_N4M32_END_PACK:     \n"
        
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

    int KB_D =  (Kb /12)*12;
    int KB_R = Kb % 12;
    //Wb = 12;
    //printf("begin\n");

    //Edge = 14;
    for(hh = 0; hh < out_H; hh = hh + Hb)
    {
        Hb = 4;                 
        if (out_H - hh < 4)
        {
            Hb = out_H - hh;
        }
        for(ww = 0; ww < out_W-32; ww = ww + Wb)
        {
            
            Wb = 32;
            Edge = 34;

            if (W - ww < 32)
            {
                Wb = W - ww;
                Edge = Wb + 1;
            }
            
            input0 = input + hh * W  + ww;
            
            Edge_stride = Edge * Cb;
            
            
            //if(ww != 0)
            //    input0 = input0 - 1;
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
                
                if(Wb == 32){
                    
                    
                    for(kk=0; kk<KB_D; kk+=12){
                        filter0 = filter + kk*9*Cb;
                        output1 = output0 + kk * stride_out + h*out_W;
                        direct_3x3_N12M32_AVX512_pack(input1, input_c0, filter0, output1, Cb, c_tag, stride_in_size, hh_tag, h, kk, ww, out_W<<2);
                    }
                    filter0 = filter + kk*9*Cb;
                    output1 = output0 + kk * stride_out + h*out_W;
                    direct_3x3_N4M32_AVX512_pack(input1, input_c0, filter0, output1, Cb, c_tag, stride_in_size, hh_tag, h, kk, ww, out_W<<2);
                }

                if(hh_tag == 0 ){
                    input_c0 -= Edge_stride;
                }
                input_c0 += Edge_stride;
            }
        }
            
        Wb = 32;
        Edge = 34;

        input0 = input + hh * W  + ww;
        Edge_stride = Edge * Cb;
        input0 = input0 - 1;
        output0 = output + hh * out_W + ww;
        input_c0 = input_c;
        for(h=0; h<Hb; h++){
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
            if(Wb == 32){
                for(kk=0; kk<KB_D; kk+=12){
                    filter0 = filter + kk*9*Cb;
                    output1 = output0 + kk * stride_out + h*out_W;
                    direct_3x3_N12M32_AVX512_pack_end(input1, input_c0, filter0, output1, Cb, c_tag, stride_in_size, hh_tag, h, kk, ww, out_W<<2);
                }
                filter0 = filter + kk*9*Cb;
                output1 = output0 + kk * stride_out + h*out_W;
                direct_3x3_N4M32_AVX512_pack_end(input1, input_c0, filter0, output1, Cb, c_tag, stride_in_size, hh_tag, h, kk, ww, out_W<<2);
            }
            if(hh_tag == 0 ){
                input_c0 -= Edge_stride;
            }
            input_c0 += Edge_stride;
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
    float *input_c = ( float * ) malloc( NUM * CONV_C * 14 * 34 * sizeof( float ));


    #pragma omp parallel num_threads(NUM)
    {

        int i, ii, jj, kk, cc, hh, h, ww;
        int Cb, Kb, Wb, Hb;
        bool C_end = 0;
        //float *temp_MVWB = MVWB;


        int id = omp_get_thread_num();

        float *input_c0 = input_c + id * CONV_C * 14 * 34;

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
    int array_C[25] = {64};
    int array_K[25] = {64};
    int array_H[25] = {224};
    int array_W[25] = {224};
    int array_R[25] = {3};
    int array_S[25] = {3};
    int array_pad[25] = {1};
    int array_str[25] = {1};
    int z,k;
    FILE *fp;
    if ( (fp = fopen("ndirect3x3.txt", "a+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
    for(k=0;k<5;k++){
        for(z=0;z<1;z++){
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
            int N = NUM;
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
            float *filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 512);
            //float *filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 128);
            if(filter == NULL ){
                printf("filter aligned error!\n");
                return 0;
                
            }
            
            float *trans_filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 512);
            //float *trans_filter = (float*)_mm_malloc((size_t)K*C*R*S*sizeof(float), 128);
            //float *trans_filter = (float*)_mm_malloc(K*C*3*3*sizeof(float), 32);
            
            if(filter == NULL ){
                printf("trans_filter aligned error!\n");
                return 0;
                
            }
            
            float *input =  (float*)_mm_malloc((size_t)N*C*H*W*sizeof(float), 512);
            //float *input =  (float*)_mm_malloc((size_t)N*C*H*W*sizeof(float), 128);
            if(input == NULL ){
                printf("input aligned error!\n");
                return 0;
                
            }
            
            float *output = (float*)_mm_malloc((size_t)N*K*out_H*out_W*sizeof(float), 512);
            //float *output = (float*)_mm_malloc((size_t)N*K*out_H*out_W*sizeof(float), 128);
            if(output == NULL ){
                printf("output aligned error!\n");
                return 0;
                
            }
            
            float *output1 = (float*)malloc(N*K*out_H*out_W*sizeof(float));
            float *data_col = (float*)malloc(N*C*(H/stride*W/stride)*R*S*sizeof(float));
            //float *output1 = (float*)_mm_malloc(N*K*H*W*sizeof(float), 32);
            //printf("N = %d, K = %d, C = %d, H = %d, W = %d\n", N, K, C, H, W);
            

            /*
            float *filter = (float*)malloc((size_t)K*C*R*S*sizeof(float));

            
            float *trans_filter = (float*)malloc((size_t)K*C*R*S*sizeof(float));

            
            float *input =  (float*)malloc((size_t)N*C*H*W*sizeof(float));

            
            float *output = (float*)malloc((size_t)N*K*out_H*out_W*sizeof(float));
            float *output1 = (float*)malloc(N*K*out_H*out_W*sizeof(float));
            float *data_col = (float*)malloc(N*C*(H/stride*W/stride)*R*S*sizeof(float));
            */


            for(int pc = 0; pc < 1; pc++)
            {
                random_matrix(K, C * R *S, filter);
                random_matrix_1(N, C * H * W, input);
                
                //printf("1\n");
                transform_filter(K, C, R, S, filter, trans_filter);

                
                for(i = 0; i < 1; i++)
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
            

            if(flag != 0)
                printf("结果错误\n");
            */

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