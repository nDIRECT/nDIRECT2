#include <malloc.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
void direct_7x7_N32M12_AVX512(float *img_start_0, float *kernal, float *out, float *out_buffer, int ic_count, int stride_in_size, int stride_out_size, int hh_tag, int ww_tag, int W_size){

    asm volatile(
        ".macro KERNEL32x10_K7     \n"
        // 1-1
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vbroadcastss 12(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 28(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        
        "   vbroadcastss 36(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 44(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        
        "   vbroadcastss 52(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"
        
        "   vbroadcastss 60(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 68(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        "   vbroadcastss 76(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"

        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        
        //1-2
        "   vbroadcastss (%%r10), %%zmm27    \n "
        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 16(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 32(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 48(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        
        "   vbroadcastss 56(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"
        
        "   vbroadcastss 64(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 72(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        "   vbroadcastss 80(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        //1-3
        "   vbroadcastss 4(%%r10), %%zmm27    \n "
        "   vbroadcastss 12(%%r10), %%zmm1    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 20(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 28(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 36(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 44(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 52(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        
        "   vbroadcastss 60(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"
        
        "   vbroadcastss 68(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 76(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        "   vbroadcastss 84(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        //1-4
        "   vbroadcastss (%%r10), %%zmm1    \n "
        "   vbroadcastss 8(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 32(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 48(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 56(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        
        "   vbroadcastss 64(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"
        
        "   vbroadcastss 72(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 80(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        "   vbroadcastss 88(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        
        //1-5
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vbroadcastss 12(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 28(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 36(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 44(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 52(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 60(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        
        "   vbroadcastss 68(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"
        
        "   vbroadcastss 76(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 84(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        "   vbroadcastss 92(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"

        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        //1-6
        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        "   vbroadcastss 16(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 32(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 48(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 56(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 64(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        
        "   vbroadcastss 72(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"
        
        "   vbroadcastss 80(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 88(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        "   vbroadcastss 96(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"

        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        //1-7
        "   vbroadcastss 12(%%r10), %%zmm1    \n "
        "   vbroadcastss 20(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 28(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 36(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 44(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 52(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 60(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 68(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        
        "   vbroadcastss 76(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"
        
        "   vbroadcastss 84(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 92(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        "   vbroadcastss 100(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        
        ".endm                          \n"
        
        // W > 12
        ".macro KERNEL32x12_K7     \n"
        // 1-1
        "   vbroadcastss (%%r10), %%zmm1    \n "
        "   vbroadcastss 8(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 32(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        
        "   vbroadcastss 48(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 56(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"
        
        "   vbroadcastss 64(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"
        
        "   vbroadcastss 72(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 80(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"
        
        "   vbroadcastss 88(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        //1-2
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vbroadcastss 12(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 28(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 36(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 44(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        "   vbroadcastss 52(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 60(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        
        "   vbroadcastss 68(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"

        
        "   vbroadcastss 76(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 84(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        
        "   vbroadcastss 92(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        //1-3
        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        "   vbroadcastss 16(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 32(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 48(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        
        "   vbroadcastss 56(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 64(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        
        "   vbroadcastss 72(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"

        
        "   vbroadcastss 80(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 88(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        
        "   vbroadcastss 96(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        //1-4
        "   vbroadcastss 12(%%r10), %%zmm1    \n "
        "   vbroadcastss 20(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 28(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 36(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 44(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 52(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        
        "   vbroadcastss 60(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 68(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        
        "   vbroadcastss 76(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"

        
        "   vbroadcastss 84(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 92(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        
        "   vbroadcastss 100(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        
        //1-5
        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vbroadcastss 24(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 32(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 48(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 56(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        
        "   vbroadcastss 64(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 72(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        
        "   vbroadcastss 80(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"

        
        "   vbroadcastss 88(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 96(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        
        "   vbroadcastss 104(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        
        //1-6
        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vbroadcastss 28(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 36(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 44(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 52(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 60(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        
        "   vbroadcastss 68(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 76(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        
        "   vbroadcastss 84(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"

        
        "   vbroadcastss 92(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 100(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        
        "   vbroadcastss 108(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        
        //1-7
        "   vbroadcastss 24(%%r10), %%zmm1    \n "
        "   vbroadcastss 32(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 48(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        
        "   vbroadcastss 56(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        
        "   vbroadcastss 64(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm6 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm18 \n"
        
        
        "   vbroadcastss 72(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm7 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm19 \n"
        
        "   vbroadcastss 80(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm8 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm20 \n"

        
        "   vbroadcastss 88(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm9 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm21 \n"

        
        "   vbroadcastss 96(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm10 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm22 \n"
        
        "   vbroadcastss 104(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm11 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm23 \n"

        
        "   vbroadcastss 112(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm12 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm24 \n"

        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm13 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm25 \n"
        
        ".endm                          \n"
        
        // W = final
        ".macro KERNEL32x4_K7     \n"
        // 1-1
        "   vbroadcastss (%%r10), %%zmm1    \n "
        "   vbroadcastss 8(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"

        //1-2
        "   vbroadcastss 4(%%r10), %%zmm1    \n "
        "   vbroadcastss 12(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 28(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"

        //1-3
        "   vbroadcastss 8(%%r10), %%zmm1    \n "
        "   vbroadcastss 16(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 24(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 32(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"

        //1-4
        "   vbroadcastss 12(%%r10), %%zmm1    \n "
        "   vbroadcastss 20(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 28(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 36(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"

        
        //1-5
        "   vbroadcastss 16(%%r10), %%zmm1    \n "
        "   vbroadcastss 24(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 32(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm27    \n "
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm5 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm17 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        
        //1-6
        "   vbroadcastss 20(%%r10), %%zmm1    \n "
        "   vbroadcastss 28(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 36(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        "   addq $128,%%r11       \n"
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        
        //1-7
        "   vbroadcastss 24(%%r10), %%zmm1    \n "
        "   vbroadcastss 32(%%r10), %%zmm27    \n "
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm2 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm14 \n"
        
        "   vbroadcastss 40(%%r10), %%zmm1    \n "
        "   vfmadd231ps %%zmm0, %%zmm27, %%zmm3 \n"
        "   vfmadd231ps %%zmm26, %%zmm27, %%zmm15 \n"
        
        "   vfmadd231ps %%zmm0, %%zmm1, %%zmm4 \n"
        "   vfmadd231ps %%zmm26, %%zmm1, %%zmm16 \n"
        

        
        ".endm                          \n"
        
        
        ".macro AVX512_N32M12_ST   \n"
        
        "   shl $2, %%r11     \n"
        "   shl $3, %%r14       \n"
        "   addq %%r11, %%r8    \n"      // output 5
        "   addq %%r14, %%r9    \n"      // output 9
        "   addq %%r11, %%r14    \n"
        
        "   shl $1, %%rbx       \n"     //inc 2
        "   addq %%r14, %%r10    \n"     //output 13
        
        "   sub %%r15, %%r11    \n"     //inc 3
        "   addq %%r15, %%r14    \n"
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
        
        ".macro AVX512_N32M4_ST   \n"
        
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
        
        ".macro AVX512_N32M12_ST_Continue           \n"
        
        //  1 5 9 13
        "   shl $2, %%r11     \n"        //inc 4
        "   shl $1, %%rbx       \n"     //inc 2
        "   shl $3, %%r14       \n"     //inc 8
        "   addq %%r11, %%r8    \n"      // output 5
        "   addq %%r11, %%r9    \n"      // output 5
        "   addq %%r14, %%r10    \n"      // output 5
        
        
        "   addq %%r15, %%r8        \n"     // output 6
        "   addq %%rbx, %%r9           \n"  // output 7
        "   subq %%r15, %%r10           \n" // output 8
        
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
        "   vmovups %%ymm30, 384(%%rcx)            \n"     //input 9
        
        "   vmovups %%ymm29, 192(%%rcx)         \n"     //  input 5
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 576(%%rcx)            \n"     //input 13
        //  
        "   vunpcklps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpcklps %%zmm13, %%zmm12, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r12)         \n"     //  input 1
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 5
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 9
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 13
        
        "   vmovups %%xmm28, 224(%%rcx)         \n"     //  input 5
        "   vmovups %%xmm29, 416(%%rcx)         \n"     //  input 9
        "   vmovups %%xmm30, 608(%%rcx)         \n"     //  input 13
        
        "   add %%r15, %%r12                    \n"
        
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
        "   vmovups %%ymm30, 432(%%rcx)            \n"     //input 10
        
        "   vmovups %%ymm29, 240(%%rcx)         \n"     //  input 6
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 624(%%rcx)            \n"     //input 14
        
        //  
        "   vunpcklps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpcklps %%zmm13, %%zmm12, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 32(%%r12)         \n"     //  input 2
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 6
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 10
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 14
        
        "   vmovups %%xmm28, 272(%%rcx)         \n"     //  input 6
        "   vmovups %%xmm29, 464(%%rcx)         \n"     //  input 10
        "   vmovups %%xmm30, 656(%%rcx)         \n"     //  input 14
        
        "   add %%r15, %%r12                    \n"
        
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
        "   vmovups %%ymm30, 480(%%rcx)            \n"     //input 11
        
        "   vmovups %%ymm29, 288(%%rcx)         \n"     //  input 7
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 672(%%rcx)            \n"     //input 15
        
        //  
        "   vunpckhps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpckhps %%zmm13, %%zmm12, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r12)         \n"     //  input 3
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 7
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 11
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 15
        
        "   vmovups %%xmm28, 320(%%rcx)         \n"     //  input 7
        "   vmovups %%xmm29, 512(%%rcx)         \n"     //  input 11
        "   vmovups %%xmm30, 704(%%rcx)         \n"     //  input 15
        
        "   add %%r15, %%r12                    \n"
        
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
        "   vmovups %%ymm30, 528(%%rcx)            \n"     //input 12
        
        "   vmovups %%ymm29, 336(%%rcx)         \n"     //  input 8
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 720(%%rcx)            \n"     //input 16
        
        //  
        "   vunpckhps %%zmm11, %%zmm10, %%zmm0    \n"
        "   vunpckhps %%zmm13, %%zmm12, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 32(%%r12)         \n"     //  input 4
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 8
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 12
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 16
        "   add %%r15, %%r12    \n"
        "   movl $0x3f, %%edi              \n"
        "   vmovups %%xmm28, 368(%%rcx)         \n"     //  input 8
        "   vmovups %%xmm29, 560(%%rcx)         \n"     //  input 12
        "   kmovd   %%edi, %%k4            \n"
        "   vmovups %%xmm30, 752(%%rcx)         \n"     //  input 16
        
        "   vmovupd 192(%%rcx), %%zmm2%{%%k4%}             \n"  //5
        "   vmovupd 240(%%rcx), %%zmm3%{%%k4%}             \n"  //6
        "   vmovupd 288(%%rcx), %%zmm4%{%%k4%}             \n"  //7
        "   vmovupd 336(%%rcx), %%zmm5%{%%k4%}             \n"  //8
        "   vmovupd 384(%%rcx), %%zmm6%{%%k4%}             \n"  //9
        "   vmovupd 432(%%rcx), %%zmm7%{%%k4%}             \n"  //10
        "   vmovupd 480(%%rcx), %%zmm8%{%%k4%}             \n"  //11
        "   vmovupd 528(%%rcx), %%zmm9%{%%k4%}             \n"  //12
        "   vmovupd 576(%%rcx), %%zmm10%{%%k4%}             \n"  //13
        "   vmovupd 624(%%rcx), %%zmm11%{%%k4%}             \n"  //14
        "   vmovupd 672(%%rcx), %%zmm12%{%%k4%}             \n"  //15
        "   vmovupd 720(%%rcx), %%zmm13%{%%k4%}             \n"  //16
        
        
        
        
       //17 21 25 29
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm19, %%zmm18, %%zmm1    \n"
        "   vmovupd %%zmm2, (%%r12)%{%%k4%}         \n"
        "   vmovupd %%zmm3, (%%r8)%{%%k4%}         \n"
        "   vmovupd %%zmm4, (%%r9)%{%%k4%}         \n"
        "   vmovupd %%zmm5, (%%r10)%{%%k4%}         \n"
        
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        "   vunpcklps %%zmm21, %%zmm20, %%zmm27    \n"
        "   add %%r11, %%r12    \n"
        "   add %%r11, %%r8    \n"
        "   add %%r11, %%r9    \n"
        "   add %%r11, %%r10    \n"
        
        "   movl    $0xaa, %%r13d       \n"
        "   movl    $0xcc, %%eax       \n"
        "   movl    $0x33, %%edx       \n"
        
        "   vmovupd %%zmm6, (%%r12)%{%%k4%}         \n"
        "   vmovupd %%zmm7, (%%r8)%{%%k4%}         \n"
        "   vmovupd %%zmm8, (%%r9)%{%%k4%}         \n"
        "   vmovupd %%zmm9, (%%r10)%{%%k4%}         \n"
        
        
        
        "   kmovd   %%r13d, %%k1            \n"
        "   kmovd   %%eax, %%k2            \n"
        "   kmovd   %%edx, %%k3            \n"
        
        "   add %%r11, %%r12    \n"
        "   add %%r11, %%r8    \n"
        "   add %%r11, %%r9    \n"
        "   add %%r11, %%r10    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"
        
        "   vmovupd %%zmm10, (%%r12)%{%%k4%}         \n"
        "   vmovupd %%zmm11, (%%r8)%{%%k4%}         \n"
        "   vmovupd %%zmm12, (%%r9)%{%%k4%}         \n"
        "   vmovupd %%zmm13, (%%r10)%{%%k4%}         \n"
        
        "   add %%r11, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"
        
        "   vmovups %%ymm0, (%%r12)         \n"     //  input 17
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, 1152(%%rcx)            \n"     //input 25
        
        "   vmovups %%ymm29, 960(%%rcx)         \n"     //  input 21
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 1344(%%rcx)            \n"     //input 29
        
        //  
        "   vunpcklps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpcklps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r12)         \n"     //  input 17
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 21
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 25
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 29
        
        "   vmovups %%xmm28, 992(%%rcx)         \n"     //  input 21
        "   vmovups %%xmm29, 1184(%%rcx)         \n"     //  input 25
        "   vmovups %%xmm30, 1376(%%rcx)         \n"     //  input 29
        
        "   add %%r15, %%r12    \n"
        
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
        "   vmovups %%ymm30, 1200(%%rcx)            \n"     //input 26
        
        "   vmovups %%ymm29, 1008(%%rcx)         \n"     //  input 22
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 1392(%%rcx)            \n"     //input 30
        
        //  
        "   vunpcklps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpcklps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 32(%%r12)         \n"     //  input 18
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 22
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 26
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 30
        
        "   vmovups %%xmm28, 1040(%%rcx)         \n"     //  input 22
        "   vmovups %%xmm29, 1232(%%rcx)         \n"     //  input 26
        "   vmovups %%xmm30, 1424(%%rcx)         \n"     //  input 30
        
        "   add %%r15, %%r12    \n"
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
        "   vmovups %%ymm30, 1248(%%rcx)            \n"     //input 27
        
        "   vmovups %%ymm29, 1056(%%rcx)         \n"     //  input 23
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 1440(%%rcx)            \n"     //input 31
        
        //  
        "   vunpckhps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpckhps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r12)         \n"     //  input 19
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 23
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 27
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 31
        
        "   vmovups %%xmm28, 1088(%%rcx)         \n"     //  input 23
        "   vmovups %%xmm29, 1280(%%rcx)         \n"     //  input 27
        "   vmovups %%xmm30, 1472(%%rcx)         \n"     //  input 31
        
        "   add %%r15, %%r12    \n"
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
        "   vmovups %%ymm30, 1296(%%rcx)            \n"     //input 28
        
        "   vmovups %%ymm29, 1104(%%rcx)         \n"     //  input 24
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 1488(%%rcx)            \n"     //input 32
        
        //  
        "   vunpckhps %%zmm23, %%zmm22, %%zmm0    \n"
        "   vunpckhps %%zmm25, %%zmm24, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 32(%%r12)         \n"     //  input 20
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 24
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 28
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 32
        
        "   vmovups %%xmm28, 1136(%%rcx)         \n"     //  input 24
        "   vmovups %%xmm29, 1328(%%rcx)         \n"     //  input 28
        "   vmovups %%xmm30, 1520(%%rcx)         \n"     //  input 32
        
        "   add %%r15, %%r12                    \n"
        
        "   vmovupd 960(%%rcx), %%zmm2%{%%k4%}             \n"  //21
        "   vmovupd 1008(%%rcx), %%zmm3%{%%k4%}             \n"  //22
        "   vmovupd 1056(%%rcx), %%zmm4%{%%k4%}             \n"  //23
        "   vmovupd 1104(%%rcx), %%zmm5%{%%k4%}             \n"  //24
        
        "   vmovupd %%zmm2, (%%r12)%{%%k4%}         \n"
        "   vmovupd %%zmm3, (%%r8)%{%%k4%}         \n"
        "   vmovupd %%zmm4, (%%r9)%{%%k4%}         \n"
        "   vmovupd %%zmm5, (%%r10)%{%%k4%}         \n"
        
        "   add %%r11, %%r12    \n"
        "   add %%r11, %%r8    \n"
        "   add %%r11, %%r9    \n"
        "   add %%r11, %%r10    \n"
        
        "   vmovupd 1152(%%rcx), %%zmm6%{%%k4%}             \n"  //25
        "   vmovupd 1200(%%rcx), %%zmm7%{%%k4%}             \n"  //26
        "   vmovupd 1248(%%rcx), %%zmm8%{%%k4%}             \n"  //27
        "   vmovupd 1296(%%rcx), %%zmm9%{%%k4%}             \n"  //28
        
        "   vmovupd %%zmm6, (%%r12)%{%%k4%}         \n"
        "   vmovupd %%zmm7, (%%r8)%{%%k4%}         \n"
        "   vmovupd %%zmm8, (%%r9)%{%%k4%}         \n"
        "   vmovupd %%zmm9, (%%r10)%{%%k4%}         \n"
        
        "   add %%r11, %%r12    \n"
        "   add %%r11, %%r8    \n"
        "   add %%r11, %%r9    \n"
        "   add %%r11, %%r10    \n"
        
        "   vmovupd 1344(%%rcx), %%zmm10%{%%k4%}             \n"  //29
        "   vmovupd 1392(%%rcx), %%zmm11%{%%k4%}             \n"  //30
        "   vmovupd 1440(%%rcx), %%zmm12%{%%k4%}             \n"  //31
        "   vmovupd 1488(%%rcx), %%zmm13%{%%k4%}             \n"  //32
        
        "   vmovupd %%zmm10, (%%r12)%{%%k4%}         \n"
        "   vmovupd %%zmm11, (%%r8)%{%%k4%}         \n"
        "   vmovupd %%zmm12, (%%r9)%{%%k4%}         \n"
        "   vmovupd %%zmm13, (%%r10)%{%%k4%}         \n"
        
        ".endm                          \n"
        
        ".macro AVX512_N32M4_ST_Continue   \n"
        
        "   shl $2, %%r11     \n"        //inc 4
        "   shl $1, %%rbx       \n"     //inc 2
        "   shl $3, %%r14       \n"     //inc 8
        "   addq %%r11, %%r8    \n"      // output 5
        "   addq %%r11, %%r9    \n"      // output 5
        "   addq %%r14, %%r10    \n"      // output 5
        
        
        "   addq %%r15, %%r8        \n"     // output 6
        "   addq %%rbx, %%r9           \n"  // output 7
        "   subq %%r15, %%r10           \n" // output 8
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
        
        "   vmovups %%xmm28, 64(%%rcx)         \n"     //  input 5
        "   vmovups %%xmm29, 128(%%rcx)         \n"     //  input 9
        "   vmovups %%xmm30, 192(%%rcx)         \n"     //  input 13
        
        "   add %%r15, %%r12    \n"

        
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
        
        "   vmovups %%xmm28, 80(%%rcx)         \n"     //  input 6
        "   vmovups %%xmm29, 144(%%rcx)         \n"     //  input 10
        "   vmovups %%xmm30, 208(%%rcx)         \n"     //  input 14
        
        "   add %%r15, %%r12    \n"

        //3 7 11 15
        "   movl    $0xaa, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, (%%r12)         \n"     //  input 3
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 7
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 11
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 15
        
        "   vmovups %%xmm28, 96(%%rcx)         \n"     //  input 7
        "   vmovups %%xmm29, 160(%%rcx)         \n"     //  input 11
        "   vmovups %%xmm30, 224(%%rcx)         \n"     //  input 15
        "   add %%r15, %%r12    \n"

        
        
        //4 8 12 16
        "   movl    $0x55, %%r13d       \n"
        
        "   kmovd   %%r13d, %%k1            \n"
        
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        
        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, (%%r12)         \n"     //  input 4
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 8
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 12
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 16
        
        "   add %%r15, %%r12    \n"
        "   vmovups %%xmm28, 112(%%rcx)         \n"     //  input 8
        "   vmovups %%xmm29, 176(%%rcx)         \n"     //  input 12
        "   vmovups %%xmm30, 240(%%rcx)         \n"     //  input 16
        
        "   vmovups 64(%%rcx), %%xmm2             \n"  //5
        "   vmovups 80(%%rcx), %%xmm3             \n"  //6
        "   vmovups 96(%%rcx), %%xmm4             \n"  //7
        "   vmovups 112(%%rcx), %%xmm5             \n"  //8
        
        "   vmovups %%xmm2, (%%r12)         \n"
        "   vmovups %%xmm3, (%%r8)         \n"
        "   vmovups %%xmm4, (%%r9)         \n"
        "   vmovups %%xmm5, (%%r10)         \n"
        
        "   add %%r11, %%r12    \n"
        "   add %%r11, %%r8    \n"
        "   add %%r11, %%r9    \n"
        "   add %%r11, %%r10    \n"
        
        "   vmovups 128(%%rcx), %%xmm6             \n"  //9
        "   vmovups 144(%%rcx), %%xmm7             \n"  //10
        "   vmovups 160(%%rcx), %%xmm8             \n"  //11
        "   vmovups 176(%%rcx), %%xmm9             \n"  //12
        
        "   vmovups %%xmm6, (%%r12)         \n"
        "   vmovups %%xmm7, (%%r8)         \n"
        "   vmovups %%xmm8, (%%r9)         \n"
        "   vmovups %%xmm9, (%%r10)         \n"
        
        "   add %%r11, %%r12    \n"
        "   add %%r11, %%r8    \n"
        "   add %%r11, %%r9    \n"
        "   add %%r11, %%r10    \n"
        
        "   vmovups 192(%%rcx), %%xmm10             \n"  //13
        "   vmovups 208(%%rcx), %%xmm11             \n"  //14
        "   vmovups 224(%%rcx), %%xmm12             \n"  //15
        "   vmovups 240(%%rcx), %%xmm13             \n"  //16
        
        "   vmovups %%xmm10, (%%r12)         \n"
        "   vmovups %%xmm11, (%%r8)         \n"
        "   vmovups %%xmm12, (%%r9)         \n"
        "   vmovups %%xmm13, (%%r10)         \n"
        
        "   add %%r11, %%r12    \n"
        "   add %%r14, %%r8    \n"
        "   add %%r14, %%r9    \n"
        "   add %%r14, %%r10    \n"
        
        
        //17 21 25 29
        "   movl    $0xaa, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, (%%r12)         \n"     //  input 17
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 21
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 25
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 29
        
        "   vmovups %%xmm28, 320(%%rcx)         \n"     //  input 21
        "   vmovups %%xmm29, 384(%%rcx)         \n"     //  input 25
        "   vmovups %%xmm30, 448(%%rcx)         \n"     //  input 29

        
        "   add %%r15, %%r12    \n"

        //18 22 26 30
        "   movl    $0x55, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpcklps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpcklps %%zmm17, %%zmm16, %%zmm26    \n"
        
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, (%%r12)         \n"     //  input 18
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 22
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 26
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 30
        
        "   vmovups %%xmm28, 336(%%rcx)         \n"     //  input 22
        "   vmovups %%xmm29, 400(%%rcx)         \n"     //  input 26
        "   vmovups %%xmm30, 464(%%rcx)         \n"     //  input 30
        
        
        "   add %%r15, %%r12    \n"

        
        
        //19 23 27 31
        "   movl    $0xaa, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, (%%r12)         \n"     //  input 19
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"    //  input 23
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"    //  input 27
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"    //  input 21
        
        "   vmovups %%xmm28, 352(%%rcx)         \n"     //  input 23
        "   vmovups %%xmm29, 416(%%rcx)         \n"     //  input 27
        "   vmovups %%xmm30, 480(%%rcx)         \n"     //  input 31

        
        "   add %%r15, %%r12    \n"

        //20 24 28 32
        "   movl    $0x55, %%r13d       \n"
        "   kmovd   %%r13d, %%k1            \n"
        "   vunpckhps %%zmm15, %%zmm14, %%zmm0    \n"
        "   vunpckhps %%zmm17, %%zmm16, %%zmm26    \n"
        
        "   vpermq   $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, (%%r12)         \n"     //  input 20
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"    //  input 24
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"    //  input 28
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"    //  input 32
        
        "   vmovups %%xmm28, 368(%%rcx)         \n"     //  input 24
        "   vmovups %%xmm29, 432(%%rcx)         \n"     //  input 28
        "   vmovups %%xmm30, 496(%%rcx)         \n"     //  input 32
        
        "   add %%r15, %%r12                    \n"
        
        "   vmovups 320(%%rcx), %%xmm2             \n"  //21
        "   vmovups 336(%%rcx), %%xmm3             \n"  //22
        "   vmovups 352(%%rcx), %%xmm4             \n"  //23
        "   vmovups 368(%%rcx), %%xmm5             \n"  //24
        
        "   vmovups %%xmm2, (%%r12)         \n"
        "   vmovups %%xmm3, (%%r8)         \n"
        "   vmovups %%xmm4, (%%r9)         \n"
        "   vmovups %%xmm5, (%%r10)         \n"
        
        "   add %%r11, %%r12    \n"
        "   add %%r11, %%r8    \n"
        "   add %%r11, %%r9    \n"
        "   add %%r11, %%r10    \n"
        
        "   vmovups 384(%%rcx), %%xmm6             \n"  //25
        "   vmovups 400(%%rcx), %%xmm7             \n"  //26
        "   vmovups 416(%%rcx), %%xmm8             \n"  //27
        "   vmovups 432(%%rcx), %%xmm9             \n"  //28
        
        "   vmovups %%xmm6, (%%r12)         \n"
        "   vmovups %%xmm7, (%%r8)         \n"
        "   vmovups %%xmm8, (%%r9)         \n"
        "   vmovups %%xmm9, (%%r10)         \n"
        
        "   add %%r11, %%r12    \n"
        "   add %%r11, %%r8    \n"
        "   add %%r11, %%r9    \n"
        "   add %%r11, %%r10    \n"
        
        "   vmovups 448(%%rcx), %%xmm10             \n"  //29
        "   vmovups 464(%%rcx), %%xmm11             \n"  //30
        "   vmovups 480(%%rcx), %%xmm12             \n"  //31
        "   vmovups 496(%%rcx), %%xmm13             \n"  //32
        
        "   vmovups %%xmm10, (%%r12)         \n"
        "   vmovups %%xmm11, (%%r8)         \n"
        "   vmovups %%xmm12, (%%r9)         \n"
        "   vmovups %%xmm13, (%%r10)         \n"
        
        
        ".endm      \n"
        
        "CONV_N32M12_ENTRANCE:            \n"
        "   mov %[img_start_0], %%r10    \n"
        "   mov %[img_start_0], %%rdi    \n"

        "   mov %[kernal], %%r11 \n"

        "   movl %[ic_count], %%r13d  \n"           //
        "   movl %[stride_in_size], %%r14d    \n"
        "   movl %[W_size], %%r15d    \n"

        "   movl %[hh_tag], %%eax   \n"
        "   movl %[ww_tag], %%ecx        \n"


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
        
        
        "   movl $7, %%r9d       \n"
        "   cmp $2, %%eax                  \n"
        "   je CMP_W            \n"
        
        "   movl $4, %%r9d       \n"
        "   cmp $0, %%eax                  \n"
        "   je  CMP_W           \n"
        
        "   movl $6, %%r9d       \n"
        "   cmp $1, %%eax                  \n"
        "   je  CMP_W           \n"
        "   movl $5, %%r9d       \n"
        
        "CMP_W:                 \n"
        "   cmp $1, %%ecx       \n"
        "   je CONV_7x7_W1_Main       \n"
        "   cmp $0, %%ecx               \n"
        "   je CONV_7x7_W0_Main             \n"
        "   jmp CONV_7x7_W2_Main            \n"
        

        
        "CONV_7x7_W0_Pre:              \n"
        "   addq $128,%%r11       \n"
        "   addq %%r14, %%r10   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%r10)                 \n"
        
        "CONV_7x7_W0_Main:         \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x10_K7                      \n"
        "   subl $1, %%r13d       \n"
        "   jne     CONV_7x7_W0_Pre       \n"
        "   subl $1, %%r9d          \n"
        "   je      CONV_7x7_12_STORE     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     CONV_7x7_W0_Main            \n"
        
        "CONV_7x7_W1_Pre:              \n"
        "   addq $128,%%r11       \n"
        "   addq %%r14, %%r10   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%r10)                 \n"
        
        "CONV_7x7_W1_Main:         \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x12_K7                      \n"
        "   subl $1, %%r13d       \n"
        "   jne     CONV_7x7_W1_Pre       \n"
        "   subl $1, %%r9d          \n"
        "   je      CONV_7x7_12_STORE     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     CONV_7x7_W1_Main            \n"
        
        "CONV_7x7_W2_Pre:              \n"
        "   addq $128,%%r11       \n"
        "   addq %%r14, %%r10   \n"
        "   prefetcht0 (%%r11)                 \n"
        "   prefetcht0 (%%r10)                 \n"
        
        "CONV_7x7_W2_Main:         \n"
        "   vmovups (%%r11), %%zmm0        \n"          
        "   vmovups 64(%%r11), %%zmm26        \n"
        "   KERNEL32x4_K7                      \n"
        "   subl $1, %%r13d       \n"
        "   jne     CONV_7x7_W2_Pre       \n"
        "   subl $1, %%r9d          \n"
        "   je      CONV_7x7_4_STORE     \n"
        "   addq %%r15, %%rdi        \n"
        "   addq $128,%%r11       \n"
        "   movl %[ic_count], %%r13d  \n"
        "   mov %%rdi, %%r10         \n"
        "   jmp     CONV_7x7_W2_Main            \n"
        
        
        //  ADD or ST
        "CONV_7x7_12_STORE:       \n"
        
        "   mov %[out], %%r12    \n"
        "   mov %[out], %%r8    \n"
        "   mov %[out], %%r9    \n"
        "   mov %[out], %%r10    \n"
        "   mov %[out_buffer], %%rcx        \n"
        "   movl %[stride_out_size], %%ebx   \n"
        "   movl %[stride_out_size], %%r15d   \n"
        "   movl %[stride_out_size], %%r14d   \n"
        "   movl %[stride_out_size], %%r11d   \n"
        //"   AVX512_N32M12_ST_Continue     \n"
        "   AVX512_N32M12_ST     \n"
        
        "   jmp CONV_N32M12_EXIT        \n"
        
        "CONV_7x7_4_STORE:       \n"
        "   mov %[out], %%r12    \n"
        "   mov %[out], %%r8    \n"
        "   mov %[out], %%r9    \n"
        "   mov %[out], %%r10    \n"
        "   mov %[out_buffer], %%rcx        \n"
        "   movl %[stride_out_size], %%ebx   \n"
        "   movl %[stride_out_size], %%r15d   \n"
        "   movl %[stride_out_size], %%r14d   \n"
        "   movl %[stride_out_size], %%r11d   \n"
        //"   AVX512_N32M4_ST_Continue     \n"
        "   AVX512_N32M4_ST     \n"
        
        "CONV_N32M12_EXIT:            \n"
        
    :
    
    
    :
        [img_start_0]                 "m"     (img_start_0),
        [kernal]                      "m"     (kernal),
        [out]                       "m"     (out),
        [out_buffer]                "m"     (out_buffer),
        [ic_count]                  "m"     (ic_count),
        [stride_in_size]            "m"     (stride_in_size),
        [stride_out_size]            "m"     (stride_out_size),
        [hh_tag]                         "m"     (hh_tag),
        [ww_tag]                         "m"     (ww_tag),
        [W_size]                         "m"     (W_size)

    
    
    :
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi","r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory","k0","k1","k2","k3","k4"
    
    );
    
    

}

void avx512_Kernel_n32m12_7x7s2(float *filter, float *input, float *output, float *output_buffer, int K, int Kb, int H, int R, int Cb)
{

    int ww, hh, Hb, Wb, kk;
    int W = H;
    int S = R;
    int out_H = (H - R + 2*3)/2 + 1;;
    int out_W = out_H;
    int stride_in = H * W;
    int stride_out = out_H * out_W;

    float *input0, *input1, *input2, *output0, *filter0, *output1, *output2;
    
    int stride_in_size = stride_in * sizeof(float);
    
    int stride_out_size = stride_out * sizeof(float);
    
    int hh_tag, ww_tag;
    
    Wb = 12;
    
    int last_H = out_H - 1;
    int last_W = out_W - 12;
    //h = 0
    hh = 0;
    ww = 0;
    hh_tag = 0;
    ww_tag = 0;
    output0 = output;
    input0 = input;

    output1 = output0;
    input1 = input0;
    int h0_size = 32*S*Cb*3;
    int h1_size = 32*S*Cb;
    for(kk=0; kk<Kb; kk+=32){
        
        filter0 = filter + kk*49*Cb;
        
        filter0 += h0_size;
        output2 = output1 + kk * stride_out;

        direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size, hh_tag, ww_tag, W<<2);
    }
    
    ww_tag = 1;
    
    input0 -= 3;
    
    for(ww = 12; ww < last_W; ww = ww + Wb)
    {
        output1 = output0 + ww;

        input1 = input0 + 2* ww;

        for(kk=0; kk<Kb; kk+=32){
            
            filter0 = filter + kk*49*Cb;
            filter0 += h0_size;
            output2 = output1 + kk * stride_out;
            direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size, hh_tag, ww_tag, W<<2);
        }
    }
    
    ww_tag = 2;
    output1 = output0 + ww;

    input1 = input0 + 2* ww;

    for(kk=0; kk<Kb; kk+=32){
        
        filter0 = filter + kk*49*Cb;
        filter0 += h0_size;
        output2 = output1 + kk * stride_out;
        direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size,hh_tag, ww_tag, W<<2);
    }
    
    //h = 1
    hh = 1;
    ww = 0;
    hh_tag = 1;
    ww_tag = 0;
    output0 = output + out_W;
    input0 = input;
    
    output1 = output0;

    input1 = input0;

    for(kk=0; kk<Kb; kk+=32){
        
        filter0 = filter + kk*49*Cb;
        filter0 += h1_size;
        output2 = output1 + kk * stride_out;
        direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size,hh_tag, ww_tag, W<<2);
    }
    
    ww_tag = 1;
    
    input0 -= 3;
    
    for(ww = 12; ww < last_W; ww = ww + Wb)
    {
        output1 = output0 + ww;

        input1 = input0 + 2* ww;

        for(kk=0; kk<Kb; kk+=32){
            
            filter0 = filter + kk*49*Cb;
            
            filter0 += h1_size;
            
            output2 = output1 + kk * stride_out;
            direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size,hh_tag, ww_tag, W<<2);
        }
    }
    
    ww_tag = 2;
    output1 = output0 + ww;

    input1 = input0 + 2* ww;

    for(kk=0; kk<Kb; kk+=32){
        
        filter0 = filter + kk*49*Cb;
        
        filter0 += h1_size;
        
        output2 = output1 + kk * stride_out;

        direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size,hh_tag, ww_tag, W<<2);
    }
    // h < (out_H -1)
    hh_tag = 2;
    output0 = output + out_W;

    for(hh=2; hh<last_H; hh++){
        ww = 0;
        ww_tag = 0;
        output0 += out_W;
        input0 = input + (2 * hh - 3)* W;
        
        output1 = output0;

        input1 = input0;

        for(kk=0; kk<Kb; kk+=32){
            
            filter0 = filter + kk*49*Cb;
            output2 = output1 + kk * stride_out;
            direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size,hh_tag, ww_tag, W<<2);
        }
        
        ww_tag = 1;
        
        input0 -= 3;
        
        for(ww = 12; ww < last_W; ww = ww + Wb)
        {
            output1 = output0 + ww;

            input1 = input0 + 2* ww;

            for(kk=0; kk<Kb; kk+=32){
                
                filter0 = filter + kk*49*Cb;
                
                output2 = output1 + kk * stride_out;
                direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size,hh_tag, ww_tag, W<<2);
            }
        }
        
        ww_tag = 2;
        output1 = output0 + ww;

        input1 = input0 + 2* ww;

        for(kk=0; kk<Kb; kk+=32){
            
            filter0 = filter + kk*49*Cb;
            
            output2 = output1 + kk * stride_out;
            direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size,hh_tag, ww_tag, W<<2);
        }
    }
    // h = last_H
    hh_tag = 3;
    output0 = output + last_H * out_W;
    ww = 0;
    ww_tag = 0;
    input0 = input + (2 * last_H - 3)* W;
    
    output1 = output0;

    input1 = input0;

    for(kk=0; kk<Kb; kk+=32){
        
        filter0 = filter + kk*49*Cb;
        
        output2 = output1 + kk * stride_out;
        direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size,hh_tag, ww_tag, W<<2);
    }
    
    ww_tag = 1;
    
    input0 -= 3;
    
    for(ww = 12; ww < last_W; ww = ww + Wb)
    {
        output1 = output0 + ww;

        input1 = input0 + 2* ww;

        for(kk=0; kk<Kb; kk+=32){
            
            filter0 = filter + kk*49*Cb;
            
            output2 = output1 + kk * stride_out;
            direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size,hh_tag, ww_tag, W<<2);
        }
    }
    
    ww_tag = 2;
    output1 = output0 + ww;

    input1 = input0 + 2* ww;

    for(kk=0; kk<Kb; kk+=32){
        
        filter0 = filter + kk*49*Cb;
        
        output2 = output1 + kk * stride_out;
        direct_7x7_N32M12_AVX512(input1, filter0, output2, output_buffer, Cb, stride_in_size, stride_out_size,hh_tag, ww_tag, W<<2);
    }

}


void avx512_n32m12_7x7s2(float* filter, float* input, float* output, int K, int C, int H, int R)
{
    int W = H;
    int S = R;

    int out_H = (H - R + 2*3)/2 + 1;
    int out_W = out_H;
    
    int stride_in = H * W;
    int stride_out = out_W * out_H;
    //float *ptr = (float *)_mm_malloc();
    //posix_memalign(&ptr, 64, NUM * CONV_C_7X7 * 14 * 14 * sizeof( float ));
    //float *input_c = (float *)ptr;
    float *output_buffer = ( float * ) malloc( NUM * 12 * 32 * sizeof( float ));

    #pragma omp parallel num_threads(NUM)
    {

        int i, ii, jj, kk, cc, hh, h, ww;
        int Cb, Kb, Wb, Hb;
        int id = omp_get_thread_num();

        for(cc = 0; cc < C; cc = cc + Cb)
        {
            Cb = CONV_C_7X7;
            if (C - cc <= CONV_C_7X7)
            {
                Cb = C - cc;
            }

            for(ii = 0; ii < K; ii = ii + Kb)
            {
                Kb = CONV_K_7X7;

                if (K - ii < CONV_K_7X7)
                {
                    Kb = K - ii;
                }
                
                float *filter0 = filter + ii * 49 * Cb + 49 * cc * K;
                float *input0 = input + id * C * stride_in + cc * stride_in;
                float *output0 = output + id * K * stride_out + ii * stride_out;
                float *output_buffer0 = output_buffer + id * 12 * 32;
                avx512_Kernel_n32m12_7x7s2(filter0, input0, output0, output_buffer0, K, Kb, H, R, Cb);
            }
        }
    
        
    }
    //free(input_c);
    free(output_buffer);
}
