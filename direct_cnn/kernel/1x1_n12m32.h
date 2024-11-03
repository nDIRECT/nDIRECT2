#include <malloc.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
void direct_1x1_N12M32_AVX512_pack(float *output, float *trans_filter, float *input, int Kb, int LEN_HWb, int Cb, int input_HW_size, float *input_buffer, int cc, int EDGE_Kb){
    asm volatile(


        ".macro KERNEL12x32_PACK_K1                         \n"

        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm9          \n"
        "   vmovups         %%zmm4, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm11         \n"
        "   add             %%r8, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm13         \n"
        "   prefetcht2      128(%%rax)                      \n"


        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm15         \n"
        "   vmovups         %%zmm5, 64(%%r14)                 \n"
        //"   prefetcht2      192(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm17         \n"
        "   vmovups         (%%rax), %%zmm6                     \n"
        "   prefetcht2      192(%%rax)                      \n"
        
        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm19         \n"
        "   add             $128, %%r14                     \n"
        
        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm21         \n"
        "   vmovups         64(%%rax), %%zmm7                     \n"

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
        "   add             $48, %%rbx                                 \n"
        
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm28         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm29         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        
        
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm31         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"
        

        ".endm                                              \n"


        ".macro KERNEL12x32_PACK_K2                         \n"

        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        "   vmovups         %%zmm6, (%%r14)                 \n"


        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"
        "   add             %%r8, %%rax                      \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"
        "   prefetcht2      128(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"
        "   vmovups         %%zmm7, 64(%%r14)                 \n"
        //"   prefetcht1      192(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm17         \n"
        "   vmovups         (%%rax), %%zmm4                     \n"
        "   prefetcht2      192(%%rax)                      \n"
        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm19         \n"
        "   add             $128, %%r14                     \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm21         \n"
        "   vmovups         64(%%rax), %%zmm5                     \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm23         \n"
        "   prefetcht0      48(%%rbx)                      \n"
        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm25         \n"
        //"   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm27         \n"
        "   add             $48, %%rbx                                 \n"


        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm29         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm31         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"


        ".endm                                              \n"



        ".macro KERNEL12x32_PACK_END_K                      \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"

        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        "   vmovups         %%zmm6, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"
        //"   prefetcht2      128(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm15         \n"
        "   vmovups         %%zmm7, 64(%%r14)                 \n"
        //"   prefetcht2      192(%%rax)                      \n"

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
        "   add             $48, %%rbx                                 \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm29         \n"


        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm31         \n"

        ".endm                                              \n"



        ".macro KERNEL12x32_K1                              \n"
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

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%zmm1, %%zmm5, %%zmm19         \n"
        "   vmovups         64(%%rax), %%zmm7                     \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vfmadd231ps     %%zmm2, %%zmm5, %%zmm21         \n"
        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm5, %%zmm23         \n"
        //"   prefetcht0      48(%%rbx)                      \n"

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


        ".macro KERNEL12x32_K2                              \n"
        //"   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        //"   prefetcht0      128(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%zmm1, %%zmm7, %%zmm11         \n"
        "   add             $128, %%rax                      \n"
        
        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%zmm2, %%zmm7, %%zmm13         \n"
        "   prefetcht0      64(%%rax)                      \n"

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
        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%zmm3, %%zmm7, %%zmm23         \n"
        //"   prefetcht0      48(%%rbx)                      \n"

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



        ".macro KERNEL12x32_END_K                           \n"


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


        ".macro ST_12x32   \n"
        
        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)              \n"
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)              \n"

        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%zmm19, 64(%%r11)              \n"
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%zmm21, 64(%%r12)              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%zmm23, 64(%%r13)              \n" 
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%zmm24, (%%r10)                \n"
        "   vmovups         %%zmm25, 64(%%r10)              \n"
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%zmm26, (%%r11)                \n"
        "   vmovups         %%zmm27, 64(%%r11)              \n"
        "   vmovups         %%zmm28, (%%r12)                \n"
        "   vmovups         %%zmm29, 64(%%r12)              \n"
        "   vmovups         %%zmm30, (%%r13)                \n"
        "   vmovups         %%zmm31, 64(%%r13)              \n"

        
        ".endm      \n"

        ".macro ADD_C_12x32   \n"
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"
		"   vmovups 		64(%%r10), %%zmm1        		\n"
		"	vaddps 			%%zmm1, %%zmm9, %%zmm9			\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"
		"   vmovups 		64(%%r11), %%zmm3        		\n"
		"	vaddps 			%%zmm3, %%zmm11, %%zmm11		\n"
		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"
		"   vmovups 		64(%%r12), %%zmm5        		\n"
		"	vaddps 			%%zmm5, %%zmm13, %%zmm13		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"
		"   vmovups 		64(%%r13), %%zmm7        		\n"
		"	vaddps 			%%zmm7, %%zmm15, %%zmm15		\n"

        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)              \n"

		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm16, %%zmm16		\n"
		"   vmovups 		64(%%r10), %%zmm1        		\n"
		"	vaddps 			%%zmm1, %%zmm17, %%zmm17		\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm18, %%zmm18		\n"
		"   vmovups 		64(%%r11), %%zmm3        		\n"
		"	vaddps 			%%zmm3, %%zmm19, %%zmm19		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm20, %%zmm20		\n"
		"   vmovups 		64(%%r12), %%zmm5        		\n"
		"	vaddps 			%%zmm5, %%zmm21, %%zmm21		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm22, %%zmm22		\n"
		"   vmovups 		64(%%r13), %%zmm7        		\n"
		"	vaddps 			%%zmm7, %%zmm23, %%zmm23		\n"

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%zmm19, 64(%%r11)              \n"
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%zmm21, 64(%%r12)              \n"
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%zmm23, 64(%%r13)              \n" 


		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm24, %%zmm24		\n"
		"   vmovups 		64(%%r10), %%zmm1        		\n"
		"	vaddps 			%%zmm1, %%zmm25, %%zmm25		\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm26, %%zmm26		\n"
		"   vmovups 		64(%%r11), %%zmm3        		\n"
		"	vaddps 			%%zmm3, %%zmm27, %%zmm27		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm28, %%zmm28		\n"
		"   vmovups 		64(%%r12), %%zmm5        		\n"
		"	vaddps 			%%zmm5, %%zmm29, %%zmm29		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm30, %%zmm30		\n"
		"   vmovups 		64(%%r13), %%zmm7        		\n"
		"	vaddps 			%%zmm7, %%zmm31, %%zmm31		\n"

        "   vmovups         %%zmm24, (%%r10)                \n"
        "   vmovups         %%zmm25, 64(%%r10)              \n"
        "   vmovups         %%zmm26, (%%r11)                \n"
        "   vmovups         %%zmm27, 64(%%r11)              \n"
        "   vmovups         %%zmm28, (%%r12)                \n"
        "   vmovups         %%zmm29, 64(%%r12)              \n"
        "   vmovups         %%zmm30, (%%r13)                \n"
        "   vmovups         %%zmm31, 64(%%r13)              \n"
        
        ".endm      \n"

        ".macro KERNEL8x32_K1                              \n"
        "   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm9          \n"

        "   prefetcht0      192(%%rax)                      \n"
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
        "   prefetcht0      32(%%rbx)                      \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm17         \n"
        //"   prefetcht0      48(%%rbx)                      \n"

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


        ".macro KERNEL8x32_K2                              \n"
        "   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm7, %%zmm9          \n"
        
        "   prefetcht0      192(%%rax)                      \n"
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



        ".macro KERNEL8x32_END_K                           \n"


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


        ".macro ST_8x32   \n"
        
        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)              \n"
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)              \n"

        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%zmm19, 64(%%r11)              \n"
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%zmm21, 64(%%r12)              \n"
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%zmm23, 64(%%r13)              \n" 




        
        ".endm      \n"

        ".macro ADD_C_8x32   \n"
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"
		"   vmovups 		64(%%r10), %%zmm1        		\n"
		"	vaddps 			%%zmm1, %%zmm9, %%zmm9			\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"
		"   vmovups 		64(%%r11), %%zmm3        		\n"
		"	vaddps 			%%zmm3, %%zmm11, %%zmm11		\n"
		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"
		"   vmovups 		64(%%r12), %%zmm5        		\n"
		"	vaddps 			%%zmm5, %%zmm13, %%zmm13		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"
		"   vmovups 		64(%%r13), %%zmm7        		\n"
		"	vaddps 			%%zmm7, %%zmm15, %%zmm15		\n"

        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)              \n"

		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm16, %%zmm16		\n"
		"   vmovups 		64(%%r10), %%zmm1        		\n"
		"	vaddps 			%%zmm1, %%zmm17, %%zmm17		\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm18, %%zmm18		\n"
		"   vmovups 		64(%%r11), %%zmm3        		\n"
		"	vaddps 			%%zmm3, %%zmm19, %%zmm19		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm20, %%zmm20		\n"
		"   vmovups 		64(%%r12), %%zmm5        		\n"
		"	vaddps 			%%zmm5, %%zmm21, %%zmm21		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm22, %%zmm22		\n"
		"   vmovups 		64(%%r13), %%zmm7        		\n"
		"	vaddps 			%%zmm7, %%zmm23, %%zmm23		\n"

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)              \n"
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%zmm19, 64(%%r11)              \n"
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%zmm21, 64(%%r12)              \n"
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%zmm23, 64(%%r13)              \n" 
        
        ".endm      \n"

        ".macro KERNEL4x32_K1                              \n"
        "   prefetcht0      128(%%rax)                      \n"
        "   prefetcht0      192(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%zmm0, %%zmm5, %%zmm9          \n"
        "   add             $128, %%rax                      \n"
        "   prefetcht0      16(%%rbx)                      \n"
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


        ".macro KERNEL4x32_K2                              \n"
        "   prefetcht0      128(%%rax)                      \n"
        "   prefetcht0      192(%%rax)                      \n"
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



        ".macro KERNEL4x32_END_K                           \n"


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


        ".macro ST_4x32   \n"
        
        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)              \n"





        
        ".endm      \n"

        ".macro ADD_C_4x32   \n"
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"
		"   vmovups 		64(%%r10), %%zmm1        		\n"
		"	vaddps 			%%zmm1, %%zmm9, %%zmm9			\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"
		"   vmovups 		64(%%r11), %%zmm3        		\n"
		"	vaddps 			%%zmm3, %%zmm11, %%zmm11		\n"
		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"
		"   vmovups 		64(%%r12), %%zmm5        		\n"
		"	vaddps 			%%zmm5, %%zmm13, %%zmm13		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"
		"   vmovups 		64(%%r13), %%zmm7        		\n"
		"	vaddps 			%%zmm7, %%zmm15, %%zmm15		\n"

        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%zmm9, 64(%%r10)               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%zmm11, 64(%%r11)              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%zmm13, 64(%%r12)              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%zmm15, 64(%%r13)              \n"
        
        ".endm      \n"

        "CONV_KERNEL12x32:                                \n"

        "   mov     %[output], %%rcx                             \n"
        "   mov     %[trans_filter], %%rbx                             \n"
        "   mov     %[input], %%r9                             \n"

        //"   prefetcht0      (%%rax)                         \n"
        "   movl     %[LEN_HWb], %%esi                   \n"
        "   movl     %[Cb], %%edx                             \n"  // Cb
        "   movl     %[input_HW_size], %%r8d                             \n" 
        "   mov         %[input_buffer], %%r14                            \n"   //input_b
        "   movl        %[Kb], %%edi                             \n"

        //------------------- loop body
        "BEGIN_PACK:                                        \n"


        //"   shl $2, %%r8                \n"
        "   mov     %%rcx, %%r10                            \n"  // C0
        "   mov     %%r9, %%rax                         \n"
        "   prefetcht0      (%%rax)                         \n" 
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   mov     %%r9, %%rax                         \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   prefetcht0      (%%rbx)                         \n" 
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   prefetcht0      64(%%rax)                         \n"
        
        //"   shr $2, %%r8                \n"
        

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
        "   ADD_C_12x32                                     \n"
        "   jmp PACK_Kb_END                 \n"
        "PACK_ST_C:                                            \n"
        "   ST_12x32                                      \n"

        "PACK_Kb_END:               \n"
        "   sub    $12, %%rdi       \n"
        "   je     EDGE_CASE                              \n"

        "BEGIN_M:                                           \n"
        
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
        "BEGIN_K:                                           \n"

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
        "BEGIN_SAVE:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C                                  \n"
        "   ADD_C_12x32                                     \n"
        "   jmp Kb_END                  \n"
        "ST_C:                                            \n"
        "   ST_12x32                                      \n"
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
        
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
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
        "   vpxorq      %%zmm24, %%zmm24, %%zmm24           \n"
        "   vpxorq      %%zmm25, %%zmm25, %%zmm25           \n" 
        "   vpxorq      %%zmm26, %%zmm26, %%zmm26           \n" 
        "   vpxorq      %%zmm27, %%zmm27, %%zmm27           \n"
        "   vpxorq      %%zmm28, %%zmm28, %%zmm28           \n"
        "   vpxorq      %%zmm29, %%zmm29, %%zmm29           \n" 
        "   vpxorq      %%zmm30, %%zmm30, %%zmm30           \n" 
        "   vpxorq      %%zmm31, %%zmm31, %%zmm31           \n"   
        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_8:                                            \n"


        "   KERNEL8x32_K1                                  \n"
        "   KERNEL8x32_K2                                  \n"
        "   KERNEL8x32_K1                                  \n"
        "   KERNEL8x32_K2                                  \n"
        "   KERNEL8x32_K1                                  \n"
        "   KERNEL8x32_K2                                  \n"
        "   KERNEL8x32_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_8                                  \n"
        "   KERNEL8x32_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_8                                  \n"
        "EDGE_K_8:                                            \n"
        
        "   KERNEL8x32_END_K                               \n"

        "BEGIN_SAVE_8:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_8                                  \n"
        "   ADD_C_8x32                                     \n"
        "   jmp END_M                  \n"
        "ST_C_8:                                            \n"
        "   ST_8x32                                      \n"
        "   jmp     END_M                               \n"

        "EDGE_4:                    \n"
        
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
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

        "MAIN_K_4:                                            \n"


        "   KERNEL4x32_K1                                  \n"
        "   KERNEL4x32_K2                                  \n"
        "   KERNEL4x32_K1                                  \n"
        "   KERNEL4x32_K2                                  \n"
        "   KERNEL4x32_K1                                  \n"
        "   KERNEL4x32_K2                                  \n"
        "   KERNEL4x32_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_4                                  \n"
        "   KERNEL4x32_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_4                                  \n"
        "EDGE_K_4:                                            \n"
        
        "   KERNEL4x32_END_K                               \n"

        "BEGIN_SAVE_4:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_4                                  \n"
        "   ADD_C_4x32                                     \n"
        "   jmp END_M                  \n"
        "ST_C_4:                                            \n"
        "   ST_4x32                                      \n"

        "END_M:                                             \n"
        
        "   add $128, %%rcx               \n"
        "   add $128, %%r9               \n"
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

void direct_1x1_N12M17_AVX512_pack(float *output, float *trans_filter, float *input, int Kb, int Cb, int input_HW_size, float *input_buffer, int cc, int EDGE_Kb){
    asm volatile(


        ".macro KERNEL12x17_PACK_K1                         \n"

        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm5, %%xmm9          \n"
        "   vmovups         %%zmm4, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm5, %%xmm11         \n"
        "   add             %%r8, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm5, %%xmm13         \n"
        "   prefetcht2      128(%%rax)                      \n"


        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm5, %%xmm15         \n"
        "   vmovups         %%xmm5, 64(%%r14)                 \n"
        "   prefetcht2      192(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%xmm0, %%xmm5, %%xmm17         \n"
        "   vmovups         (%%rax), %%zmm6                     \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%xmm1, %%xmm5, %%xmm19         \n"
        "   add             $128, %%r14                     \n"
        
        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vfmadd231ps     %%xmm2, %%xmm5, %%xmm21         \n"
        "   vmovups         64(%%rax), %%xmm7%{%%k1%}                     \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"
        "   vfmadd231ps     %%xmm3, %%xmm5, %%xmm23         \n"
        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm24         \n"
        "   vfmadd231ps     %%xmm0, %%xmm5, %%xmm25         \n"


        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm26         \n"
        "   vfmadd231ps     %%xmm1, %%xmm5, %%xmm27         \n"
        "   add             $48, %%rbx                                 \n"
        
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm28         \n"
        "   vfmadd231ps     %%xmm2, %%xmm5, %%xmm29         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        
        
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm30         \n"
        "   vfmadd231ps     %%xmm3, %%xmm5, %%xmm31         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"
        

        ".endm                                              \n"


        ".macro KERNEL12x17_PACK_K2                         \n"

        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm9          \n"
        "   vmovups         %%zmm6, (%%r14)                 \n"


        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm11         \n"
        "   add             %%r8, %%rax                      \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm13         \n"
        "   prefetcht2      128(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm15         \n"
        "   vmovups         %%xmm7, 64(%%r14)                 \n"
        "   prefetcht2      192(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm17         \n"
        "   vmovups         (%%rax), %%zmm4                     \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm19         \n"
        "   add             $128, %%r14                     \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm21         \n"
        "   vmovups         64(%%rax), %%xmm5%{%%k1%}                   \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm23         \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm25         \n"
        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm27         \n"
        "   add             $48, %%rbx                                 \n"


        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm29         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm31         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"


        ".endm                                              \n"



        ".macro KERNEL12x17_PACK_END_K                      \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"

        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm9          \n"
        "   vmovups         %%zmm6, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm11         \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm13         \n"
        "   prefetcht2      128(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm15         \n"
        "   vmovups         %%xmm7, 64(%%r14)                 \n"
        "   prefetcht2      192(%%rax)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm17         \n"

        
        
        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm19         \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm21         \n"

        
        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm23         \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm25         \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm27         \n"
        "   add             $48, %%rbx                                 \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm29         \n"


        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm31         \n"

        ".endm                                              \n"



        ".macro KERNEL12x17_K1                              \n"
        "   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm5, %%xmm9          \n"
        "   prefetcht0      192(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm5, %%xmm11         \n"
        "   add             $128, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm5, %%xmm13         \n"
        //"   prefetcht0      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm5, %%xmm15         \n"
        "   vmovups         (%%rax), %%zmm6                     \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%xmm0, %%xmm5, %%xmm17         \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%xmm1, %%xmm5, %%xmm19         \n"
        "   vmovups         64(%%rax), %%xmm7                     \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vfmadd231ps     %%xmm2, %%xmm5, %%xmm21         \n"


        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"
        "   vfmadd231ps     %%xmm3, %%xmm5, %%xmm23         \n"
        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm24         \n"
        "   vfmadd231ps     %%xmm0, %%xmm5, %%xmm25         \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm26         \n"
        "   vfmadd231ps     %%xmm1, %%xmm5, %%xmm27         \n"
        "   add             $48, %%rbx                      \n"
        
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm28         \n"
        "   vfmadd231ps     %%xmm2, %%xmm5, %%xmm29         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm30         \n"
        "   vfmadd231ps     %%xmm3, %%xmm5, %%xmm31         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL12x17_K2                              \n"
        "   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm9          \n"
        "   prefetcht0      192(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm11         \n"
        "   add             $128, %%rax                      \n"
        
        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm13         \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm15         \n"
        "   vmovups         (%%rax), %%zmm4                     \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm17         \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm19         \n"
        "   vmovups         64(%%rax), %%xmm5                     \n"
        
        
        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm21         \n"


        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm23         \n"
        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm25         \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm27         \n"
        "   add             $48, %%rbx                      \n"
        
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm29         \n"

        "   vbroadcastss    (%%rbx), %%zmm0                \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm31         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"

        ".endm                                              \n"



        ".macro KERNEL12x17_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm9          \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm11         \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm13         \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm15         \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm17         \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm19         \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm21         \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm23         \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm25         \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm27         \n"
        "   add             $48, %%rbx                      \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm29         \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm31         \n"
        

        ".endm                                              \n"


        ".macro ST_12x17   \n"
        //"   movl $0x1, %%r8d                   \n"
        //"   kmovd %%r8d, %%k1           \n"
        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%xmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%xmm11, 64(%%r11)%{%%k1%}              \n"
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%xmm13, 64(%%r12)%{%%k1%}              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%xmm15, 64(%%r13)%{%%k1%}              \n"

        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%xmm17, 64(%%r10)%{%%k1%}              \n"
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%xmm19, 64(%%r11)%{%%k1%}              \n"
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%xmm21, 64(%%r12)%{%%k1%}              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%xmm23, 64(%%r13)%{%%k1%}              \n" 
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%zmm24, (%%r10)                \n"
        "   vmovups         %%xmm25, 64(%%r10)%{%%k1%}              \n"
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%zmm26, (%%r11)                \n"
        "   vmovups         %%xmm27, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm28, (%%r12)                \n"
        "   vmovups         %%xmm29, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm30, (%%r13)                \n"
        "   vmovups         %%xmm31, 64(%%r13)%{%%k1%}              \n"

        
        ".endm      \n"

        ".macro ADD_C_12x17   \n"
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"
		"   vmovups 		64(%%r10), %%xmm1%{%%k1%}        		\n"
		"	vaddps 			%%xmm1, %%xmm9, %%xmm9			\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"
		"   vmovups 		64(%%r11), %%xmm3%{%%k1%}        		\n"
		"	vaddps 			%%xmm3, %%xmm11, %%xmm11		\n"
		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"
		"   vmovups 		64(%%r12), %%xmm5%{%%k1%}        		\n"
		"	vaddps 			%%xmm5, %%xmm13, %%xmm13		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"
		"   vmovups 		64(%%r13), %%xmm7%{%%k1%}        		\n"
		"	vaddps 			%%xmm7, %%xmm15, %%xmm15		\n"

        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%xmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%xmm11, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%xmm13, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%xmm15, 64(%%r13)%{%%k1%}              \n"

		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm16, %%zmm16		\n"
		"   vmovups 		64(%%r10), %%xmm1%{%%k1%}        		\n"
		"	vaddps 			%%xmm1, %%xmm17, %%xmm17		\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm18, %%zmm18		\n"
		"   vmovups 		64(%%r11), %%xmm3%{%%k1%}        		\n"
		"	vaddps 			%%xmm3, %%xmm19, %%xmm19		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm20, %%zmm20		\n"
		"   vmovups 		64(%%r12), %%xmm5%{%%k1%}        		\n"
		"	vaddps 			%%xmm5, %%xmm21, %%xmm21		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm22, %%zmm22		\n"
		"   vmovups 		64(%%r13), %%xmm7%{%%k1%}        		\n"
		"	vaddps 			%%xmm7, %%xmm23, %%xmm23		\n"

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)%{%%k1%}              \n"
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%zmm19, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%zmm21, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%zmm23, 64(%%r13)%{%%k1%}              \n" 


		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm24, %%zmm24		\n"
		"   vmovups 		64(%%r10), %%xmm1%{%%k1%}        		\n"
		"	vaddps 			%%xmm1, %%xmm25, %%xmm25		\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm26, %%zmm26		\n"
		"   vmovups 		64(%%r11), %%xmm3%{%%k1%}        		\n"
		"	vaddps 			%%xmm3, %%xmm27, %%xmm27		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm28, %%zmm28		\n"
		"   vmovups 		64(%%r12), %%xmm5%{%%k1%}        		\n"
		"	vaddps 			%%xmm5, %%xmm29, %%xmm29		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm30, %%zmm30		\n"
		"   vmovups 		64(%%r13), %%xmm7%{%%k1%}        		\n"
		"	vaddps 			%%xmm7, %%xmm31, %%xmm31		\n"

        "   vmovups         %%zmm24, (%%r10)                \n"
        "   vmovups         %%xmm25, 64(%%r10)%{%%k1%}              \n"
        "   vmovups         %%zmm26, (%%r11)                \n"
        "   vmovups         %%xmm27, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm28, (%%r12)                \n"
        "   vmovups         %%xmm29, 64(%%r12)%{%%k1%}             \n"
        "   vmovups         %%zmm30, (%%r13)                \n"
        "   vmovups         %%xmm31, 64(%%r13)%{%%k1%}              \n"
        
        ".endm      \n"

        ".macro KERNEL8x17_K1                              \n"
        "   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm5, %%xmm9          \n"
        "   add             $128, %%rax                      \n"

        "   prefetcht0      192(%%rax)                      \n"
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm5, %%xmm11         \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm5, %%xmm13         \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm5, %%xmm15         \n"
        "   vmovups         (%%rax), %%zmm6                     \n"
        "   prefetcht0      32(%%rbx)                      \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"
        "   vfmadd231ps     %%xmm0, %%xmm5, %%xmm17         \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   vfmadd231ps     %%xmm1, %%xmm5, %%xmm19         \n"
        "   vmovups         64(%%rax), %%xmm7                     \n"
        "   add             $32, %%rbx                      \n"

        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"
        "   vfmadd231ps     %%xmm2, %%xmm5, %%xmm21         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"
        "   vfmadd231ps     %%xmm3, %%xmm5, %%xmm23         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL8x17_K2                              \n"

        "   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm9          \n"
        "   prefetcht0      192(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm11         \n"
        "   add             $128, %%rax                      \n"
        
        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm13         \n"
        "   prefetcht0      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm15         \n"
        "   vmovups         (%%rax), %%zmm4                     \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm17         \n"
        "   prefetcht0      32(%%rbx)                      \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm19         \n"
        "   vmovups         64(%%rax), %%xmm5                     \n"
        "   add             $32, %%rbx                      \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm21         \n"
        "   vbroadcastss    (%%rbx), %%zmm0                \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm23         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"

        ".endm                                              \n"



        ".macro KERNEL8x17_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm9          \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm11         \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm13         \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm15         \n"

        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm17         \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm19         \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm21         \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm23         \n"
        "   add             $32, %%rbx                      \n"

        

        ".endm                                              \n"


        ".macro ST_8x17   \n"

        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%xmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%xmm11, 64(%%r11)%{%%k1%}              \n"
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%xmm13, 64(%%r12)%{%%k1%}              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%xmm15, 64(%%r13)%{%%k1%}              \n"

        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%xmm17, 64(%%r10)%{%%k1%}              \n"
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%xmm19, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%xmm21, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%xmm23, 64(%%r13)%{%%k1%}              \n" 
        ".endm      \n"

        ".macro ADD_C_8x17   \n"
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"
		"   vmovups 		64(%%r10), %%xmm1%{%%k1%}        		\n"
		"	vaddps 			%%xmm1, %%xmm9, %%xmm9			\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"
		"   vmovups 		64(%%r11), %%xmm3%{%%k1%}        		\n"
		"	vaddps 			%%xmm3, %%xmm11, %%xmm11		\n"
		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"
		"   vmovups 		64(%%r12), %%xmm5%{%%k1%}        		\n"
		"	vaddps 			%%xmm5, %%xmm13, %%xmm13		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"
		"   vmovups 		64(%%r13), %%xmm7%{%%k1%}        		\n"
		"	vaddps 			%%xmm7, %%xmm15, %%xmm15		\n"

        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%xmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%xmm11, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%xmm13, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%xmm15, 64(%%r13)%{%%k1%}              \n"

		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm16, %%zmm16		\n"
		"   vmovups 		64(%%r10), %%xmm1%{%%k1%}        		\n"
		"	vaddps 			%%xmm1, %%xmm17, %%xmm17		\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm18, %%zmm18		\n"
		"   vmovups 		64(%%r11), %%xmm3%{%%k1%}        		\n"
		"	vaddps 			%%xmm3, %%xmm19, %%xmm19		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm20, %%zmm20		\n"
		"   vmovups 		64(%%r12), %%xmm5%{%%k1%}        		\n"
		"	vaddps 			%%xmm5, %%xmm21, %%xmm21		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm22, %%zmm22		\n"
		"   vmovups 		64(%%r13), %%xmm7%{%%k1%}        		\n"
		"	vaddps 			%%xmm7, %%xmm23, %%xmm23		\n"

        "   vmovups         %%zmm16, (%%r10)                \n"
        "   vmovups         %%zmm17, 64(%%r10)%{%%k1%}              \n"
        "   vmovups         %%zmm18, (%%r11)                \n"
        "   vmovups         %%zmm19, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm20, (%%r12)                \n"
        "   vmovups         %%zmm21, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm22, (%%r13)                \n"
        "   vmovups         %%zmm23, 64(%%r13)%{%%k1%}              \n"  
        ".endm      \n"

        ".macro KERNEL4x17_K1                              \n"

        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm5, %%xmm9          \n"
        "   add             $128, %%rax                      \n"

        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm5, %%xmm11         \n"
        "   vmovups         (%%rax), %%zmm6                     \n"
        "   add             $16, %%rbx                      \n"

        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm5, %%xmm13         \n"
        "   vmovups         64(%%rax), %%xmm7                     \n"
        "   vbroadcastss    (%%rbx), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm5, %%xmm15         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL4x17_K2                              \n"

        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm9          \n"
        "   add             $128, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm11         \n"
        "   vmovups         (%%rax), %%zmm4                     \n"
        "   add             $16, %%rbx                      \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm13         \n"
        "   vmovups         64(%%rax), %%xmm5                     \n"
        "   vbroadcastss    (%%rbx), %%zmm0                \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm15         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"

        ".endm                                              \n"



        ".macro KERNEL4x17_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"
        "   vfmadd231ps     %%xmm0, %%xmm7, %%xmm9          \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"
        "   vfmadd231ps     %%xmm1, %%xmm7, %%xmm11         \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"
        "   vfmadd231ps     %%xmm2, %%xmm7, %%xmm13         \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"
        "   vfmadd231ps     %%xmm3, %%xmm7, %%xmm15         \n"
        "   add             $16, %%rbx                      \n"
        
        ".endm                                              \n"


        ".macro ST_4x17   \n"

        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%xmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%xmm11, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%xmm13, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%xmm15, 64(%%r13)%{%%k1%}              \n"
        

        ".endm      \n"

        ".macro ADD_C_4x17   \n"
        
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"
		"   vmovups 		64(%%r10), %%xmm1%{%%k1%}        		\n"
		"	vaddps 			%%xmm1, %%xmm9, %%xmm9			\n"
		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"
		"   vmovups 		64(%%r11), %%xmm3%{%%k1%}        		\n"
		"	vaddps 			%%xmm3, %%xmm11, %%xmm11		\n"
		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"
		"   vmovups 		64(%%r12), %%xmm5%{%%k1%}        		\n"
		"	vaddps 			%%xmm5, %%xmm13, %%xmm13		\n"
		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"
		"   vmovups 		64(%%r13), %%xmm7%{%%k1%}        		\n"
		"	vaddps 			%%xmm7, %%xmm15, %%xmm15		\n"

        "   vmovups         %%zmm8, (%%r10)                 \n"
        "   vmovups         %%xmm9, 64(%%r10)%{%%k1%}               \n"
        "   vmovups         %%zmm10, (%%r11)                \n"
        "   vmovups         %%xmm11, 64(%%r11)%{%%k1%}              \n"
        "   vmovups         %%zmm12, (%%r12)                \n"
        "   vmovups         %%xmm13, 64(%%r12)%{%%k1%}              \n"
        "   vmovups         %%zmm14, (%%r13)                \n"
        "   vmovups         %%xmm15, 64(%%r13)%{%%k1%}              \n"
        
        ".endm      \n"

        "CONV_KERNEL12x17:                                \n"
        "   mov     %[output], %%rcx                             \n"
        "   mov     %[trans_filter], %%rbx                             \n"
        "   mov     %[input], %%r9                             \n"
        //"   prefetcht0      (%%rax)                         \n"
        "   movl $0x1, %%r15d                   \n"
        "   movl     %[Cb], %%edx                             \n"  // Cb
        "   movl     %[input_HW_size], %%r8d                             \n" 
        "   mov         %[input_buffer], %%r14                            \n"
        "   movl        %[Kb], %%edi                             \n"
        "   kmovd %%r15d, %%k1           \n"
        
        //------------------- loop body
        "BEGIN_PACK_12x17:                                        \n"

        "   mov     %%rcx, %%r10                            \n"  // C0
        "   mov     %%r9, %%rax                         \n"
        "   prefetcht0      (%%rax)                         \n" 
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   prefetcht0      64(%%rax)                         \n" 
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   prefetcht0      (%%rbx)                         \n" 
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  
        "   vmovups     64(%%rax), %%xmm5%{%%k1%}                   \n"
        
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
        
        "MAIN_PACK_K_12x17:                                       \n"
        "   KERNEL12x17_PACK_K1                             \n"
        "   KERNEL12x17_PACK_K2                             \n"
        "   KERNEL12x17_PACK_K1                             \n"
        "   KERNEL12x17_PACK_K2                             \n"
        "   KERNEL12x17_PACK_K1                             \n"
        "   KERNEL12x17_PACK_K2                             \n"
        "   KERNEL12x17_PACK_K1                             \n"
        "   cmp     $0, %%rdx                               \n"
        "   je      PACK_SAVE_12x17                             \n"
        "   KERNEL12x17_PACK_K2                             \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_PACK_K_12x17                             \n"
        
        "PACK_SAVE_12x17:                                       \n"
        "   movl %[cc], %%r15d              \n"
        "   KERNEL12x17_PACK_END_K                          \n"
        "   mov     %[input_buffer], %%r14                            \n"
        "   cmp     $0, %%r15                               \n"
        "   je      PACK_ST_C_12x17                                  \n"
        "   ADD_C_12x17                                     \n"
        "   jmp PACK_Kb_END_12x17                 \n"
        "PACK_ST_C_12x17:                                            \n"
        "   ST_12x17                                      \n"

        "PACK_Kb_END_12x17:               \n"
        "   sub    $12, %%rdi       \n"
        "   je     EDGE_CASE_17                              \n"
        
        "BEGIN_M_12x17:                                           \n"
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
        "BEGIN_K_12x17:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  
        "   vmovups     64(%%rax), %%xmm5                   \n"

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

        "MAIN_K_12x17:                                            \n"


        "   KERNEL12x17_K1                                  \n"
        "   KERNEL12x17_K2                                  \n"
        "   KERNEL12x17_K1                                  \n"
        "   KERNEL12x17_K2                                  \n"
        "   KERNEL12x17_K1                                  \n"
        "   KERNEL12x17_K2                                  \n"
        "   KERNEL12x17_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_12x17                                  \n"
        "   KERNEL12x17_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_12x17                                  \n"
        
        "EDGE_K_12x17:                                            \n"
        "   KERNEL12x17_END_K                               \n"
        "BEGIN_SAVE_12x17:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_12x17                                  \n"
        "   ADD_C_12x17                                     \n"
        "   jmp Kb_END_12x17                  \n"
        "ST_C_12x17:                                            \n"
        "   ST_12x17                                      \n"
        
        "Kb_END_12x17:                    \n"
        "   sub     $12, %%rdi       \n"
        "   jne     BEGIN_M_12x17                              \n"
        
        "EDGE_CASE_17:                          \n"
        "   movl    %[EDGE_Kb], %%edi                             \n"
        "   cmp $8, %%edi       \n"
        "   je  EDGE_8_17         \n"
        "   cmp $4, %%edi       \n"
        "   je  EDGE_4_17        \n"
        "   jmp     END_M_12x17           \n"

        "EDGE_8_17:                    \n"
        
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
        "BEGIN_K_8_17:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  
        "   vmovups     64(%%rax), %%xmm5                   \n"

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

        "MAIN_K_8_17:                                            \n"


        "   KERNEL8x17_K1                                  \n"
        "   KERNEL8x17_K2                                  \n"
        "   KERNEL8x17_K1                                  \n"
        "   KERNEL8x17_K2                                  \n"
        "   KERNEL8x17_K1                                  \n"
        "   KERNEL8x17_K2                                  \n"
        "   KERNEL8x17_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_8_17                                  \n"
        "   KERNEL8x17_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_8_17                                  \n"
        "EDGE_K_8_17:                                            \n"
        
        "   KERNEL8x17_END_K                               \n"

        "BEGIN_SAVE_8_17:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_8_17                                  \n"
        "   ADD_C_8x17                                     \n"
        "   jmp END_M_12x17                  \n"
        "ST_C_8_17:                                            \n"
        "   ST_8x17                                      \n"
        "   jmp     END_M_12x17                               \n"

        "EDGE_4_17:                    \n"
        
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
        "BEGIN_K_4_17:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  
        "   vmovups     64(%%rax), %%xmm5                   \n"

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

        "MAIN_K_4_17:                                            \n"
        "   KERNEL4x17_K1                                  \n"
        "   KERNEL4x17_K2                                  \n"
        "   KERNEL4x17_K1                                  \n"
        "   KERNEL4x17_K2                                  \n"
        "   KERNEL4x17_K1                                  \n"
        "   KERNEL4x17_K2                                  \n"
        "   KERNEL4x17_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_4_17                                  \n"
        "   KERNEL4x17_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_4_17                                  \n"
        "EDGE_K_4_17:                                            \n"
        
        "   KERNEL4x17_END_K                               \n"

        "BEGIN_SAVE_4_17:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_4_17                                  \n"
        "   ADD_C_4x17                                     \n"
        "   jmp END_M_12x17                  \n"
        "ST_C_4_17:                                            \n"
        "   ST_4x17                                      \n"

        "END_M_12x17:                                             \n"
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


void direct_1x1_N12M16_AVX512_pack(float *output, float *trans_filter, float *input, int Kb, int Cb, int input_HW_size, float *input_buffer, int cc, int EDGE_Kb){
    asm volatile(


        ".macro KERNEL12x16_PACK_K1                         \n"
        //"   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"

        "   vmovups         %%zmm4, (%%r14)                 \n"
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"

        "   add             %%r8, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   prefetcht2      128(%%rax)                      \n"
        "   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"


        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"

        "   vmovups         (%%rax), %%zmm6                     \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"
        "   prefetcht0      48(%%rbx)                      \n"
        "   add             $64, %%r14                     \n"
        
        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"

        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm24         \n"
  


        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm26         \n"

        "   add             $48, %%rbx                                 \n"
        
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm28         \n"

        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm30         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"
        

        ".endm                                              \n"


        ".macro KERNEL12x16_PACK_K2                         \n"
        //"   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"

        "   vmovups         %%zmm6, (%%r14)                 \n"


        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"

        "   add             %%r8, %%rax                      \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"

        "   prefetcht2      128(%%rax)                      \n"
        "   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"


        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm16         \n"
        "   prefetcht0      48(%%rbx)                      \n"
        "   vmovups         (%%rax), %%zmm4                     \n"

        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm18         \n"

        "   add             $64, %%r14                     \n"

        "   vbroadcastss    32(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"

        "   vbroadcastss    36(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"


        "   vbroadcastss    40(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm24         \n"

        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    44(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm26         \n"

        "   add             $48, %%rbx                                 \n"


        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"

        "   vbroadcastss    (%%rbx), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"

        "   vbroadcastss    4(%%rbx), %%zmm1                \n"


        ".endm                                              \n"



        ".macro KERNEL12x16_PACK_END_K                      \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"

        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"

        "   vmovups         %%zmm6, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"

        //"   prefetcht2      64(%%rax)                      \n"

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

        "   add             $48, %%rbx                                 \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm28         \n"
        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm30         \n"


        ".endm                                              \n"



        ".macro KERNEL12x16_K1                              \n"
        "   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"

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


        ".macro KERNEL12x16_K2                              \n"
        "   prefetcht0      64(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"



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



        ".macro KERNEL12x16_END_K                           \n"


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


        ".macro ST_12x16   \n"
        //"   movl $0x1, %%r8d                   \n"
        //"   kmovd %%r8d, %%k1           \n"
        "   vmovups         %%zmm8, (%%r10)                 \n"

        "   vmovups         %%zmm10, (%%r11)                \n"

        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%zmm12, (%%r12)                \n"

        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%zmm14, (%%r13)                \n"


        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%zmm16, (%%r10)                \n"

        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%zmm18, (%%r11)                \n"

        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%zmm20, (%%r12)                \n"

        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%zmm22, (%%r13)                \n"

        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%zmm24, (%%r10)                \n"

        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%zmm26, (%%r11)                \n"

        "   vmovups         %%zmm28, (%%r12)                \n"

        "   vmovups         %%zmm30, (%%r13)                \n"


        
        ".endm      \n"

        ".macro ADD_C_12x16   \n"
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"

		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"

		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"


        "   vmovups         %%zmm8, (%%r10)                 \n"

        "   vmovups         %%zmm10, (%%r11)                \n"

        "   vmovups         %%zmm12, (%%r12)                \n"

        "   vmovups         %%zmm14, (%%r13)                \n"


		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm16, %%zmm16		\n"

		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm18, %%zmm18		\n"


		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm20, %%zmm20		\n"

		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm22, %%zmm22		\n"


        "   vmovups         %%zmm16, (%%r10)                \n"

        "   vmovups         %%zmm18, (%%r11)                \n"

        "   vmovups         %%zmm20, (%%r12)                \n"

        "   vmovups         %%zmm22, (%%r13)                \n"



		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm24, %%zmm24		\n"

		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm26, %%zmm26		\n"


		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm28, %%zmm28		\n"

		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm30, %%zmm30		\n"


        "   vmovups         %%zmm24, (%%r10)                \n"

        "   vmovups         %%zmm26, (%%r11)                \n"

        "   vmovups         %%zmm28, (%%r12)                \n"

        "   vmovups         %%zmm30, (%%r13)                \n"
        ".endm      \n"

        ".macro KERNEL8x16_K1                              \n"
        "   prefetcht0      64(%%rax)                      \n"
        //"   prefetcht0      128(%%rax)                      \n"
        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"

        "   add             $64, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"


        "   vbroadcastss    16(%%rbx), %%zmm0               \n"
        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"
        "   prefetcht0      32(%%rbx)                      \n"

        "   vbroadcastss    20(%%rbx), %%zmm1               \n"
        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"

        "   vmovups         (%%rax), %%zmm6                     \n"
        
        "   vbroadcastss    24(%%rbx), %%zmm2               \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm16         \n"


        "   vbroadcastss    28(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm18         \n"

        "   add             $32, %%rbx                      \n"

        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm20         \n"

        "   vbroadcastss    (%%rbx), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm22         \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL8x16_K2                              \n"
        "   prefetcht0      64(%%rax)                      \n"
        //"   prefetcht0      128(%%rax)                      \n"
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

        "   vbroadcastss    (%%rbx), %%zmm0                \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"

        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"

        ".endm                                              \n"



        ".macro KERNEL8x16_END_K                           \n"


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


        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm20         \n"


        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm22         \n"

        "   add             $32, %%rbx                      \n"

        

        ".endm                                              \n"


        ".macro ST_8x16   \n"

        "   vmovups         %%zmm8, (%%r10)                 \n"

        "   vmovups         %%zmm10, (%%r11)                \n"

        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%zmm12, (%%r12)                \n"
 
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%zmm14, (%%r13)                \n"


        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%zmm16, (%%r10)                \n"

        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%zmm18, (%%r11)                \n"

        "   vmovups         %%zmm20, (%%r12)                \n"

        "   vmovups         %%zmm22, (%%r13)                \n"

        ".endm      \n"

        ".macro ADD_C_8x16   \n"
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"

		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"

		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"


        "   vmovups         %%zmm8, (%%r10)                 \n"

        "   vmovups         %%zmm10, (%%r11)                \n"

        "   vmovups         %%zmm12, (%%r12)                \n"

        "   vmovups         %%zmm14, (%%r13)                \n"

		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm16, %%zmm16		\n"

		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm18, %%zmm18		\n"


		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm20, %%zmm20		\n"

		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm22, %%zmm22		\n"


        "   vmovups         %%zmm16, (%%r10)                \n"

        "   vmovups         %%zmm18, (%%r11)                \n"

        "   vmovups         %%zmm20, (%%r12)                \n"

        "   vmovups         %%zmm22, (%%r13)                \n"
 
        ".endm      \n"

        ".macro KERNEL4x16_K1                              \n"

        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm4, %%zmm8          \n"

        "   add             $64, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm4, %%zmm10         \n"

        "   vmovups         (%%rax), %%zmm6                     \n"
        "   add             $16, %%rbx                      \n"

        "   vfmadd231ps     %%zmm2, %%zmm4, %%zmm12         \n"

        "   vbroadcastss    (%%rbx), %%zmm0                 \n"

        "   vfmadd231ps     %%zmm3, %%zmm4, %%zmm14         \n"
 
        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL4x16_K2                              \n"

        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"

        "   add             $64, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"

        "   vmovups         (%%rax), %%zmm4                     \n"
        "   add             $16, %%rbx                      \n"

        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"

        "   vbroadcastss    (%%rbx), %%zmm0                \n"

        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"

        "   vbroadcastss    4(%%rbx), %%zmm1                 \n"

        ".endm                                              \n"



        ".macro KERNEL4x16_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%zmm2                \n"
        "   vfmadd231ps     %%zmm0, %%zmm6, %%zmm8          \n"

        
        "   vbroadcastss    12(%%rbx), %%zmm3               \n"
        "   vfmadd231ps     %%zmm1, %%zmm6, %%zmm10         \n"


        "   vfmadd231ps     %%zmm2, %%zmm6, %%zmm12         \n"


        "   vfmadd231ps     %%zmm3, %%zmm6, %%zmm14         \n"

        "   add             $16, %%rbx                      \n"
        
        ".endm                                              \n"


        ".macro ST_4x16   \n"

        "   vmovups         %%zmm8, (%%r10)                 \n"

        "   vmovups         %%zmm10, (%%r11)                \n"

        "   vmovups         %%zmm12, (%%r12)                \n"

        "   vmovups         %%zmm14, (%%r13)                \n"

        

        ".endm      \n"

        ".macro ADD_C_4x16   \n"
        
		"   vmovups 		(%%r10), %%zmm0        			\n"
		"	vaddps 			%%zmm0, %%zmm8, %%zmm8			\n"

		"   vmovups 		(%%r11), %%zmm2        			\n"
		"	vaddps 			%%zmm2, %%zmm10, %%zmm10		\n"

		"   vmovups 		(%%r12), %%zmm4        			\n"
		"	vaddps 			%%zmm4, %%zmm12, %%zmm12		\n"

		"   vmovups 		(%%r13), %%zmm6        			\n"
		"	vaddps 			%%zmm6, %%zmm14, %%zmm14		\n"


        "   vmovups         %%zmm8, (%%r10)                 \n"

        "   vmovups         %%zmm10, (%%r11)                \n"

        "   vmovups         %%zmm12, (%%r12)                \n"

        "   vmovups         %%zmm14, (%%r13)                \n"

        
        ".endm      \n"

        "CONV_KERNEL12x16:                                \n"
        "   mov     %[output], %%rcx                             \n"
        "   mov     %[trans_filter], %%rbx                             \n"
        "   mov     %[input], %%r9                             \n"
        //"   prefetcht0      (%%rax)                         \n"

        "   movl     %[Cb], %%edx                             \n"  // Cb
        "   movl     %[input_HW_size], %%r8d                             \n" 
        "   mov         %[input_buffer], %%r14                            \n"
        "   movl        %[Kb], %%edi                             \n"

        
        //------------------- loop body
        "BEGIN_PACK_12x16:                                        \n"

        "   mov     %%rcx, %%r10                            \n"  // C0
        "   mov     %%r9, %%rax                         \n"
        "   prefetcht0      (%%rax)                         \n" 
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   prefetcht0      (%%rbx)                         \n" 
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  

        
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
        
        "MAIN_PACK_K_12x16:                                       \n"
        "   KERNEL12x16_PACK_K1                             \n"
        "   KERNEL12x16_PACK_K2                             \n"
        "   KERNEL12x16_PACK_K1                             \n"
        "   KERNEL12x16_PACK_K2                             \n"
        "   KERNEL12x16_PACK_K1                             \n"
        "   KERNEL12x16_PACK_K2                             \n"
        "   KERNEL12x16_PACK_K1                             \n"
        "   cmp     $0, %%rdx                               \n"
        "   je      PACK_SAVE_12x16                             \n"
        "   KERNEL12x16_PACK_K2                             \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_PACK_K_12x16                             \n"
        
        "PACK_SAVE_12x16:                                       \n"
        "   movl %[cc], %%r15d              \n"
        "   KERNEL12x16_PACK_END_K                          \n"
        "   mov     %[input_buffer], %%r14                            \n"
        "   cmp     $0, %%r15                               \n"
        "   je      PACK_ST_C_12x16                                  \n"
        "   ADD_C_12x16                                     \n"
        "   jmp PACK_Kb_END_12x16                 \n"
        "PACK_ST_C_12x16:                                            \n"
        "   ST_12x16                                      \n"

        "PACK_Kb_END_12x16:               \n"
        "   sub    $12, %%rdi       \n"
        "   je     EDGE_CASE_16                              \n"
        
        "BEGIN_M_12x16:                                           \n"
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
        "BEGIN_K_12x16:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"
        "   vpxorq      %%zmm9, %%zmm9, %%zmm9              \n" 
        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
        "   vpxorq      %%zmm11, %%zmm11, %%zmm11           \n"  

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

        "MAIN_K_12x16:                                            \n"


        "   KERNEL12x16_K1                                  \n"
        "   KERNEL12x16_K2                                  \n"
        "   KERNEL12x16_K1                                  \n"
        "   KERNEL12x16_K2                                  \n"
        "   KERNEL12x16_K1                                  \n"
        "   KERNEL12x16_K2                                  \n"
        "   KERNEL12x16_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_12x16                                  \n"
        "   KERNEL12x16_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_12x16                                  \n"
        
        "EDGE_K_12x16:                                            \n"
        "   KERNEL12x16_END_K                               \n"
        "BEGIN_SAVE_12x16:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_12x16                                  \n"
        "   ADD_C_12x16                                     \n"
        "   jmp Kb_END_12x16                  \n"
        "ST_C_12x16:                                            \n"
        "   ST_12x16                                      \n"
        
        "Kb_END_12x16:                    \n"
        "   sub     $12, %%rdi       \n"
        "   jne     BEGIN_M_12x16                              \n"
        
        "EDGE_CASE_16:                          \n"
        "   movl    %[EDGE_Kb], %%edi                             \n"
        "   cmp $8, %%edi       \n"
        "   je  EDGE_8_16         \n"
        "   cmp $4, %%edi       \n"
        "   je  EDGE_4_16        \n"
        "   jmp     END_M_12x16           \n"

        "EDGE_8_16:                    \n"
        
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
        "BEGIN_K_8_16:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"

        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 
 

        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"

        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 

        "   vpxorq      %%zmm16, %%zmm16, %%zmm16           \n"

        "   vpxorq      %%zmm18, %%zmm18, %%zmm18           \n" 


        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

        "   vpxorq      %%zmm20, %%zmm20, %%zmm20           \n"

        "   vpxorq      %%zmm22, %%zmm22, %%zmm22           \n" 

        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_8_16:                                            \n"


        "   KERNEL8x16_K1                                  \n"
        "   KERNEL8x16_K2                                  \n"
        "   KERNEL8x16_K1                                  \n"
        "   KERNEL8x16_K2                                  \n"
        "   KERNEL8x16_K1                                  \n"
        "   KERNEL8x16_K2                                  \n"
        "   KERNEL8x16_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_8_16                                  \n"
        "   KERNEL8x16_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_8_16                                  \n"
        "EDGE_K_8_16:                                            \n"
        
        "   KERNEL8x16_END_K                               \n"

        "BEGIN_SAVE_8_16:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_8_16                                  \n"
        "   ADD_C_8x16                                     \n"
        "   jmp END_M_12x16                  \n"
        "ST_C_8_16:                                            \n"
        "   ST_8x16                                      \n"
        "   jmp     END_M_12x16                               \n"

        "EDGE_4_16:                    \n"
        
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
        "BEGIN_K_4_16:                                           \n"

        "   vmovups     (%%rax), %%zmm4                     \n"
        "   vpxorq      %%zmm8, %%zmm8, %%zmm8              \n"

        "   vpxorq      %%zmm10, %%zmm10, %%zmm10           \n" 


        "   vpxorq      %%zmm12, %%zmm12, %%zmm12           \n"

        "   vpxorq      %%zmm14, %%zmm14, %%zmm14           \n" 


        "   vbroadcastss    (%%rbx), %%zmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%zmm1                \n"

        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_4_16:                                            \n"
        "   KERNEL4x16_K1                                  \n"
        "   KERNEL4x16_K2                                  \n"
        "   KERNEL4x16_K1                                  \n"
        "   KERNEL4x16_K2                                  \n"
        "   KERNEL4x16_K1                                  \n"
        "   KERNEL4x16_K2                                  \n"
        "   KERNEL4x16_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_4_16                                  \n"
        "   KERNEL4x16_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_4_16                                  \n"
        "EDGE_K_4_16:                                            \n"
        
        "   KERNEL4x16_END_K                               \n"

        "BEGIN_SAVE_4_16:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_4_16                                  \n"
        "   ADD_C_4x16                                     \n"
        "   jmp END_M_12x16                  \n"
        "ST_C_4_16:                                            \n"
        "   ST_4x16                                      \n"

        "END_M_12x16:                                             \n"
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

void direct_1x1_N12M4_AVX512_pack(float *output, float *trans_filter, float *input, int Kb, int Cb, int input_HW_size, float *input_buffer, int cc, int EDGE_Kb){
    asm volatile(


        ".macro KERNEL12x4_PACK_K1                         \n"

        "   vbroadcastss    8(%%rbx), %%xmm2                \n"
        "   vfmadd231ps     %%xmm0, %%xmm4, %%xmm8          \n"

        "   vmovups         %%xmm4, (%%r14)                 \n"
        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm4, %%xmm10         \n"

        "   add             %%r8, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm4, %%xmm12         \n"
        //"   prefetcht2      64(%%rax)                      \n"
        "   vbroadcastss    20(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm4, %%xmm14         \n"


        "   vbroadcastss    24(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm4, %%xmm16         \n"

        "   vmovups         (%%rax), %%xmm6                     \n"

        "   vbroadcastss    28(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm4, %%xmm18         \n"

        "   add             $16, %%r14                     \n"
        
        "   vbroadcastss    32(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm4, %%xmm20         \n"

        "   vbroadcastss    36(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm4, %%xmm22         \n"

        "   vbroadcastss    40(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm4, %%xmm24         \n"
  


        "   vbroadcastss    44(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm4, %%xmm26         \n"

        "   add             $48, %%rbx                                 \n"
        
        "   vfmadd231ps     %%xmm2, %%xmm4, %%xmm28         \n"

        "   vbroadcastss    (%%rbx), %%xmm0                 \n"
        "   vfmadd231ps     %%xmm3, %%xmm4, %%xmm30         \n"
        "   vbroadcastss    4(%%rbx), %%xmm1                \n"
        

        ".endm                                              \n"


        ".macro KERNEL12x4_PACK_K2                         \n"

        "   vbroadcastss    8(%%rbx), %%xmm2                \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm8          \n"

        "   vmovups         %%xmm6, (%%r14)                 \n"


        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm10         \n"

        "   add             %%r8, %%rax                      \n"


        "   vbroadcastss    16(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm12         \n"

        "   prefetcht2      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm14         \n"


        "   vbroadcastss    24(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm16         \n"

        "   vmovups         (%%rax), %%xmm4                     \n"

        "   vbroadcastss    28(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm18         \n"

        "   add             $16, %%r14                     \n"

        "   vbroadcastss    32(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm20         \n"

        "   vbroadcastss    36(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm22         \n"


        "   vbroadcastss    40(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm24         \n"

        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    44(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm26         \n"

        "   add             $48, %%rbx                                 \n"


        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm28         \n"

        "   vbroadcastss    (%%rbx), %%xmm0                 \n"

        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm30         \n"

        "   vbroadcastss    4(%%rbx), %%xmm1                \n"


        ".endm                                              \n"



        ".macro KERNEL12x4_PACK_END_K                      \n"


        "   vbroadcastss    8(%%rbx), %%xmm2                \n"

        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm8          \n"

        "   vmovups         %%xmm6, (%%r14)                 \n"
        
        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm10         \n"


        "   vbroadcastss    16(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm12         \n"

        //"   prefetcht2      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm14         \n"


        "   vbroadcastss    24(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm16         \n"


        
        
        "   vbroadcastss    28(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm18         \n"


        "   vbroadcastss    32(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm20         \n"


        
        "   vbroadcastss    36(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm22         \n"


        "   vbroadcastss    40(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm24         \n"


        "   vbroadcastss    44(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm26         \n"

        "   add             $48, %%rbx                                 \n"

        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm28         \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm30         \n"


        ".endm                                              \n"



        ".macro KERNEL12x4_K1                              \n"

        "   vbroadcastss    8(%%rbx), %%xmm2                \n"
        "   vfmadd231ps     %%xmm0, %%xmm4, %%xmm8          \n"

        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm4, %%xmm10         \n"

        "   add             $16, %%rax                      \n"

        "   vbroadcastss    16(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm4, %%xmm12         \n"

        //"   prefetcht0      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm4, %%xmm14         \n"

        "   vmovups         (%%rax), %%xmm6                     \n"
        
        "   vbroadcastss    24(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm4, %%xmm16         \n"


        "   vbroadcastss    28(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm4, %%xmm18         \n"


        "   vbroadcastss    32(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm4, %%xmm20         \n"



        "   vbroadcastss    36(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm4, %%xmm22         \n"

        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm4, %%xmm24         \n"


        "   vbroadcastss    44(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm4, %%xmm26         \n"

        "   add             $48, %%rbx                      \n"
        
        "   vfmadd231ps     %%xmm2, %%xmm4, %%xmm28         \n"

        "   vbroadcastss    (%%rbx), %%xmm0                 \n"
        
        "   vfmadd231ps     %%xmm3, %%xmm4, %%xmm30         \n"

        "   vbroadcastss    4(%%rbx), %%xmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL12x4_K2                              \n"

        "   vbroadcastss    8(%%rbx), %%xmm2                \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm8          \n"



        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm10         \n"

        "   add             $16, %%rax                      \n"
        
        "   vbroadcastss    16(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm12         \n"

        "   prefetcht0      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm14         \n"

        "   vmovups         (%%rax), %%xmm4                     \n"
        
        "   vbroadcastss    24(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm16         \n"


        "   vbroadcastss    28(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm18         \n"


        
        
        "   vbroadcastss    32(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm20         \n"



        "   vbroadcastss    36(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm22         \n"

        "   prefetcht0      48(%%rbx)                      \n"

        "   vbroadcastss    40(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm24         \n"


        "   vbroadcastss    44(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm26         \n"

        "   add             $48, %%rbx                      \n"
        
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm28         \n"


        "   vbroadcastss    (%%rbx), %%xmm0                \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm30         \n"

        "   vbroadcastss    4(%%rbx), %%xmm1                 \n"

        ".endm                                              \n"



        ".macro KERNEL12x4_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%xmm2                \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm8          \n"

        
        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm10         \n"



        "   vbroadcastss    16(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm12         \n"


        "   vbroadcastss    20(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm14         \n"


        "   vbroadcastss    24(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm16         \n"


        "   vbroadcastss    28(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm18         \n"


        "   vbroadcastss    32(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm20         \n"


        "   vbroadcastss    36(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm22         \n"


        "   vbroadcastss    40(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm24         \n"


        "   vbroadcastss    44(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm26         \n"

        "   add             $48, %%rbx                      \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm28         \n"


        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm30         \n"

        

        ".endm                                              \n"


        ".macro ST_12x4   \n"
        //"   movl $0x1, %%r8d                   \n"
        //"   kmovd %%r8d, %%k1           \n"
        "   vmovups         %%xmm8, (%%r10)                 \n"

        "   vmovups         %%xmm10, (%%r11)                \n"

        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%xmm12, (%%r12)                \n"

        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%xmm14, (%%r13)                \n"


        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%xmm16, (%%r10)                \n"

        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%xmm18, (%%r11)                \n"

        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%xmm20, (%%r12)                \n"

        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%xmm22, (%%r13)                \n"

        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%xmm24, (%%r10)                \n"

        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%xmm26, (%%r11)                \n"

        "   vmovups         %%xmm28, (%%r12)                \n"

        "   vmovups         %%xmm30, (%%r13)                \n"


        
        ".endm      \n"

        ".macro ADD_C_12x4   \n"
		"   vmovups 		(%%r10), %%xmm0        			\n"
		"	vaddps 			%%xmm0, %%xmm8, %%xmm8			\n"

		"   vmovups 		(%%r11), %%xmm2        			\n"
		"	vaddps 			%%xmm2, %%xmm10, %%xmm10		\n"

		"   vmovups 		(%%r12), %%xmm4        			\n"
		"	vaddps 			%%xmm4, %%xmm12, %%xmm12		\n"

		"   vmovups 		(%%r13), %%xmm6        			\n"
		"	vaddps 			%%xmm6, %%xmm14, %%xmm14		\n"


        "   vmovups         %%xmm8, (%%r10)                 \n"

        "   vmovups         %%xmm10, (%%r11)                \n"

        "   vmovups         %%xmm12, (%%r12)                \n"

        "   vmovups         %%xmm14, (%%r13)                \n"


		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%xmm0        			\n"
		"	vaddps 			%%xmm0, %%xmm16, %%xmm16		\n"

		"   vmovups 		(%%r11), %%xmm2        			\n"
		"	vaddps 			%%xmm2, %%xmm18, %%xmm18		\n"


		"   vmovups 		(%%r12), %%xmm4        			\n"
		"	vaddps 			%%xmm4, %%xmm20, %%xmm20		\n"

		"   vmovups 		(%%r13), %%xmm6        			\n"
		"	vaddps 			%%xmm6, %%xmm22, %%xmm22		\n"


        "   vmovups         %%xmm16, (%%r10)                \n"

        "   vmovups         %%xmm18, (%%r11)                \n"

        "   vmovups         %%xmm20, (%%r12)                \n"

        "   vmovups         %%xmm22, (%%r13)                \n"



		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%xmm0        			\n"
		"	vaddps 			%%xmm0, %%xmm24, %%xmm24		\n"

		"   vmovups 		(%%r11), %%xmm2        			\n"
		"	vaddps 			%%xmm2, %%xmm26, %%xmm26		\n"


		"   vmovups 		(%%r12), %%xmm4        			\n"
		"	vaddps 			%%xmm4, %%xmm28, %%xmm28		\n"

		"   vmovups 		(%%r13), %%xmm6        			\n"
		"	vaddps 			%%xmm6, %%xmm30, %%xmm30		\n"


        "   vmovups         %%xmm24, (%%r10)                \n"

        "   vmovups         %%xmm26, (%%r11)                \n"

        "   vmovups         %%xmm28, (%%r12)                \n"

        "   vmovups         %%xmm30, (%%r13)                \n"
        ".endm      \n"

        ".macro KERNEL8x4_K1                              \n"

        "   vbroadcastss    8(%%rbx), %%xmm2                \n"
        "   vfmadd231ps     %%xmm0, %%xmm4, %%xmm8          \n"

        "   add             $16, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm4, %%xmm10         \n"


        "   vbroadcastss    16(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm4, %%xmm12         \n"


        "   vbroadcastss    20(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm4, %%xmm14         \n"

        "   vmovups         (%%rax), %%xmm6                     \n"
        
        "   vbroadcastss    24(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm4, %%xmm16         \n"


        "   vbroadcastss    28(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm4, %%xmm18         \n"

        "   add             $32, %%rbx                      \n"

        "   vfmadd231ps     %%xmm2, %%xmm4, %%xmm20         \n"

        "   vbroadcastss    (%%rbx), %%xmm0                 \n"

        "   vfmadd231ps     %%xmm3, %%xmm4, %%xmm22         \n"
        "   vbroadcastss    4(%%rbx), %%xmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL8x4_K2                              \n"

        "   vbroadcastss    8(%%rbx), %%xmm2                \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm8          \n"

        "   prefetcht0      128(%%rax)                      \n"

        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm10         \n"

        "   add             $16, %%rax                      \n"
        
        "   vbroadcastss    16(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm12         \n"

        "   prefetcht0      64(%%rax)                      \n"

        "   vbroadcastss    20(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm14         \n"

        "   vmovups         (%%rax), %%xmm4                     \n"
        
        "   vbroadcastss    24(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm16         \n"


        "   vbroadcastss    28(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm18         \n"

        "   add             $32, %%rbx                      \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm20         \n"

        "   vbroadcastss    (%%rbx), %%xmm0                \n"

        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm22         \n"

        "   vbroadcastss    4(%%rbx), %%xmm1                 \n"

        ".endm                                              \n"



        ".macro KERNEL8x4_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%xmm2                \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm8          \n"

        
        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm10         \n"



        "   vbroadcastss    16(%%rbx), %%xmm0               \n"
        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm12         \n"


        "   vbroadcastss    20(%%rbx), %%xmm1               \n"
        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm14         \n"


        "   vbroadcastss    24(%%rbx), %%xmm2               \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm16         \n"


        "   vbroadcastss    28(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm18         \n"


        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm20         \n"


        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm22         \n"

        "   add             $32, %%rbx                      \n"

        

        ".endm                                              \n"


        ".macro ST_8x4   \n"

        "   vmovups         %%xmm8, (%%r10)                 \n"

        "   vmovups         %%xmm10, (%%r11)                \n"

        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C0
        "   vmovups         %%xmm12, (%%r12)                \n"
 
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   vmovups         %%xmm14, (%%r13)                \n"


        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2

        "   vmovups         %%xmm16, (%%r10)                \n"

        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups         %%xmm18, (%%r11)                \n"

        "   vmovups         %%xmm20, (%%r12)                \n"

        "   vmovups         %%xmm22, (%%r13)                \n"

        ".endm      \n"

        ".macro ADD_C_8x4   \n"
		"   vmovups 		(%%r10), %%xmm0        			\n"
		"	vaddps 			%%xmm0, %%xmm8, %%xmm8			\n"

		"   vmovups 		(%%r11), %%xmm2        			\n"
		"	vaddps 			%%xmm2, %%xmm10, %%xmm10		\n"

		"   vmovups 		(%%r12), %%xmm4        			\n"
		"	vaddps 			%%xmm4, %%xmm12, %%xmm12		\n"

		"   vmovups 		(%%r13), %%xmm6        			\n"
		"	vaddps 			%%xmm6, %%xmm14, %%xmm14		\n"


        "   vmovups         %%xmm8, (%%r10)                 \n"

        "   vmovups         %%xmm10, (%%r11)                \n"

        "   vmovups         %%xmm12, (%%r12)                \n"

        "   vmovups         %%xmm14, (%%r13)                \n"

		"	leaq  			(%%r13, %%r8), %%r10 		\n"  // C0
		"	leaq 			(%%r10, %%r8), %%r11 		\n"	 // C1
		"	leaq 			(%%r11, %%r8), %%r12 		\n"  // C2
		"	leaq 			(%%r12, %%r8), %%r13 		\n"  // C3

		"   vmovups 		(%%r10), %%xmm0        			\n"
		"	vaddps 			%%xmm0, %%xmm16, %%xmm16		\n"

		"   vmovups 		(%%r11), %%xmm2        			\n"
		"	vaddps 			%%xmm2, %%xmm18, %%xmm18		\n"


		"   vmovups 		(%%r12), %%xmm4        			\n"
		"	vaddps 			%%xmm4, %%xmm20, %%xmm20		\n"

		"   vmovups 		(%%r13), %%xmm6        			\n"
		"	vaddps 			%%xmm6, %%xmm22, %%xmm22		\n"


        "   vmovups         %%xmm16, (%%r10)                \n"

        "   vmovups         %%xmm18, (%%r11)                \n"

        "   vmovups         %%xmm20, (%%r12)                \n"

        "   vmovups         %%xmm22, (%%r13)                \n"
 
        ".endm      \n"

        ".macro KERNEL4x4_K1                              \n"

        "   vbroadcastss    8(%%rbx), %%xmm2                \n"
        "   vfmadd231ps     %%xmm0, %%xmm4, %%xmm8          \n"

        "   add             $16, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm4, %%xmm10         \n"

        "   vmovups         (%%rax), %%xmm6                     \n"
        "   add             $16, %%rbx                      \n"

        "   vfmadd231ps     %%xmm2, %%xmm4, %%xmm12         \n"

        "   vbroadcastss    (%%rbx), %%xmm0                 \n"

        "   vfmadd231ps     %%xmm3, %%xmm4, %%xmm14         \n"
 
        "   vbroadcastss    4(%%rbx), %%xmm1                 \n"
        
        ".endm                                              \n"


        ".macro KERNEL4x4_K2                              \n"

        "   vbroadcastss    8(%%rbx), %%xmm2                \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm8          \n"

        "   add             $16, %%rax                      \n"

        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm10         \n"

        "   vmovups         (%%rax), %%xmm4                     \n"
        "   add             $16, %%rbx                      \n"

        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm12         \n"

        "   vbroadcastss    (%%rbx), %%xmm0                \n"

        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm14         \n"

        "   vbroadcastss    4(%%rbx), %%xmm1                 \n"

        ".endm                                              \n"



        ".macro KERNEL4x4_END_K                           \n"


        "   vbroadcastss    8(%%rbx), %%xmm2                \n"
        "   vfmadd231ps     %%xmm0, %%xmm6, %%xmm8          \n"

        
        "   vbroadcastss    12(%%rbx), %%xmm3               \n"
        "   vfmadd231ps     %%xmm1, %%xmm6, %%xmm10         \n"


        "   vfmadd231ps     %%xmm2, %%xmm6, %%xmm12         \n"


        "   vfmadd231ps     %%xmm3, %%xmm6, %%xmm14         \n"

        "   add             $16, %%rbx                      \n"
        
        ".endm                                              \n"


        ".macro ST_4x4   \n"

        "   vmovups         %%xmm8, (%%r10)                 \n"

        "   vmovups         %%xmm10, (%%r11)                \n"

        "   vmovups         %%xmm12, (%%r12)                \n"

        "   vmovups         %%xmm14, (%%r13)                \n"

        

        ".endm      \n"

        ".macro ADD_C_4x4   \n"
        
		"   vmovups 		(%%r10), %%xmm0        			\n"
		"	vaddps 			%%xmm0, %%xmm8, %%xmm8			\n"

		"   vmovups 		(%%r11), %%xmm2        			\n"
		"	vaddps 			%%xmm2, %%xmm10, %%xmm10		\n"

		"   vmovups 		(%%r12), %%xmm4        			\n"
		"	vaddps 			%%xmm4, %%xmm12, %%xmm12		\n"

		"   vmovups 		(%%r13), %%xmm6        			\n"
		"	vaddps 			%%xmm6, %%xmm14, %%xmm14		\n"


        "   vmovups         %%xmm8, (%%r10)                 \n"

        "   vmovups         %%xmm10, (%%r11)                \n"

        "   vmovups         %%xmm12, (%%r12)                \n"

        "   vmovups         %%xmm14, (%%r13)                \n"

        
        ".endm      \n"

        "CONV_KERNEL12x4:                                \n"
        "   mov     %[output], %%rcx                             \n"
        "   mov     %[trans_filter], %%rbx                             \n"
        "   mov     %[input], %%r9                             \n"
        //"   prefetcht0      (%%rax)                         \n"

        "   movl     %[Cb], %%edx                             \n"  // Cb
        "   movl     %[input_HW_size], %%r8d                             \n" 
        "   mov         %[input_buffer], %%r14                            \n"
        "   movl        %[Kb], %%edi                             \n"

        
        //------------------- loop body
        "BEGIN_PACK_12x4:                                        \n"

        "   mov     %%rcx, %%r10                            \n"  // C0
        "   mov     %%r9, %%rax                         \n"
        "   prefetcht0      (%%rax)                         \n" 
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   prefetcht0      (%%rbx)                         \n" 
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3
        "   vmovups     (%%rax), %%xmm4                     \n"
        "   vpxorq      %%xmm8, %%xmm8, %%xmm8              \n"

        "   vpxorq      %%xmm10, %%xmm10, %%xmm10           \n" 
        "   vpxorq      %%xmm12, %%xmm12, %%xmm12           \n"

        "   vpxorq      %%xmm14, %%xmm14, %%xmm14           \n" 

        "   vpxorq      %%xmm16, %%xmm16, %%xmm16           \n"

        "   vpxorq      %%xmm18, %%xmm18, %%xmm18           \n" 

        "   vbroadcastss    (%%rbx), %%xmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%xmm1                \n"
        "   vpxorq      %%xmm20, %%xmm20, %%xmm20           \n"

        //"   prefetcht2      64(%%r10)                       \n"
        "   vpxorq      %%xmm22, %%xmm22, %%xmm22           \n" 

        //"   prefetcht2      64(%%r11)                       \n"
        "   vpxorq      %%xmm24, %%xmm24, %%xmm24           \n"

        //"   prefetcht2      64(%%r12)                       \n"
        "   vpxorq      %%xmm26, %%xmm26, %%xmm26           \n" 

        //"   prefetcht2      64(%%r13)                       \n"
        "   vpxorq      %%xmm28, %%xmm28, %%xmm28           \n"

        "   vpxorq      %%xmm30, %%xmm30, %%xmm30           \n" 

        "   sub    $8, %%rdx                               \n"
        
        "MAIN_PACK_K_12x4:                                       \n"
        "   KERNEL12x4_PACK_K1                             \n"
        "   KERNEL12x4_PACK_K2                             \n"
        "   KERNEL12x4_PACK_K1                             \n"
        "   KERNEL12x4_PACK_K2                             \n"
        "   KERNEL12x4_PACK_K1                             \n"
        "   KERNEL12x4_PACK_K2                             \n"
        "   KERNEL12x4_PACK_K1                             \n"
        "   cmp     $0, %%rdx                               \n"
        "   je      PACK_SAVE_12x4                             \n"
        "   KERNEL12x4_PACK_K2                             \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_PACK_K_12x4                             \n"
        
        "PACK_SAVE_12x4:                                       \n"
        "   movl %[cc], %%r15d              \n"
        "   KERNEL12x4_PACK_END_K                          \n"
        "   mov     %[input_buffer], %%r14                            \n"
        "   cmp     $0, %%r15                               \n"
        "   je      PACK_ST_C_12x4                                  \n"
        "   ADD_C_12x4                                     \n"
        "   jmp PACK_Kb_END_12x4                 \n"
        "PACK_ST_C_12x4:                                            \n"
        "   ST_12x4                                      \n"

        "PACK_Kb_END_12x4:               \n"
        "   sub    $12, %%rdi       \n"
        "   je     EDGE_CASE_4                              \n"
        
        "BEGIN_M_12x4:                                           \n"
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
        "BEGIN_K_12x4:                                           \n"

        "   vmovups     (%%rax), %%xmm4                     \n"
        "   vpxorq      %%xmm8, %%xmm8, %%xmm8              \n"

        "   vpxorq      %%xmm10, %%xmm10, %%xmm10           \n" 


        "   vpxorq      %%xmm12, %%xmm12, %%xmm12           \n"

        "   vpxorq      %%xmm14, %%xmm14, %%xmm14           \n" 

        "   vpxorq      %%xmm16, %%xmm16, %%xmm16           \n"

        "   vpxorq      %%xmm18, %%xmm18, %%xmm18           \n" 


        "   vbroadcastss    (%%rbx), %%xmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%xmm1                \n"

        "   vpxorq      %%xmm20, %%xmm20, %%xmm20           \n"

        "   vpxorq      %%xmm22, %%xmm22, %%xmm22           \n" 

        "   vpxorq      %%xmm24, %%xmm24, %%xmm24           \n"

        "   vpxorq      %%xmm26, %%xmm26, %%xmm26           \n" 

        "   vpxorq      %%xmm28, %%xmm28, %%xmm28           \n"

        "   vpxorq      %%xmm30, %%xmm30, %%xmm30           \n" 
   
        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_12x4:                                            \n"


        "   KERNEL12x4_K1                                  \n"
        "   KERNEL12x4_K2                                  \n"
        "   KERNEL12x4_K1                                  \n"
        "   KERNEL12x4_K2                                  \n"
        "   KERNEL12x4_K1                                  \n"
        "   KERNEL12x4_K2                                  \n"
        "   KERNEL12x4_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_12x4                                  \n"
        "   KERNEL12x4_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_12x4                                  \n"
        
        "EDGE_K_12x4:                                            \n"
        "   KERNEL12x4_END_K                               \n"
        "BEGIN_SAVE_12x4:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_12x4                                  \n"
        "   ADD_C_12x4                                     \n"
        "   jmp Kb_END_12x4                  \n"
        "ST_C_12x4:                                            \n"
        "   ST_12x4                                      \n"
        
        "Kb_END_12x4:                    \n"
        "   sub     $12, %%rdi       \n"
        "   jne     BEGIN_M_12x4                              \n"
        
        "EDGE_CASE_4:                          \n"
        "   movl    %[EDGE_Kb], %%edi                             \n"
        "   cmp $8, %%edi       \n"
        "   je  EDGE_8_4         \n"
        "   cmp $4, %%edi       \n"
        "   je  EDGE_4_4        \n"
        "   jmp     END_M_12x4           \n"

        "EDGE_8_4:                    \n"
        
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
        "BEGIN_K_8_4:                                           \n"

        "   vmovups     (%%rax), %%xmm4                     \n"
        "   vpxorq      %%xmm8, %%xmm8, %%xmm8              \n"

        "   vpxorq      %%xmm10, %%xmm10, %%xmm10           \n" 
 

        "   vpxorq      %%xmm12, %%xmm12, %%xmm12           \n"

        "   vpxorq      %%xmm14, %%xmm14, %%xmm14           \n" 

        "   vpxorq      %%xmm16, %%xmm16, %%xmm16           \n"

        "   vpxorq      %%xmm18, %%xmm18, %%xmm18           \n" 


        "   vbroadcastss    (%%rbx), %%xmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%xmm1                \n"

        "   vpxorq      %%xmm20, %%xmm20, %%xmm20           \n"

        "   vpxorq      %%xmm22, %%xmm22, %%xmm22           \n" 

        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_8_4:                                            \n"


        "   KERNEL8x4_K1                                  \n"
        "   KERNEL8x4_K2                                  \n"
        "   KERNEL8x4_K1                                  \n"
        "   KERNEL8x4_K2                                  \n"
        "   KERNEL8x4_K1                                  \n"
        "   KERNEL8x4_K2                                  \n"
        "   KERNEL8x4_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_8_4                                  \n"
        "   KERNEL8x4_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_8_4                                  \n"
        "EDGE_K_8_4:                                            \n"
        
        "   KERNEL8x4_END_K                               \n"

        "BEGIN_SAVE_8_4:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_8_4                                  \n"
        "   ADD_C_8x4                                     \n"
        "   jmp END_M_12x4                  \n"
        "ST_C_8_4:                                            \n"
        "   ST_8x4                                      \n"
        "   jmp     END_M_12x4                               \n"

        "EDGE_4_4:                    \n"
        
        "   leaq    (%%r13, %%r8), %%r10                 \n"  // C3
        "   movl %[cc], %%r15d              \n"
        "   leaq    (%%r10, %%r8), %%r11                 \n"  // C1
        "   movl %[Cb], %%edx           \n"
        "   leaq    (%%r11, %%r8), %%r12                 \n"  // C2
        "   mov     %%r14, %%rax                            \n"  // Bc
        "   leaq    (%%r12, %%r8), %%r13                 \n"  // C3

        
        "BEGIN_K_4_4:                                           \n"

        "   vmovups     (%%rax), %%xmm4                     \n"
        "   vpxorq      %%xmm8, %%xmm8, %%xmm8              \n"

        "   vpxorq      %%xmm10, %%xmm10, %%xmm10           \n" 


        "   vpxorq      %%xmm12, %%xmm12, %%xmm12           \n"

        "   vpxorq      %%xmm14, %%xmm14, %%xmm14           \n" 


        "   vbroadcastss    (%%rbx), %%xmm0                 \n"
        "   vbroadcastss    4(%%rbx), %%xmm1                \n"

        
        "   sub $8, %%rdx                               \n"

        "MAIN_K_4_4:                                            \n"
        "   KERNEL4x4_K1                                  \n"
        "   KERNEL4x4_K2                                  \n"
        "   KERNEL4x4_K1                                  \n"
        "   KERNEL4x4_K2                                  \n"
        "   KERNEL4x4_K1                                  \n"
        "   KERNEL4x4_K2                                  \n"
        "   KERNEL4x4_K1                                  \n"
        "   cmp     $0, %%edx                               \n"
        "   je      EDGE_K_4_4                                  \n"
        "   KERNEL4x4_K2                                  \n"
        "   sub    $8, %%rdx                               \n"
        "   jmp     MAIN_K_4_4                                  \n"
        "EDGE_K_4_4:                                            \n"
        
        "   KERNEL4x4_END_K                               \n"

        "BEGIN_SAVE_4_4:                                        \n"
        "   cmp     $0, %%r15d                               \n"
        "   je      ST_C_4_4                                  \n"
        "   ADD_C_4x4                                     \n"
        "   jmp END_M_12x4                  \n"
        "ST_C_4_4:                                            \n"
        "   ST_4x4                                      \n"

        "END_M_12x4:                                             \n"
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


void avx512_1x1s1(int H, int W, int N, int C, float *input, int K, int R, float *filter, float *output)
{
    int S = R;

    int out_W = W;
    int out_H = H;
    
    int stride_in = H * W;
    int stride_out = out_W * out_H;
    
    void *ptr, *ptr1;
    int Tn = NUM;               // N
    int Tm = NUM / Tn;          // 1


    float *NDIRECT_input = (float*)_mm_malloc((size_t)NUM * CONV_C_1X1 * 32 *sizeof(float), 64);
    

    int input_HW_size = H * W;

    int Num_K_block = K /CONV_K_1X1;
    int Num_C_block = C / CONV_C_1X1;
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
            for (cc = ccs; cc < C_to; cc = cc + Cb)         // 0 C Cb
            {

                Cb = CONV_C_1X1;
                if (C_to - cc < CONV_C_1X1)
                    Cb = C_to - cc;

                float *buffer_input = input + ((id % Tn) / img_Tn)  * thread_num_inputs * C * input_HW_size + cc * input_HW_size + jj;      //input + id * CHW + cc* HW + 0(jj)
                float *buffer_filter = filter + cc * K;
                for ( ii = iis ; ii < K_to; ii = ii + Kb)   // 0 K Kb
                {
                    
                    Kb = CONV_K_1X1;
                    if (K_to - ii < CONV_K_1X1)
                    {
                        Kb = K_to - ii;
                    }

                    float *buffer_filter1 = buffer_filter + ii * Cb; //filter + kC + 0(cc)
                    float *buffer_output = output + ( (id % Tn) / img_Tn) *thread_num_inputs * K * input_HW_size + ii * input_HW_size + jj ;    
                    int EDGE_Kb = Kb % 12;
                    
                    int D_Kb = Kb-EDGE_Kb;
                    
                    for(nn = 0; nn < thread_num_inputs; nn++)      
                    {
                        float *buffer_output1 = buffer_output + nn * K * input_HW_size; //output + id * KHW + ii* HW + 0(jj)
                        float *buffer_input1 = buffer_input + nn * C * input_HW_size; //input + id * CHW + cc* HW + 0(jj)
                        int EDGE_HWb = HWb % 32;
                        int LEN_HWb = HWb - EDGE_HWb;
                        if(LEN_HWb > 0){

                            direct_1x1_N12M32_AVX512_pack(buffer_output1, buffer_filter1, buffer_input1, D_Kb, LEN_HWb/32, Cb, input_HW_size<<2, &NDIRECT_input[id * CONV_C_1X1 * 32], cc, EDGE_Kb);
                        }

                        if(EDGE_HWb != 0)  
                        {
                            float *temp_buffer_input = buffer_input1 + LEN_HWb;
                            float *temp_buffer_output = buffer_output1 + LEN_HWb;
                            if (EDGE_HWb == 17)
                            {
                                direct_1x1_N12M17_AVX512_pack(temp_buffer_output, buffer_filter1, temp_buffer_input, D_Kb, Cb, input_HW_size<<2, &NDIRECT_input[id * CONV_C_1X1 * 32], cc, EDGE_Kb);                            }

                            else if (EDGE_HWb == 16)
                            {
                                direct_1x1_N12M16_AVX512_pack(temp_buffer_output, buffer_filter1, temp_buffer_input, D_Kb, Cb, input_HW_size<<2, &NDIRECT_input[id * CONV_C_1X1 * 32], cc, EDGE_Kb);                            }
                            else if (EDGE_HWb == 4)
                            {
                                direct_1x1_N12M4_AVX512_pack(temp_buffer_output, buffer_filter1, temp_buffer_input, D_Kb, Cb, input_HW_size<<2, &NDIRECT_input[id * CONV_C_1X1 * 32], cc, EDGE_Kb);                            }
                        }
                        
                    }


                }
            }

        }
        
        
    }
    
    free(NDIRECT_input);
}
