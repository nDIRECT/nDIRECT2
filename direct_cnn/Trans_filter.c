#include "NDIRECT2.h"


void NDIRECT2_flush()
{

  float *flush = ( float * ) malloc( 10000 * 10000* sizeof( float ) );
  float *dirty = ( float * ) malloc( 10000 * 10000* sizeof( float ) );
  #pragma omp parallel for num_threads(NUM)
  for(int i =0 ; i < 10000 * 10000; i ++)
  {
  	flush[i] = 0.5 + 0.00001 * (i % 100) ;
  	dirty[i] = 0.00001 * (i % 1000) + 0.123;
  }

  #pragma omp parallel for num_threads(NUM)
  for(int i =0 ; i < 10000 * 10000; i ++)
  {
  	dirty[i] = dirty[i % 64] + 0.1 * flush[i];
  }
  free(flush);
  free(dirty);
}

void transform2_filter_1x1(const int outch, const int inch, float* kernel, float* out_kernel)
{
    int mr = CONV_K_1x1, cr = CONV_C_1x1,zr;

    int i, j , k, ii,jj,kk, h,w,z;
    int st = 0;

    for(j = 0; j < inch; j = j + cr)
    {
        cr = CONV_C_1x1;
        if(inch - j < CONV_C_1x1)
            cr = inch - j;
        for(i = 0; i < outch; i = i + mr)
        {    
            zr = 12;
            mr = CONV_K_1x1;
            if(outch - i < CONV_K_1x1)
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

void transform2_filter_3x3(const int outch, const int inch, float* kernel, float* out_kernel)
{
    int mr = 32, cr = CONV_C_3x3;

    int i, j , k, ii,jj,kk, h,w;
    int st = 0;

    for(j = 0; j < inch; j = j + cr)
    {
        cr = CONV_C_3x3;
        if(inch - j < CONV_C_3x3)
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

void transform2_filter_7x7(const int outch, const int inch, const int k_h, 
                const int k_w, float* kernel, float* out_kernel)
{
    int mr = 16, cr = CONV_C_7x7;

    int i, j , k, ii,jj,kk, h,w;
    int st = 0;

    for(j = 0; j < inch; j = j + cr)
    {

        cr = CONV_C;
        if(inch - j < CONV_C_7x7)
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

