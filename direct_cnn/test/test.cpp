#include <malloc.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "NDIRECT2.h"

using namespace std;

static double gtod_ref_time_sec = 0.0;

void random_matrix(int m, int n, float *a)
{
  //srand48((unsigned)time(NULL));
  double drand48();
  int i,j;
  for ( i=0; i< m; i++ )
    for( j =0; j < n; j++)
    {
        a[i*n+j]= 2.0 * (float)drand48( ) - 1.0;

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

bool is_a_ge_zero_and_a_lt_b(int a, int b) 
{
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

int main(int argc, char **argv){

    
    long loop = 5;
    long N = 48;
  
    long C=strtol(argv[1], NULL, 10);
    
    long K=strtol(argv[2], NULL, 10);
    
    long H=strtol(argv[3], NULL, 10);
    
    long W = H;
    
    long R=strtol(argv[4], NULL, 10);
    
    long S = R;
    
    long stride=strtol(argv[5], NULL, 10);
    
    long pad = strtol(argv[6], NULL, 10);
    
    //long N = strtol(argv[7], NULL, 10);

    
    long out_h = (H - R + 2 * pad)/stride + 1;
    
    long out_w = out_h;
    
    long i, j;
    
    long lda = K, ldb = out_h * out_w, ldc = R * R * C;
    
    long in_size = N * C * H * W;
    
    long out_size = N * K * out_h * out_w;
    
    double start, end, cost;
    
    double ops = (double) N * K *C *out_h * out_w * R * S *1.0e-09 * 2;
    
    double run_flops;
    
    
    long per_img_batch = C * H * W; 
    
    long per_datacol_batch = out_h * out_w * C * R * S;
    
    long per_out_batch = K * out_h * out_w;
    
    
    double d_time[loop];
    
    double d_flops[loop];
    
    long loop_index;
    
    long itr = 20;

    for(loop_index = 0; loop_index<loop; loop_index++){

        float *kernel=(float*)malloc(K* C * R * S * sizeof(float));
        
        float *img=( float * ) malloc( in_size * sizeof( float ) );
        
        float *data_col=( float * ) malloc( ( N * out_h * out_w * C * R * S)*sizeof( float ) );
        
        float *i_out=( float * ) malloc( out_size * sizeof( float ) );

        float *d_out=( float * ) malloc( out_size * sizeof( float ) );
        
        
        random_matrix(K, C * R * S, kernel);
        
        random_matrix(N * C, H * W, img);

        LIB_R7_s2(H, W, N, C, img, K, R, S, kernel, pad, pad, stride, d_out);



        //direct run
        start = dclock();

        for(j=0; j<itr; j++){
            
            LIB_R7_s2(H, W, N, C, img, K, R, S, kernel, pad, pad, stride, d_out);
        }

        cost = ((dclock()- start) / (itr * 1.0));

        run_flops = ops / cost;
        
        d_time[loop_index] = cost*1000;
        
        d_flops[loop_index] = run_flops;

        

        
        
        long out_num = N * K * out_h * out_w;
        

        

        
        free(kernel);
        free(img);
        free(data_col);
       
        free(i_out);
        free(d_out);
    }
    
    printf("N=%ld, C=%ld, K=%ld, H=%ld, R=%ld, pad=%ld, stride=%ld\n", N, C, K, H, R, pad, stride);
    
    for(loop_index=0; loop_index<loop; loop_index++){
        
        printf("\n");
        
        printf("Loop=%ld:\n", loop_index);
        
        printf("    Direct_time       = %.4f ms\n", d_time[loop_index]);
        printf("    Direct_gflops     = %lf\n", d_flops[loop_index]);
        
        printf("-----------------------------------------\n");
        
    }
    
    return 0;
}

