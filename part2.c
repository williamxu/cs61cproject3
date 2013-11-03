#include <emmintrin.h>
#include <omp.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    int kern_cent_X = (KERNX - 1)/2;
    int kern_cent_Y = (KERNY - 1)/2;
    int pad = (KERNX-1)/2;
    int pad_x = data_size_X+pad*2;
    int pad_y = data_size_Y+pad*2;
    int size = pad_x*pad_y;
    float in_modified[size];

    __m128 zeros = _mm_setzero_ps();
    #pragma parallel omp for
    for (int x = 0; x < pad_x; x++){
        in_modified[x] = 0;
        in_modified[x+(data_size_Y+1)*pad_x] = 0;
    }
    #pragma parallel omp for
    for (int y = 0; y < data_size_Y; y++){
        in_modified[(y+1) * pad_x] = 0;
        for (int x = 0; x < data_size_X; x++){
            in_modified[x+1+(y+1)*pad_x] = in[x + y*data_size_X];
        }
        in_modified[(y+2)*pad_x - 1] = 0;
    }

    __m128 kern0 = _mm_set1_ps(kernel[8]);
    __m128 kern1 = _mm_set1_ps(kernel[7]);
    __m128 kern2 = _mm_set1_ps(kernel[6]);
    __m128 kern3 = _mm_set1_ps(kernel[5]);
    __m128 kern4 = _mm_set1_ps(kernel[4]);
    __m128 kern5 = _mm_set1_ps(kernel[3]);
    __m128 kern6 = _mm_set1_ps(kernel[2]);
    __m128 kern7 = _mm_set1_ps(kernel[1]);
    __m128 kern8 = _mm_set1_ps(kernel[0]);

    // main convolution loop
    #pragma omp parallel for
    for(int y = 0; y < data_size_Y; y++){ 
        int x;
        for(x = 0; x < data_size_X/4*4; x+=4){
            __m128 sum = _mm_setzero_ps();
            sum = _mm_add_ps(sum, _mm_mul_ps(kern0, _mm_loadu_ps(in_modified + x + y * pad_x)));           
            sum = _mm_add_ps(sum, _mm_mul_ps(kern1, _mm_loadu_ps(in_modified + x + 1 + y * pad_x)));        
            sum = _mm_add_ps(sum, _mm_mul_ps(kern2, _mm_loadu_ps(in_modified + x + 2 + y * pad_x)));         
            sum = _mm_add_ps(sum, _mm_mul_ps(kern3, _mm_loadu_ps(in_modified + x + (y + 1) * pad_x)));       
            sum = _mm_add_ps(sum, _mm_mul_ps(kern4, _mm_loadu_ps(in_modified + x + 1 + (y + 1) * pad_x)));  
            sum = _mm_add_ps(sum, _mm_mul_ps(kern5, _mm_loadu_ps(in_modified + x + 2 + (y + 1) * pad_x)));  
            sum = _mm_add_ps(sum, _mm_mul_ps(kern6, _mm_loadu_ps(in_modified + x + (y + 2) * pad_x)));     
            sum = _mm_add_ps(sum, _mm_mul_ps(kern7, _mm_loadu_ps(in_modified + x + 1 + (y + 2) * pad_x)));  
            sum = _mm_add_ps(sum, _mm_mul_ps(kern8, _mm_loadu_ps(in_modified + x + 2 + (y + 2) * pad_x)));  
            _mm_storeu_ps(out+x+y*data_size_X, sum);
        }
        for(; x < data_size_X; x++){
            out[x+y*data_size_X] = kernel[8] * in_modified[x + y*pad_x] + 
                                kernel[7] * in_modified[(x+1) + y*pad_x] +
                                kernel[6] * in_modified[(x+2) + y*pad_x] +
                                kernel[5] * in_modified[x + (y+1)*pad_x] +
                                kernel[4] * in_modified[(x+1) + (y+1)*pad_x] +
                                kernel[3] * in_modified[(x+2)+ (y+1)*pad_x] +
                                kernel[2] * in_modified[x + (y+2)*pad_x] +
                                kernel[1] * in_modified[(x+1) + (y+2)*pad_x] +
                                kernel[0] * in_modified[(x+2) + (y+2)*pad_x];
        }
    }
    return 1;
}