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
    int counter = 0;
    int index;
    int check = pad_x*pad - 4;
    for (index = 0; index < check; index += 4) {
        _mm_storeu_ps(in_modified+index, zeros);
    }
    for (; index < pad_x*pad; index++){
        in_modified[index] = 0;
    }
    int row = pad;
    check = pad_x * (pad + data_size_Y) - 4;
    for (; index < check; index += pad_x) {
        int t = index;
        in_modified[t] = 0;
        t++;
        int check1 = row*data_size_X - 4;
        for (; counter < check1; counter += 4) {
            __m128 vals = _mm_loadu_ps(in+counter);
            _mm_storeu_ps(in_modified+t, vals);
            t += 4;
        }
        for (; counter < row*data_size_X; counter++){
            in_modified[t] = in[counter];
            t++;
        }
        in_modified[t] = 0;
        row++;
    }
    check = size - 4;
    for (; index < check; index += 4) {
        _mm_storeu_ps(in_modified+index, zeros);
    }
    for (; index < size; index++){
        in_modified[index] = 0;
    }

    // float flipped_kernel[KERNX*KERNY];
    // for (int n = 0; n < KERNX*KERNY; n++){
    //     flipped_kernel[n] = kernel[KERNX*KERNY-1-n];
    // }
    float flipped_kernel[12] = {kernel[8],kernel[7],kernel[6],0,
                                kernel[5],kernel[4],kernel[3],0,
                                kernel[2],kernel[1],kernel[0],0};

    __m128 kern1 = _mm_loadu_ps(flipped_kernel); //0-3
    __m128 kern2 = _mm_loadu_ps(flipped_kernel + 4); //4-7
    __m128 kern3 = _mm_loadu_ps(flipped_kernel + 8); //8-11

    // main convolution loop
    #pragma omp parallel for
    for(int y = 0; y < data_size_Y; y++){
        for(int x = 0; x < data_size_X; x++){
            float sum = 0;
            float temp[4];
            __m128 part1 = _mm_loadu_ps(in_modified + (x+y*pad_x));
            __m128 part2 = _mm_loadu_ps(in_modified + (x+(y+1)*pad_x));
            __m128 part3 = _mm_loadu_ps(in_modified + (x+(y+2)*pad_x));
            
            part1 = _mm_mul_ps(kern1, part1);
            part2 = _mm_mul_ps(kern2, part2);
            part3 = _mm_mul_ps(kern3, part3);

            __m128 parts = _mm_add_ps(_mm_add_ps(part1, part2),part3);
            _mm_storeu_ps(temp, parts);
            for (int t = 0; t < 4; t++)
                sum += temp[t];
            out[x+y*data_size_X] += sum;
        }
    }
    return 1;
}