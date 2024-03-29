#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    
    /*
        Here, we create a new float array that holds the input array, along with 0 padding on the sides. 
        This will allow us to eliminate the if statement in the main loop.
    */
    int pad = (KERNX-1)/2; //assuming KERNEL is always square, the pad amount is the same for x and y
    int pad_x = data_size_X+pad*2; // x-size of the padded input
    int pad_y = data_size_Y+pad*2; // y-size of the padded input
    int size = pad_x*pad_y;
    float in_modified[size]; //padded input of size pad_x * pad_y
    /*
        Populate in_modified with 0 pads and original in values. Ex:
        |00000|
        |0xxx0|
        |0xxx0|
        |0xxx0|
        |00000|

    */      
    __m128 zeros = _mm_setzero_ps();
    int counter = 0;
    int index;
    int check = pad_x*pad - 4;
    for (index = 0; index < check; index += 4) {
        _mm_storeu_ps(in_modified+index, zeros);
    }
    for (; index < pad_x*pad; index++){
        in_modified[index] = 0; //first #pad rows are all 0s
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
        in_modified[index] = 0; //fill last #pad rows with 0s
    }

    //flip the kernel ahead of time, to reduce the number of instructions
    float flipped_kernel[KERNX*KERNY];
    for (int n = 0; n < KERNX*KERNY; n++){
        flipped_kernel[n] = kernel[KERNX*KERNY-1-n];
    }

    // main convolution loop
    for(int y = 0; y < data_size_Y; y++){ // the y coordinate of the output location we're focusing on
        for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
            // for(int j = 0; j < KERNX; j++){ // kernel unflipped y coordinate
                // for(int i = 0; i < KERNY; i++){ // kernel unflipped x coordinate
                    // only do the operation if not out of bounds
                    out[x+y*data_size_X] = flipped_kernel[0] * in_modified[x + y*pad_x] + 
                                            flipped_kernel[1] * in_modified[(x+1) + y*pad_x] +
                                            flipped_kernel[2] * in_modified[(x+2) + y*pad_x] +
                                            flipped_kernel[3] * in_modified[x + (y+1)*pad_x] +
                                            flipped_kernel[4] * in_modified[(x+1) + (y+1)*pad_x] +
                                            flipped_kernel[5] * in_modified[(x+2)+ (y+1)*pad_x] +
                                            flipped_kernel[6] * in_modified[x + (y+2)*pad_x] +
                                            flipped_kernel[7] * in_modified[(x+1) + (y+2)*pad_x] +
                                            flipped_kernel[8] * in_modified[(x+2) + (y+2)*pad_x];
                // }
            // }
        }
    }
    return 1;
}

