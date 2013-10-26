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
    //TODO loop is inefficient, use SIMD
    int counter = 0;
	int index;
    for (index = 0; index < pad_x*pad; index++) {
    	in_modified[index] = 0; //first #pad rows are all 0s
    }
    for (; index < pad_x * (pad + data_size_Y); index++) {
    	int t = index % pad_x;
    	if (t < pad || pad_x - 1 - t < pad) {
    		in_modified[index] = 0; //first #pad and last #pad column are 0s
    	}
    	else {
    		in_modified[index] = in[counter]; //rest of the values are from original input
    		counter++;
    	}
    }
    for (; index < size; index++) {
    	in_modified[index] = 0; //fill last #pad rows with 0s
    }
    // main convolution loop
    for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
        for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
            float temp = 0; //by adding the values up first, we utilize register blocking.
            for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
                for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
                    // only do the operation if not out of bounds
                    temp += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in_modified[(x+i+pad) + (y+j+pad)*pad_x];
                }
            }
            out[x+y*data_size_X] += temp;
        }
    }
	return 1;
}

