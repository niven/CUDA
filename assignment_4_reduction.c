// MP 4 Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];
// Due Tuesday, January 15, 2013 at 11:59 p.m. PST

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void total(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ float sub_array[BLOCK_SIZE];
  
  	int tix = threadIdx.x;
  	int index = blockIdx.x * BLOCK_SIZE + tix;
  
  	sub_array[tix] = index < len ? input[index] : 0.0f;
   	__syncthreads(); // all done copying

  	//@@ Traverse the reduction tree
    int stride = 1;
  	while( stride < BLOCK_SIZE ) {
      int idx = (tix+1) * stride * 2 -1;
      if( idx < BLOCK_SIZE ) {
        sub_array[idx] += sub_array[ idx-stride ];        
      }
      stride *= 2;
      __syncthreads();
  	}
      
      
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    output[ blockIdx.x ] = sub_array[BLOCK_SIZE - 1];
}

int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);
  
    numOutputElements = ceil((float)numInputElements / (float)BLOCK_SIZE);

  	size_t size_input = numInputElements * sizeof(float);
  	size_t size_output = numOutputElements * sizeof(float);

	hostOutput = (float*) malloc( size_output );

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The max index of output elements is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here X
    wbCheck( cudaMalloc((void **) &deviceInput, size_input ) );
    wbCheck( cudaMalloc((void **) &deviceOutput, size_output ) );
  
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here X
	wbCheck( cudaMemcpy( deviceInput, hostInput, size_input, cudaMemcpyHostToDevice ) );
      
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here

    dim3 dimBlock( BLOCK_SIZE, 1, 1 );
	dim3 dimGrid( numOutputElements, 1, 1);
  
    wbLog( TRACE, "Grids: ", numOutputElements );
    wbLog( TRACE, "Block size: ", BLOCK_SIZE );
      
      
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here X
	total<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements);
  
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here X
	wbCheck( cudaMemcpy( hostOutput, deviceOutput, size_output, cudaMemcpyDeviceToHost ) );
    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
	for(int j=0; j<numOutputElements; j++) {
        wbLog(TRACE, "Output element ", j, " is ", hostOutput[j]);
  	}
  
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here X
	cudaFree( deviceInput ); cudaFree( deviceOutput );
  
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}

