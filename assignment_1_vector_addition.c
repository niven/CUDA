// MP 1
#include	<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
}

void checkError( cudaError_t err, char* file, int line ) {

	if( err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), file, line);
      // no e_x_i_t allowed in sandbox mode
	}
}


int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

  	int size = inputLength * sizeof(float);
  
    wbLog(TRACE, "The input length is ", inputLength, " (", size, " bytes)");

	wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	checkError( cudaMalloc((void **) &deviceInput1, size), __FILE__, __LINE__  );
	checkError( cudaMalloc((void **) &deviceInput2, size), __FILE__, __LINE__  );
	checkError( cudaMalloc((void **) &deviceOutput, size), __FILE__, __LINE__  );

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	checkError( cudeMemcpy( hostInput1, deviceInput1, size, cudeMemcpyHostToDevice ) );
	checkError( cudeMemcpy( hostInput2, deviceInput2, size, cudeMemcpyHostToDevice ) );

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here


    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	checkError( cudaMemcpy( hostOutput, deviceOutput, size, cudeMemcpyDeviceToHost ) );
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree( deviceInput1 ); cudaFree( deviceInput2 ); cudaFree( deviceOutput );
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}