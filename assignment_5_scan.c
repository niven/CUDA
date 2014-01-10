// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}
// Due Tuesday, January 22, 2013 at 11:59 p.m. PST

#include    <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            wbLog(ERROR, cudaGetErrorString( err ) );      \            
            return -1;                                     \
        }                                                  \
    } while(0)

      
// not efficient, but all threads in a warp are occupied anyway
// also, this is mind-blowingly awesome!
/*
	What happens:
	array init:[0 1 2 3		4 		5 		6 		7]
    if>=1: add [- 0 1 2 	3 		4 		5 		6]
    if>=2: add [- - 0 (1,0) (2,1)	(3,2)	(4,3)	(5,4)]
    if>=4: add [- - - -		0		(1,0)	(2,1,0) (3,2,1,0)]	
    
    Note: I wasted around an hour because I forgot that partials needs to
    be volatile to force the writes to happen. (alternative is to __syncthreads() after every if)
*/
__device__ void scan_warp(volatile float* partials, int index) {
  
  int lane = threadIdx.x & 31;
  
  if ( lane >=  1 ) partials[index] += partials[index -  1];
  if ( lane >=  2 ) partials[index] += partials[index -  2]; 
  if ( lane >=  4 ) partials[index] += partials[index -  4]; 
  if ( lane >=  8 ) partials[index] += partials[index -  8]; 
  if ( lane >= 16 ) partials[index] += partials[index - 16]; 
  
}

// this just does a scan for the block, but returns the reduction for the block if we need to
// do more than 1 block
__device__ void scan_block(volatile float* partials, int index) {

  
  // just assume block size is multiple of 32
  __shared__ float warp_partials[ BLOCK_SIZE/32 ];
  
  int tix = threadIdx.x;
 
  scan_warp(partials, index);
  __syncthreads();
  
  // all the last threads in the warp write their result to the partials
  int last_thread_in_warp = (tix & 31) == 31;
  int warp_id = tix >> 5; // map 31, 63, 95, 127, .. to 0,1,2,3,..
  if( last_thread_in_warp ) {
    warp_partials[ warp_id ] = partials[ index ];
  }
  __syncthreads();
  
  // the first warp then does a scan on the warp_partials
  // (turns out you only need 1 warp since the max threads in a block is 1024 meaning you have a max of 32 partials per block) 
  if( warp_id == 0 ) {
    scan_warp( warp_partials, tix ); // since this is warp 0, thread indices are [0-31]
  }
  __syncthreads();
  
  // now take the partial sums and distribute them over the warps
  // (this is the fanning bit)
  if( warp_id > 0 ) {
    partials[ index ] += warp_partials[ warp_id -1 ];
  }
  __syncthreads();

}
     
__global__ void scan(float* input, float* output, float* block_sums, int len) {
    
 	__shared__ volatile float partials[ BLOCK_SIZE ];
  
  	int tix = threadIdx.x;
  	int index = blockIdx.x * BLOCK_SIZE + tix;
  
  	// copy input to shared memory
  	partials[ tix ] = index < len ? input[ index ] : 0.0f;
  	__syncthreads();

    // scan the entire block (modifies partials)
    scan_block( partials, tix );
    __syncthreads();

    // copy partial results to out    
  	if( index < len ) {
      output[ index] = partials[ tix ];
  	}
	
  	// last thread has the reduction of the entire block, save it
  	if( tix == BLOCK_SIZE -1 ) {
      	block_sums[ blockIdx.x ] = partials[ tix ];
  	}

}

// distribute all the block sums over the output
__global__ void scan_fan(float* output, float* block_sums, int len) {
  
  int tix = threadIdx.x;
  int index = blockIdx.x * BLOCK_SIZE + tix;
  
  // scan the block sums, otherwise the fanning doesn't work
  scan_block(block_sums, tix);
  
  if( blockIdx.x > 0 && index < len ) {
	  output[ index ] += block_sums[ blockIdx.x -1 ];
  }
  
}

//@@ Modify the body of this function to complete the functionality of
//@@ the scan on the device
//@@ You may need multiple kernel calls; write your kernels before this
//@@ function and call them from here
  
/*
	In case anyone is reading this and wondering what all this extra code is:
    I implemented 2 algorithms, scan_parallel_inclusive() is the one requested in
    this assigment. (The other one is nicer and also faster)
*/
__global__ void scan_parallel_inclusive(float* input, float* output, float* block_sums, int len) {

	__shared__ float partials[ BLOCK_SIZE ];
  
  	int tix = threadIdx.x;
  	int index = blockIdx.x * BLOCK_SIZE + tix;
  
  	// copy input to shared memory
  	partials[ tix ] = index < len ? input[ index ] : 0.0f;

  	__syncthreads();
 
  	// upsweep part (where we add stuff to the value of our index FROM somewhere else)
  	int stride = 1;
  	int idx = -1;
  	while( stride < BLOCK_SIZE ) {
  		idx = (tix+1) * stride * 2 -1;
      	if( idx < BLOCK_SIZE ) {
      		partials[ idx ] += partials[ idx - stride ];
        }
    	stride *= 2;
      	__syncthreads();
    }
   
  	// downsweep part (where we add the value from our index TO somewhere else)
  	// stride is now equal to BLOCK_SIZE
  	stride /= 2;
  	while( stride > 0 ) {
  		idx = (tix+1) * stride * 2 -1;
      	if( idx+stride < BLOCK_SIZE ) {
      		partials[ idx+stride ] += partials[ idx ];
       }
    	stride /= 2;
      	__syncthreads();
    }
 
  	output[ index ] = partials[ tix ];
  	__syncthreads();
  
  	// save the intermediary results
  	if( tix == BLOCK_SIZE-1 ) {
  		block_sums[ blockIdx.x ] = partials[ tix ];
    }
}


int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    float* device_block_sums;
    float* host_sum_blocks;
  
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
  
    wbTime_stop(Generic, "Importing data and creating memory on host");
  	
/* For debugging
  	for(int i=0; i<64; i++){
  		hostInput[ i ] = i;
  	}
  	
  	for(int i=1000; i<1048; i+=4) {
      wbLog(TRACE, i, " to ", i+4, " [", hostInput[i+0], ",", hostInput[i+1], ",", hostInput[i+2], ",", 
 		 hostInput[i+3], "]");
    }
*/  
    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the device
    

    int grid_width = (numElements-1) / BLOCK_SIZE +1;
	wbLog(TRACE, "Grid width/num block sums: ", grid_width);

  	size_t block_sum_size = grid_width*sizeof(float);
    wbCheck(cudaMalloc((void**)&device_block_sums, block_sum_size));
  
    dim3 dimBlock( BLOCK_SIZE, 1, 1 );
	dim3 dimGrid( grid_width, 1, 1);
//    scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, device_block_sums, numElements);
    scan_parallel_inclusive<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, device_block_sums, numElements);
	scan_fan<<<dimGrid, dimBlock>>>(deviceOutput, device_block_sums, numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
/* For debugging
    host_sum_blocks = (float*) malloc( block_sum_size );
    
  	if( host_sum_blocks == NULL ) {
      wbLog(ERROR, "Could not allocate host_sum_blocks (requested ", block_sum_size, " bytes)");
  	}
    wbCheck(cudaMemcpy(host_sum_blocks, device_block_sums, block_sum_size, cudaMemcpyDeviceToHost));

    for(int i=0; i<grid_width; i++) {
      wbLog(TRACE, "sum_blocks[", i, "] = ", host_sum_blocks[i]);
  	}
*/
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");
/* For debug
    for(int i=1000; i<1048; i+=4) {
      wbLog(TRACE, i, " to ", i+4, " [", hostOutput[i+0], ",", hostOutput[i+1], ",", hostOutput[i+2], ",", 
            hostOutput[i+3], "]");
  	}
*/
  
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree( device_block_sums );
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

