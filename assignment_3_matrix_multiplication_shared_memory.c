// MP 3: Due Sunday, Dec 30, 2012 at 11:59 p.m. PST
#include    <wb.h>

#define TILE_SIZE 16

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            wbLog(ERROR, cudaGetErrorString( err ) );      \
            return -1;                                     \
        }                                                  \
    } while(0)

      
// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C, int c_rows, int c_cols, int length ) {
    //@@ Insert code to implement matrix multiplication here X
    //@@ You have to use shared memory for this MP
    
    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];  
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];  
    
    int thread_x = threadIdx.x; 
  	int thread_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y; 
 
	int row = block_y * TILE_SIZE + thread_y;
    int col = block_x * TILE_SIZE + thread_x;

  	float sum = 0;

 	if( (row < c_rows) && (col < c_cols) ) {
      
      for(int m=0; m < length/TILE_SIZE; m++) {
      
         ds_A[thread_y][thread_x] = A[ row*length + (m*TILE_SIZE + thread_x)];
         ds_B[thread_y][thread_x] = B[ col + (m*TILE_SIZE + thread_y) * c_cols];
         __syncthreads(); // everybody done copying
  
        
         for(int k=0; k<TILE_SIZE; ++k) {
            sum += ds_A[thread_y][k] * ds_B[k][thread_x];
         }
  		 __syncthreads(); // everyone done multiplying
      }
	
      C[row * c_cols + col] = sum;

    }
      
  
}      
      
int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    // matrix mult must always have dimensions [LxM] x [MxN] => [LxN]
    numCRows = numARows;
    numCColumns = numBColumns;
  
    size_t a_size = numARows * numAColumns * sizeof(float);
    size_t b_size = numBRows * numBColumns * sizeof(float);
    size_t c_size = numCRows * numCColumns * sizeof(float);
  
    //@@ Allocate the hostC matrix
    hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));
  	if( hostC == NULL ) {
  		wbLog( TRACE, "Could not allocate memory for output matrix C" );
      	return -1;
    }
    wbTime_stop(Generic, "Importing data and creating memory on host");

  
    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    wbCheck( cudaMalloc((void **) &deviceA, a_size) );
    wbCheck( cudaMalloc((void **) &deviceB, b_size) );
    wbCheck( cudaMalloc((void **) &deviceC, c_size) );

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbLog(TRACE, "Copying bytes H2D for A: ", a_size); 
    wbCheck( cudaMemcpy( deviceA, hostA, a_size, cudaMemcpyHostToDevice ) );
  
    wbLog(TRACE, "Copying bytes H2D for B: ", b_size);
    wbCheck( cudaMemcpy( deviceB, hostB, b_size, cudaMemcpyHostToDevice ) );
  
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
	int grid_width =  ( numCColumns / TILE_SIZE ) + 1 ;
	int grid_height = ( numCRows / TILE_SIZE ) + 1;

    dim3 dimBlock( TILE_SIZE, TILE_SIZE, 1 );
	dim3 dimGrid( grid_width, grid_height, 1);
  
    wbLog( TRACE, "Grid: ", grid_width, " x ", grid_height );
    wbLog( TRACE, "Blocks: ", TILE_SIZE, " x ", TILE_SIZE );
  
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    // numAColumns must be equal to numBRows, I just assume it is. 
  	wbLog( TRACE, "C Rows/C Cols/Vector length: ", numCRows, "/", numCColumns, "/", numAColumns);
	matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numCRows, numCColumns, numAColumns);

      
    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	wbCheck( cudaMemcpy( hostC, deviceC, c_size, cudaMemcpyDeviceToHost ) );
    wbLog(TRACE, "Copying bytes D2H for C: ", c_size);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree( deviceA ); cudaFree( deviceB ); cudaFree( deviceC );

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

