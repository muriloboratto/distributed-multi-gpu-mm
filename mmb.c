#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define TILE_DIM 32 //Sets the tile size to 32 × 32 elements per block

extern void ABMultiply(double *a, double *b, double *c, int m, int n, int k,int lda, int ldb,int ldc, int w);

int main(int argc, char* argv[])
{
  int myRank, nRanks;
  char name[50];
  int resultlen;

  //initializing MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  MPI_Get_processor_name(name, &resultlen);

  if (argc != 4)
  {
    if (myRank == 0)
        fprintf(stderr, "Usage: %s <device_id> <matrix_size> <communication_configuration>\n", argv[0]);
  
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  
  }else if (strlen(argv[3]) != 3 || strspn(argv[3], "MCN") != 3)
  {
    if (myRank == 0)
        fprintf(stderr, "Invalid communication configuration. Use exactly 3 characters from: M, C, N.\n");
    
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  double start_time, stop_time, elapsed_time, global_elapsed_time;               
 
  int deviceId = atoi(argv[1]);          
  int matrix_size = atoi(argv[2]);      
  char communication_libraries[10];          // communication_library[i]=M,C,N --> MPI, CAM, NCCL 
  strcpy(communication_libraries,argv[3]);   // i=0 matrix A, i=1 matrix B, i=2 matrix C
  
  ncclUniqueId id;
  ncclComm_t comm;
  cudaStream_t s;

  //Get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) 
    ncclGetUniqueId(&id);
  
  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  //Picking a GPU based on "localRank"
  cudaSetDevice(deviceId);
  cudaStreamCreate(&s);

  //Initializing NCCL
  ncclCommInitRank(&comm, nRanks, id, myRank);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceId);
     
  printf("mmb Node name=%s\t nRanks=%d\t myRank=%d\t device=%d: %s\n", name, nRanks, myRank, deviceId, deviceProp.name);

  double *A, *B, *C, *lA, *lC, *d_A, *d_B, *d_C, *d_lA, *d_lC;
    
  //Default values
  int m  = matrix_size;
  int n  = matrix_size;
  int k  = matrix_size;
  int w  = TILE_DIM;
  int mi = matrix_size / nRanks;     
                                                  
  //Allocate in each CPU: B, lA, lC 
  B  = (double*) calloc (k * n, sizeof(double));
  lA = (double*) calloc (mi * k, sizeof(double));
  lC = (double*) calloc (mi * n, sizeof(double));
                                                       
  // Allocate in each GPU: d_B, d_lA, d_lC
  cudaMalloc(&d_B,   k * n * sizeof(double)) ;
  cudaMalloc(&d_lA, mi * k * sizeof(double)) ;
  cudaMalloc(&d_lC, mi * n * sizeof(double)) ;
     
  //Allocate A and C in each CPU process
  A = (double*) malloc (m * k * sizeof(double));
  C = (double*) malloc (m * n * sizeof(double));
                     
  // Allocate complete matrices d_A and d_C in each GPU.
  cudaMalloc(&d_A, m * k * sizeof(double));
  cudaMalloc(&d_C, m * n * sizeof(double));
  
  //Set initial values for matrices: CPU_0 to GPU_0
  if (myRank == 0)
  {
      for(int i = 0; i < (m*k); i++)
        A[i] = (double) 1.;
      for(int i = 0; i < (k*n); i++)
        B[i] = (double) 2.;

     //Copy data from main host memory to main device memory
     cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
     cudaMemcpy(d_B, B, k * n * sizeof(double), cudaMemcpyHostToDevice);

  }                                                                                                                     

  int loop_count = 10; // Number of benchmark iterations

  /////////////////////////////////////////////////////////////////////////
                     MPI_Barrier(MPI_COMM_WORLD);
                     start_time = MPI_Wtime();
  /////////////////////////////////////////////////////////////////////////
                   
  for (int i = 1; i <= loop_count; i++)
  {
  //////////////////////////////////////////////////////
  // 1. Sending matrices A and B from GPU_0 to all GPUs
  //////////////////////////////////////////////////////
    
     // 1.1 - scattering matrix A
     switch (communication_libraries[0])
     { 
        case 'M':
             // 1.1.option 'M' - MPI: scatter 
             (myRank == 0) && cudaMemcpy(A, d_A, m * k * sizeof(double), cudaMemcpyDeviceToHost);
             MPI_Scatter(A, mi * k, MPI_DOUBLE, lA, mi * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);    
             cudaMemcpy(d_lA, lA, mi * k * sizeof(double), cudaMemcpyHostToDevice);
             break;
        case 'C':
             // 1.1.option 'C' - CUDA AWARE MPI: scatter
             MPI_Scatter(d_A, mi * k, MPI_DOUBLE, d_lA, mi * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);    
             break;
        //case 'N':
              // 1.1.option 'N' - NCCL: broadcast 
        //    ncclBcast(d_A, m*k, ncclDouble, 0,comm, s); 
        //    d_lA = &d_A[myRank*mi*k]; 
        //    cudaStreamSynchronize(s);
        //    break;
        case 'N':
             // 1.1.option 'N' - NCCL: broadcast 
             ncclBcast(d_A, m*k, ncclDouble, 0, comm, s);
             cudaStreamSynchronize(s);
             cudaMemcpy(d_lA, d_A + (size_t)myRank * mi * k, (size_t)mi * k * sizeof(double), cudaMemcpyDeviceToDevice);
             break;
     } 
     
     // 1.2 - broadcasting matrix B
     switch (communication_libraries[1])
     { 
       case 'M':
            // 1.2.option 'M' - MPI: broadcast 
            (myRank == 0) && cudaMemcpy(B, d_B, k * n * sizeof(double), cudaMemcpyDeviceToHost);
            MPI_Bcast(B,k*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
            (myRank != 0) && cudaMemcpy(d_B, B, k * n * sizeof(double), cudaMemcpyHostToDevice);
            break;
       case 'C':
            // 1.2.option 'C' - CUDA Aware MPI: broadcast 
            MPI_Bcast(d_B,k*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
            break;
       case 'N':
            // 1.2.option 'N' - NCCL: broadcast 
            ncclBcast(d_B, k*n, ncclDouble, 0,comm, s);
            cudaStreamSynchronize(s);
            break;            
     }
      
   //////////////////////////////////////////////////////
   // 2. Calling the kernel
   //////////////////////////////////////////////////////
   
    int lda = k;
    int ldb = n;
    int ldc = n;
   
    ABMultiply(d_lA, d_B, d_lC, mi, n, k, lda, ldb, ldc, w);
  
   //////////////////////////////////////////////////////
   // 3. Gathering matrix C from all GPUs towards GPU_0
   //////////////////////////////////////////////////////
     
    switch (communication_libraries[2])
    { 
        case 'M':
             // 3.option 'M' - MPI: gather 
             cudaMemcpy(lC, d_lC, mi * n * sizeof(double), cudaMemcpyDeviceToHost);
             MPI_Gather(lC, mi * n, MPI_DOUBLE, C, mi * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
             (myRank == 0) && cudaMemcpy(d_C, C, m * n * sizeof(double), cudaMemcpyHostToDevice);
             break;
             
        case 'C':
             // 3.option 'C' - CUDA-AWARE MPI: gather 
             MPI_Gather(d_lC, mi * n, MPI_DOUBLE, d_C, mi * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
             break;
       
        case 'N':
             // 3.option 'N' - NCCL: Allgather 
             ncclAllGather(d_lC, d_C, mi * n, ncclDouble, comm, s);      
             cudaStreamSynchronize(s);
             break;                               
    }
    
  }
  
  ///////////////////////////////////////////////////////////////
                      MPI_Barrier(MPI_COMM_WORLD);
                      stop_time = MPI_Wtime(); 
  //////////////////////////////////////////////////////////////
  
  elapsed_time = stop_time - start_time;
  MPI_Reduce(&elapsed_time, &global_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (myRank == 0)
  {
    double avg_execution_time = global_elapsed_time / (double)loop_count;
    printf("\n\n mmb myRank=%d\t Libraries=%s\t RESULT: Matrix size: %10d\t Time (seconds): %8.3f\n\n", myRank, communication_libraries, matrix_size, avg_execution_time);
 
  }

  cudaFree(d_B);
  cudaFree(d_lA);
  cudaFree(d_lC);
  cudaFree(d_A);
  cudaFree(d_C);
                                                         
  free(A);
  free(B);
  free(C);
  free(lA);
  free(lC);
                                                                                         
  ncclCommDestroy(comm);
  cudaStreamDestroy(s);
  MPI_Finalize();

  return 0;

}

