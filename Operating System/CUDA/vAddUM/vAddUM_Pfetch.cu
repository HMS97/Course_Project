#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void vecAdd(int n, float *a, float *b, float *c)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i+=stride)
    c[i] = a[i] + b[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y, *z;

  float msec;
  cudaEvent_t start, stop;

  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  cudaMallocManaged(&z, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Prefetch the data to the GPU
  int device;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL);
  cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL);



  int blockSize = 256;
  int numBlocks = 12; // good enough for P620 
  cudaEventRecord(start);
  vecAdd<<<numBlocks, blockSize>>>(N, x, y, z);
  cudaEventRecord(stop);
  // Wait for GPU to finish before accessing on host
  cudaEventSynchronize(stop);  

  cudaEventElapsedTime(&msec, start, stop);
  printf("Kernel time: %f ms\n", msec);
  
  cudaMemPrefetchAsync(z, N*sizeof(float), cudaCpuDeviceId, NULL);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(z[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  
  return 0;
}
