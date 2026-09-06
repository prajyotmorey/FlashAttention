 %%writefile vector_add.cu

#include<cuda_runtime.h>
#include<iostream>

//Kernel
__global__ void add_one(float* x, float* y,int n)
{
  i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i<n)
  {
    y[i] = x[i] + 1.0f;
  }
}


int main()
{

  const int N=10,000,000

  size_t bytes =  N * sizeof(float);

  // Host memory
   float* h_x = new float[N];

   floa
}