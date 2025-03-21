#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <float.h>

#include <cmath>

// width , height  are planes
// depth is the len
// first dimension
__global__ void edt_depth(
    int* d_output, size_t stride, uint rank, uint d, uint len, uint width, uint height,
    size_t width_stride, size_t height_stride)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  uint y = blockIdx.y * blockDim.y + threadIdx.y;  // height
  if (x >= width || y >= height) return;

  // supposed d_output has been initialized

  int* d_output_start = d_output + x * width_stride + y * height_stride;
  int l = -1, ii, maxl, idx1, idx2, jj;
  // __shared__ int f[512][3];
  // __shared__ int g[512];
  int f[512][3];
  int g[512];

  // __shared__ int shared_coor[3*16*16];
  // int tid = threadIdx.y * blockDim.x + threadIdx.x;

  // int *coor = shared_coor + tid * 3;
  int coor[3];
  coor[d] = 0;
  coor[(d + 1) % 3] = x;
  coor[(d + 2) % 3] = y;
  // __syncthreads();

  for (ii = 0; ii < len; ii++) {
    for (jj = 0; jj < rank; jj++) {
      f[ii][jj] = d_output_start[ii * stride + jj];
      // d_output_start[ii * stride + jj] = coor[jj];
    }
  }
  // __syncthreads();
  // return;

  for (ii = 0; ii < len; ii++) {
    if (f[ii][0] >= 0) {
      int fd = f[ii][d];
      int wR = 0.0;
      for (jj = 0; jj < rank; jj++) {
        if (jj != d) {
          int tw = (f[ii][jj] - coor[jj]);
          wR += tw * tw;
        }
      }
      while (l >= 1) {
        int a, b, c, uR = 0.0, vR = 0.0, f1;
        idx1 = g[l];
        f1 = f[idx1][d];
        idx2 = g[l - 1];
        a = f1 - f[idx2][d];
        b = fd - f1;
        c = a + b;
        for (jj = 0; jj < rank; jj++) {
          if (jj != d) {
            int cc = coor[jj];
            int tu = f[idx2][jj] - cc;
            int tv = f[idx1][jj] - cc;
            uR += tu * tu;
            vR += tv * tv;
          }
        }
        if (c * vR - b * uR - a * wR - a * b * c <= 0.0) { break; }
        --l;
      }
      ++l;
      g[l] = ii;
    }
  }
  maxl = l;
  if (maxl >= 0) {
    l = 0;
    for (ii = 0; ii < len; ii++) {
      int delta1 = 0.0, t;
      for (jj = 0; jj < rank; jj++) {
        t = jj == d ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];

        delta1 += t * t;
      }
      while (l < maxl) {
        int delta2 = 0.0;
        for (jj = 0; jj < rank; jj++) {
          t = jj == d ? f[g[l + 1]][jj] - ii : f[g[l + 1]][jj] - coor[jj];

          delta2 += t * t;
        }
        if (delta1 <= delta2) break;
        delta1 = delta2;
        ++l;
      }
      idx1 = g[l];
      for (jj = 0; jj < rank; jj++) d_output_start[ii * stride + jj] = f[idx1][jj];
    }
  }
}

__global__ void init_edt_3d(
    char* input, int* output, char b_tag, int rank, uint width, uint height, uint depth)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;  // slowest dimension 
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;

  if (input[idx] == b_tag) {
    output[idx * rank] = z;
    output[idx * rank + 1] = y;
    output[idx * rank + 2] = x;
  }
  else {
    output[idx * rank] = -1;
    // output[idx*rank + 1] = 0;
    // output[idx*rank + 2] = 0;
  }
}

__global__ void calculate_distance(
    int* output, float* distance, int rank, uint width, uint height, uint depth)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;
  double d = 0;
  d += (output[idx * rank] - z) * (output[idx * rank] - z);
  d += (output[idx * rank + 1] - y) * (output[idx * rank + 1] - y);
  d += (output[idx * rank + 2] - x) * (output[idx * rank + 2] - x);
  distance[idx] = sqrt(d);
}



// search the current dimension and fins the cloest boundary point. 
// the current dimension is preset to zero if it's not a boundary point 
// if its a boundary point, no need to search, simply return. // or juse use bounday to check?(could be less efficient)
__global__ void edt_brute_force()
{

  

}