#ifndef boundary_hpp
#define boundary_hpp

#include "compensation.hpp"
template <typename T_data>
__global__ void get_boundary(
    T_data* input, char* output, char b_tag, int rank, int width, int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || z == 0 || z == depth - 1) {
    output[idx] = 0;
    return;
  }
  T_data cur_idx = input[idx];
  // x is the slowest dimension
  T_data idx_left = input[(x - 1) + y * width + z * width * height];
  T_data idx_right = input[(x + 1) + y * width + z * width * height];
  T_data idx_up = input[x + (y - 1) * width + z * width * height];
  T_data idx_down = input[x + (y + 1) * width + z * width * height];
  T_data idx_front = input[x + y * width + (z - 1) * width * height];
  T_data idx_back = input[x + y * width + (z + 1) * width * height];

  if (cur_idx != idx_left || cur_idx != idx_right || cur_idx != idx_up || cur_idx != idx_down ||
      cur_idx != idx_front || cur_idx != idx_back) {
    output[idx] = b_tag;
  }
  else {
    output[idx] = 0;
  }
}

template <typename T_data>
__global__ void get_boundary_and_sign_map(
    T_data* input, char* output, char* sign_map, char b_tag, int rank, int width, int height,
    int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || z == 0 || z == depth - 1) {
    output[idx] = 0;
    return;
  }
  T_data cur_idx = input[idx];
  // check the boundary

  // x is the slowest dimension
  T_data idx_left = input[(x - 1) + y * width + z * width * height];
  T_data idx_right = input[(x + 1) + y * width + z * width * height];
  T_data idx_up = input[x + (y - 1) * width + z * width * height];
  T_data idx_down = input[x + (y + 1) * width + z * width * height];
  T_data idx_front = input[x + y * width + (z - 1) * width * height];
  T_data idx_back = input[x + y * width + (z + 1) * width * height];

  //   T_data signs[6] = {idx_left-cur_idx, idx_right-cur_idx,
  //                     idx_up-cur_idx, idx_down-cur_idx,
  //                     idx_front-cur_idx, idx_back-cur_idx};
  T_data signs[6] = {idx_up - cur_idx,    idx_down - cur_idx,  idx_left - cur_idx,
                     idx_right - cur_idx, idx_front - cur_idx, idx_back - cur_idx};

  double grad_x = abs(idx_right - idx_left) * 0.5;
  double grad_y = abs(idx_up - idx_down) * 0.5;
  double grad_z = abs(idx_front - idx_back) * 0.5;
  double max_grad = grad_x;
  if (grad_y > max_grad) max_grad = grad_y;
  if (grad_z > max_grad) max_grad = grad_z;

  if (cur_idx != idx_left || cur_idx != idx_right || cur_idx != idx_up || cur_idx != idx_down ||
      cur_idx != idx_front || cur_idx != idx_back) {
    output[idx] = b_tag;
    if (max_grad >= 1.0) {
      sign_map[idx] = 0;
      return;
    }
    for (int i = 0; i < 6; i++) {
      if (signs[i] != 0) {
        sign_map[idx] = (signs[i] > 0) ? 1 : -1;
        break;
      }
    }
  }
  else {
    output[idx] = 0;
  }
}

template <typename T_sign, typename T_boundary, typename T_index, typename T_data>
void __global__ fill_sign_and_compensation(
    T_sign* sign_map, T_boundary* boundary, T_index* edt_index, T_data magnitude, int width,
    int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;
  int eZ = edt_index[idx * 3];
  int eY = edt_index[idx * 3 + 1];
  int eX = edt_index[idx * 3 + 2];
  size_t edt_index_idx = eZ * height * width + eY * width + eX;
  if (boundary[idx] == 0) { sign_map[idx] = sign_map[edt_index_idx]; }
}

template <typename T_boundary>
void __global__ filter_boundary(
    T_boundary* orig_bounday, T_boundary* new_boundary, T_boundary tag, int width, int height,
    int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  size_t idx = x + y * width + z * width * height;
  if (orig_bounday[idx] == tag && new_boundary[idx] == tag) 
  { new_boundary[idx] = 0; }
}

#endif