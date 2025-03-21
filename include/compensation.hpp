#ifndef compensation
#define compensation

// input: quantization index [0 as the middle point]
// output: the boundary of the quantization index
// b_tag: the boundary tag
// rank: the number of dimensions
template <typename T_data>
__global__ void get_boundary(
  T_data* input, char* output, char b_tag, int rank, uint width, uint height, uint depth)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || z == 0 || z == depth - 1) {
    output[idx] = 0;
    return;
  }
  // check the boundary
  int idx_left = (x - 1) + y * width + z * width * height;
  int idx_right = (x + 1) + y * width + z * width * height;
  int idx_up = x + (y - 1) * width + z * width * height;
  int idx_down = x + (y + 1) * width + z * width * height;
  int idx_front = x + y * width + (z - 1) * width * height;
  int idx_back = x + y * width + (z + 1) * width * height;

  if (input[idx] != input[idx_left] || input[idx] != input[idx_right] ||
      input[idx] != input[idx_up] || input[idx] != input[idx_down] ||
      input[idx] != input[idx_front] || input[idx] != input[idx_back])
    output[idx] = b_tag;
  else
    output[idx] = 0;
}

// given the quantization index map and the boundary map
// calculate the sign of the edges

template <typename T_data_sign>
__device__ char get_sign(T_data_sign data) {
  char sign = (char)(((double)data > 0.0) - ((double)data < 0.0));
  return sign;
}



__global__ void get_sign_map(
    int* d_quant_inds, char* d_boundary, char* d_sign_map, int rank, uint width, uint height,
    uint depth)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // fastest dimension depth 
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;  // slowest dimension width
  if (x >= width || y >= height || z >= depth) return;
  size_t stride_x = 1; 
  size_t stride_y = width;
  size_t stride_z = width * height;
  size_t strides[3] = {stride_x, stride_y, stride_z}; 
  int dims[3] = {width, height, depth}; // fastest to slowest 
  int idx = x*stride_x + y*stride_y + z*stride_z; // fast to slowest 
  if (d_boundary[idx] == 0) return;  // if this is not a boundary point, return
  int cur_quant_index = d_quant_inds[idx]; // the current quantization index
  int tx, ty, tz;  
  char signs[6] = {0, 0, 0, 0, 0, 0};
  int distances[6] = {0, 0, 0, 0, 0, 0};
  int gradient[3] = {0,0,0}; 
  double max_grad = -1; 
  gradient[0] = abs(d_quant_inds[idx + stride_x] - d_quant_inds[idx - stride_x]);
  gradient[1] = abs(d_quant_inds[idx + stride_y] - d_quant_inds[idx - stride_y]);
  gradient[2] = abs(d_quant_inds[idx + stride_z] - d_quant_inds[idx - stride_z]);
  for (int i = 0; i < 3; i++)
  {
    if ((gradient[i]) > max_grad)
    {
      max_grad = (gradient[i])*1.0;
    }
  }
  max_grad = max_grad * 0.5; 
  if(max_grad >= 1.0)
  {
    d_sign_map[idx] = 0 ; 
    return; 
  } 


  // first dimension 
  uint index; 

  index = 2;
  tx = x - 1; ty = y; tz = z; 
  while (tx >0)
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(cur_quant_index - d_quant_inds[cur_idx]);
      break; 
    }
    tx--;
  }
  distances[index] = x - tx -1;

  index = 3; 
  tx = x + 1; ty = y; tz = z;
  while (tx < dims[0])
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(d_quant_inds[cur_idx] -cur_quant_index);
      break; 
    }
    tx++;
  }
  distances[index] = tx - x - 1;

  // second dimension
  index = 0; 
  tx = x; ty = y - 1; tz = z;
  while (ty > 0)
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(cur_quant_index - d_quant_inds[cur_idx]);
      break; 
    }
    ty--;
  }
  distances[index] = y - ty - 1;

  index = 1; 
  tx = x; ty = y + 1; tz = z;
  while (ty < dims[1])
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(d_quant_inds[cur_idx] - cur_quant_index);
      break; 
    }
    ty++;
  }
  distances[index] = ty - y - 1;

  // third dimension
  index = 4;
  tx = x; ty = y; tz = z - 1;
  while (tz > 0)
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(cur_quant_index - d_quant_inds[cur_idx]);
      break; 
    }
    tz--;
  }
  distances[index] = z - tz - 1;

  index = 5;
  tx = x; ty = y; tz = z + 1;
  while (tz < dims[2] )
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(d_quant_inds[cur_idx] - cur_quant_index);
      break; 
    }
    tz++;
  }
  distances[index] = tz - z - 1;

  // calculate the sign map
  char sign = 0;
  int direction  = 0; 
  int min_distance = distances[0];  
  for (int i = 0; i < 6; i++)
  {
    if (distances[i] < min_distance)
    {
      min_distance = distances[i];
      direction = i;
    }
  }
  sign = (direction % 2 == 0) ? -1.0f : 1.0f; 
  sign = sign * signs[direction]; 
  d_sign_map[idx] = sign;
}

template <typename T_distance, typename T_data, typename T_index> 
__global__ void compensation_idw(char* boundary, T_distance* d_edge, T_index* idx_edge,  T_distance* d_neutral, 
                                char* sign_map, T_data* quantized_data, T_data magnitude, uint width, uint height, uint depth) 
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // fastest dimension depth 
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;  // slowest dimension width
  if (x >= width || y >= height || z >= depth) return;
  size_t stride_x = 1; 
  size_t stride_y = width;
  size_t stride_z = width * height;
  int idx = x*stride_x + y*stride_y + z*stride_z; // fast to slowest 
  size_t edge_index = idx_edge[idx*3]*stride_z + idx_edge[idx*3+1]*stride_y + idx_edge[idx*3+2]*stride_x;
  
  char sign = sign_map[edge_index]; 
  T_distance d1 = d_edge[idx]+0.5;
  T_distance d2 = d_neutral[idx]+0.5;
  double val = (1/d1) / (1/d1 + 1/d2) * sign * magnitude;
  quantized_data[idx] = val + quantized_data[idx];
}

#endif