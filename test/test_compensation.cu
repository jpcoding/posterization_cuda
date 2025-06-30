#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
// #include "SZ3/api/sz.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/quantizer/LinearQuantizer.hpp"
#include<algorithm>
#include "compensation.hpp"
#include "boundary.hpp" 
#include "edt.hpp"

namespace SZ=SZ3; 

template <typename Type>
void verify(Type *ori_data, Type *data, size_t num_elements, double &psnr, double &nrmse, double &max_diff) {
    size_t i = 0;
    double Max = ori_data[0];
    double Min = ori_data[0];
    max_diff = fabs(data[0] - ori_data[0]);
    double diff_sum = 0;
    double maxpw_relerr = 0;
    double sum1 = 0, sum2 = 0, l2sum = 0;
    for (i = 0; i < num_elements; i++) {
        sum1 += ori_data[i];
        sum2 += data[i];
        l2sum += data[i] * data[i];
    }
    double mean1 = sum1 / num_elements;
    double mean2 = sum2 / num_elements;

    double sum3 = 0, sum4 = 0;
    double sum = 0, prodSum = 0, relerr = 0;

    double *diff = (double *)malloc(num_elements * sizeof(double));

    for (i = 0; i < num_elements; i++) {
        diff[i] = data[i] - ori_data[i];
        diff_sum += data[i] - ori_data[i];
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];
        double err = fabs(data[i] - ori_data[i]);
        if (ori_data[i] != 0) {
            relerr = err / fabs(ori_data[i]);
            if (maxpw_relerr < relerr) maxpw_relerr = relerr;
        }

        if (max_diff < err) max_diff = err;
        prodSum += (ori_data[i] - mean1) * (data[i] - mean2);
        sum3 += (ori_data[i] - mean1) * (ori_data[i] - mean1);
        sum4 += (data[i] - mean2) * (data[i] - mean2);
        sum += err * err;
    }
    double std1 = sqrt(sum3 / num_elements);
    double std2 = sqrt(sum4 / num_elements);
    double ee = prodSum / num_elements;
    double acEff = ee / std1 / std2;

    double mse = sum / num_elements;
    double sse = sum; // sum of square error
    double range = Max - Min;
    psnr = 20 * log10(range) - 10 * log10(mse);
    nrmse = sqrt(mse) / range;

    double normErr = sqrt(sum);
    double normErr_norm = normErr / sqrt(l2sum);

    printf("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
    printf("Max absolute error = %.2G\n", max_diff);
    printf("Max relative error = %.2G\n", max_diff / (Max - Min));
    printf("Max pw relative error = %.2G\n", maxpw_relerr);
    printf("PSNR = %f, NRMSE= %.10G\n", psnr, nrmse);
    printf("normError = %f, normErr_norm = %f\n", normErr, normErr_norm);
    printf("acEff=%f\n", acEff);
    printf("SSE=%f\n", sse);
    printf("MSE=%f\n", mse);
    //        printf("errAutoCorr=%.10f\n", autocorrelation1DLag1<double>(diff, num_elements, diff_sum / num_elements));
    free(diff);
}

void run_cuda(int* quant_inds, float* quantized_data, size_t size, uint width, uint height, uint depth, double magnitude, bool use_chunck= true)
{
    // allocate memory on the device
    int* d_quant_inds;
    float* d_quantized_data;
    char* d_boundary;
    char* d_boundary_neutral;
    float* distance_edge;
    int* index_edge;
    float* distance_neutral;
    int* index_neutral;
    char* d_sign_map;
    

    cudaMalloc(&d_quant_inds, size*sizeof(int));
    cudaMalloc(&d_quantized_data, size*sizeof(float));
    cudaMalloc(&d_boundary, size*sizeof(char));
    cudaMalloc(&d_boundary_neutral, size*sizeof(char));
    cudaMalloc(&distance_edge, size*sizeof(float));
    cudaMalloc(&index_edge, size*sizeof(int)*3);
    cudaMalloc(&distance_neutral, size*sizeof(float));
    cudaMalloc(&index_neutral, size*sizeof(int)*3);
    cudaMalloc(&d_sign_map, size*sizeof(char));



    // copy the data to the device
    cudaMemcpy(d_quant_inds, quant_inds, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_quantized_data, quantized_data, size*sizeof(float), cudaMemcpyHostToDevice);


    auto start = std::chrono::high_resolution_clock::now();
    // boundary detect 
    {
        char b_tag = 1;
        dim3 block(8, 8, 8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);
        // get_boundary<int><<<grid, block>>>(d_quant_inds, d_boundary, b_tag, 3, width, height, depth); 
        // get_sign_map<<<grid, block>>>(d_quant_inds,d_boundary,  d_sign_map, 3, width, height, depth);
        get_boundary_and_sign_map<int><<<grid, block>>>(d_quant_inds, d_boundary, d_sign_map, b_tag, 3, width, height, depth); 
        // cudaDeviceSynchronize();
        // use sign map to create neutral boundary 
        // fill boundary and compensation map 
        // dump the boundary to the host 
        if(0)
        {
            std::vector<char> boundary(size, 0);
            cudaMemcpy(boundary.data(), d_boundary, size*sizeof(char), cudaMemcpyDeviceToHost);
            SZ::writefile("boundary.uint8", boundary.data(), size);
        }
        if(0)
        {
            std::vector<char> sign_map(size, 0);
            cudaMemcpy(sign_map.data(), d_sign_map, size*sizeof(char), cudaMemcpyDeviceToHost);
            SZ::writefile("sign_map.uint8", sign_map.data(), size);
        }
    }
    // edt core 1
    if(1){
        if(!use_chunck) edt_3d(d_boundary, index_edge , distance_edge,   width, height, depth);
        // edt_3d_bf(d_boundary, index_edge , distance_edge,   width, height, depth);
        if(use_chunck)  edt_3d_chunck(d_boundary, index_edge , distance_edge,   width, height, depth);

        if(0)
        {
            std::vector<float> distance_host(size, 0);  
            cudaMemcpy(distance_host.data(), distance_edge, size*sizeof(float), cudaMemcpyDeviceToHost);
            SZ::writefile("distance_edge.dat", distance_host.data(), size);

            std::vector<int> index_host(size*3, 0);
            cudaMemcpy(index_host.data(), index_edge, size*sizeof(int)*3, cudaMemcpyDeviceToHost);
            SZ::writefile("index_edge.dat", index_host.data(), size*3);
        }
    }

    if(1){
        // fill sign and compensation 
        dim3 block(8, 8, 8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);
        fill_sign_and_compensation<<<grid, block>>>(d_sign_map, d_boundary, index_edge,magnitude, width, height, depth);
        cudaDeviceSynchronize();
        if(0)
        {
            std::vector<char> sign_map(size, 0);
            cudaMemcpy(sign_map.data(), d_sign_map, size*sizeof(char), cudaMemcpyDeviceToHost);
            SZ::writefile("sign_map.uint8", sign_map.data(), size);
        }
    }

    if(1){
        char b_tag = 1; 
        dim3 block(8, 8, 8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);
        get_boundary<char><<<grid, block>>>(d_sign_map, d_boundary_neutral, b_tag, 3, width, height, depth);
        cudaDeviceSynchronize();
        // filter boundary 
        if(1)filter_boundary<<<grid, block>>>(d_boundary, d_boundary_neutral, b_tag, width, height, depth);
        cudaDeviceSynchronize();
        if(0)
        {
            std::vector<char> boundary(size, 0);
            cudaMemcpy(boundary.data(), d_boundary_neutral, size*sizeof(char), cudaMemcpyDeviceToHost);
            SZ::writefile("boundary_neutral.uint8", boundary.data(), size);
        }
    }

    // second round of edts
    if(1){
        if(!use_chunck) edt_3d(d_boundary_neutral, index_neutral , distance_neutral,   width, height, depth);
        if(use_chunck) edt_3d_chunck(d_boundary_neutral, index_neutral , distance_neutral,   width, height, depth);

        // edt_3d_bf(d_boundary_neutral, index_neutral , distance_neutral,   width, height, depth);

    }
    // compensation
    if(1){
        dim3 block(8, 8, 8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);
        // template <typename T_distance, typename T_data, typename T_index> 
        // __global__ void compensation_idw(char* boundary, T_distance* d_edge, T_index* idx_edge,  T_distance* d_neutral, 
        //                                 char* sign_map, T_data* quantized_data, T_data magnitude, uint width, uint height, uint depth) 
        compensation_idw<float, float, int><<<grid, block>>>(d_boundary, distance_edge, index_edge, distance_neutral,
                                         d_sign_map, d_quantized_data, magnitude, width, height, depth);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    printf("Elapsed time: %f\n", diff.count());

    // copy the data back to the host
    // cudaMemcpy(quant_inds, d_quant_inds, size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(quantized_data, d_quantized_data, size*sizeof(float), cudaMemcpyDeviceToHost);
    // free the memory
    cudaFree(d_quant_inds);
    cudaFree(d_quantized_data);
    cudaFree(d_boundary);
    cudaFree(d_boundary_neutral);
    cudaFree(distance_edge);
    cudaFree(index_edge);
    cudaFree(distance_neutral);
    cudaFree(index_neutral);
    cudaFree(d_sign_map);
}




int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <input_file> <relative_error_bound>\n", argv[0]);
        printf("Example: %s testfloat_8_8_128.dat 32768 0 0 \n", argv[0]);
        return 0;
    }
    std::filesystem::path p{argv[1]} ;
    if (!std::filesystem::exists(p)) {
        printf("File %s does not exist\n", argv[1]);
        return 0;
    }
    size_t file_size = std::filesystem::file_size(p)/sizeof(float);
    std::vector<int> quant_inds(file_size, 0);
    std::vector<float> input_data(file_size, 0);
    SZ::readfile(argv[1],  file_size, input_data.data());
    std::vector<float> input_copy(file_size, 0); 
    std::copy(input_data.begin(), input_data.end(), input_copy.begin());
    float max = *std::max_element(input_data.begin(), input_data.end());
    float min = *std::min_element(input_data.begin(), input_data.end());
    printf("max: %f, min: %f\n", max, min);
    double eb = atof(argv[2])*(max - min);
    printf("relative eb: %.6f\n", atof(argv[2]));
    printf("absolute eb: %.6f\n", eb);
    // create a linear quantizer
    auto quantizer = SZ::LinearQuantizer<float>();
    quantizer.set_eb(eb);
    bool use_chunck = bool(atoi(argv[3]));

    std::cout << "use chunck edt: " << use_chunck << std::endl; 
    // iterate the input data and quantize it
    for (size_t i = 0; i < file_size; i++) {
        quant_inds[i] = quantizer.quantize_and_overwrite(input_data[i],0)-32768;
    }
    double psnr, nrmse, max_diff;
    verify(input_copy.data(), input_data.data(), file_size, psnr, nrmse, max_diff);
    run_cuda(quant_inds.data(), input_data.data(), file_size, 384, 384, 256, eb*0.9, use_chunck);
    double psnr2, nrmse2, max_diff2;
    verify(input_copy.data(), input_data.data(), file_size, psnr2, nrmse2, max_diff2);
    // write the quantized data to a file
    std::string output_file =  p.filename().string()+argv[2] + ".quant.i32";
    std::string out_data_file =  p.filename().string()+argv[2] + ".out";
    printf("Writing quantized data to %s\n", output_file.c_str());
    SZ::writefile(output_file.c_str(), quant_inds.data(),file_size);
    SZ::writefile(out_data_file.c_str(), input_data.data(),file_size);

    return 0;
}