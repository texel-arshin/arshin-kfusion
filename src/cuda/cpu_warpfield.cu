#include <kfusion/cuda/device_array.hpp>
#include "safe_call.hpp"
#include "device.hpp"
#include "kfusion/cpu/cpu_knn.hpp"
#include "kfusion/cpu/cpu_warpfield.hpp"
#include "kfusion/cuda/warpfield.hpp"
#include <stdio.h>


namespace cpu_warp {
    warp::WarpField WarpField_Host::get_gpu_warpfield() {
        int node_count = warpfield_ptr->node_count;
        gpu_node_coords_data.create(node_count * sizeof(float) * 3);
        gpu_node_weights_data.create(node_count * sizeof(float));
        gpu_node_params_data.create(node_count * sizeof(DualQuaternion));
        gpu_warpfield_ptr->warp_coords = gpu_node_coords_data.ptr<float>();
        gpu_warpfield_ptr->warp_weights = gpu_node_weights_data.ptr<float>();
        gpu_warpfield_ptr->warp_params = gpu_node_params_data.ptr<DualQuaternion>();
        gpu_warpfield_ptr->node_count = warpfield_ptr->node_count;
        cudaMemcpy(gpu_warpfield_ptr->warp_coords, warpfield_ptr->warp_coords, warpfield_ptr->node_count * sizeof(float) * 3, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_warpfield_ptr->warp_weights, warpfield_ptr->warp_weights, warpfield_ptr->node_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_warpfield_ptr->warp_params, warpfield_ptr->warp_params, warpfield_ptr->node_count * sizeof(DualQuaternion),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_warpfield_ptr->knn_field_gpu.field_ptr, warpfield_ptr->knn_field_gpu.field_ptr,
                   knn_field.get_field_total_size() * sizeof(cpu_knn::DiscreteBruteFieldKNN::elem_type), cudaMemcpyHostToDevice);
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());
        return *gpu_warpfield_ptr;
    }
}