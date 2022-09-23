#include "kfusion/cpu/cpu_node_storage.hpp"
#include "kfusion/cpu/cpu_knn.hpp"
#include "kfusion/dual_quaternion.hpp"

namespace cpu_warp {

    void CpuNodeStorage::create_mem(int node_count)
    {
        node_coords = std::shared_ptr<float>(new float[node_count*3]);
        node_weights = std::shared_ptr<float>(new float[node_count]);
        node_params = std::shared_ptr<DualQuaternion>(new DualQuaternion[node_count]);
    }

    void CpuNodeStorage::reset_nodes(float *warp_coords, float *warp_weights,
                     int node_count, int data_node_count)
    {
        if (node_count<data_node_count)
        {
            throw std::invalid_argument("node_count<data_node_count");
        }
        if (this->all_node_count != node_count)
        {
            create_mem(node_count);
            this->all_node_count = node_count;
        }
        this->data_node_count = data_node_count;
        std::memcpy(node_coords.get(), warp_coords, node_count * sizeof(float) * 3);
        std::memcpy(node_weights.get(), warp_weights, node_count * sizeof(float));
    }

    void CpuNodeStorage::reset_data_nodes(float *warp_coords, float *warp_weights)
    {
        std::memcpy(node_coords.get(), warp_coords, data_node_count * sizeof(float) * 3);
        std::memcpy(node_weights.get(), warp_weights, data_node_count * sizeof(float));
    }

    void CpuNodeStorage::reset_node_params(DualQuaternion *warp_params)
    {
        std::memcpy(node_params.get(), warp_params, all_node_count * sizeof(DualQuaternion));
    }

    void CpuNodeStorage::reset_data_node_params(DualQuaternion *warp_params)
    {
        std::memcpy(node_params.get(), warp_params, data_node_count * sizeof(DualQuaternion));
    }
}