#pragma once

#include "kfusion/dual_quaternion.hpp"
#include "kfusion/cpu/cpu_knn.hpp"

namespace cpu_warp {
    struct CpuNodeStorage {
        int data_node_count, all_node_count;
        std::shared_ptr<float> node_coords, node_weights;
        std::shared_ptr <DualQuaternion> node_params;

        CpuNodeStorage(): data_node_count(0), all_node_count(0) {}

        void create_mem(int node_count);

        void reset_nodes(float *warp_coords, float *warp_weights,
                         int node_count, int data_node_count);

        void reset_data_nodes(float *warp_coords, float *warp_weights);

        void reset_node_params(DualQuaternion *warp_params);

        void reset_data_node_params(DualQuaternion *warp_params);

    };
}