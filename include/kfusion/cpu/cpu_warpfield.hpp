#pragma once
#include <memory>
#include "cpu_knn.hpp"
#include "kfusion/cuda/warpfield.hpp"
#include "kfusion/cuda/knn.hpp"
#include "kfusion/dual_quaternion.hpp"
#include "cpu_node_storage.hpp"

namespace cpu_warp {

//    struct DualQuaternion {
////    union
////    {
////        //float data[8];
////        //float rot_w, rot_x, rot_y, rot_z, tr_w, tr_x, tr_y, tr_z;
////    };
//        union {
//            struct {
//                float4 rot, tr;
//            };
//            float data[8];
//        };
//
//        static const int PARAMS_LENGTH = 8;
//        static const int ROTATION_PARAMS_LENGTH = 4;
//        void zero_val() {
//            rot = make_float4(0.,0.,0.,0.);
//            tr = make_float4(0.,0.,0.,0.);
//        }
//        //DualQuaternion& operator+=(const DualQuaternion &a);
//    };

    typedef DualQuaternion DerivativeWrtDQ;

    struct WarpFieldCache
    {
        cpu_knn::DiscreteBruteFieldKNN::elem_type *V_knn;
        //V = warp(Par), V - 3-dim, Par - 5x8 dim; dV/dPar - 3x(5*8)
        float dV_wrt_params[3*cpu_knn::DiscreteBruteFieldKNN::k*DualQuaternion::PARAMS_LENGTH];
        float dN_wrt_params[3*cpu_knn::DiscreteBruteFieldKNN::k*DualQuaternion::ROTATION_PARAMS_LENGTH];
        float3 last_V, last_N;
        bool warp_applied;
        float warp_omega_weights[cpu_knn::DiscreteBruteFieldKNN::k];
        float b_inv_norm, b_inv_norm_sq;
        //float dc_wrt_db[DualQuaternion::PARAMS_LENGTH*DualQuaternion::PARAMS_LENGTH];
        DerivativeWrtDQ dV_wrt_dc[3], dN_wrt_dc[3], dV_wrt_db[3], dN_wrt_db[3];
        DualQuaternion b, c;
    };

    struct WarpField {

        cpu_knn::DiscreteBruteFieldKNN knn_field_gpu;
        float *warp_coords, *warp_weights;
        DualQuaternion *warp_params;
        int node_count;

        WarpField(cpu_knn::DiscreteBruteFieldKNN knn_field_gpu):
            knn_field_gpu(knn_field_gpu) {}

        WarpField(float *warp_coords, float *warp_weights,
                  DualQuaternion *warp_params, int node_count,
                  cpu_knn::DiscreteBruteFieldKNN knn_field_gpu) :
                knn_field_gpu(knn_field_gpu), warp_coords(warp_coords),
                warp_weights(warp_weights), warp_params(warp_params),
                node_count(node_count) {}

        float3 warp_vertex(WarpFieldCache &wfc, float3 vertex) const;
        float3 warp_normal(WarpFieldCache &wfc, float3) const;
        void compute_dN_wrt_dc(WarpFieldCache &wfc) const;
        void compute_dV_wrt_dc(WarpFieldCache &wfc) const;
        void compute_dN_wrt_db(WarpFieldCache &wfc) const;
        void compute_dV_wrt_db(WarpFieldCache &wfc) const;
        void compute_dV_wrt_dq(WarpFieldCache &wfc) const;
        void compute_dN_wrt_dq(WarpFieldCache &wfc) const;
        void warp_forward(WarpFieldCache &wfc, float3 *vertex_ptr_in, float3 *vertex_ptr_out, float3 *normal_ptr_in, float3* normal_ptr_out) const;
        void warp_backward(WarpFieldCache &wfc, float *gradV_memory, float *gradN_memory, long *knn_memory) const;
    };

    float3 warp_vertex_nograd(const WarpField wf, const float3 vertex);

    class WarpField_Host {
        std::shared_ptr<WarpField> warpfield_ptr;
        std::shared_ptr<warp::WarpField> gpu_warpfield_ptr;
        CpuNodeStorage *node_storage_ptr;

        cpu_knn::DiscreteBruteFieldKNN_Host knn_field;
        knn::DiscreteBruteFieldKNN_Host gpu_knn_field;

        std::shared_ptr<float> node_coords_data, node_weights_data;
        std::shared_ptr<DualQuaternion> node_params_data;

        kfusion::cuda::CudaData gpu_node_coords_data, gpu_node_weights_data, gpu_node_params_data;
    public:
        WarpField_Host(int3 volume_dims, float volume_cell_size, int knn_field_divider, CpuNodeStorage * node_storage_ptr) :
        node_storage_ptr(node_storage_ptr),
        knn_field(volume_dims,volume_cell_size, knn_field_divider),
        gpu_knn_field(volume_dims,volume_cell_size, knn_field_divider) {
            warpfield_ptr = std::shared_ptr<WarpField>(new WarpField(knn_field.get_field()));
            warpfield_ptr->warp_params = nullptr;
            warpfield_ptr->warp_coords = nullptr;

            gpu_warpfield_ptr = std::shared_ptr<warp::WarpField>(new warp::WarpField(gpu_knn_field.get_field()));
            gpu_warpfield_ptr->warp_params = nullptr;
            gpu_warpfield_ptr->warp_coords = nullptr;

        }

        void reset_nodes(float *warp_coords, float *warp_weights, cpu_knn::DiscreteBruteFieldKNN::elem_type *field_data, int field_data_bytes, int node_count);

        warp::WarpField get_gpu_warpfield();

        void reset_node_params(DualQuaternion *warp_params);

        void check_warpfield() const
        {
            if (warpfield_ptr->warp_params == nullptr || warpfield_ptr->warp_coords == nullptr)
                throw std::runtime_error("Warpfield was not set up correctly");
        }

        WarpField get_warpfield() const
        {
            check_warpfield();
            return *warpfield_ptr;
        }

        int get_node_count() const
        {
            return warpfield_ptr->node_count;
        }

        void warp_with_grad(int verts_count, const float *verts_in, float *verts_out, const float *norms_in, float *norms_out,
                            float *gradV_memory, float *gradN_memory, long *knn_memory);

        void dual_quaternion_blending(int verts_count, const float *verts_in, float *params_out);

        void get_unsupported_mask(int verts_count, const float *verts_in, unsigned char *mask_out);
    };
}
