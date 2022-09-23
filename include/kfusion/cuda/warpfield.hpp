#ifndef KFUSION_WARPFIELD_HPP
#define KFUSION_WARPFIELD_HPP

#include <kfusion/cuda/device_array.hpp>
#include <memory>
#include "knn.hpp"
#include "kfusion/dual_quaternion.hpp"

namespace warp {

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
//        __host__ __device__ void zero_val() {
//            rot = make_float4(0.,0.,0.,0.);
//            tr = make_float4(0.,0.,0.,0.);
//        }
//        //__kf_device__ DualQuaternion& operator*(float w);
//        //__kf_device__ DualQuaternion& operator+=(const DualQuaternion &a);
//    };

    typedef DualQuaternion DerivativeWrtDQ;

    struct WarpFieldCache
    {
        knn::DiscreteBruteFieldKNN::elem_type *V_knn;
        //V = warp(Par), V - 3-dim, Par - 5x8 dim; dV/dPar - 3x(5*8)
        float dV_wrt_params[3*knn::DiscreteBruteFieldKNN::k*DualQuaternion::PARAMS_LENGTH];
        float dN_wrt_params[3*knn::DiscreteBruteFieldKNN::k*DualQuaternion::ROTATION_PARAMS_LENGTH];
        float3 last_V, last_N;
        float pretransformed_V[3], pretransformed_N[3];
        bool warp_applied;
        float warp_omega_weights[knn::DiscreteBruteFieldKNN::k];
        float b_inv_norm, b_inv_norm_sq;
        //float dc_wrt_db[DualQuaternion::PARAMS_LENGTH*DualQuaternion::PARAMS_LENGTH];
        DerivativeWrtDQ dV_wrt_db[3], dN_wrt_db[3];
        DualQuaternion b;
        //kfusion::device::Aff3f vol2cam;
    };

    struct WarpField {

        knn::DiscreteBruteFieldKNN knn_field_gpu;
        float *warp_coords, *warp_weights;
        DualQuaternion *warp_params;
        int node_count;

        WarpField(knn::DiscreteBruteFieldKNN knn_field_gpu): knn_field_gpu(knn_field_gpu) {}

        WarpField(float *warp_coords, float *warp_weights, DualQuaternion *warp_params,
                  int node_count, knn::DiscreteBruteFieldKNN knn_field_gpu) :
                knn_field_gpu(knn_field_gpu),
                warp_coords(warp_coords),
                warp_weights(warp_weights),
                warp_params(warp_params),
                node_count(node_count)
        {}

        __kf_device__ float3 warp_vertex(WarpFieldCache &wfc, float3 vertex) const;
        __kf_device__ float3 warp_normal(WarpFieldCache &wfc, float3) const;
        __kf_device__ void compute_dpN_wrt_db(WarpFieldCache &wfc) const;
        __kf_device__ void compute_dpV_wrt_db(WarpFieldCache &wfc) const;
        __kf_device__ void compute_dN_wrt_db(WarpFieldCache &wfc) const;
        __kf_device__ void compute_dV_wrt_db(WarpFieldCache &wfc) const;
        __kf_device__ void compute_dV_wrt_dq(WarpFieldCache &wfc) const;
        __kf_device__ void compute_dN_wrt_dq(WarpFieldCache &wfc) const;
        __kf_device__ void warp_forward(WarpFieldCache &wfc, float3 *vertex_ptr_in, float3 *vertex_ptr_out, float3 *normal_ptr_in, float3* normal_ptr_out) const;
        __kf_device__ void warp_backward(WarpFieldCache &wfc, float *gradV_memory, float *gradN_memory, int64_t *knn_memory) const;
    };

    __device__ float4 warp_vertex_nograd(const WarpField &wf, const float3 vertex);

    class WarpField_Host {
        std::shared_ptr<WarpField> warpfield_ptr;
        knn::DiscreteBruteFieldKNN_Host knn_field;
        kfusion::cuda::CudaData node_coords_data, node_weights_data, node_params_data;
    public:
        WarpField_Host(int3 volume_dims, float volume_cell_size, int knn_field_divider) : knn_field(volume_dims,volume_cell_size, knn_field_divider) {
            warpfield_ptr = std::shared_ptr<WarpField>(new WarpField(knn_field.get_field()));
            warpfield_ptr->warp_params = nullptr;
            warpfield_ptr->warp_coords = nullptr;
        }

//        ~WarpField_Host()
//        {
//            delete warpfield_ptr;
//        }

        void reset_nodes(float *warp_coords, float *warp_weights, int node_count);

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
                            float *gradV_memory, float *gradN_memory, int64_t *knn_memory);

        void warp_nograd(int verts_count, const float *verts_in, float *verts_out, const float *norms_in, float *norms_out);

        void dual_quaternion_blending(int verts_count, const float *verts_in, float *params_out);

        void get_unsupported_mask(int verts_count, const float *verts_in, unsigned char *mask_out);
    };
}
#endif
