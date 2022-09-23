#include <kfusion/cuda/device_array.hpp>
#include "kfusion/cpu/cpu_knn.hpp"
#include "kfusion/cpu/cpu_warpfield.hpp"
#include "kfusion/cuda/warpfield.hpp"
#include "kfusion/cpu/temp_utils.hpp"
#include <stdio.h>

using namespace cpu_warp;

DualQuaternion & operator+=(DualQuaternion &a, const DualQuaternion &b) {
    a.rot += b.rot;
    a.tr += b.tr;
    return a;
}

namespace cpu_warp {

    static const float MINIMUM_SQ_NORM = 1e-10;

    void pstatus(DualQuaternion *params) {
        for (int i = 0; i < 7; i++) {
            printf("[%d] Curr param: %f %f %f %f %f %f %f %f\n", i, params[i].rot.x, params[i].rot.y, params[i].rot.z, params[i].rot.w,
                   params[i].tr.x, params[i].tr.y, params[i].tr.z, params[i].tr.w);
            printf("DATA [%d] Curr param: ", i);
            for (int j = 0; j < 8; j++) {
                printf("%f ", params[i].data[j]);
            }
            printf("\n");
        }
    }

    DualQuaternion operator*(const DualQuaternion x, const float w) {
        DualQuaternion a;
        a.rot = make_float4(x.rot.x * w, x.rot.y * w, x.rot.z * w, x.rot.w * w);
        a.tr = make_float4(x.tr.x * w, x.tr.y * w, x.tr.z * w, x.tr.w * w);
        return a;
    }




    float compute_omega_w(float3 vertex, float3 node, float weigth) {
        float3 vec = vertex - node;
        float sqdist = dot(vec, vec);
        float sqweight = weigth * weigth;
//    if (isnan(__expf(-sqdist/(2*weigth*weigth))))
//    {
//        printf("NAN in compute_omega_w, sqdist %f, weight %f, -sqdist/(2*weigth*weigth) %f\n", sqdist, weigth, -sqdist/(2*weigth*weigth));
//    }
        if (sqdist > 9 * sqweight) {
            return 0;
        }
        return expf(-sqdist / (2 * sqweight));
    }


    float3 apply_dq(DualQuaternion c, float3 vertex) {
        float3 tvec, warped_vertex;
        tvec.x = 2 * (-c.tr.w * c.rot.x + c.tr.x * c.rot.w - c.tr.y * c.rot.z + c.tr.z * c.rot.y);
        tvec.y = 2 * (-c.tr.w * c.rot.y + c.tr.x * c.rot.z + c.tr.y * c.rot.w - c.tr.z * c.rot.x);
        tvec.z = 2 * (-c.tr.w * c.rot.z - c.tr.x * c.rot.y + c.tr.y * c.rot.x + c.tr.z * c.rot.w);

        warped_vertex.x =
                (1 - 2 * c.rot.y * c.rot.y - 2 * c.rot.z * c.rot.z) * vertex.x +
                2 * (c.rot.x * c.rot.y - c.rot.w * c.rot.z) * vertex.y +
                2 * (c.rot.x * c.rot.z + c.rot.w * c.rot.y) * vertex.z;
        warped_vertex.y =
                2 * (c.rot.x * c.rot.y + c.rot.w * c.rot.z) * vertex.x +
                (1 - 2 * c.rot.x * c.rot.x - 2 * c.rot.z * c.rot.z) * vertex.y +
                2 * (c.rot.y * c.rot.z - c.rot.w * c.rot.x) * vertex.z;
        warped_vertex.z =
                2 * (c.rot.x * c.rot.z - c.rot.w * c.rot.y) * vertex.x +
                2 * (c.rot.y * c.rot.z + c.rot.w * c.rot.x) * vertex.y +
                (1 - 2 * c.rot.x * c.rot.x - 2 * c.rot.y * c.rot.y) * vertex.z;
        warped_vertex += tvec;
        return warped_vertex;
    }

    DualQuaternion compute_dqb(const WarpField wf, const float3 vertex) {
        int3 knn_field_coords = make_int3(
                __float2int_rd(vertex.x / wf.knn_field_gpu.field_cell_size),
                __float2int_rd(vertex.y / wf.knn_field_gpu.field_cell_size),
                __float2int_rd(vertex.z / wf.knn_field_gpu.field_cell_size));
        //searching NN
        cpu_knn::DiscreteBruteFieldKNN::elem_type *knn_ptr = cpu_knn::get_knn_field_coords(wf.knn_field_gpu, knn_field_coords);
        //weighting & summing
        DualQuaternion bt, ct;
        bt.zero_val();
        for (int i = 0; (i < cpu_knn::DiscreteBruteFieldKNN::k) && (knn_ptr[i] != cpu_knn::DiscreteBruteFieldKNN::UNDEFINED_OFFSET); i++) {
            int curr_offset = (int) knn_ptr[i];
            float3 curr_node_coords = make_float3(wf.warp_coords[3 * curr_offset], wf.warp_coords[3 * curr_offset + 1],
                                                  wf.warp_coords[3 * curr_offset + 2]);
            float w = compute_omega_w(vertex, curr_node_coords, wf.warp_weights[curr_offset]);
            bt += (wf.warp_params[curr_offset] * w);
        }
        //normalize
        float bt_norm = dot(bt.rot, bt.rot);
        if (bt_norm < MINIMUM_SQ_NORM) {
            bt.zero_val();
            bt.rot.w = 1;
            return bt;
        }
        ct = bt * rsqrtf(bt_norm);
        return ct;
    }

    bool is_unsupported(const WarpField wf, const float3 vertex) {
        int3 knn_field_coords = make_int3(
                __float2int_rd(vertex.x / wf.knn_field_gpu.field_cell_size),
                __float2int_rd(vertex.y / wf.knn_field_gpu.field_cell_size),
                __float2int_rd(vertex.z / wf.knn_field_gpu.field_cell_size));
        //searching NN
        cpu_knn::DiscreteBruteFieldKNN::elem_type *knn_ptr = cpu_knn::get_knn_field_coords(wf.knn_field_gpu, knn_field_coords);
        for (int i = 0; (i < cpu_knn::DiscreteBruteFieldKNN::k) && (knn_ptr[i] != cpu_knn::DiscreteBruteFieldKNN::UNDEFINED_OFFSET); i++) {
            int curr_offset = (int) knn_ptr[i];
            float3 curr_node_coords = make_float3(wf.warp_coords[3 * curr_offset], wf.warp_coords[3 * curr_offset + 1],
                                                  wf.warp_coords[3 * curr_offset + 2]);
            float3 vec = vertex - curr_node_coords;
            float sqdist = dot(vec, vec);
            float w = wf.warp_weights[curr_offset];
            if (sqdist / (w * w) < 1) {
                return false;
            }
        }
        return true;
    }


    float3 WarpField::warp_vertex(WarpFieldCache &wfc, float3 vertex) const {
        wfc.last_V = vertex;
        int3 knn_field_coords = make_int3(
                __float2int_rd(vertex.x / knn_field_gpu.field_cell_size),
                __float2int_rd(vertex.y / knn_field_gpu.field_cell_size),
                __float2int_rd(vertex.z / knn_field_gpu.field_cell_size));
//searching NN
        wfc.V_knn = cpu_knn::get_knn_field_coords(knn_field_gpu, knn_field_coords);
//weighting & summing
        DualQuaternion b;
        b.zero_val();
        for (int i = 0; i < cpu_knn::DiscreteBruteFieldKNN::k; i++) {
            int curr_offset = (int) wfc.V_knn[i];
            float3 curr_node_coords = make_float3(warp_coords[3 * curr_offset], warp_coords[3 * curr_offset + 1], warp_coords[3 * curr_offset + 2]);
            float w = compute_omega_w(vertex, curr_node_coords, warp_weights[curr_offset]);
            wfc.warp_omega_weights[i] = w;
//        if (isnan(warp_weights[curr_offset]))
//{
//            printf("NAN WARP_WEIGHT at node %d weight %f, field coords (%d, %d, %d) \n", curr_offset, warp_weights[curr_offset], knn_field_coords.x, knn_field_coords.y, knn_field_coords.z);
//}
//        if (isnan(w))
//{
//float3 vec = vertex - curr_node_coords;
//float sqdist = dot(vec,vec);
//float weight = warp_weights[curr_offset];
//printf("NAN in compute_omega_w, sqdist %f, weight %f, -sqdist/(2*weight*weight) %f\n", sqdist, weight, -sqdist/(2*weight*weight));
//printf("NAN WEIGHT at node %d weight %f, field coords (%d, %d, %d) \n", curr_offset, warp_weights[curr_offset], knn_field_coords.x, knn_field_coords.y, knn_field_coords.z);
//printf("NAN WEIGHT at (%.3f, %.3f, %.3f), for node %d at (%.3f, %.3f, %.3f), expf(0) = %f\n", vertex.x, vertex.y, vertex.z, curr_offset, curr_node_coords.x, curr_node_coords.y, curr_node_coords.z, __expf(0.f));
//}
            b += (warp_params[curr_offset] * w);
        }
//pstatus(warp_params);
//normalize

//printf("b_norm: %f\n", b_norm);
//printf("knn_field_coords: (%d, %d, %d), vertex knn: %d %d %d %d %d\n", knn_field_coords.x, knn_field_coords.y, knn_field_coords.z,
//(int)V_knn[0], (int)V_knn[1], (int)V_knn[2], (int)V_knn[3], (int)V_knn[4]);
        float b_norm_sq = dot(b.rot, b.rot);
        if (b_norm_sq < MINIMUM_SQ_NORM) {
            wfc.warp_applied = false;
            wfc.b_inv_norm = 0;
            wfc.b_inv_norm_sq = 0;
            wfc.c.zero_val();
            return vertex;
        }
        wfc.b_inv_norm = rsqrtf(b_norm_sq);
        wfc.b_inv_norm_sq = wfc.b_inv_norm * wfc.b_inv_norm;
        wfc.c = b * wfc.b_inv_norm;
//    if (isnan(b_norm_sq))
//{
//        printf("NAN at (%.3f, %.3f, %.3f), weights are (%f, %f, %f, %f, %f)\n", vertex.x, vertex.y, vertex.z,
//warp_omega_weights[0], warp_omega_weights[1],warp_omega_weights[2],warp_omega_weights[3],warp_omega_weights[4]);
//}
//printf("b_norm_sq: %f, b_inv_norm: %f\n", b_norm_sq, b_inv_norm);
        wfc.warp_applied = true;
//warp vertex
        wfc.b = b;
        return apply_dq(wfc.c, vertex);


    }


    float3 WarpField::warp_normal(WarpFieldCache &wfc, float3 normal) const { //call ONLY after warp_vertex
        wfc.last_N = normal;
        float3 warped_normal;
        DualQuaternion &c = wfc.c;
        warped_normal.x =
                (1 - 2 * c.rot.y * c.rot.y - 2 * c.rot.z * c.rot.z) * normal.x +
                2 * (c.rot.x * c.rot.y - c.rot.w * c.rot.z) * normal.y +
                2 * (c.rot.x * c.rot.z + c.rot.w * c.rot.y) * normal.z;
        warped_normal.y =
                2 * (c.rot.x * c.rot.y + c.rot.w * c.rot.z) * normal.x +
                (1 - 2 * c.rot.x * c.rot.x - 2 * c.rot.z * c.rot.z) * normal.y +
                2 * (c.rot.y * c.rot.z - c.rot.w * c.rot.x) * normal.z;
        warped_normal.z =
                2 * (c.rot.x * c.rot.z - c.rot.w * c.rot.y) * normal.x +
                2 * (c.rot.y * c.rot.z + c.rot.w * c.rot.x) * normal.y +
                (1 - 2 * c.rot.x * c.rot.x - 2 * c.rot.y * c.rot.y) * normal.z;
        return warped_normal;
    }


    void WarpField::compute_dN_wrt_dc(WarpFieldCache &wfc) const {
        float3 &n = wfc.last_N;
        DualQuaternion &c = wfc.c;
        wfc.dN_wrt_dc[0].rot.x = 2 * (c.rot.y * n.y + c.rot.z * n.z);
        wfc.dN_wrt_dc[0].rot.y = 2 * (-2 * c.rot.y * n.x + c.rot.x * n.y + c.rot.w * n.z);
        wfc.dN_wrt_dc[0].rot.z = 2 * (-2 * c.rot.z * n.x - c.rot.w * n.y + c.rot.x * n.z);
        wfc.dN_wrt_dc[0].rot.w = 2 * (-c.rot.z * n.y + c.rot.y * n.z);

        wfc.dN_wrt_dc[1].rot.x = 2 * (c.rot.y * n.x - 2 * c.rot.x * n.y - c.rot.w * n.z);
        wfc.dN_wrt_dc[1].rot.y = 2 * (c.rot.x * n.x + c.rot.z * n.z);
        wfc.dN_wrt_dc[1].rot.z = 2 * (c.rot.w * n.x - 2 * c.rot.z * n.y + c.rot.y * n.z);
        wfc.dN_wrt_dc[1].rot.w = 2 * (c.rot.z * n.x - c.rot.x * n.z);

        wfc.dN_wrt_dc[2].rot.x = 2 * (c.rot.z * n.x + c.rot.w * n.y - 2 * c.rot.x * n.z);
        wfc.dN_wrt_dc[2].rot.y = 2 * (-c.rot.w * n.x + c.rot.z * n.y - 2 * c.rot.y * n.z);
        wfc.dN_wrt_dc[2].rot.z = 2 * (c.rot.x * n.x + c.rot.y * n.y);
        wfc.dN_wrt_dc[2].rot.w = 2 * (-c.rot.y * n.x + c.rot.x * n.y);


        for (int i = 0; i < 3; i++) {
            wfc.dN_wrt_dc[i].tr = make_float4(0., 0., 0., 0.);
        }
    }


    void WarpField::compute_dV_wrt_dc(WarpFieldCache &wfc) const {
        float3 &v = wfc.last_V;
        DualQuaternion &c = wfc.c;

        wfc.dV_wrt_dc[0].rot.x = 2 * (c.rot.y * v.y + c.rot.z * v.z) - 2 * c.tr.w;
        wfc.dV_wrt_dc[0].rot.y = 2 * (-2 * c.rot.y * v.x + c.rot.x * v.y + c.rot.w * v.z) + 2 * c.tr.z;
        wfc.dV_wrt_dc[0].rot.z = 2 * (-2 * c.rot.z * v.x - c.rot.w * v.y + c.rot.x * v.z) - 2 * c.tr.y;
        wfc.dV_wrt_dc[0].rot.w = 2 * (-c.rot.z * v.y + c.rot.y * v.z) + 2 * c.tr.x;

        wfc.dV_wrt_dc[1].rot.x = 2 * (c.rot.y * v.x - 2 * c.rot.x * v.y - c.rot.w * v.z) - 2 * c.tr.z;
        wfc.dV_wrt_dc[1].rot.y = 2 * (c.rot.x * v.x + c.rot.z * v.z) - 2 * c.tr.w;
        wfc.dV_wrt_dc[1].rot.z = 2 * (c.rot.w * v.x - 2 * c.rot.z * v.y + c.rot.y * v.z) + 2 * c.tr.x;
        wfc.dV_wrt_dc[1].rot.w = 2 * (c.rot.z * v.x - c.rot.x * v.z) + 2 * c.tr.y;

        wfc.dV_wrt_dc[2].rot.x = 2 * (c.rot.z * v.x + c.rot.w * v.y - 2 * c.rot.x * v.z) + 2 * c.tr.y;
        wfc.dV_wrt_dc[2].rot.y = 2 * (-c.rot.w * v.x + c.rot.z * v.y - 2 * c.rot.y * v.z) - 2 * c.tr.x;
        wfc.dV_wrt_dc[2].rot.z = 2 * (c.rot.x * v.x + c.rot.y * v.y) - 2 * c.tr.w;
        wfc.dV_wrt_dc[2].rot.w = 2 * (-c.rot.y * v.x + c.rot.x * v.y) + 2 * c.tr.z;

        wfc.dV_wrt_dc[0].tr.x = 2 * c.rot.w;
        wfc.dV_wrt_dc[0].tr.y = -2 * c.rot.z;
        wfc.dV_wrt_dc[0].tr.z = 2 * c.rot.y;
        wfc.dV_wrt_dc[0].tr.w = -2 * c.rot.x;

        wfc.dV_wrt_dc[1].tr.x = 2 * c.rot.z;
        wfc.dV_wrt_dc[1].tr.y = 2 * c.rot.w;
        wfc.dV_wrt_dc[1].tr.z = -2 * c.rot.x;
        wfc.dV_wrt_dc[1].tr.w = -2 * c.rot.y;

        wfc.dV_wrt_dc[2].tr.x = -2 * c.rot.y;
        wfc.dV_wrt_dc[2].tr.y = 2 * c.rot.x;
        wfc.dV_wrt_dc[2].tr.z = 2 * c.rot.w;
        wfc.dV_wrt_dc[2].tr.w = -2 * c.rot.z;

    }


    void WarpField::compute_dN_wrt_db(WarpFieldCache &wfc) const {
        for (int i = 0; i < 3; i++) {
            float precomp_xi = 0;
            for (int j = 0; j < DualQuaternion::ROTATION_PARAMS_LENGTH; j++) {
                precomp_xi += wfc.b.data[j] * wfc.dN_wrt_dc[i].data[j];
            }
            precomp_xi *= wfc.b_inv_norm_sq;

            for (int j = 0; j < DualQuaternion::ROTATION_PARAMS_LENGTH; j++) {
                wfc.dN_wrt_db[i].data[j] = wfc.b_inv_norm * (wfc.dN_wrt_dc[i].data[j] - wfc.b.data[j] * precomp_xi);
            }
        }

        for (int i = 0; i < 3; i++) {
            wfc.dN_wrt_db[i].tr = make_float4(0., 0., 0., 0.);
        }
    }


    void WarpField::compute_dV_wrt_db(WarpFieldCache &wfc) const {
        for (int i = 0; i < 3; i++) {
            float precomp_xi = 0;
            for (int j = 0; j < DualQuaternion::PARAMS_LENGTH; j++) {
                precomp_xi += wfc.b.data[j] * wfc.dV_wrt_dc[i].data[j];
            }
            precomp_xi *= wfc.b_inv_norm_sq;


//TODO: check it!
            for (int j = 0; j < DualQuaternion::ROTATION_PARAMS_LENGTH; j++) {
                wfc.dV_wrt_db[i].data[j] = wfc.b_inv_norm * (wfc.dV_wrt_dc[i].data[j] - wfc.b.data[j] * precomp_xi);
            }

            for (int j = DualQuaternion::ROTATION_PARAMS_LENGTH; j < DualQuaternion::PARAMS_LENGTH; j++) {
                wfc.dV_wrt_db[i].data[j] = wfc.b_inv_norm * wfc.dV_wrt_dc[i].data[j];
            }
        }
    }


    void WarpField::compute_dV_wrt_dq(WarpFieldCache &wfc) const {
        for (int nn = 0; nn < cpu_knn::DiscreteBruteFieldKNN::k; nn++) {
            for (int vdim = 0; vdim < 3; vdim++) {
                for (int pnum = 0; pnum < DualQuaternion::PARAMS_LENGTH; pnum++) {
                    wfc.dV_wrt_params[(nn * 3 + vdim) * DualQuaternion::PARAMS_LENGTH + pnum] =
                            wfc.dV_wrt_db[vdim].data[pnum] * wfc.warp_omega_weights[nn];
                }
            }
        }
    }


    void WarpField::compute_dN_wrt_dq(WarpFieldCache &wfc) const {
        for (int nn = 0; nn < cpu_knn::DiscreteBruteFieldKNN::k; nn++) {
            for (int ndim = 0; ndim < 3; ndim++) {
                for (int pnum = 0; pnum < DualQuaternion::ROTATION_PARAMS_LENGTH; pnum++) {
                    wfc.dN_wrt_params[(nn * 3 + ndim) * DualQuaternion::ROTATION_PARAMS_LENGTH + pnum] =
                            wfc.dN_wrt_db[ndim].data[pnum] * wfc.warp_omega_weights[nn];
                }
            }
        }
    }


    void
    WarpField::warp_forward(WarpFieldCache &wfc, float3 *vertex_ptr_in, float3 *vertex_ptr_out, float3 *normal_ptr_in, float3 *normal_ptr_out) const {
        *vertex_ptr_out = warp_vertex(wfc, *vertex_ptr_in);
        if (wfc.warp_applied) {
            *normal_ptr_out = warp_normal(wfc, *normal_ptr_in);
        } else {
            *normal_ptr_out = *normal_ptr_in;
        }
    }

//Compute gradients V_warped_wrt_params_of_all_knn_nodes

    void
    WarpField::warp_backward(WarpFieldCache &wfc, float *gradV_memory, float *gradN_memory, long *knn_memory) const { //call ONLY after forward pass
        if (wfc.warp_applied) {
            compute_dN_wrt_dc(wfc);
            compute_dV_wrt_dc(wfc);
            compute_dN_wrt_db(wfc);
            compute_dV_wrt_db(wfc);
            compute_dN_wrt_dq(wfc);
            compute_dV_wrt_dq(wfc);
            for (int i = 0; i < 3 * cpu_knn::DiscreteBruteFieldKNN::k * DualQuaternion::PARAMS_LENGTH; i++) {
                gradV_memory[i] = wfc.dV_wrt_params[i];
            }
            for (int i = 0; i < 3 * cpu_knn::DiscreteBruteFieldKNN::k * DualQuaternion::ROTATION_PARAMS_LENGTH; i++) {
                gradN_memory[i] = wfc.dN_wrt_params[i];
            }
        } else {
            for (int i = 0; i < 3 * cpu_knn::DiscreteBruteFieldKNN::k * DualQuaternion::PARAMS_LENGTH; i++) {
                gradV_memory[i] = 0;
            }
            for (int i = 0; i < 3 * cpu_knn::DiscreteBruteFieldKNN::k * DualQuaternion::ROTATION_PARAMS_LENGTH; i++) {
                gradN_memory[i] = 0;
            }
        }
        for (int i = 0; i < cpu_knn::DiscreteBruteFieldKNN::k; i++) {

            knn_memory[i] = ((long) wfc.V_knn[i]);//*cpu_knn::DiscreteBruteFieldKNN::k+i;
        }
    }

    void warp_with_grad_kernel(const WarpField wf, int verts_count, const float *verts_in, float *verts_out, const float *norms_in, float *norms_out,
                               float *gradV_memory, float *gradN_memory, long *knn_memory) {
        for (int off = 0; off < verts_count; off++) {
            int f3off = 3 * off;
            WarpFieldCache wfc;
            wf.warp_forward(wfc, (float3 * )(verts_in + f3off), (float3 * )(verts_out + f3off), (float3 * )(norms_in + f3off),
                            (float3 * )(norms_out + f3off));
            //pstatus(wf.warp_params);
            wf.warp_backward(wfc, gradV_memory + off * 3 * cpu_knn::DiscreteBruteFieldKNN::k * DualQuaternion::PARAMS_LENGTH,
                             gradN_memory + off * 3 * cpu_knn::DiscreteBruteFieldKNN::k * DualQuaternion::ROTATION_PARAMS_LENGTH,
                             knn_memory + off * cpu_knn::DiscreteBruteFieldKNN::k);
        }
    }

    void dual_quaternion_blending_kernel(const WarpField wf, int verts_count, const float *verts_in, DualQuaternion *params_out) {
        for (int off = 0; off < verts_count; off++) {
            params_out[off] = compute_dqb(wf, *((float3 * )(verts_in + 3 * off)));
        }
    }

    void get_unsupported_mask_kernel(const WarpField wf, int verts_count, const float *verts_in, unsigned char *mask_out) {
        for (int off = 0; off < verts_count; off++) {
            mask_out[off] = is_unsupported(wf, *((float3 * )(verts_in + 3 * off))) ? 1 : 0;
        }
    }

    void WarpField_Host::warp_with_grad(int verts_count, const float *verts_in, float *verts_out, const float *norms_in, float *norms_out,
                                        float *gradV_memory, float *gradN_memory, long *knn_memory) {
        check_warpfield();
        warp_with_grad_kernel(*warpfield_ptr, verts_count, verts_in, verts_out, norms_in, norms_out, gradV_memory, gradN_memory, knn_memory);
    }

    void WarpField_Host::dual_quaternion_blending(int verts_count, const float *verts_in, float *params_out) {
        check_warpfield();
        dual_quaternion_blending_kernel(*warpfield_ptr, verts_count, verts_in, reinterpret_cast<DualQuaternion *>(params_out));
    }

    void WarpField_Host::get_unsupported_mask(int verts_count, const float *verts_in, unsigned char *mask_out) {
        check_warpfield();
        get_unsupported_mask_kernel(*warpfield_ptr, verts_count, verts_in, mask_out);
    }

    void WarpField_Host::reset_nodes(float *warp_coords, float *warp_weights, cpu_knn::DiscreteBruteFieldKNN::elem_type *field_data, int field_data_bytes, int node_count) {
        if (node_count != node_storage_ptr->data_node_count)
        {
            throw std::invalid_argument("Node count for resetting is different than node_storage.data_node_count");
        }
        node_storage_ptr->reset_data_nodes(warp_coords, warp_weights);
        warpfield_ptr->warp_coords = node_storage_ptr->node_coords.get();
        warpfield_ptr->warp_weights = node_storage_ptr->node_weights.get();
        warpfield_ptr->warp_params = node_storage_ptr->node_params.get();
        warpfield_ptr->node_count = node_storage_ptr->data_node_count;
        knn_field.set_field(field_data, field_data_bytes);
    }

    void WarpField_Host::reset_node_params(DualQuaternion *warp_params) {
        node_storage_ptr->reset_data_node_params(warp_params);
    }

}