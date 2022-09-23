#ifndef KFUSION_KNN_HPP
#define KFUSION_KNN_HPP


#include <kfusion/cuda/device_array.hpp>
#include <memory>
#include "kfusion/types.hpp"
#include "vector_functions.h"

namespace knn {
    struct DiscreteBruteFieldKNN {
        typedef unsigned short elem_type;
        static const elem_type UNDEFINED_OFFSET = 65535;
        static const int k = 5;
        elem_type *field_ptr;
        const float field_cell_size;
        const int field_divider;
        const int3 field_dims;

        DiscreteBruteFieldKNN(
                elem_type *field_ptr,
                int3 field_dims,
                float field_cell_size,
                int field_divider) :
                field_ptr(field_ptr),
                field_cell_size(field_cell_size),
                field_divider(field_divider),
                field_dims(field_dims)
                {}


    };

    class DiscreteBruteFieldKNN_Host {
        typedef DiscreteBruteFieldKNN::elem_type elem_type;
        kfusion::cuda::CudaData field_data;
        elem_type *field_ptr;
        std::shared_ptr<DiscreteBruteFieldKNN> discrete_field;
        int field_total;
    public:
        DiscreteBruteFieldKNN_Host(int3 volume_dims, float volume_cell_size, int field_divider) {
            int3 field_dims = make_int3(volume_dims.x / field_divider + 1,
                                        volume_dims.y / field_divider + 1,
                                        volume_dims.z / field_divider + 1);
            field_total = (field_dims.x) * (field_dims.y) * (field_dims.z) * DiscreteBruteFieldKNN::k;
            field_data.create(field_total * sizeof(elem_type));
            field_ptr = field_data.ptr<elem_type>();
            discrete_field = std::shared_ptr<DiscreteBruteFieldKNN>(new DiscreteBruteFieldKNN(field_ptr, field_dims, volume_cell_size*field_divider, field_divider));
        }

//        ~DiscreteBruteFieldKNN_Host() {
//            delete discrete_field;
//        }

        int get_field_total_size() const {return field_total;}

        static constexpr int get_k() { return DiscreteBruteFieldKNN::k; }

        DiscreteBruteFieldKNN get_field()
        {
            return *discrete_field;
        }

        void recompute_field(float *all_nodes_ptr, elem_type new_nodes_start_off, int new_nodes_count, bool use_kdtree = true);

        void clear();

        void query_field(int3 coords, elem_type* answers); //super slow, used only for debugging

    };

    __device__ DiscreteBruteFieldKNN::elem_type* get_knn_field_coords(const DiscreteBruteFieldKNN &df, int3 coords);
    __device__ DiscreteBruteFieldKNN::elem_type* get_knn_tsdf_coords(const DiscreteBruteFieldKNN &df, int3 coords);
}






//        struct DiscreteBlockFieldKNN {
//            typedef unsigned char elem_type;
//            const int k;
//            elem_type *field;
//            int *blocks;
//            const int field_divider;
//            const int block_divider;
//            int all_divider;
//            const int block_size;
//            const int3 field_dims;
//            const int3 block_dims;
//
//            DiscreteBlockFieldKNN(
//                    int k_neighbours,
//                    elem_type *field,
//                    int *blocks,
//                    int3 field_dims,
//                    int3 block_dims,
//                    int field_divider,
//                    int block_divider) :
//                    k(k_neighbours),
//                    field_divider(field_divider),
//                    block_divider(block_divider),
//                    field(field),
//                    blocks(blocks),
//                    field_dims(field_dims),
//                    block_dims(block_dims)
//            {
//                block_size = (sizeof(elem_type) * 8 - 3);
//                all_divider = block_divider * field_divider;
//            }
//
//            int linearize_block_coords(int3 coords);
//
//            bool add_node(int3 coords, int offset) {
//                int3 block_coord = coords / all_divider;
//                int *block = blocks[linearize_block_coords(block_coord)] + 1;
//                int records_count = block[-1];
//                if (records_count == block_size - 1)
//                    return false;
//                block[records_count] = offset;
//                block[-1]++;
//                return true;
//            }
//
//            __kf_device__ void recompute_field_kernel();
//
//
//        };
//
//
//        class DiscreteBlockFieldKNN_Host {
//            typedef unsigned char elem_type;
//            int k;
//            CudaData field_data, blocks_data;
//            elem_type *field;
//            int *blocks;
//            DiscreteBlockFieldKNN *discrete_field;
//        public:
//            DiscreteBlockFieldKNN_Host(int k_neighbours, int3 volume_dims, int field_divider, int block_divider) : k(
//                    k_neighbours) {
//                int3 field_dims = volume_dims / field_divider + 1;
//                int3 block_dims = field_dims / block_divider + 1;
//                int field_total = (field_dims.x) * (field_dims.y) * (field_dims.z);
//                int blocks_total = (block_dims.x) * (block_dims.y) * (block_dims.z);
//                int block_mem = (sizeof(elem_type) * 8 - 3) * sizeof(int);
//                field_data.create(field_total * sizeof(elem_type))
//                blocks_data.create(block_mem * blocks_total);
//                field = field_data.ptr<elem_type>();
//                blocks = field_data.ptr<int>();
//                discrete_field = new DiscreteBlockFieldKNN(k_neighbours, field, blocks, field_dims, block_dims,
//                                                      field_divider, block_divider);
//            }
//
//            ~DiscreteBlockFieldKNN_Host() {
//                delete discrete_field;
//            }
//
//            void add_nodes(int3 *coords, int nodes_count, int start_offset) {
//                int curr_offset = start_offset;
//                int3 *curr_coords = coords;
//                bool result;
//                for (int i = 0; i < nodes_count; i++, offset++, curr_coords++) {
//                    result = discrete_field->add_node(*curr_coords, offset);
//                    if (!result) {
//                        throw std::exception("No place for new nodes!");
//                    }
//                }
//                recompute_field(coords, nodes_count);
//            }
//
//            void recompute_field(int3 *coords, int nodes_count);
//
//        };




#endif