#pragma once
#include "kfusion/types.hpp"
#include "vector_functions.h"
#include <memory>


namespace cpu_knn {
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
        std::shared_ptr<elem_type> field_data;
        elem_type *field_ptr;
        std::shared_ptr<DiscreteBruteFieldKNN> discrete_field;
        int field_total;
    public:
        DiscreteBruteFieldKNN_Host(int3 volume_dims, float volume_cell_size, int field_divider) {
            int3 field_dims = make_int3(volume_dims.x / field_divider + 1,
                                        volume_dims.y / field_divider + 1,
                                        volume_dims.z / field_divider + 1);
            field_total = (field_dims.x) * (field_dims.y) * (field_dims.z) * DiscreteBruteFieldKNN::k;
            field_data = std::shared_ptr<elem_type>(new elem_type[field_total]);
            field_ptr = field_data.get();
            discrete_field = std::shared_ptr<DiscreteBruteFieldKNN>(new DiscreteBruteFieldKNN(field_ptr, field_dims, volume_cell_size*field_divider, field_divider));
        }

        static constexpr int get_k() { return DiscreteBruteFieldKNN::k; }

        DiscreteBruteFieldKNN get_field()
        {
            return *discrete_field;
        }

        int get_field_total_size() const {return field_total;}

        void set_field(elem_type *data, int total_bytes)
        {
            if (total_bytes != static_cast<int>(field_total*sizeof(elem_type)))
            {
                throw std::invalid_argument("Total bytes are different");
            }
            std::memcpy(field_ptr, data, total_bytes);
        }
//
//        void recompute_field(float *all_nodes_ptr, elem_type new_nodes_start_off, int new_nodes_count, bool use_kdtree = true);
//
//        void clear();

        void query_field(int3 coords, elem_type* answers); //super slow, used only for debugging

    };

    DiscreteBruteFieldKNN::elem_type* get_knn_field_coords(DiscreteBruteFieldKNN df, int3 coords);
    DiscreteBruteFieldKNN::elem_type* get_knn_tsdf_coords(DiscreteBruteFieldKNN df, int3 coords);
}
