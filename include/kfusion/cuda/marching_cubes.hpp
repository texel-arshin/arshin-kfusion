#pragma once

/* eigen includes */
#include <Eigen/Core>

/* kinfu includes */
#include <kfusion/cuda/device_array.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/types.hpp>
#include <memory>

namespace kfusion {
    namespace cuda {

/** \brief MarchingCubes implements MarchingCubes functionality for TSDF volume on GPU
 * \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */
        class KF_EXPORTS MarchingCubes {
                public:
                /** \brief Default size for triangles buffer */
                enum { POINTS_PER_TRIANGLE = 3, DEFAULT_TRIANGLES_BUFFER_SIZE = 2 * 1000 * 1000 * POINTS_PER_TRIANGLE };

                /** \brief Smart pointer. */
                typedef std::shared_ptr<MarchingCubes> Ptr;

                /** \brief Default constructor */
                MarchingCubes();
                /** \brief Destructor */
                ~MarchingCubes();

                /* set volume pose */
                void setPose(const kfusion::Affine3f& pose);

                /* run mc to extract the zero level set */
                int run(TsdfVolume& volume, DeviceArray<Point>& vertices_buffer,
                DeviceArray<Normal>& normals_buffer,
                DeviceArray<int>& vertexindices_buffer);

                private:
                /** \brief Edge table for marching cubes  */
                DeviceArray<int> edgeTable_;

                /** \brief Number of vertextes table for marching cubes  */
                DeviceArray<int> numVertsTable_;

                /** \brief Triangles table for marching cubes  */
                DeviceArray<int> triTable_;

                /** \brief Temporary buffer used by marching cubes (first row stores occuped voxes id, second number of vetexes,
                 * third poits offsets */
                DeviceArray2D<int> occupied_voxels_buffer_;

                /* volume pose */
                Affine3f pose;
        };
    }  // namespace cuda
}  // namespace kfusion
