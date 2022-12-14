#pragma once

#include <kfusion/cuda/device_array.hpp>
#include "safe_call.hpp"
#include <cuda_fp16.h>
#include "kfusion/cuda/warpfield.hpp"

//#define USE_DEPTH

namespace kfusion
{
    namespace device
    {
        typedef float4 Normal;
        typedef float4 Point;

        typedef unsigned short ushort;
        typedef unsigned char uchar;

        typedef PtrStepSz<half> Dists;
        typedef DeviceArray2D<ushort> Depth;
        typedef DeviceArray2D<Normal> Normals;
        typedef DeviceArray2D<Point> Points;
        typedef DeviceArray2D<uchar4> Image;

        typedef int3   Vec3i;
        typedef float3 Vec3f;
        struct Mat3f { float3 data[3]; };
        struct Aff3f { Mat3f R; Vec3f t; };

        struct Surface {
            DeviceArray<float4> vertices;
            DeviceArray<float4> normals;
        };

        struct TsdfVolume
        {
        public:
            typedef half2 elem_type;

            elem_type *const data;
            const int3 dims;
            const float3 voxel_size;
            const float trunc_dist;
            const float max_weight;

            TsdfVolume(elem_type* data, int3 dims, float3 voxel_size, float trunc_dist, float max_weight);
            //TsdfVolume(const TsdfVolume&);

            __kf_device__ elem_type* operator()(int x, int y, int z);
            __kf_device__ const elem_type* operator() (int x, int y, int z) const ;
            __kf_device__ elem_type* beg(int x, int y) const;
            __kf_device__ elem_type* zstep(elem_type *const ptr) const;
        private:
            TsdfVolume& operator=(const TsdfVolume&);
        };

        struct CubeIndexEstimator {
            CubeIndexEstimator(const TsdfVolume &vol) : volume(vol) { isoValue = 0.f; }

            const TsdfVolume volume;
            float isoValue;

            __kf_device__ int computeCubeIndex(int x, int y, int z, float f[8]) const;
            __kf_device__ void readTsdf(int x, int y, int z, float &f, float &weight) const;
        };

        struct OccupiedVoxels : public CubeIndexEstimator {
            OccupiedVoxels(const TsdfVolume &vol) : CubeIndexEstimator({vol}) {}

            enum {
                CTA_SIZE_X = 32,
                CTA_SIZE_Y = 8,
                CTA_SIZE   = CTA_SIZE_X * CTA_SIZE_Y,

                LOG_WARP_SIZE = 5,
                WARP_SIZE     = 1 << LOG_WARP_SIZE,
                WARPS_COUNT   = CTA_SIZE / WARP_SIZE,
            };

            mutable int *voxels_indices;
            mutable int *vertices_number;
            int max_size;

            __kf_device__ void operator()() const;
        };

        struct TrianglesGenerator : public CubeIndexEstimator {
            TrianglesGenerator(const TsdfVolume &vol) : CubeIndexEstimator({vol}) {volume_yzdim = (vol.dims.y+1)*(vol.dims.z+1);}

            enum { CTA_SIZE = 256, MAX_GRID_SIZE_X = 65536 };

            const int *occupied_voxels;
            const int *vertex_ofssets;
            int volume_yzdim;
            int voxels_count;
            float3 cell_size;

            Aff3f pose;

            mutable Point *outputVertices;
            mutable Normal *outputNormals;
            mutable int *outputVertexIndices;

            __kf_device__ void operator()() const;

            __kf_device__ void store_point(float4 *ptr, int index, const float3 &vertex) const;

            __kf_device__ float3 get_node_coo(int x, int y, int z) const;
            __kf_device__ float3 vertex_interp(float3 p0, float3 p1, float f0, float f1) const;
            __kf_device__ int vertex_id(int vx, int vy, int vz, int edge_index) const;

        };  // namespace device

        struct Projector
        {
            float2 f, c;
            Projector(){}
            Projector(float fx, float fy, float cx, float cy);
            __kf_device__ float2 operator()(const float3& p) const;
        };

        struct Reprojector
        {
            Reprojector() {}
            Reprojector(float fx, float fy, float cx, float cy);
            float2 finv, c;
            __kf_device__ float3 operator()(int x, int y, float z) const;
        };

        struct ComputeIcpHelper
        {
            struct Policy;
            struct PageLockHelper
            {
                float* data;
                PageLockHelper();
                ~PageLockHelper();
            };

            float min_cosine;
            float dist2_thres;

            Aff3f aff;

            float rows, cols;
            float2 f, c, finv;

            PtrStep<ushort> dcurr;
            PtrStep<Normal> ncurr;
            PtrStep<Point> vcurr;

            ComputeIcpHelper(float dist_thres, float angle_thres);
            void setLevelIntr(int level_index, float fx, float fy, float cx, float cy);

            void operator()(const Depth& dprev, const Normals& nprev, DeviceArray2D<float>& buffer, float* data, cudaStream_t stream);
            void operator()(const Points& vprev, const Normals& nprev, DeviceArray2D<float>& buffer, float* data, cudaStream_t stream);

            static void allocate_buffer(DeviceArray2D<float>& buffer, int partials_count = -1);

            //private:
            __kf_device__ int find_coresp(int x, int y, float3& n, float3& d, float3& s) const;
            __kf_device__ void partial_reduce(const float row[7], PtrStep<float>& partial_buffer) const;
            __kf_device__ float2 proj(const float3& p) const;
            __kf_device__ float3 reproj(float x, float y, float z)  const;
        };

        //tsdf volume functions
        void clear_volume(TsdfVolume volume);
        void integrate(const Dists& depth, TsdfVolume& volume, const Aff3f& aff, const Projector& proj, warp::WarpField wf);
        void integrate_nowarp(const Dists& depth, TsdfVolume& volume, const Aff3f& aff, const Projector& proj);
        void query_volume_with_grad(const TsdfVolume& volume, const DeviceArray<float> &verts, DeviceArray<float> &tsdfs, DeviceArray<float> &tsdfs_grad);

        void raycast(const TsdfVolume& volume, const Aff3f& aff, const Mat3f& Rinv,
                     const Reprojector& reproj, Depth& depth, Normals& normals, float step_factor, float delta_factor);

        void raycast(const TsdfVolume& volume, const Aff3f& aff, const Mat3f& Rinv,
                     const Reprojector& reproj, Points& points, Normals& normals, float step_factor, float delta_factor);

        __kf_device__ half2 pack_tsdf(float tsdf, float weight);
        __kf_device__ float unpack_tsdf(half2 value, float& weight);
        __kf_device__ float unpack_tsdf(half2 value);

        /*
         * MARCHING CUBES FUNCTIONS
         */

        /** \brief binds marching cubes tables to texture references */
        void bindTextures(const int *edgeBuf, const int *triBuf, const int *numVertsBuf);

        /** \brief unbinds */
        void unbindTextures();

        /** \brief scans TSDF volume and retrieves occuped voxes
         * \param[in] volume TSDF volume
         * \param[out] occupied_voxels buffer for occupied voxels; the function fills the first row with voxel id's and second
         * row with the no. of vertices
         * \return number of voxels in the buffer
         */
        int getOccupiedVoxels(const TsdfVolume &volume, DeviceArray2D<int> &occupied_voxels);

        /** \brief computes total number of vertices for all voxels and offsets of vertices in final triangle array
         * \param[out] occupied_voxels buffer with occupied voxels; the function fills 3rd only with offsets
         * \return total number of vertexes
         */
        int computeOffsetsAndTotalVertices(DeviceArray2D<int> &occupied_voxels);

        /** \brief generates final triangle array
         * \param[in] volume TSDF volume
         * \param[in] occupied_voxels occupied voxel ids (1st row), number of vertices (2nd row), offsets (3rd row).
         * \param[in] volume_size volume size in meters
         * \param[out] output triangle array
         */
        void generateTriangles(const TsdfVolume &volume, const DeviceArray2D<int> &occupied_voxels, const float3 &volume_size,
                const Aff3f &pose, DeviceArray<Point> &outputVertices, DeviceArray<Normal> &outputNormals, DeviceArray<int>& outputVertexIndices);



        //image proc functions
        void compute_dists(const Depth& depth, Dists dists, float2 f, float2 c);

        void truncateDepth(Depth& depth, float max_dist /*meters*/);
        void bilateralFilter(const Depth& src, Depth& dst, int kernel_size, float sigma_spatial, float sigma_depth);
        void depthPyr(const Depth& source, Depth& pyramid, float sigma_depth);

        void resizeDepthNormals(const Depth& depth, const Normals& normals, Depth& depth_out, Normals& normals_out);
        void resizePointsNormals(const Points& points, const Normals& normals, Points& points_out, Normals& normals_out);

        void computeNormalsAndMaskDepth(const Reprojector& reproj, Depth& depth, Normals& normals);
        void computePointNormals(const Reprojector& reproj, const Depth& depth, Points& points, Normals& normals);

        void renderImage(const Depth& depth, const Normals& normals, const Reprojector& reproj, const Vec3f& light_pose, Image& image);
        void renderImage(const Points& points, const Normals& normals, const Reprojector& reproj, const Vec3f& light_pose, Image& image);
        void renderTangentColors(const Normals& normals, Image& image);


        //exctraction functionality
        size_t extractCloud(const TsdfVolume& volume, const Aff3f& aff, PtrSz<Point> output);
        void extractNormals(const TsdfVolume& volume, const PtrSz<Point>& points, const Aff3f& aff, const Mat3f& Rinv, float gradient_delta_factor, float4* output);

        struct float8  { float x, y, z, w, c1, c2, c3, c4; };
        struct float12 { float x, y, z, w, normal_x, normal_y, normal_z, n4, c1, c2, c3, c4; };
        void mergePointNormal(const DeviceArray<Point>& cloud, const DeviceArray<float8>& normals, const DeviceArray<float12>& output);
    }
}
