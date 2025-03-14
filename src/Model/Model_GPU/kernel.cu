#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include <iostream>
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_particles) return;
	accelerationsGPU[i].x = 0.0f;
	accelerationsGPU[i].y = 0.0f;
	accelerationsGPU[i].z = 0.0f;
	__shared__  float3 posj[128];
	__shared__  float masses[128];
	float3 posi = positionsGPU[i];
	float3 acc = {0.0f, 0.0f, 0.0f};

	for(int block = 0; block < (n_particles + blockDim.x - 1) / blockDim.x; block++){
		int j = block*blockDim.x + threadIdx.x;
		if (j < n_particles) {
			posj[threadIdx.x] = positionsGPU[j];
			masses[threadIdx.x] = massesGPU[j];
		} else {
			posj[threadIdx.x] = {0.0f, 0.0f, 0.0f};
			masses[threadIdx.x] = 0.0f;
		}

		__syncthreads();

		for(int k = 0; k < blockDim.x; k++){
			if (!(block == blockIdx.x && k == threadIdx.x)) {
				
				const float diffx = posj[k].x - posi.x;
				const float diffy = posj[k].y - posi.y;
				const float diffz = posj[k].z - posi.z;

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0){
					dij = 10.0f;
				} else {
					dij = std::sqrt(dij);
					dij = 10.0 / (dij * dij * dij);
				}

				acc.x += diffx * dij * masses[k];
				acc.y += diffy * dij * masses[k];
				acc.z += diffz * dij * masses[k];
			}
		}

		__syncthreads();
	}

	accelerationsGPU[i] = acc;
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_particles) return;

	velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
	velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
	velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
	positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
	positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
	positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	cudaDeviceSynchronize();
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU