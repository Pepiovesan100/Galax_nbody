#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"

#define THD (512)

__global__ void compute_acc(float4 * __restrict__ positionsGPU, float3 * __restrict__ accelerationsGPU, int n_particles){
	unsigned int i = fmaf(blockIdx.x , blockDim.x , threadIdx.x);
	float4 posi;
	posi = (i < n_particles) ? __ldg(&positionsGPU[i]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	
	float3 acc = {0.0f, 0.0f, 0.0f};

	__shared__ float4 shared_particles[THD];

	#pragma unroll 16
	for (int j = 0; j < n_particles; j += blockDim.x) {
		// Load a tile of particles into shared memory
		int k = j + threadIdx.x;
		shared_particles[threadIdx.x] = (k < n_particles) ? __ldg(&positionsGPU[k]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		
		__syncthreads(); // Ensure all threads have loaded the tile

		#pragma unroll 128
		for (int l = 0; l < blockDim.x; l++) {
			int idx = j + l; // Global index of the particle
			if (idx >= n_particles) break;

			float4 posj = shared_particles[l];
			float3 diff = (float3){posj.x - posi.x, posj.y - posi.y, posj.z - posi.z};

			float dij = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
			
			dij = rsqrtf(fmaxf(dij,1.0f));
			dij = posj.w * (dij * dij * dij);

			acc.x = fmaf(diff.x, dij, acc.x);
			acc.y = fmaf(diff.y, dij, acc.y);
			acc.z = fmaf(diff.z, dij, acc.z);
		}

		__syncthreads(); // Ensure all threads load the data before computation
		}
	accelerationsGPU[i].x = acc.x * 10.0f;
	accelerationsGPU[i].y = acc.y * 10.0f;
	accelerationsGPU[i].z = acc.z * 10.0f;

}

__global__ void maj_pos(float4 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_particles) return;
    float3 acc;
	acc.x = __ldg(&accelerationsGPU[idx].x);
	acc.y = __ldg(&accelerationsGPU[idx].y);
	acc.z = __ldg(&accelerationsGPU[idx].z);
    float3 vel = velocitiesGPU[idx];
    float4 pos = positionsGPU[idx];

    vel.x += acc.x * 2.0f;
    vel.y += acc.y * 2.0f;
    vel.z += acc.z * 2.0f;

    pos.x += vel.x * 0.1f;
    pos.y += vel.y * 0.1f;
    pos.z += vel.z * 0.1f;

    velocitiesGPU[idx] = vel;
    positionsGPU[idx] = pos;

}

void update_position_cu(float4* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles)
{
	int nthreads = THD;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;
	
	compute_acc<<<nblocks, nthreads>>>(positionsGPU, accelerationsGPU, n_particles);
	cudaDeviceSynchronize();
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU