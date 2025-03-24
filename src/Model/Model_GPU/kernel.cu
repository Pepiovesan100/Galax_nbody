#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"

#define THD (512)

__global__ void compute_acc(ParticleSoA * positionsGPU, float3 * accelerationsGPU, int n_particles){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	float posix;
	float posiy;
	float posiz;
	float posiw;
	if (i >= n_particles){
		posix = 0.0f;
		posiy = 0.0f;
		posiz = 0.0f;
		posiw = 0.0f;
	} else {
		posix = positionsGPU->x[i];
		posiy = positionsGPU->y[i];
		posiz = positionsGPU->z[i];
		posiw = positionsGPU->w[i];
	}
	
	float3 acc = {0.0f, 0.0f, 0.0f};

	__shared__ float4 shared_particles[THD];

	#pragma unroll 16
	for (int j = 0; j < n_particles; j += blockDim.x) {
		// Load a tile of particles into shared memory
		int k = j + threadIdx.x;
		if (k < n_particles) {
			shared_particles[threadIdx.x].x = positionsGPU->x[k];
			shared_particles[threadIdx.x].y = positionsGPU->y[k];
			shared_particles[threadIdx.x].z = positionsGPU->z[k];
			shared_particles[threadIdx.x].w = positionsGPU->w[k];
		} else {
			shared_particles[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		}
		
		__syncthreads(); // Ensure all threads have loaded the tile

		#pragma unroll 128
		for (int l = 0; l < blockDim.x; l++) {
			int idx = j + l; // Global index of the particle
			if (idx >= n_particles) break;

			float4 posj = shared_particles[l];
			float3 diff = (float3){posj.x - posix, posj.y - posiy, posj.z - posiz};

			float dij = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
			
			dij = rsqrtf(fmaxf(dij,1.0f));
			dij = posj.w * (dij * dij * dij);

			// float mul = dij * posj.w;
			// float mul = __fmul_rn(dij, posj.w);

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

__global__ void maj_pos(ParticleSoA * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_particles) return;

	velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
	velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
	velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
	positionsGPU->x[i] += velocitiesGPU[i].x * 0.1f;
	positionsGPU->y[i] += velocitiesGPU[i].y * 0.1f;
	positionsGPU->z[i] += velocitiesGPU[i].z * 0.1f;

}

void update_position_cu(ParticleSoA* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles)
{
	int nthreads = THD;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;
	
	compute_acc<<<nblocks, nthreads>>>(positionsGPU, accelerationsGPU, n_particles);
	cudaDeviceSynchronize();
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU