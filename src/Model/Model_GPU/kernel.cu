#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"

#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float4 * positionsGPU, float3 * accelerationsGPU, int n_particles){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_particles) return;
	float4 posi = positionsGPU[i];
	float3 acc = {0.0f, 0.0f, 0.0f};

	for(int j = 0; j < n_particles; j++){
		const float diffx = positionsGPU[j].x - posi.x;
		const float diffy = positionsGPU[j].y - posi.y;
		const float diffz = positionsGPU[j].z - posi.z;

		float dij = diffx * diffx + diffy * diffy + diffz * diffz;


		dij = sqrtf(fmaxf(dij,1.0f));
		dij = 10.0 / (dij * dij * dij);

		acc.x += diffx * dij * positionsGPU[j].w;
		acc.y += diffy * dij * positionsGPU[j].w;
		acc.z += diffz * dij * positionsGPU[j].w;
	}
	accelerationsGPU[i] = acc;
}

__global__ void maj_pos(float4 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
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

void update_position_cu(float4* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, accelerationsGPU, n_particles);
	// cudaDeviceSynchronize();
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU