#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

double RNG_RANDOM()
{
	return rand() / (double)RAND_MAX;
}

double RNG_UNIFORM(int a, int b)
{
	return (a + (b - a) * RNG_RANDOM());
}

typedef struct Particle
{
	double* position_i;
	double* velocity_i;
	double* pos_best_i;
	double err_best_i = -1;
	double err_i = -1;
} Particle;

__device__ void update_pbest(Particle* swarm, int i)
{
	if (swarm[i].err_i < swarm[i].err_best_i || swarm[i].err_i == -1)
	{
		swarm[i].pos_best_i = swarm[i].position_i;
		swarm[i].err_best_i = swarm[i].err_i;
	}
}

__device__ void update_gbest(Particle* swarm, int i, double* pos_best_g, double* err_best_g) {
	if (swarm[i].err_i < *err_best_g || *err_best_g == -1)
	{
		pos_best_g[0] = swarm[i].position_i[0];
		pos_best_g[1] = swarm[i].position_i[1];
		*err_best_g = swarm[i].err_i;
	}
}

__device__ void update_position_velocity(int num_dimensions, Particle* swarm, int i, double* bounds, double* pos_best_g)
{
	double w = 0.5;
	int c1 = 1;
	int c2 = 2;

	curandState_t state;
	curand_init(0, 0, 0, &state);

	for (int j = 0; j < num_dimensions; j++)
	{
		swarm[i].position_i[j] = swarm[i].position_i[j] + swarm[i].velocity_i[j];

		if (swarm[i].position_i[j] > bounds[j * num_dimensions + 1])
		{
			swarm[i].position_i[j] = bounds[j * num_dimensions + 1];
		}

		if (swarm[i].position_i[j] < bounds[j * num_dimensions + 0])
		{
			swarm[i].position_i[j] = bounds[j * num_dimensions + 0];
		}

		double r1 = curand_uniform(&state);
		double r2 = curand_uniform(&state);
		double vel_cognitive = c1 * r1 * (swarm[i].pos_best_i[j] - swarm[i].position_i[j]);
		double vel_social = c2 * r2 * (pos_best_g[j] - swarm[i].position_i[j]);
		swarm[i].velocity_i[j] = w * swarm[i].velocity_i[j] + vel_cognitive + vel_social;
	}
}

__global__ void minimize(const double initial[], double* bounds, int num_dimensions, const int num_particle,
	double* pos_best_g, double* err_best_g, Particle* swarm)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < num_particle)
	{
		double err = 0.0;
		for (int k = 0; k <= num_dimensions; k++)
		{
			err += pow(swarm[i].position_i[k], 2);
		}
		swarm[i].err_i = err;

		update_pbest(swarm, i);
		update_gbest(swarm, i, pos_best_g, err_best_g);
		update_position_velocity(num_dimensions, swarm, i, bounds, pos_best_g);
	}
}

void alloc_memory(Particle* particle, int num_dimensions, double* x0)
{
	double* velocity_i = (double*)malloc(sizeof(double) * num_dimensions);
	double* position_i = (double*)malloc(sizeof(double) * num_dimensions);
	double* pos_best_i = (double*)malloc(sizeof(double) * num_dimensions);

	cudaMalloc(&particle->position_i, sizeof(double) * num_dimensions);
	cudaMalloc(&particle->velocity_i, sizeof(double) * num_dimensions);
	cudaMalloc(&particle->pos_best_i, sizeof(double) * num_dimensions);

	for (int i = 0; i < num_dimensions; i++)
	{
		velocity_i[i] = RNG_UNIFORM(-1, 1);
		position_i[i] = x0[i];
		pos_best_i[i] = 0.0;
	}

	cudaMemcpy(particle->velocity_i, velocity_i, sizeof(double) * num_dimensions, cudaMemcpyHostToDevice);
	cudaMemcpy(particle->position_i, velocity_i, sizeof(double) * num_dimensions, cudaMemcpyHostToDevice);
	cudaMemcpy(particle->pos_best_i, velocity_i, sizeof(double) * num_dimensions, cudaMemcpyHostToDevice);
}

void pso_single_swarm()
{
	// Variaveis de inicialização
	double initial[] = { 5, 5 };
	const int num_dimension = 2;
	const int num_particle = 4096;
	double h_BOUNDS[] = { -10,10,-10,10 };
	double* BOUNDS;

	double* pos_best_g;
	double* err_best_g;
	Particle* swarm;

	float time;
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMallocManaged(&pos_best_g, sizeof(double) * num_dimension);
	cudaMallocManaged(&err_best_g, sizeof(double));
	cudaMallocManaged(&swarm, sizeof(Particle) * num_particle);

	cudaMalloc(&BOUNDS, sizeof(double) * num_dimension * num_dimension);
	cudaMemcpy(BOUNDS, h_BOUNDS, sizeof(double) * num_dimension * num_dimension, cudaMemcpyHostToDevice);

	*err_best_g = -1;

	for (int i = 0; i < num_particle; i++)
	{
		Particle particle, p;
		alloc_memory(&particle, num_dimension, initial);
		swarm[i] = particle;
	}

	const int Thread_Per_block = 128;
	const int BLOCKS = num_particle / Thread_Per_block;

	int maxiter = 30;

	int i = 0;

	while (i < maxiter) {
		minimize << <BLOCKS, Thread_Per_block >> > (initial, BOUNDS, num_dimension, num_particle, pos_best_g, err_best_g, swarm);
		i++;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	cudaDeviceSynchronize();
	printf("[%.20f, % .20f] error: % .20f\n", pos_best_g[0], pos_best_g[1], *err_best_g);
	//printf("Time elapsed on CPU: %.4f ms.\n", time / 1000000);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(swarm);
	cudaFree(pos_best_g);
	cudaFree(swarm);
	cudaFree(BOUNDS);
}