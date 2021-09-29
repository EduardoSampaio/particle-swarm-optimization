#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "benchmark.h"

typedef double (*Function)(double* x, int N);

// generates a double between (0, 1)
#define RNG_UNIFORM() (rand() / (double)RAND_MAX)

// generate an int between 0 and s (exclusive)
#define RNG_UNIFORM_INT(s) (rand() % s)

double uniform(int min, int max)
{
	return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

typedef struct Particle
{
	double* position_i;     // particle position
	double* velocity_i;     // particle velocity
	double* pos_best_i;     // best position individual
	double err_best_i = -1; // best error individual
	double err_i = -1;      // error individual

} Particle;


Particle createParticle(int num_dimensions)
{
	Particle particle;
	particle.velocity_i = new double[num_dimensions];
	particle.position_i = new double[num_dimensions];
	particle.pos_best_i = new double[num_dimensions];
	return particle;
}

void fitness(Particle* particle, int num_dimensions, double* x0)
{
	for (int i = 0; i < num_dimensions; i++)
	{
		particle->velocity_i[i] = uniform(-1, 1);
		particle->position_i[i] = x0[i];
		particle->pos_best_i[i] = 0.0;
	}
}
// evaluate current fitness
void evaluate(Particle* particle, Function function, int num_dimensions)
{
	particle->err_i = function(particle->position_i, num_dimensions);
	// check to see if the current position is an individual best
	if (particle->err_i < particle->err_best_i || particle->err_i == -1)
	{
		particle->pos_best_i = particle->position_i;
		particle->err_best_i = particle->err_i;
	}
}

// update new particle velocity
void update_velocity(Particle* particle, double* pos_best_g, int num_dimensions)
{
	double w = 0.9; // constant inertia weight
	int c1 = 2;     // cognitive constant
	int c2 = 2;     // social constant

	for (int i = 0; i < num_dimensions; i++)
	{
		double r1 = uniform(0, 1);
		double r2 = uniform(0, 1);
		double vel_cognitive = c1 * r1 * (particle->pos_best_i[i] - particle->position_i[i]);
		double vel_social = c2 * r2 * (pos_best_g[i] - particle->position_i[i]);
		particle->velocity_i[i] = w * particle->velocity_i[i] + vel_cognitive + vel_social;
	}
}

// update the particle position based new velocity updates
void update_position(Particle* particle, double bounds[2][2], int N)
{
	for (int i = 0; i < N; i++)
	{
		particle->position_i[i] = particle->position_i[i] + particle->velocity_i[i];

		if (particle->position_i[i] > bounds[i][1])
		{
			particle->position_i[i] = bounds[i][1];
		}

		if (particle->position_i[i] < bounds[i][0])
		{
			particle->position_i[i] = bounds[i][0];
		}
	}
}

void minimize(Function func, double* x0, double bounds[2][2], int num_dimensions, int num_particle, int maxiter)
{
	Particle* swarm = new Particle[num_particle];
	double* pos_best_g = new double[num_dimensions]; // best position for group
	double err_best_g = -1;                          // best error for group

	// establish the swarm
	for (int i = 0; i < num_particle; i++)
	{
		Particle particle = createParticle(num_dimensions);
		fitness(&particle, num_dimensions, x0);
		swarm[i] = particle;
	}

	int i = 0;
	while (i < maxiter)
	{
		// cycle through particles in swarm and evaluate fitness
		for (int j = 0; j < num_particle; j++)
		{
			evaluate(&swarm[j], func, num_dimensions);

			// determine if current particle is the best (globally)
			if (swarm[j].err_i < err_best_g || err_best_g == -1)
			{
				pos_best_g = swarm[j].position_i;
				err_best_g = swarm[j].err_i;
			}
		}

		// cycle through swarm and update velocities and position
		for (int j = 0; j < num_particle; j++)
		{
			update_velocity(&swarm[j], pos_best_g, num_dimensions);
			update_position(&swarm[j], bounds, num_dimensions);
		}

		i++;
		///printf("iteration: %d  best solution: [%f , %f] error: %f \n",
			//i, pos_best_g[0], pos_best_g[0], err_best_g);
	}

	printf("Final Solution: [%f , %f] error: %f\n", pos_best_g[0], pos_best_g[1], err_best_g);
}


void pso_sequencial()
{
	double initial[] = { 5, 5 };                            // initial starting location [x1,x2...]
	double bounds[2][2] = { { -10, 10}, { -10, 10 } };     //  input bounds
	int num_dimension = 2;
	minimize(schwefel, initial, bounds, num_dimension, 15, 30);
}
