
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "pso-single-swarm.cuh"
//#include "pso.h"

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>



int main()
{
	pso_single_swarm();
	//pso_execute();
}
