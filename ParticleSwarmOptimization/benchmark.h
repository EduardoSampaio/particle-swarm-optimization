#include <iostream>
#include <vector>
#include <math.h>
#include <float.h>
#include <time.h>

using namespace std;

double BOUNDS_SPHERE[] = { -10, 10,-10, 10 };
double BOUNDS_ROSENBROCK[] = { -2048, 2048, -2048, 2048 };
double BOUNDS_RASTRINGIN[] = { -5.12, 5.12, -5.12, 5.12 };
double BOUNDS_SCHWEFEL[] = { -500, 500, -500, 500 };

typedef double (*Function)(double* x, int num_dimensions);

/**
 * @brief
 * https://www.cs.bham.ac.uk/research/projects/ecb/data/2/sphere.png
 * @param x
 * @return double
 */
double sphere(double* x, int N)
{
    double total = 0.0;
    for (int i = 0; i <= N; i++)
    {
        total += pow(x[i], 2);
    }
    return total;
}

/**
 * @brief
 * https://upload.wikimedia.org/wikipedia/commons/3/32/Rosenbrock_function.svg
 * @param x
 * @return double
 */
double rosenbrock(double* x, int N)
{
    double total = 0.0;
    for (int i = 0; i <= N; i++)
    {
        total += 100 * (pow(x[i], 2) - pow(x[i + 1], 2)) + pow(1 - x[i], 2);
    }
    return total;
}

/**
 * @brief
 * https://www.cs.bham.ac.uk/research/projects/ecb/data/1/rast.png
 * @param x
 * @return double
 */
double rastrigin(double* x, int N)
{
    double total = 0.0;
    for (int i = 0; i <= N; i++)
    {
        total += (pow(x[i], 2) - (10 * cos(2 * 3.14159265359 * x[i])) + 10);
    }
    return total;
}

/**
 * @brief
 * https://www.sfu.ca/~ssurjano/schwef.png
 * @param x
 * @return double
 */
double schwefel(double* x, int N)
{
    double total = 0.0;
    for (int i = 0; i <= N; i++)
    {
        total += (x[i] * sin(sqrt(fabs(x[i]))));
    }
    return -total;
}
