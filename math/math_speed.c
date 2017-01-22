/*
Test speed of different math function libraries.

This source code provides a separate executable for testing each libm,
amdlibm, svml, depending on compilation.  (mkl_vml requires different source)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef AMD  // (compile with -DAMD)
    #include "amdlibm.h"
#endif
#define PREC 10000000
#define MAXVEC 1000000
#define ALIGN 64

int vecsize = 1000;  // length of vector to compute on
int loops = 1000;   // number of times to compute each complete vector
float min = 0;     // use uniform dist between (min,max), affects math funcs
float max = 1;
int memalign = 1;  // boolean, whether to align vectory memory to cache
int warmup = 1;  // boolean, whether to perform warmup run of each computation


void do_math_d( double (*func)(double), double* x, double* y, char* name)
{
    clock_t begin, end;
    int i, j;

    // warmup run
    if( warmup > 0 )
    {
        for(j = 0; j < loops; j++){
            for(i = 0; i < vecsize; i++)
                y[i] = (*func)(x[i]);
            y[0] += 1.0;  // a little extra calculation to make sure it executes
        }
    }

    // timed run
    begin = clock();
    for(j = 0; j < loops; j++){
        for(i = 0; i < vecsize; i++)
            y[i] = (*func)(x[i]);
        y[0] += 1.0;  // a little extra calculation to make sure it executes
    }
    end = clock();
    printf("%s: %f s\n", name, (double) (end - begin) / CLOCKS_PER_SEC);
}


void do_math_f( float (*func)(float), float* x, float* y, char* name)
{
    clock_t begin, end;
    int i, j;

    // warmup run
    if( warmup > 0 )
    {
        for(j = 0; j < loops; j++){
            for(i = 0; i < vecsize; i++)
                y[i] = (*func)(x[i]);
            y[0] += 1.0;  // a little extra calculation to make sure it executes
        }
    }

    // timed run
    begin = clock();
    for(j = 0; j < loops; j++){
        for(i = 0; i < vecsize; i++)
            y[i] = (*func)(x[i]);
        y[0] += 1.0;  // a little extra calculation to make sure it executes
    }
    end = clock();
    printf("%s: %f s\n", name, (double) (end - begin) / CLOCKS_PER_SEC);
}


void run_all(double* x, double* y, float* xs, float* ys)
{
    srand(time(NULL));
    int i, j;
    clock_t begin, end;

    // generate the random numbers used as function arguments
    begin = clock();
    for(i = 0; i < vecsize; i++){
        x[i] = (double) (max - min) * (rand() % PREC) / PREC + min;
        xs[i] = (float) x[i];
    }
    end = clock();
    printf("random number generation: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);

    // Whatever math functions desired to test.
    do_math_d( tanh, x, y, "tanh");
    do_math_f( tanhf, xs, ys, "tanhf");
    do_math_d( expm1, x, y, "expm1");
    do_math_f( expm1f, xs, ys, "expm1f");
    do_math_d( sin, x, y, "sin");
    do_math_f( sinf, xs, ys, "sinf");

    // make sure it actually computes the values.
    printf("\nMake sure some values are actually computed:\n");
    printf("x[0]: %f, y[0]: %f\n", x[0], y[0]);
    printf("xs[0]: %f, ys[0]: %f\n", xs[0], ys[0]);
}


int main(int argc, char** argv)
{
    // get numeric command-line arguments
    if( argc==1 )
        printf("optional arguments: [min max vecsize loops memalign warmup]\n");
    if( argc>1 )
        sscanf(argv[1], "%f", &min);
    if( argc>2 )
        sscanf(argv[2], "%f", &max);
    if( argc>3 )
        sscanf(argv[3], "%d", &vecsize);
    if( argc>4 )
        sscanf(argv[4], "%d", &loops);
    if( argc>5 )
        sscanf(argv[5], "%d", &memalign); // used as: True if > 0
    if( argc>6 )
        sscanf(argv[6], "%d", &warmup);  // used as: True if > 0

    // check number of threads
    if( vecsize<1 || vecsize>MAXVEC )
    {
        printf("vecsize must be between 1 and %d\n", MAXVEC);
        return 1;
    }

    printf("Performing this many loops: %d\n", loops);
    printf("On a vector of length: %d\n", vecsize);
    printf("Using random numbers between (%f, %f)\n", min, max);
    printf("Memory Aligned: %s\n", memalign > 0 ? "Yes" : "No");
    printf("Pre-timing warmup loops: %s\n\n", warmup > 0 ? "Yes" : "No");

    if( memalign > 0)
    {
        void *xmem, *ymem, *xsmem, *ysmem;
        int pr;
        pr = posix_memalign(&xmem, ALIGN, vecsize * sizeof(double));
        pr = posix_memalign(&ymem, ALIGN, vecsize * sizeof(double));
        pr = posix_memalign(&xsmem, ALIGN, vecsize * sizeof(float));
        pr = posix_memalign(&ysmem, ALIGN, vecsize * sizeof(float));
        double *x = (double *)xmem;
        double *y = (double *)ymem;
        float *xs = (float *)xsmem;
        float *ys = (float *)ysmem;

        int memalign_success = 1;
        if( (long long int) (x) % ALIGN != 0)
            memalign_success = 0;
        if( (long long int) (y) % ALIGN != 0)
            memalign_success = 0;
        if( (long long int) (xs) % ALIGN != 0)
            memalign_success = 0;
        if( (long long int) (ys) % ALIGN != 0)
            memalign_success = 0;
        if( memalign_success == 0)
            printf("WARNING: memory alignmed did not succeed.");

        run_all(x, y, xs, ys);
    }
    else
    {
        double x[vecsize], y[vecsize];
        float xs[vecsize], ys[vecsize];

        int memalign_success = 0;
        if( (long long int) (x) % ALIGN == 0)
            memalign_success = 1;
        if( (long long int) (y) % ALIGN == 0)
            memalign_success = 1;
        if( (long long int) (xs) % ALIGN == 0)
            memalign_success = 1;
        if( (long long int) (ys) % ALIGN == 0)
            memalign_success = 1;
        if( memalign_success == 1)
            printf("WARNING: at least one variable accidentally memory aligned.\n\n");

        run_all(x, y, xs, ys);
    }

    return 0;
}
