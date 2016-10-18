/*
Test libm math function speed
(or Intel short vector math (svml) if compiled with icc).
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include "mkl.h"
// #include "amdlibm.h"
#include <time.h>
#define PREC 10000000
#define MAXVEC 100000

int vecsize = 1000;  // length of vector to compute on
int loops = 1000;   // number of times to compute each complete vector
float min = 0;     // use uniform dist between (min,max), affects math funcs
float max = 1;

int main(int argc, char** argv)
{
    double x[MAXVEC], a[MAXVEC], b[MAXVEC], c[MAXVEC];
    float xs[MAXVEC], as[MAXVEC], bs[MAXVEC], cs[MAXVEC];
    srand(time(NULL));
    int i, j;
    clock_t begin, end;

    // get numeric command-line arguments
    if( argc==1 )
        printf("optional arguments: [min max vecsize loops]\n");
    if( argc>1 )
        sscanf(argv[1], "%f", &min);
    if( argc>2 )
        sscanf(argv[2], "%f", &max);
    if( argc>3 )
        sscanf(argv[3], "%d", &vecsize);
    if( argc>4 )
        sscanf(argv[4], "%d", &loops);

    // check number of threads
    if( vecsize<1 || vecsize>MAXVEC )
    {
        printf("vecsize must be between 1 and %d\n", MAXVEC);
        return 1;
    }

    printf("Performing %d loops on vectors of length %d\n", loops, vecsize);
    printf("Using random numbers between (%f, %f)\n\n", min, max);

    // generate the random numbers used as function arguments
    begin = clock();
    for(i = 0; i < vecsize; i++){
        x[i] = (double) (max - min) * (rand() % PREC) / PREC + min;
        xs[i] = (float) x[i];
    }
    end = clock();
    printf("random number generation: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);


    // tanh (double)
    begin = clock();
    for(j = 0; j < loops; j++){
        for(i = 0; i < vecsize; i++){
            a[i] = tanh(x[i]);
        }
        a[0] += 1.0;  // maybe needed for the intel compiler not to skip computation
    }
    end = clock();
    printf("tanh: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);

    // tanh (single)
    begin = clock();
    for(j = 0; j < loops; j++){
        for(i = 0; i < vecsize; i++){
            as[i] = tanhf(xs[i]);
        }
        as[0] += 1.0;
    }
    end = clock();
    printf("tanhf: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);

    // expm1 (double)
    begin = clock();
    for(j = 0; j < loops; j++){
        for(i = 0; i < vecsize; i++){
            b[i] = expm1(x[i]);
        }
        b[0] += 1.0;
    }
    end = clock();
    printf("expm1: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);

    // expm1 (single)
    begin = clock();
    for(j = 0; j < loops; j++){
        for(i = 0; i < vecsize; i++){
            bs[i] = expm1f(xs[i]);
        }
        bs[0] += 1.0;
    }
    end = clock();
    printf("expm1f: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);

    // sin (double)
    begin = clock();
    for(j = 0; j < loops; j++){
        for(i = 0; i < vecsize; i++){
            c[i] = sin(x[i]);
        }
        c[0] += 1.0;
    }
    end = clock();
    printf("sin: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);

    // sin (single)
    begin = clock();
    for(j = 0; j < loops; j++){
        for(i = 0; i < vecsize; i++){
            cs[i] = sinf(xs[i]);
        }
        cs[0] += 1.0;
    }
    end = clock();
    printf("sinf: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);

    // make sure it actually computes the values.
    printf("\nMake sure the values are actually computed:\n");
    printf("x[0]: %f, a[0]: %f, b[0]: %f, c[0]: %f\n", x[0], a[0], b[0], c[0]);
    printf("xs[0]: %f, as[0]: %f, bs[0]: %f, cs[0]: %f\n", xs[0], as[0], bs[0], cs[0]);

    return 0;
}
