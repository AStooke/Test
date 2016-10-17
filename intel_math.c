/*
Trying to test intel math function speed.
*/

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <math.h>
#include <time.h>
#define NUM 100000
#define NUM1 1000
#define SCALE 0.01


void main()
{
    double x[NUM], y[NUM], z[NUM];
    srand(time(NULL));
    int i, j;
    clock_t begin, end;

    begin = clock();
    for(i = 0; i < NUM; i++){
        x[i] = (double) SCALE * (rand() % 100);
    }
    end = clock();
    printf("random generation: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);
    // /opt/intel/compilers_and_libraries_2017.0.098/linux/mkl/include/

    begin = clock();
    for(j = 0; j < NUM1; j++){
        for(i = 0; i < NUM; i++){
            y[i] = tanh(x[i]);
        }
    }
    end = clock();
    printf("tanh: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);

    begin = clock();
    for(j = 0; j < NUM1; j++){
        for(i = 0; i < NUM; i++){
            z[i] = expm1(x[i]);
        }
    }
    end = clock();
    printf("expm1: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);
    printf("x[0]: %f, y[0]: %f, z[0]: %f\n", x[0], y[0], z[0]);

}
