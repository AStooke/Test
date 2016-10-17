//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright (C) 2016 Roboti LLC.   //
//-----------------------------------//


#include "mujoco.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


// enable compilation with and without OpenMP support
#if defined(_OPENMP)
    #include <omp.h>
#else
    // omp timer replacement
    #include <chrono>
    double omp_get_wtime(void)
    {
        static std::chrono::system_clock::time_point _start = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - _start;
        return elapsed.count();
    }

    // omp functions used below
    void omp_set_dynamic(int) {}
    void omp_set_num_threads(int) {}
    int omp_get_num_procs(void) {return 1;}
#endif


// gloval variables: internal
const int MAXTHREAD = 512;   // maximum number of threads allowed
const int MAXEPOCH = 100;   // maximum number of epochs


// global variables: user-defined, with defaults
int nthread = 0;            // number of parallel threads (default set later)
int nepoch = 20;            // number of timing epochs
int nstep = 500;            // number of simulation steps per epoch


// worker function for parallel finite-difference computation of derivatives
void worker(const mjModel* m, const mjData* dmain, mjData* d, int id)
{
    int nv = m->nv;

    // // copy state and control from dmain to thread-specific d
    // d->time = dmain->time;
    // mju_copy(d->qpos, dmain->qpos, m->nq);
    // mju_copy(d->qvel, dmain->qvel, m->nv);
    // mju_copy(d->qacc, dmain->qacc, m->nv);
    // mju_copy(d->qfrc_applied, dmain->qfrc_applied, m->nv);
    // mju_copy(d->xfrc_applied, dmain->xfrc_applied, 6*m->nbody);
    // mju_copy(d->ctrl, dmain->ctrl, m->nu);


    // advance this thread's simulation for nstep
    for( int i=0; i<nstep; i++ )
        mj_step(m, d);
}




// main function
int main(int argc, char** argv)
{
    // print help if not enough arguments
    if( argc<2 )
    {
        printf("\n Arguments: modelfile [nthread nepoch nstep]\n\n");
        return 1;
    }

    // default nthread = number of logical cores (usually optimal)
    nthread = omp_get_num_procs();

    // get numeric command-line arguments
    if( argc>2 )
        sscanf(argv[2], "%d", &nthread);
    if( argc>3 )
        sscanf(argv[3], "%d", &nepoch);
    if( argc>4 )
        sscanf(argv[4], "%d", &nstep);

    // check number of threads
    if( nthread<1 || nthread>MAXTHREAD )
    {
        printf("nthread must be between 1 and %d\n", MAXTHREAD);
        return 1;
    }

    // check number of epochs
    if( nepoch<1 || nepoch>MAXEPOCH )
    {
        printf("nepoch must be between 1 and %d\n", MAXEPOCH);
        return 1;
    }

    // activate and load model
    mj_activate("mjkey.txt");
    mjModel* m = 0;
    if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
        m = mj_loadModel(argv[1], NULL, 0);
    else
        m = mj_loadXML(argv[1], NULL, NULL, 0);
    if( !m )
    {
        printf("Could not load modelfile '%s'\n", argv[1]);
        return 1;
    }

    // print arguments
#if defined(_OPENMP)
    printf("\nnthread : %d (OpenMP)\n", nthread);
#else
    printf("\nnthread : %d (serial)\n", nthread);
#endif
    printf("nepoch  : %d\n", nepoch);
    printf("nstep   : %d\n", nstep);

    // make mjData: main, per-thread
    mjData* dmain = mj_makeData(m);
    mjData* d[MAXTHREAD];
    for( int n=0; n<nthread; n++ )
    {
        d[n] = mj_makeData(m);
        d[n]->time = dmain->time;
        mju_copy(d[n]->qpos, dmain->qpos, m->nq);
        mju_copy(d[n]->qvel, dmain->qvel, m->nv);
        mju_copy(d[n]->qacc, dmain->qacc, m->nv);
        mju_copy(d[n]->qfrc_applied, dmain->qfrc_applied, m->nv);
        mju_copy(d[n]->xfrc_applied, dmain->xfrc_applied, 6*m->nbody);
        mju_copy(d[n]->ctrl, dmain->ctrl, m->nu);
    }

    // set up OpenMP (if not enabled, this does nothing)
    omp_set_dynamic(0);
    omp_set_num_threads(nthread);

    // allocate statistics
    double cputm[MAXEPOCH];

    // run epochs, collect statistics
    for( int epoch=0; epoch<nepoch; epoch++ )
    {

        // start timer
        double starttm = omp_get_wtime();

        // run worker threads in parallel if OpenMP is enabled
        #pragma omp parallel for schedule(static)
        for( int n=0; n<nthread; n++ )
            worker(m, dmain, d[n], n);

        // record duration in ms
        cputm[epoch] = 1000*(omp_get_wtime() - starttm);

        // // advance main simulation for nstep
        // for( int i=0; i<nstep; i++ )
        //     mj_step(m, dmain);
    }

    // compute statistics
    printf("\n epoch times:\n");
    double mcputm = 0;
    for( int epoch=0; epoch<nepoch; epoch++ )
    {
        printf("%.0f, ", cputm[epoch]);
        mcputm += cputm[epoch];
    }

    // print sizes, timing, accuracy
    printf("\n\navg stepping time, per epoch: %.2f ms", mcputm/nepoch);
    printf("\navg stepping time, per thread: %.2f ms\n\n", mcputm/nthread);


    // shut down
    mj_deleteData(dmain);
    for( int n=0; n<nthread; n++ )
        mj_deleteData(d[n]);
    mj_deleteModel(m);
    mj_deactivate();
    return 0;
}
