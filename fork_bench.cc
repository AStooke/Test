#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>


static long numForks = 100;


double
secsPerFork(const timeval& tv_start, const timeval& tv_end)
{
    double diff = (tv_end.tv_sec * 1000000 + tv_end.tv_usec)
                       - (tv_start.tv_sec * 1000000 + tv_start.tv_usec);
    double secsPerForkVal = ((diff / 1000000) / numForks);
    return secsPerForkVal;
}


double
doForks()
{
    timeval tv_start;
    gettimeofday(&tv_start, NULL);
    for (int i = 0; i < numForks; i++)
    {
        pid_t child = fork();
        if (child)
        {
            waitpid(child, NULL, 0);
        }
        else
        {
            exit(0);
        }
    }
    timeval tv_end;
    gettimeofday(&tv_end, NULL);

    return secsPerFork(tv_start, tv_end);
}


int
main(int argc, char *argv[])
{
    std::cout << "Time taken per fork:\t\t" << doForks() << std::endl;

    long mallocSize = 500 * 1024 * 1024;

    void *ptr = malloc(mallocSize);
    memset(ptr, 0,  mallocSize);
    std::cout << "Time taken per fork (500MB):\t" << doForks() << std::endl;

    ptr = malloc(mallocSize);
    memset(ptr, 0,  mallocSize);
    std::cout << "Time taken per fork (1000MB):\t" << doForks() << std::endl;

    ptr = malloc(mallocSize);
    memset(ptr, 0,  mallocSize);
    ptr = malloc(mallocSize);
    memset(ptr, 0,  mallocSize);
    std::cout << "Time taken per fork (2000MB):\t" << doForks() << std::endl;

    return 0;
}
