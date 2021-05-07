#include <cstdio>

struct Array {
    int c[10];
    Array() {
        for(int i = 0; i<10; i++)
            c[i] = 0;
    }
    void operator+=(Array b) {
        for(int i = 0; i<10; i++) 
            c[i] += b.c[i]; 
    }

 };

#pragma omp declare reduction (ArrayPlus : Array : omp_out += omp_in)

void taskloop_reduce(long c) 
{
    Array aa;

#pragma omp taskloop default(none) shared(c) reduction(ArrayPlus: aa)
    for (unsigned j = 0; j < c; j++)
    { }

}

int main(int argc, char *argv[])
{
    printf("-- call with argc ?? -- \n");
    taskloop_reduce(argc);
    printf("-- call with argc ok -- \n");
    for(int i=0; i<10000; ++i)
    {
        printf("-- call with %d ?? -- \n", i);
        taskloop_reduce(i);
        printf("-- call with %d ok -- \n", i);
    }
    printf("finished\n");
}