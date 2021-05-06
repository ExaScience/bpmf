/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <unistd.h>

#include "bpmf.h"


int main(int argc, char *argv[])
{
    for(int i=0; i<10000; ++i)
    {
        printf("%i\n", i);
        computeMuLambda_2lvls_standalone(i);
    }
    printf("Success!\n");
}