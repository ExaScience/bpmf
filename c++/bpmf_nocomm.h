/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#define SYS Sys

void Sys::Init()
{
    Sys::procid = 0;
    Sys::nprocs = 1;
}

void Sys::Finalize()
{
#ifdef BPMF_PROFILING
        perf_data.print();
#endif

}

void Sys::sync() {}
void Sys::Abort(int) { abort(); }
