/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <iostream>

#include "bpmf.h"
#include "bpmf_mpi_isendirecv.h"

int main() {
        Sys::Init();
        {

        SendBuffer<double> sb(Sys::procid, 3);
        RecvBuffer<double> rb(Sys::procid, 3);
        
        sb.put(42.0);
        sb.put(420.0);
        sb.put(4200.0);
        Sys::cout() << "a = " << rb.get() << std::endl;
        Sys::cout() << "a = " << rb.get() << std::endl;
        Sys::cout() << "a = " << rb.get() << std::endl;

        }
        Sys::Finalize();

        return 0;
}

