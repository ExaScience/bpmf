/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#ifndef COUNTERS_H
#define COUNTERS_H

#ifdef BPMF_PROFILING

#include <iostream>
#include <iomanip>
#include <sstream>
#include <mutex>

#include "bpmf.h"

Counter::Counter(std::string name)
    : name(name), diff(0), total_counter(false)
{
    start = tick();
}

Counter::Counter() 
    : name(std::string()), diff(0), total_counter(true)
{} 

Counter::~Counter() {
    static std::mutex mtx;

    if(total_counter) return;
    stop = tick();
    diff = stop - start;

    mtx.lock();
    perf_data[name] += *this;
    mtx.unlock();
}

void Counter::operator+=(const Counter &other) {
    if (name.empty()) name = other.name;
    diff += other.diff;
}

std::string Counter::as_string(const Counter &total) {
    std::ostringstream os;
    int percent = 100.0 * diff / (total.diff + 0.000001);
    os << ">> " << name << ":\t" << std::fixed << std::setw(11)
        << std::setprecision(4) << diff << "\t(" << percent << "%)\n";
    return os.str();
}

TotalsCounter perf_data;

TotalsCounter::TotalsCounter() {}

void TotalsCounter::print() {
    std::cerr << "\nTotals on " << Sys::procid << ":\n";
    for(auto &t : data)
        std::cerr << t.second.as_string(data["main"]);
}

#endif // BPMF_PROFILING

#endif // COUNTERS_H
