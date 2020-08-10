/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <chrono>

#ifdef BPMF_PROFILING

#include <cmath>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <cmath>

#include "counters.h"
#include "bpmf.h"

static thread_vector<Counter *> active_counters(0);
thread_vector<TotalsCounter> perf_data;

void perf_data_init()
{
    active_counters.init();
    perf_data.init();
}

void perf_data_print() {
    int threadid = 0;
    for(auto &p : perf_data)
    {
        p.print(threadid++);
    }
}

Counter::Counter(std::string name)
    : name(name), diff(0), count(1), total_counter(false)
{
    parent = active_counters.local();
    active_counters.local() = this;

    fullname = (parent) ? parent->fullname + "/" + name : name; 

    start = tick();
}

Counter::Counter()
    : parent(0), name(std::string()), fullname(std::string()), diff(0), count(0), total_counter(true)
{
} 

Counter::~Counter() {
    if(total_counter) return;

    stop = tick();
    diff = stop - start;

    perf_data.local()[fullname] += *this;
    active_counters.local() = parent;
}

void Counter::operator+=(const Counter &other) {
    if (name.empty()) 
    {
        name = other.name;
        fullname = other.fullname;
    }
    diff += other.diff;
    count += other.count;
}

std::string Counter::as_string(const Counter &total) const {
    std::ostringstream os;
    int percent = round(100.0 * diff / (total.diff + 0.000001));
    os << ">> " << fullname << ":\t" << std::fixed << std::setw(11)
       << std::setprecision(4) << diff << "\t(" << percent << "%) in\t" << count << "\n";
    return os.str();
}

std::string Counter::as_string() const
{
    std::ostringstream os;
    os << ">> " << fullname << ":\t" << std::fixed << std::setw(11)
       << std::setprecision(4) << diff << " in\t" << count << "\n";
    return os.str();
}


TotalsCounter::TotalsCounter(int p) : procid(p) {}

void TotalsCounter::print(int threadid) const {
    if (data.empty()) return;
    char hostname[1024];
    gethostname(hostname, 1024);
    Sys::cout() << "\nTotals on " << hostname << " (" << procid << ") / thread " << threadid << ":\n";
    const auto total = data.find("main");
    for(auto &t : data)
        if (total != data.end())
            Sys::cout() << t.second.as_string(total->second);
        else
            Sys::cout() << t.second.as_string();
}

#endif // BPMF_PROFILING

double tick() 
{
   return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
