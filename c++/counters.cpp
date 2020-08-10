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
thread_vector<TotalsCounter> hier_perf_data, flat_perf_data;

void perf_data_init()
{
    active_counters.init();
    hier_perf_data.init();
    flat_perf_data.init();
}

void perf_data_print(const thread_vector<TotalsCounter> &data, bool hier) {
    std::string title = hier ? 
        "\nHierarchical view:\n==================\n" :
        "\nFlat view:\n==========\n";
    Sys::cout() << title;
    TotalsCounter sum_all_threads;
    int num_active_threads = 0;
    int threadid = 0;
    for(auto &d : data)
    {
        if (!d.empty())
        {
            d.print(threadid, hier);
            sum_all_threads += d;
            num_active_threads++;
        }
    }

    if (num_active_threads > 1)
        sum_all_threads.print(hier);
}

void perf_data_print() {
    perf_data_print(hier_perf_data, true);
    perf_data_print(flat_perf_data, false);
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
    if (name == "main")
    {
        //init performance counters
        perf_data_init();
    }

}

Counter::~Counter() {
    if(total_counter) return;

    stop = tick();
    diff = stop - start;

    hier_perf_data.local()[fullname] += *this;
    flat_perf_data.local()[name] += *this;
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

std::string Counter::as_string(const Counter &total, bool hier) const {
    std::ostringstream os;
    std::string n = hier ? fullname : name;
    int percent = round(100.0 * diff / (total.diff + 0.000001));
    os << ">> " << n << ":\t" << std::fixed << std::setw(11)
       << std::setprecision(4) << diff << "\t(" << percent << "%) in\t" << count << "\n";
    return os.str();
}

std::string Counter::as_string(bool hier) const
{
    std::ostringstream os;
    std::string n = hier ? fullname : name;
    os << ">> " << n << ":\t" << std::fixed << std::setw(11)
       << std::setprecision(4) << diff << " in\t" << count << "\n";
    return os.str();
}

TotalsCounter::TotalsCounter(int p) : procid(p) {}

void TotalsCounter::operator+=(const TotalsCounter &other)
{
    for(auto &t : other.data) data[t.first] += t.second;
}

void TotalsCounter::print(bool hier) const {
    print_body("sum of all threads / ", hier);
}

void TotalsCounter::print(const int threadid, bool hier) const {
    std::ostringstream s;
    s << "thread " << threadid << " / ";
    print_body(s.str(), hier);
}

void TotalsCounter::print_body(const std::string &thread_str, bool hier) const {
    if (data.empty()) return;
    char hostname[1024];
    gethostname(hostname, 1024);
    Sys::cout() << "\nTotals on " << hostname << " (" << procid << ") / " << thread_str;
    Sys::cout() << (hier ? "hierarchical\n" : "flat\n");

    const auto total = data.find("main");
    for(auto &t : data)
    {
        auto parent_name = t.first.substr(0, t.first.find_last_of("/"));
        const auto parent = data.find(parent_name);
        if (hier && parent != data.end())
            Sys::cout() << t.second.as_string(parent->second, hier);
        else if (!hier && total != data.end())
            Sys::cout() << t.second.as_string(total->second, hier);
        else
            Sys::cout() << t.second.as_string(hier);
    }
}

#endif // BPMF_PROFILING

double tick() 
{
   return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
