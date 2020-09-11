/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#pragma once

#ifdef BPMF_PROFILING

#include <string>
#include <map>
#include "thread_vector.h"

#define BPMF_COUNTER(name) Counter c(name)

struct Counter {
    Counter *parent;
    std::string name, fullname;
    double start, stop, diff; // wallclock time
    long long count;

    bool total_counter;

    Counter(std::string name);
    Counter(); // needed for std::map in TotalsCounter

    ~Counter();

    void operator+=(const Counter &other);

    std::string as_string(const Counter &total, bool hier) const;
    std::string as_string(bool hier) const;
};

struct TotalsCounter {
    private:
        std::map<std::string, Counter> data;
        int procid;
        void print_body(const std::string &, bool) const;

    public:
        //c-tor starts PAPI
        TotalsCounter(int = 0);

        void operator+=(const TotalsCounter &other);

        //prints results
        void print(int, bool) const;
        void print(bool) const;

        Counter &operator[](const std::string &name) {
            return data[name];
        }

        bool empty() const { return data.empty(); }
};

extern thread_vector<TotalsCounter> hier_perf_data, flat_perf_data;

void perf_data_init();
void perf_data_print();

#else 

#define BPMF_COUNTER(name) 
inline void perf_data_init() {}
inline void perf_data_print() {}

#endif //BPMF_PROFILING

double tick();
