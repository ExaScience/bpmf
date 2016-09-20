/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#ifdef BPMF_PROFILING

#include <string>
#include <map>

#define BPMF_COUNTER(name) Counter c(name)

struct Counter {
    std::string name;
    double start, stop, diff; // wallclock time
    long long count;

    bool total_counter;

    Counter(std::string name);
    Counter();

    ~Counter();

    void operator+=(const Counter &other);

    std::string as_string(const Counter &total);
};

struct TotalsCounter {
    private:
        std::map<std::string, Counter> data;

    public:
        //c-tor starts PAPI
        TotalsCounter();

        //prints results
        void print();

        Counter &operator[](const std::string &name) {
            return data[name];
        }
};

extern TotalsCounter perf_data;
#else 

#define BPMF_COUNTER(name) 

#endif //BPMF_PROFILING
