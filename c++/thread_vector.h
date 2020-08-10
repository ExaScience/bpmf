#pragma once

#include <algorithm>
#include <numeric>
#include <vector>
#include <cassert>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif


namespace threads
{

inline void init(int n)
{
#if defined(_OPENMP)
    if (n <= 0)
    {
        n = omp_get_max_threads();
    }
    else
    {
        omp_set_num_threads(n);
    }
#else
    if (n > 0) 
    {
        std::cout << "Ignoring num-of-threads parameter (" << n << ")\n";
    }
#endif
}

inline int get_max_threads()
{
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int get_thread_num()
{
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif

}

inline bool is_master()
{
    return get_thread_num() == 0;
}

} // namespace threads

template <typename T>
class thread_vector
{
  public:
    thread_vector(const T &t = T())
    {
        init(t);
    }
    template <typename F>
    T combine(F f) const
    {
        return std::accumulate(_m.begin(), _m.end(), _i, f);
    }
    T combine() const
    {
        return std::accumulate(_m.begin(), _m.end(), _i, std::plus<T>());
    }

    T &local()
    {
        return _m.at(threads::get_thread_num());
    }
    void reset()
    {
        for (auto &t : _m)
            t = _i;
    }
    template <typename F>
    T combine_and_reset(F f) const
    {
        T ret = combine(f);
        reset();
        return ret;
    }
    T combine_and_reset()
    {
        T ret = combine();
        reset();
        return ret;
    }
    void init(const T &t = T())
    {
        _i = t;
        _m.resize(threads::get_max_threads());
        reset();
    }
    void init(const std::vector<T> &v)
    {
        assert((int)v.size() == threads::get_max_threads());
        _m.resize(threads::get_max_threads());
        _m = v;
    }

    typedef typename std::vector<T>::const_iterator const_iterator;

    const_iterator begin() const
    {
        return _m.begin();
    }

    const_iterator end() const
    {
        return _m.end();
    }

  private:
    std::vector<T> _m;
    T _i;
};