/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include "bpmf.h"

#include <random>
#include <chrono>
#include <memory>
#include <cstdio>
#include <iostream>

#ifdef BPMF_TBB_SCHED
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#include "tbb/task_scheduler_init.h"

typedef tbb::blocked_range<VectorNd::Index> Range;
#elif defined(BPMF_OMP_SCHED)
#include "omp.h"
#endif

#ifdef BPMF_OMP_SCHED
#pragma omp declare reduction (VectorPlus : VectorNd : omp_out += omp_in) initializer(omp_priv = VectorNd::Zero())
#pragma omp declare reduction (MatrixPlus : MatrixNNd : omp_out += omp_in) initializer(omp_priv = MatrixNNd::Zero())
#endif

int Sys::procid;
int Sys::nprocs;
int Sys::nthrds;

void Sys::SetupThreads(int n)
{
#ifdef BPMF_TBB_SCHED
    if (n <= 0) {
        n = tbb::task_scheduler_init::default_num_threads();
    } else {
        static tbb::task_scheduler_init init(n);
    }
#elif defined(BPMF_OMP_SCHED)
    if (n <= 0) {
        n = omp_get_num_threads();
    } else {
       omp_set_num_threads(n);
    }
#elif defined(BPMF_SER_SCHED)
    n = 1;
#else
#error No sched SetupThreads
#endif

    Sys::nthrds = n;

}

void read_sparse_float64(SparseMatrixD &m, std::string fname) 
{
    FILE *f = fopen(fname.c_str(), "r");
    uint64_t nrow, ncol, nnz;
    fread(&nrow, sizeof(uint64_t), 1, f);
    fread(&ncol, sizeof(uint64_t), 1, f);
    fread(&nnz , sizeof(uint64_t), 1, f);

    std::vector<uint32_t> rows(nnz), cols(nnz);
    std::vector<double> vals(nnz);
    fread(rows.data(), sizeof(uint32_t), nnz, f);
    fread(cols.data(), sizeof(uint32_t), nnz, f);
    fread(vals.data(), sizeof(double), nnz, f);

    struct sparse_vec_iterator {
        std::vector<uint32_t> &rows, &cols;
        std::vector<double> &vals;
        uint64_t pos;
        bool operator!=(const sparse_vec_iterator &other) const { 
            assert(&rows == &(other.rows));
            assert(&cols == &(other.cols));
            assert(&vals == &(other.vals));
            return pos != other.pos;
        }
        sparse_vec_iterator &operator++() { pos++; return *this; }
        typedef Eigen::Triplet<double> T;
        std::unique_ptr<T> operator->() {
            // also convert from 1-base to 0-base
            return std::unique_ptr<T>(new T(rows.at(pos) - 1, cols.at(pos) - 1, vals.at(pos)));
        }
    };

    sparse_vec_iterator begin = {rows, cols, vals, 0};
    sparse_vec_iterator end =   {rows, cols, vals, nnz};
    m.resize(nrow, ncol);
    m.setFromTriplets(begin, end);
}

double tick() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); 
}

Eval::Eval(std::string probename, double mean_rating, int burnin)
: mean_rating(mean_rating), burnin(burnin)
{
    SparseMatrixD tmp;
    read_sparse_float64(tmp, probename);
    P = T = tmp.transpose(); // reference and predictions
}

void Eval::predict(int n, Sys& movies, Sys& users)
{
    iter = n;
    n = (n < burnin) ? 0 : (n - burnin);
    
#ifdef BPMF_TBB_SCHED
    tbb::combinable<double> se(0.0); // squared err
    tbb::combinable<double> se_avg(0.0); // squared avg err

    tbb::parallel_for( 
        tbb::blocked_range<int>(0, T.outerSize()),
        [&](const tbb::blocked_range<int>& r) {
            for (int k=r.begin(); k<r.end(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(T,k); it; ++it)
                {
                    const double pred = movies.items().col(it.row()).dot(users.items().col(it.col())) + mean_rating;
                    se.local() += sqr(it.value() - pred);

                    // update average prediction
                    double &avg = P.coeffRef(it.row(), it.col());
                    avg = (n == 0) ? pred : (avg + (pred - avg) / n);
                    se_avg.local() += sqr(it.value() - avg);
                }
            }
        }
    );

    rmse = sqrt( se.combine(std::plus<double>()) / T.nonZeros() );
    rmse_avg = sqrt( se_avg.combine(std::plus<double>()) / T.nonZeros() );
#elif defined(BPMF_OMP_SCHED)
    double se(0.0); // squared err
    double se_avg(0.0); // squared avg err

#pragma omp parallel for reduction(+:se,se_avg)
    for(int k=0; k<T.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(T,k); it; ++it)
        {
            const double pred = movies.items().col(it.row()).dot(users.items().col(it.col())) + mean_rating;
            se += sqr(it.value() - pred);

            // update average prediction
            double &avg = P.coeffRef(it.row(), it.col());
            avg = (n == 0) ? pred : (avg + (pred - avg) / n);
            se_avg += sqr(it.value() - avg);
        }
    }

    rmse = sqrt( se / T.nonZeros() );
    rmse_avg = sqrt( se_avg / T.nonZeros() );

#elif defined(BPMF_SER_SCHED)
#error not implemented yet
#else
#error No sched predict
#endif
}

void Eval::print(double samples_per_sec, double norm_u, double norm_m) {
  printf("%d: Iteration %d:\t RMSE: %3.2f\tavg RMSE: %3.2f\tFU(%6.2f)\tFM(%6.2f)\tSamples/sec: %6.2f\n",
                    Sys::procid, iter, rmse, rmse_avg, norm_u, norm_m, samples_per_sec);
}

Sys::Sys(std::string name, std::string fname) : name(name) {
    read_sparse_float64(M, fname);
}

Sys::Sys(std::string name, const SparseMatrixD &Mt) : name(name) {
    M = Mt.transpose();
}

void Sys::init(const Sys &other)
{
    //-- M
    assert(M.rows() > 0 && M.cols() > 0);
    mean_rating = M.sum() / M.nonZeros();
    std::cout << "mean rating = " << mean_rating << std::endl;
    items().setZero();
    sum_map().setZero();
    cov_map().setZero();
    norm_map().setZero();

    std::cout << "assigning " << name << " other assigned? " << other.assigned() << std::endl;

    //-- assign items to procs according to NNZ
    const auto nprocs = Sys::nprocs;
    const auto tot = M.nonZeros();
    std::multimap<unsigned, unsigned, std::greater<unsigned>> nnz_per_col;
    for(int i=0; i<num(); i++) nnz_per_col.insert(std::make_pair(M.col(i).nonZeros(), i));

    std::vector<unsigned> nnz_per_proc(nprocs);
    std::vector<unsigned> items_per_proc(nprocs);
    item_to_proc.resize(num());
    proc_to_item.resize(nprocs);
    unsigned total_nnz = 0;
    unsigned total_items = 0;

    auto best = [&](int idx) {

        const int count = M.innerVector(idx).nonZeros();
        std::vector<unsigned> comm_per_proc(Sys::nprocs);
        if (other.assigned()) for (SparseMatrixD::InnerIterator it(M,idx); it; ++it) comm_per_proc[other.proc(it.row())]++;

        double min_cost = 1e9;
        int best_proc = -1;
        for(int i=0; i<nprocs; ++i) {
            double load_unbalance = std::max((double)nnz_per_proc[i] / (double)total_nnz, (double)items_per_proc[i] / (double)total_items);
            double comm_cost = comm_per_proc.at(i) / ((double)count + 0.0001);
            double total_cost = load_unbalance + 100 * comm_cost;
            if (total_cost > min_cost) continue;
            best_proc = i;
            min_cost = total_cost;
        }
        return best_proc;
    };

    for(auto p : nnz_per_col) {
        auto proc = best(p.second);
        item_to_proc[p.second] = proc;
        proc_to_item[proc].push_back(p.second);

        nnz_per_proc  [proc] += p.first;
        items_per_proc[proc]++;

        total_nnz += p.first;
        total_items++;
    }

    if(Sys::procid == 0) for (unsigned i=0; i<nnz_per_proc.size(); ++i) {
        std::cout << name << "@" << i << ": nnz: " << nnz_per_proc.at(i) << " (" << 100.0 * nnz_per_proc.at(i) / tot << " %)\n";
        std::cout << name << "@" << i << ": items: " << items_per_proc.at(i) << " (" << 100.0 * items_per_proc.at(i) / num() << " %)\n";
    }
}

void Sys::build_conn(const Sys& to)
{
    unsigned tot = 0;
    conn_map.resize(num());
    for (int k=0; k<num(); ++k) {
        std::bitset<max_procs> &bm = conn_map[k];
        for (SparseMatrixD::InnerIterator it(M,k); it; ++it) bm.set(to.proc(it.row()));
        // not to self
        bm.reset(Sys::procid);
        tot += bm.count();
    }

    std::cout << name << "@" << Sys::procid << ": avg comm: " << (double)tot / (double)num() << std::endl;
}

void Sys::alloc_and_init(const Sys &other)
{
    items_ptr = (double *)malloc(sizeof(double) * num_feat * num());
    sum_ptr = (double *)malloc(sizeof(double) * num_feat * Sys::nprocs);
    cov_ptr = (double *)malloc(sizeof(double) * num_feat * num_feat * Sys::nprocs);
    norm_ptr = (double *)malloc(sizeof(double) * Sys::nprocs);

    init(other);
}


class PrecomputedLLT : public Eigen::LLT<MatrixNNd>
{
  public:
    void operator=(const MatrixNNd &m) { m_matrix = m; m_isInitialized = true; m_info = Eigen::Success; }
};

VectorNd Sys::sample(long idx, const MapNXd in)
{
    // auto start = tick();
    const double alpha = 2;
    const int breakpoint1 = 1000;
    const int breakpoint2 = 1000;
    const int count = M.innerVector(idx).nonZeros();

    VectorNd rr;
    PrecomputedLLT chol;

    if( count < breakpoint1 ) {
        rr.setZero();
        chol = hp.LambdaL;
        for (SparseMatrixD::InnerIterator it(M,idx); it; ++it) {
            auto col = in.col(it.row());
            chol.rankUpdate(col, alpha);
            rr.noalias() += col * ((it.value() - mean_rating) * alpha);
        }
    } else if (count < breakpoint2) {
        rr.setZero();
        MatrixNNd MM; MM.setZero();
        for (SparseMatrixD::InnerIterator it(M,idx); it; ++it) {
            auto col = in.col(it.row());
            MM.noalias() += col * col.transpose();
            rr.noalias() += col * ((it.value() - mean_rating) * alpha);
        }
        chol.compute(hp.LambdaF + alpha * MM);

    } else {
        auto from = M.outerIndexPtr()[idx];
        auto to = M.outerIndexPtr()[idx+1];

#ifdef BPMF_TBB_SCHED
        tbb::combinable<VectorNd> s(VectorNd::Zero()); // sum
        tbb::combinable<MatrixNNd> p(MatrixNNd::Zero()); // outer prod

        tbb::parallel_for( 
            tbb::blocked_range<VectorNd::Index>(from, to),
            [&](const tbb::blocked_range<typename VectorNd::Index>& r) {
                for(auto i = r.begin(); i<r.end(); ++i) {
                    auto val = M.valuePtr()[i];
                    auto idx = M.innerIndexPtr()[i];
                    auto col = in.col(idx);
                    p.local().noalias() += col * col.transpose();
                    s.local().noalias() += col * ((val - mean_rating) * alpha);
                }
            }
        );                    

        // return sum and covariance
        rr = s.combine(std::plus<VectorNd>());
        MatrixNNd MM = p.combine(std::plus<MatrixNNd>());
#elif defined(BPMF_OMP_SCHED)
        rr.setZero();
        MatrixNNd MM(MatrixNNd::Zero()); // outer prod

#pragma omp parallel for reduction(VectorPlus:rr) reduction(MatrixPlus:MM)
        for(int i = from; i<to; ++i) {
            auto val = M.valuePtr()[i];
            auto idx = M.innerIndexPtr()[i];
            auto col = in.col(idx);
            MM.noalias() += col * col.transpose();
            rr.noalias() += col * ((val - mean_rating) * alpha);
        }

#elif defined(BPMF_SER_SCHED)
#error not implemented yet
#else
#error No sched sample_one
#endif

        chol.compute(hp.LambdaF + alpha * MM);
    }

    if(chol.info() != Eigen::Success)
        throw std::runtime_error("Cholesky Decomposition failed!");

    rr += hp.LambdaF * hp.mu;
    chol.matrixL().solveInPlace(rr);
    rr += nrandn();
    chol.matrixU().solveInPlace(rr);
    items().col(idx) = rr;

    // auto stop = tick();
    //std::cout << "  " << count << ": " << 1e6*(stop - start) << std::endl;

    return rr;
}


void Sys::sample(Sys &in) 
{
#ifdef BPMF_TBB_SCHED
    tbb::combinable<VectorNd>  s(VectorNd::Zero()); // sum
    tbb::combinable<double>    n(0.0); // squared norm
    tbb::combinable<MatrixNNd> p(MatrixNNd::Zero()); // outer prod

    tbb::parallel_for( 
        tbb::blocked_range<VectorNd::Index>(0, num()),
        [&](const tbb::blocked_range<typename VectorNd::Index>& r) {
            for(auto i = r.begin(); i<r.end(); ++i) {
                if (proc(i) != Sys::procid) continue;
                auto r = sample(i,in.items()); 
                p.local() += (r * r.transpose());
                s.local() += r;
                n.local() += r.squaredNorm();
                send_item(i);
            }
        }
    );                    

    // return sum and covariance
    auto sum = s.combine(std::plus<VectorNd>());
    auto prod = p.combine(std::plus<MatrixNNd>());
    auto norm = n.combine(std::plus<double>());
#elif defined(BPMF_OMP_SCHED)
    VectorNd  sum(VectorNd::Zero()); // sum
    double    norm(0.0); // squared norm
    MatrixNNd prod(MatrixNNd::Zero()); // outer prod

#pragma omp parallel for reduction(VectorPlus:sum) reduction(MatrixPlus:prod) reduction(+:norm)
    for(unsigned i = 0; i<my_items().size(); ++i) {
        auto r = sample(my_items().at(i),in.items()); 
        prod += (r * r.transpose());
        sum += r;
        norm += r.squaredNorm();
        send_item(my_items().at(i));
    }
#elif defined(BPMF_SER_SCHED)
    // serial 
    VectorNd sum(VectorNd::Zero()); // sum
    MatrixNNd prod(MatrixNNd::Zero()); // outer prod
    double    norm(0.0); // squared norm
    for(int i=0; i<num(); ++i) {
        if (proc(i) != Sys::procid) continue;
        auto r = sample(i,in.items());
        prod += (r * r.transpose());
        sum += r;
        norm += r.squaredNorm();
        send_item(i);
    }
#else
#error No sched sample_all
#endif

    const int N = num();
    local_sum() = sum;
    local_cov() = (prod - (sum * sum.transpose() / N)) / (N-1);
    local_norm() = norm;
}
