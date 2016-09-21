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
#include <climits>
#include <stdexcept>

#include <unsupported/Eigen/SparseExtra>

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

static const bool measure_perf = false;

std::ostream *Sys::os;
int Sys::procid;
int Sys::nprocs;
int Sys::nthrds;

int Sys::nsims;
int Sys::burnin;

bool Sys::permute = true;

unsigned Sys::grain_size;

void assert_transpose(SparseMatrixD &A, SparseMatrixD &B)
{
    SparseMatrixD At = A.transpose();
    SparseMatrixD Bt = B.transpose();
    assert(At.cols() == B.cols());
    assert(A.cols() == Bt.cols());

    for(int i=0; i<B.cols(); ++i) assert(At.col(i).nonZeros() == B.col(i).nonZeros());
    for(int i=0; i<A.cols(); ++i) assert(Bt.col(i).nonZeros() == A.col(i).nonZeros());
}

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


template <typename T>
void read_sparse_float64(SparseMatrixD &m, std::string fname) 
{
    FILE *f = fopen(fname.c_str(), "r");
    if (!f) throw std::runtime_error(std::string("Could not open " + fname));
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
 	sparse_vec_iterator(
		std::vector<uint32_t> &rows,
		std::vector<uint32_t> &cols,
		std::vector<double> &vals,
		uint64_t pos)
            : rows(rows), cols(cols), vals(vals), pos(pos) {}

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
        T v;
        T* operator->() {
            // also convert from 1-base to 0-base
            uint32_t row = rows.at(pos) - 1;
            uint32_t col = cols.at(pos) - 1;
	    double val = vals.at(pos);
            v = T(row, col, val);
            return &v;
        }
    };

    sparse_vec_iterator begin(rows, cols, vals, 0);
    sparse_vec_iterator end(rows, cols, vals, nnz);
    m.resize(nrow, ncol);
    m.setFromTriplets(begin, end);
}

double tick() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); 
}

void Sys::predict(Sys& other, bool all)
{
    int n = (iter < burnin) ? 0 : (iter - burnin);
   
#ifdef BPMF_TBB_SCHED
    tbb::combinable<double> se(0.0); // squared err
    tbb::combinable<double> se_avg(0.0); // squared avg err
        tbb::combinable<unsigned> nump(0);

    tbb::parallel_for( 
        tbb::blocked_range<int>(0, T.outerSize()),
        [&](const tbb::blocked_range<int>& r) {
            for (int k=r.begin(); k<r.end(); ++k) {
                if (all || (proc(k) == Sys::procid)) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(T,k); it; ++it)
                    {
                        auto m = items().col(it.col());
                        auto u = other.items().col(it.row());

                        assert(m.norm() > 0.0);
                        assert(u.norm() > 0.0);

                        const double pred = m.dot(u) + mean_rating;
                        se.local() += sqr(it.value() - pred);

                        // update average + variance of prediction
                        double &avg = Pavg.coeffRef(it.row(), it.col());
                        double delta = pred - avg;
                        avg = (n == 0) ? pred : (avg + delta/n);
                        double &m2 = Pm2.coeffRef(it.row(), it.col());
                        m2 = (n == 0) ? 0 : m2 + delta * (pred - avg);

                        se_avg.local() += sqr(it.value() - avg);

                        nump.local()++;
                    }
                }
            }
        }
    );

    auto tot = nump.combine(std::plus<unsigned>());
    rmse = sqrt( se.combine(std::plus<double>()) / tot );
    rmse_avg = sqrt( se_avg.combine(std::plus<double>()) / tot );
#elif defined(BPMF_OMP_SCHED) || defined(BPMF_SER_SCHED)
    double se(0.0); // squared err
    double se_avg(0.0); // squared avg err
    unsigned nump(0); // number of predictions

//#ifdef BPMF_OMP_SCHED
//#pragma omp parallel for reduction(+:se,se_avg,nump)
//#endif
    int lo = all ? 0 : from();
    int hi = all ? num() : to();
    for(int k = lo; k<hi; k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(T,k); it; ++it)
        {
            auto m = items().col(it.col());
            auto u = other.items().col(it.row());

            assert(m.norm() > 0.0);
            assert(u.norm() > 0.0);

            const double pred = m.dot(u) + mean_rating;
            se += sqr(it.value() - pred);

            // update average prediction
            double &avg = Pavg.coeffRef(it.row(), it.col());
            double delta = pred - avg;
            avg = (n == 0) ? pred : (avg + delta/n);
            double &m2 = Pm2.coeffRef(it.row(), it.col());
            m2 = (n == 0) ? 0 : m2 + delta * (pred - avg);
            se_avg += sqr(it.value() - avg);

            nump++;
        }
    }

    rmse = sqrt( se / nump );
    rmse_avg = sqrt( se_avg / nump );
#else
#error No sched predict
#endif
}

void Sys::print(double items_per_sec, double ratings_per_sec, double norm_u, double norm_m) {
  char buf[1024];
  sprintf(buf, "%d: Iteration %d:\t RMSE: %3.2f\tavg RMSE: %3.2f\tFU(%6.2f)\tFM(%6.2f)\titems/sec: %6.2f\tratings/sec: %6.2fM\n",
                    Sys::procid, iter, rmse, rmse_avg, norm_u, norm_m, items_per_sec, ratings_per_sec / 1e6);
  Sys::cout() << buf;
}

Sys::Sys(std::string name, std::string fname, std::string probename) : name(name), iter(-1), assigned(false), dom(nprocs+1) {

    if (fname.find(".mbin") != std::string::npos) read_sparse_float64(M, fname);
    else if (fname.find(".mtx") != std::string::npos) loadMarket(M, fname);
    else Sys::cout() << "input filename: expecing .mbin or .mtx, got " << fname << std::endl;

    if (probename.find(".mbin") != std::string::npos) read_sparse_float64(T, probename);
    else if (probename.find(".mtx") != std::string::npos) loadMarket(T, probename);
    else Sys::cout() << "input filename: expecing .mbin or .mtx, got " << probename << std::endl;

    auto rows = std::max(M.rows(), T.rows());
    auto cols = std::max(M.cols(), T.cols());
    M.conservativeResize(rows,cols);
    T.conservativeResize(rows,cols);
    Pm2 = Pavg = T; // reference ratings and predicted ratings
    assert(M.rows() == Pavg.rows());
    assert(M.cols() == Pavg.cols());
    assert(Sys::nprocs <= (int)Sys::max_procs);
}

Sys::Sys(std::string name, const SparseMatrixD &Mt, const SparseMatrixD &Pt) : name(name), iter(-1), assigned(false), dom(nprocs+1) {
    M = Mt.transpose();
    Pm2 = Pavg = T = Pt.transpose(); // reference ratings and predicted ratings
    assert(M.rows() == Pavg.rows());
    assert(M.cols() == Pavg.cols());
}

Sys::~Sys() 
{
    if (measure_perf) {
        Sys::cout() << " --------------------\n";
        Sys::cout() << name << ": sampling times on " << procid << "\n";
        for(int i = from(); i<to(); ++i) 
        {
            Sys::cout() << "\t" << nnz(i) << "\t" << sample_time.at(i) / nsims  << "\n";
        }
        Sys::cout() << " --------------------\n\n";
    }
}

typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PermMatrix;

void Sys::permuteCols(const PermMatrix &perm)
{
    T = T * perm;
    Pavg = Pavg * perm;
    Pm2 = Pm2 * perm;
    M = M * perm;
}

void Sys::init()
{
    //-- M
    assert(M.rows() > 0 && M.cols() > 0);
    mean_rating = M.sum() / M.nonZeros();
    items().setZero();
    sum_map().setZero();
    cov_map().setZero();
    norm_map().setZero();

    if (Sys::procid == 0) {
        Sys::cout() << "mean rating = " << mean_rating << std::endl;
        Sys::cout() << "num " << name << ": " << num() << std::endl;
    }

    if (measure_perf) sample_time.resize(num(), .0);
}

void Sys::assign(Sys &other)
{
    if (nprocs == 1) { dom[0] = 0; dom[1] = num(); return; }
    if (!permute) { 
        int p = num() / nprocs;
        int i=0; for(; i<nprocs; ++i) dom[i] = i*p;
        dom[i] = num();
        return;
    }

    std::vector<std::vector<double>> comm_cost(num());
    if (other.assigned) {
        // comm_cost[i][j] == communication cost if item i is assigned to processor j
        for(int i=0; i<num(); ++i) {
            std::vector<unsigned> comm_per_proc(nprocs);
            const int count = M.innerVector(i).nonZeros();
            for (SparseMatrixD::InnerIterator it(M,i); it; ++it) comm_per_proc.at(other.proc(it.row()))++;
            for(int j=0; j<nprocs; j++) comm_cost.at(i).push_back(count - comm_per_proc.at(j));
        }
    }

    std::vector<unsigned> nnz_per_proc(nprocs);
    std::vector<unsigned> items_per_proc(nprocs);
    std::vector<double>   work_per_proc(nprocs);

    std::vector<int> item_to_proc(num(), -1);

    unsigned total_nnz   = 1;
    unsigned total_items = 1;
    double   total_work  = 0.01;
    unsigned total_comm  = 0;

    auto best = [&](int idx, double r1, double r2) {
        double min_cost = 1e9;
        int best_proc = -1;
        for(int i=0; i<nprocs; ++i) {
            //double nnz_unbalance = (double)nnz_per_proc[i] / total_nnz;
            //double items_unbalance =  (double)items_per_proc[i] / total_items;
            //double work_unbalance = std::max(nnz_unbalance, items_unbalance);
            double work_unbalance = work_per_proc[i] / total_work;

            double comm = other.assigned ? comm_cost.at(idx).at(i) : 0.0;
            double total_cost = r1 * work_unbalance + r2 * comm;
            if (total_cost > min_cost) continue;
            best_proc = i;
            min_cost = total_cost;
        }
        return best_proc;
    };

    auto assign = [&](int item, int proc) {
        const int nnz = M.innerVector(item).nonZeros();
        double work = 10.0 + nnz; // one item is as expensive as  NZs
        item_to_proc[item] = proc;
        nnz_per_proc  [proc] += nnz;
        items_per_proc[proc]++;
        work_per_proc [proc] += work;
        total_nnz += nnz;
        total_items++;
        total_work+= work;
        total_comm += (other.assigned ? comm_cost.at(item).at(proc) : 0);
    };

    auto unassign = [&](int item) {
        int proc = item_to_proc[item];
        if (proc < 0) return;
        const int nnz = M.innerVector(item).nonZeros();
        double work = 7.1 + nnz;
        item_to_proc[item] = -1;
        nnz_per_proc  [proc] -= nnz;
        items_per_proc[proc]--;
        work_per_proc [proc] -= work;
        total_nnz -= nnz;
        total_items--;
        total_work -= work;
        total_comm -= (other.assigned ? comm_cost.at(item).at(proc) : 0);
        
    };

    // iterate once
    auto print = [&](int iter) {
        Sys::cout() << name << " -- iter " << iter << " -- \n";
        if (Sys::procid == 0) {
            int max_nnz = *std::max_element(nnz_per_proc.begin(), nnz_per_proc.end());
            int min_nnz = *std::min_element(nnz_per_proc.begin(), nnz_per_proc.end());
            int avg_nnz = nnz() / nprocs;

            int max_items = *std::max_element(items_per_proc.begin(), items_per_proc.end());
            int min_items = *std::min_element(items_per_proc.begin(), items_per_proc.end());
            int avg_items = num() / nprocs;

            double max_work = *std::max_element(work_per_proc.begin(), work_per_proc.end());
            double min_work = *std::min_element(work_per_proc.begin(), work_per_proc.end());
            double avg_work = total_work / nprocs;

            Sys::cout() << name << ": comm cost " << 100.0 * total_comm / nnz() / nprocs << "%\n";
            Sys::cout() << name << ": nnz unbalance: " << (int)(100.0 * Sys::nprocs * (max_nnz - min_nnz) / nnz()) << "%"
                << "\t(" << max_nnz << " <-> " << avg_nnz << " <-> " << min_nnz << ")\n";
            Sys::cout() << name << ": items unbalance: " << (int)(100.0 * Sys::nprocs * (max_items - min_items) / num()) << "%"
                << "\t(" << max_items << " <-> " << avg_items << " <-> " << min_items << ")\n";
            Sys::cout() << name << ": work unbalance: " << (int)(100.0 * Sys::nprocs * (max_work - min_work) / total_work) << "%"
                << "\t(" << max_work << " <-> " << avg_work << " <-> " << min_work << ")\n\n";
        }

        Sys::cout() << name << ": nnz:\t" << nnz_per_proc[procid] << " / " << nnz() << "\n";
        Sys::cout() << name << ": items:\t" << items_per_proc[procid] << " / " << num() << "\n";
        Sys::cout() << name << ": work:\t" << work_per_proc[procid] << " / " << total_work << "\n";
    };

    for(int j=0; j<3; ++j) {
        for(int i=0; i<num(); ++i) {
            unassign(i); 
            assign(i, best(i, 10000, 0));
        }
        print(j);
    }
    
    std::vector<std::vector<unsigned>> proc_to_item(nprocs);
    for(int i=0; i<num(); ++i) proc_to_item[item_to_proc[i]].push_back(i);

    // permute T
    unsigned pos = 0;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(num());
    for(auto p: proc_to_item) for(auto i: p) perm.indices()(pos++) = i;
    auto oldT = T;

    this->permuteCols(perm);
    other.M = M.transpose();
    other.Pavg = Pavg.transpose();
    other.Pm2 = Pm2.transpose();
    other.T = T.transpose();

    int i = 0;
    int n = 0;
    dom[0] = 0;
    for(auto p : items_per_proc) dom[++i] = (n += p);

#ifndef NDEBUG
    int j = 0;
    for(auto i : proc_to_item.at(0)) assert(T.col(j++).nonZeros() == oldT.col(i).nonZeros());
#endif

    Sys::cout() << name << " domain:" << std::endl;
    print_dom(Sys::cout());
    Sys::cout() << std::endl;

    assigned = true;
}

void Sys::update_conn(Sys& other)
{
    unsigned tot = 0;
    conn_map.clear();
    conn_count_map.clear();
    assert(nprocs <= (int)max_procs);

    conn_map.resize(num());
    for (int k=0; k<num(); ++k) {
        std::bitset<max_procs> &bm = conn_map[k];
        for (SparseMatrixD::InnerIterator it(M,k); it; ++it) bm.set(other.proc(it.row()));
        for (SparseMatrixD::InnerIterator it(Pavg,k); it; ++it) bm.set(other.proc(it.row()));
        bm.reset(proc(k)); // not to self
        tot += bm.count();

        // keep track of how may proc to proc sends
        auto from_proc = proc(k);
        for(int to_proc=0; to_proc<Sys::nprocs; to_proc++) {
            if (!bm.test(to_proc)) continue;
            conn_count_map[std::make_pair(from_proc, to_proc)]++;
        }
    }

    if (Sys::procid == 0) {
        Sys::cout() << name << ": avg items to send per iter: " << tot << " / " << num() << " = " << (double)tot / (double)num() << std::endl;

        Sys::cout() << name << ": messages from -> to proc\n";
        for(int i=0; i<Sys::nprocs; ++i) Sys::cout() << "\t" << i;
        Sys::cout() << "\n";
        for(int i=0; i<Sys::nprocs; ++i) {
            Sys::cout() << i;
            for(int j=0; j<Sys::nprocs; ++j) Sys::cout() << "\t" << conn_count(i,j);
            Sys::cout() << "\n";
        }
        
    }
}

void Sys::opt_conn(Sys& other)
{
    // sort internally according to hamming distance
    PermMatrix perm(num());
    perm.setIdentity();

    std::vector<std::string> s(num());

    auto v = perm.indices().data();
#ifdef BPMF_TBB_SCHED
    tbb::parallel_for(tbb::blocked_range<int>(0, nprocs, 1),
        [&](const tbb::blocked_range<int>& r) {
            for(auto p = r.begin(); p<r.end(); ++p) {
                for(int i=from(p); i<to(p); ++i) s[i] = conn(i).to_string();
                std::sort(v + from(p), v + to(p), [&](const int& a, const int& b) { return (s[a] < s[b]); });
            }
        }
    );
#elif defined(BPMF_OMP_SCHED) || defined (BPMF_SER_SCHED)
#if defined(BPMF_OMP_SCHED)
#pragma omp parallel for 
#endif
    for(auto p = 0; p < nprocs; ++p) {
	for(int i=from(p); i<to(p); ++i) s[i] = conn(i).to_string();
        std::sort(v + from(p), v + to(p), [&](const int& a, const int& b) { return (s[a] < s[b]); });
    }
#else
#error opt_conn
#endif

    permuteCols(perm);
    other.M = M.transpose();
    other.Pavg = Pavg.transpose();
    other.Pm2 = Pm2.transpose();
    other.T = T.transpose();
}


void Sys::build_conn(Sys& other)
{
    if (nprocs == 1) return;

    update_conn(other);
    opt_conn(other);
    update_conn(other);
}

class PrecomputedLLT : public Eigen::LLT<MatrixNNd>
{
  public:
    void operator=(const MatrixNNd &m) { m_matrix = m; m_isInitialized = true; m_info = Eigen::Success; }
};

VectorNd Sys::sample(long idx, const MapNXd in)
{
    auto start = tick();
    const double alpha = 2;
    const int breakpoint1 = 1000;
    const int breakpoint2 = 1000;
    const int count = M.innerVector(idx).nonZeros();

    VectorNd rr = hp.LambdaF * hp.mu;
    PrecomputedLLT chol;

    if( count < breakpoint1 ) {
        chol = hp.LambdaL;
        for (SparseMatrixD::InnerIterator it(M,idx); it; ++it) {
            auto col = in.col(it.row());
            chol.rankUpdate(col, alpha);
            rr.noalias() += col * ((it.value() - mean_rating) * alpha);
        }
    } else if (count < breakpoint2) {
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
        rr += s.combine(std::plus<VectorNd>());
        MatrixNNd MM = p.combine(std::plus<MatrixNNd>());
#elif defined(BPMF_OMP_SCHED) || defined(BPMF_SER_SCHED)
        MatrixNNd MM(MatrixNNd::Zero()); // outer prod

//#ifdef BPMF_OMP_SCHED
//#pragma omp parallel for reduction(VectorPlus:rr) reduction(MatrixPlus:MM)
//#endif
        for(int i = from; i<to; ++i) {
            auto val = M.valuePtr()[i];
            auto idx = M.innerIndexPtr()[i];
            auto col = in.col(idx);
            MM.noalias() += col * col.transpose();
            rr.noalias() += col * ((val - mean_rating) * alpha);
        }

#else
#error No sched sample_one
#endif

        chol.compute(hp.LambdaF + alpha * MM);
    }

    if(chol.info() != Eigen::Success) abort();

    chol.matrixL().solveInPlace(rr);
    rr += nrandn();
    chol.matrixU().solveInPlace(rr);
    items().col(idx) = rr;

    auto stop = tick();
    register_time(idx, 1e6 * (stop - start));
    //Sys::cout() << "  " << count << ": " << 1e6*(stop - start) << std::endl;

    assert(rr.norm() > .0);

    return rr;
}


void Sys::sample(Sys &in) 
{
    iter++;

#ifdef BPMF_TBB_SCHED
    tbb::combinable<VectorNd>  s(VectorNd::Zero()); // sum
    tbb::combinable<double>    n(0.0); // squared norm
    tbb::combinable<MatrixNNd> p(MatrixNNd::Zero()); // outer prod

    tbb::parallel_for( 
        tbb::blocked_range<VectorNd::Index>(from(), to(), grain_size),
        [&](const tbb::blocked_range<typename VectorNd::Index>& r) {
            for(auto i = r.begin(); i<r.end(); ++i) {
                assert (proc(i) == Sys::procid);
                auto r = sample(i,in.items()); 
                p.local() += (r * r.transpose());
                s.local() += r;
                n.local() += r.squaredNorm();
            }
            send_items(r.begin(), r.end());
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

#ifdef BPMF_OMP_SCHED
#pragma omp parallel for reduction(VectorPlus:sum) reduction(MatrixPlus:prod) reduction(+:norm) schedule(dynamic, 1)
#endif
    for(int i = from(); i<to(); ++i) {
        auto r = sample(i,in.items()); 
        prod += (r * r.transpose());
        sum += r;
        norm += r.squaredNorm();
        send_items(i,i+1);
    }
#elif defined(BPMF_SER_SCHED)
    // serial 
    VectorNd sum(VectorNd::Zero()); // sum
    MatrixNNd prod(MatrixNNd::Zero()); // outer prod
    double    norm(0.0); // squared norm
    for(int i=from(); i<to(); ++i) {
        assert (proc(i) == Sys::procid);
        auto r = sample(i,in.items());
        prod += (r * r.transpose());
        sum += r;
        norm += r.squaredNorm();
        send_items(i,i+1);
    }
#else
#error No sched sample_all
#endif

    const int N = num();
    local_sum() = sum;
    local_cov() = (prod - (sum * sum.transpose() / N)) / (N-1);
    local_norm() = norm;
}

void Sys::register_time(int i, double t)
{
    if (measure_perf) sample_time.at(i) += t;
}
