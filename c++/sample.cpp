/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include "bpmf.h"

#include <random>
#include <memory>
#include <cstdio>
#include <iostream>
#include <climits>
#include <stdexcept>

#include "io.h"

#if defined(BPMF_OMP_SCHED)
#include "omp.h"

#pragma omp declare reduction (VectorPlus : VectorNd : omp_out += omp_in) initializer(omp_priv = VectorNd::Zero())
#pragma omp declare reduction (MatrixPlus : MatrixNNd : omp_out += omp_in) initializer(omp_priv = MatrixNNd::Zero())
#endif

static const bool measure_perf = false;

std::ostream *Sys::os;
int Sys::procid = -1;
int Sys::nprocs = -1;
int Sys::nthrds = -1;

int Sys::nsims;
int Sys::burnin;

bool Sys::permute = true;

unsigned Sys::grain_size;

//SHAMAKINA: begin
void calc_upper_part(MatrixNNd &m, VectorNd v);         // function for calcutation of an upper part of a symmetric matrix: m = v * v.transpose(); 
void copy_lower_part(MatrixNNd &m);                     // function to copy an upper part of a symmetric matrix to a lower part
//SHAMAKINA: end

// verifies that A is transpose of B
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

bool Sys::isMasterThread()
{
#ifdef BPMF_TBB_SCHED
    assert(0);
#elif defined(BPMF_OMP_SCHED)
    return  omp_get_thread_num() == 0;
#elif defined(BPMF_SER_SCHED)
    return true;
#else
#error No sched SetupThreads
#endif
}


//
// Does predictions for prediction matrix T
// Computes RMSE (Root Means Square Error)
//
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

//
// Prints sampling progress
//
void Sys::print(double items_per_sec, double ratings_per_sec, double norm_u, double norm_m) {
  char buf[1024];
  sprintf(buf, "%d: Iteration %d:\t RMSE: %3.2f\tavg RMSE: %3.2f\tFU(%6.2f)\tFM(%6.2f)\titems/sec: %6.2f\tratings/sec: %6.2fM\n",
                    Sys::procid, iter, rmse, rmse_avg, norm_u, norm_m, items_per_sec, ratings_per_sec / 1e6);
  Sys::cout() << buf;
}

//
// Constructor with that reads MTX files
// 
Sys::Sys(std::string name, std::string fname, std::string probename)
    : name(name), iter(-1), assigned(false), dom(nprocs+1)
{

    read_matrix(fname, M);
    read_matrix(probename, T);

    auto rows = std::max(M.rows(), T.rows());
    auto cols = std::max(M.cols(), T.cols());
    M.conservativeResize(rows,cols);
    T.conservativeResize(rows,cols);
    Pm2 = Pavg = T; // reference ratings and predicted ratings
    assert(M.rows() == Pavg.rows());
    assert(M.cols() == Pavg.cols());
    assert(Sys::nprocs <= (int)Sys::max_procs);
}

//
// Constructs Sys as transpose of existing Sys
//
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

bool Sys::has_prop_posterior() const
{
    return propMu.nonZeros() > 0;
}

void Sys::add_prop_posterior(std::string mtx)
{
    if (mtx.empty()) return;

    char *cp = 0; // copy of mtx
    strcpy(cp, mtx.c_str());
    std::string mu_name = strtok (cp, ",");
    std::string lambda_name = strtok(cp, ",");

    read_matrix(mu_name, propMu);
    read_matrix(lambda_name, propLambda);

    assert(propMu.cols() == num());
    assert(propLambda.cols() == num());

    assert(propMu.rows() == num_feat);
    assert(propLambda.rows() == num_feat * num_feat);

}

typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PermMatrix;

void Sys::permuteCols(const PermMatrix &perm)
{
    T = T * perm;
    Pavg = Pavg * perm;
    Pm2 = Pm2 * perm;
    M = M * perm;
    propMu * perm;
    propLambda * perm;
}

//
// Intializes internal Matrices and Vectors
//
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
        if (has_prop_posterior())
        {
            Sys::cout() << "with propagated posterior" << std::endl;
        }
    }

    if (measure_perf) sample_time.resize(num(), .0);
}

//
// Distributes users/movies accros several nodes
// takes into account load balance and communication cost
//
void Sys::assign(Sys &other)
{
    if (nprocs == 1) {
        dom[0] = 0; 
        dom[1] = num(); 
        return; 
    }

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

    // computes best node to assign movie/user idx
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

    // update cost function when item is assigned to proc
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

    // update cost function when item is removed from proc
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

    // print cost after iterating once
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

    // permute T, P  based on assignment done before
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

//
// Update connectivity map (where to send certain items)
// based on assignment to nodes
//
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

//
// try to keep items that have to be sent to the same node next to eachothe
//
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
    //opt_conn(other);
    //update_conn(other);
}

class PrecomputedLLT : public Eigen::LLT<MatrixNNd>
{
  public:
    void operator=(const MatrixNNd &m) { m_matrix = m; m_isInitialized = true; m_info = Eigen::Success; }
};


//
// Update ONE movie or one user
//
VectorNd Sys::sample(long idx, const MapNXd in)
{
    auto start = tick();
    const double alpha = 2;       // Gaussian noice

    VectorNd hp_mu;
    MatrixNNd hp_Lambda; 
    if (has_prop_posterior())
    {
        hp_mu = propMu.col(idx);
        hp_Lambda = Eigen::Map<MatrixNNd>(propLambda.col(idx).data()); 
    }
    else
    {
        hp_mu = hp.mu;
        hp_Lambda = hp.LambdaF; 
    }


//SHAMAKINA: begin
    int breakpoint1 = 24; 
    int breakpoint2 = 10500; 
// SHAMAKINA: end    
    
    const int count = M.innerVector(idx).nonZeros(); // count of nonzeros elements in idx-th row of M matrix 
                                                     // (how many movies watched idx-th user?).

    VectorNd rr = hp_Lambda * hp.mu;                 // vector num_feat x 1, we will use it in formula (14) from the paper
    PrecomputedLLT chol;                             // matrix num_feat x num_feat, chol="lambda_i with *" from formula (14) 
    
    // if this user movie has less than 1K ratings,
    // we do a serial rank update
    if( count < breakpoint1 ) {

        chol = hp_Lambda;
        for (SparseMatrixD::InnerIterator it(M,idx); it; ++it) {
            auto col = in.col(it.row());
            chol.rankUpdate(col, alpha);
            rr.noalias() += col * ((it.value() - mean_rating) * alpha);
        }

    // else we do a serial full cholesky decomposition
    // (not used if breakpoint1 == breakpoint2)
    } else if (count < breakpoint2) {

        MatrixNNd MM(MatrixNNd::Zero());
        for (SparseMatrixD::InnerIterator it(M,idx); it; ++it) {
            auto col = in.col(it.row());
            
// SHAMAKINA: begin
            //MM.noalias() += col * col.transpose();
            calc_upper_part(MM, col);
// SHAMAKINA: end
            
            rr.noalias() += col * ((it.value() - mean_rating) * alpha);
        }

// SHAMAKINA: begin
        // Here, we copy a triangular upper part to a triangular lower part, because the matrix is symmetric.
        copy_lower_part(MM);
// SHAMAKINA: end

        chol.compute(hp_Lambda + alpha * MM);
    // for > 1K ratings, we have additional thread-level parallellism
    } else {
        auto from = M.outerIndexPtr()[idx];   // "from" belongs to [1..m], m - number of movies in M matrix 
        auto to = M.outerIndexPtr()[idx+1];   // "to"   belongs to [1..m], m - number of movies in M matrix

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
        MatrixNNd MM(MatrixNNd::Zero());               // matrix num_feat x num_feat 
 
#ifdef BPMF_OMP_SCHED

// SHAMAKINA: begin
        omp_set_nested(1);
        // #pragma omp parallel for reduction(VectorPlus:rr) reduction(MatrixPlus:MM)
        #pragma omp parallel for reduction(VectorPlus:rr) reduction(MatrixPlus:MM) num_threads(12) schedule(dynamic,200)
// SHAMAKINA: end
#endif
        for(int j = from; j<to; ++j) {                 // for each nonzeros elemen in the i-th row of M matrix
            auto val = M.valuePtr()[j];                // value of the j-th nonzeros element from idx-th row of M matrix
            auto idx = M.innerIndexPtr()[j];           // index "j" of the element [i,j] from M matrix in compressed M matrix 
            auto col = in.col(idx);                    // vector num_feat x 1 from V matrix: M[i,j] = U[i,:] x V[idx,:] 

// SHAMAKINA: begin
            //MM.noalias() += col * col.transpose();     // outer product
            calc_upper_part(MM, col);
// SHAMAKINA: end
            rr.noalias() += col * ((val - mean_rating) * alpha); // vector num_feat x 1
        }

// SHAMAKINA: begin
        copy_lower_part(MM);
// SHAMAKINA: end

#else
#error No sched sample_one
#endif

        chol.compute(hp_Lambda + alpha * MM);         // matrix num_feat x num_feat
                                                       // chol="lambda_i with *" from formula (14)
                                                       // lambda_i with * = LambdaU + alpha * MM
    }

    if(chol.info() != Eigen::Success) abort();

    // now we should calculate formula (14) from the paper
    // u_i for k-th iteration = Gaussian distribution N(u_i | mu_i with *, [lambda_i with *]^-1) =
    //                        = mu_i with * + s * [U]^-1, 
    //                        where 
    //                              s is a random vector with N(0, I),
    //                              mu_i with * is a vector num_feat x 1, 
    //                              mu_i with * = [lambda_i with *]^-1 * rr,
    //                              lambda_i with * = L * U       

    // Expression u_i = U \ (s + (L \ rr)) in Matlab looks for Eigen library like: 

    chol.matrixL().solveInPlace(rr);                    // L*Y=rr => Y=L\rr, we store Y result again in rr vector  
    rr += nrandn();                                     // rr=s+(L\rr), we store result again in rr vector
    chol.matrixU().solveInPlace(rr);                    // u_i=U\rr 
    items().col(idx) = rr;                              // we save rr vector in items matrix (it is user features matrix)

    auto stop = tick();
    register_time(idx, 1e6 * (stop - start));
    //Sys::cout() << "  " << count << ": " << 1e6*(stop - start) << std::endl;

    assert(rr.norm() > .0);

    return rr;
}

// 
// update ALL movies / users in parallel
//
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

// SHAMAKINA: begin
//#pragma omp parallel for reduction(VectorPlus:sum) reduction(MatrixPlus:prod) reduction(+:norm) schedule(dynamic, 1)
#pragma omp parallel for reduction(VectorPlus:sum) reduction(MatrixPlus:prod) reduction(+:norm) num_threads(24) schedule(dynamic,1) 
// SHAMAKINA: end

#endif
    for(int i = from(); i<to(); ++i) {
        auto r = sample(i,in.items()); 

// SHAMAKINA: begin
        //prod += (r * r.transpose());
        calc_upper_part(prod, r);
// SHAMAKINA: end

        sum += r;
        norm += r.squaredNorm();
        send_items(i,i+1);
    }

// SHAMAKINA: begin
        copy_lower_part(prod);
// SHAMAKINA: end

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

// SHAMAKINA: begin
void calc_upper_part(MatrixNNd &m, VectorNd v)
{
  // we use the formula: m = m + v * v.transpose(), but we calculate only an upper part of m matrix
  for (int j=0; j<num_feat; j++)          // columns
  {
    for(int i=0; i<=j; i++)              // rows
    {
      m(i,j) = m(i,j) + v[j] * v[i];
    }
  }
}

void copy_lower_part(MatrixNNd &m)
{
  // Here, we copy a triangular upper part to a triangular lower part, because the matrix is symmetric.
  for (int j=1; j<num_feat; j++)          // columns
  {
    for(int i=0; i<=j-1; i++)            // rows
    {
      m(j,i) = m(i,j);
    }
  }
}
// SHAMAKINA: end
