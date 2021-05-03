/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include "error.h"
#include "bpmf.h"

#include <memory>
#include <cstdio>
#include <iostream>
#include <climits>
#include <stdexcept>

typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PermMatrix;

void Sys::permuteCols(const PermMatrix &perm, Sys &other)
{
    // permute local matrices
    T = T * perm;
    Pavg = Pavg * perm;
    Pm2 = Pm2 * perm;
    M = M * perm;

    if (has_prop_posterior())
    {
        propMu *= perm;
        propLambda *= perm;
    }

    // permute other matrices
    other.T = T.transpose();
    other.Pavg = Pavg.transpose();
    other.Pm2 = Pm2.transpose();
    other.M = M.transpose();

    col_permutation = col_permutation * perm;
}

void Sys::unpermuteCols(Sys &other)
{
    auto perm = col_permutation.inverse();
    permuteCols(perm, other);

    assert_same_struct(T, Torig);
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
        std::vector<unsigned> test_ratings_per_proc(nprocs);
        for (int i = 0; i < num(); ++i)
        {
            int proc = item_to_proc[i];
            test_ratings_per_proc[proc] += T.col(i).nonZeros();
        }

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

        Sys::cout() << name << ": train nnz:\t" << nnz_per_proc[procid] << " / " << nnz() << "\n";
        Sys::cout() << name << ": test nnz:\t" << test_ratings_per_proc[procid] << " / " << nnz() << "\n";
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

    permuteCols(perm, other);

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
        Sys::cout() << name << ": avg items to send per iter: " << tot << " / " << num() << ": " << (double)tot / (double)num() << std::endl;

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
#pragma omp parallel for
    for (auto p = 0; p < nprocs; ++p)
    {
        for (int i = from(p); i < to(p); ++i)
        {
            s[i] = conn(i).to_string();
        }
        std::sort(v + from(p), v + to(p), [&](const int &a, const int &b) { return (s[a] < s[b]); });
    }

    permuteCols(perm, other);
}


void Sys::build_conn(Sys& other)
{
    if (nprocs == 1) return;

    update_conn(other);
    //opt_conn(other);
    //update_conn(other);
}