/*
 * Copyright (c) 2021-2022, etascale
 * All rights reserved.
 */

#include <mpi.h>
#include "argo.hpp"

#define SYS ARGO_Sys

struct ARGO_Sys : public Sys
{
	//-- c'tor
	ARGO_Sys(std::string name, std::string fname, std::string probename) : Sys(name, fname, probename) {}
	ARGO_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}
	~ARGO_Sys();

	virtual void alloc_and_init();

	virtual void send_item(int);
	virtual void sample(Sys &in);
};

ARGO_Sys::~ARGO_Sys()
{
	argo::codelete_array(items_ptr);
}

void ARGO_Sys::alloc_and_init()
{
	items_ptr = argo::conew_array<double>(num_latent * num());

	init();
}

void ARGO_Sys::send_item(int i)
{
#ifdef BPMF_ARGO_SELECTIVE_RELEASE
	BPMF_COUNTER("send_item");
	static std::mutex m;

	m.lock();
	auto offset = i * num_latent;
	auto size = num_latent;
	argo::backend::selective_release(items_ptr+offset, size*sizeof(double));
	m.unlock();
#endif
}

void ARGO_Sys::sample(Sys &in)
{
	{
		BPMF_COUNTER("compute");
		Sys::sample(in);
	}
	{
		BPMF_COUNTER("sync_sample");
		Sys::sync();
	}
}

void Sys::Init()
{
	// global address space size - 50GiB
	argo::init(50*1024*1024*1024UL);

	Sys::procid = argo::node_id();
	Sys::nprocs = argo::number_of_nodes();
}

void Sys::Finalize()
{
	argo::finalize();
}

void Sys::sync()
{
	argo::barrier();
}

void Sys::Abort(int err)
{
	MPI_Abort(MPI_COMM_WORLD, err);
}

void Sys::reduce_sum_cov_norm()
{
    BPMF_COUNTER("reduce_sum_cov_norm");
    MPI_Allreduce(MPI_IN_PLACE, sum.data(), num_latent, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, cov.data(), num_latent * num_latent, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}
