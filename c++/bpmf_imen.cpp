
#include <iostream>
#include <limits>
#include <iomanip>
#include <fstream>
#include <list>
#include <string>
#include <algorithm>
#include <random>
#include <shark.hpp>
#include "../src/comm_impl.hpp"
#include <unsupported/Eigen/SparseExtra>

#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#else
#include <tbb/tbb.h>
#endif

#include "bpmf.h"

using namespace std;
using std::numeric_limits;
using namespace Eigen;

using namespace shark;
using namespace shark::types2d;

const int num_feat = 10;

const double alpha = 2;
int nsims = 20;
const int burnin = 5;

double mean_rating = .0;

int num_p = 0;
int num_m = 0;

typedef SparseMatrix<double> SparseMatrixD;
SparseMatrixD M, Mt, P;

long int total_exchanged_movies = 0;
long int total_exchanged_movies_elements = 0;
long int total_exchanged_users = 0;
long int total_exchanged_users_elements = 0;

VectorXd mu_u(num_feat);
VectorXd mu_m(num_feat);
MatrixXd Lambda_u(num_feat, num_feat);
MatrixXd Lambda_m(num_feat, num_feat);
MatrixXd sample_m;

// parameters of Inv-Whishart distribution (see paper for details)

const int b0_u = 2;
const int df_u = num_feat;
VectorXd mu0_u(num_feat);

MatrixXd WI_u(num_feat, num_feat);
MatrixXd WI_m(num_feat, num_feat);

const int b0_m = 2;
const int df_m = num_feat;
VectorXd mu0_m(num_feat);

void init() {
    mean_rating = M.sum() / M.nonZeros();
    Lambda_u.setIdentity();
    Lambda_m.setIdentity();

    sample_m = MatrixXd(num_feat,M.cols());
    sample_m.setZero();

    // parameters of Inv-Whishart distribution (see paper for details)
    WI_u.setIdentity();
    mu0_u.setZero();

    WI_m.setIdentity();
    mu0_m.setZero();
}

inline double sqr(double x) { return x*x; }
std::pair<double,double> eval_probe_vec(int n, VectorXd & predictions, const MatrixXd &sample_m, const MatrixXd &sample_u, double mean_rating,
		GlobalArrayD& users)
{
    double se = 0.0, se_avg = 0.0;
    unsigned idx = 0;

    for (int k=0; k< P.outerSize(); ++k)
    {
        for (SparseMatrix<double>::InnerIterator it(P,k); it; ++it)
        {
        	if ( (users.domain().local().lower[1] <= it.row()) && (it.row() < users.domain().local().upper[1]))
        	{
        		const double pred = sample_m.col(it.col()).dot(sample_u.col(it.row() - users.domain().local().lower[1])) + mean_rating;
        		//se += (it.value() < log10(200)) == (pred < log10(200));
        		se += sqr(it.value() - pred);

        		const double pred_avg = (n == 0) ? pred : (predictions[idx] + (pred - predictions[idx]) / n);
        		//se_avg += (it.value() < log10(200)) == (pred_avg < log10(200));
        		se_avg += sqr(it.value() - pred_avg);
        		predictions[idx++] = pred_avg;
        	}
        }
    }

    const unsigned N = P.nonZeros();

    const double rmse = sqrt( se / idx );
    const double rmse_avg = sqrt( se_avg / idx );
    return std::make_pair(rmse, rmse_avg);
}

VectorXd communication_buffer(num_feat);

void sample_movies(MatrixXd &s, int mm, const SparseMatrixD &mat, double mean_rating,
const MatrixXd &samples, int alpha, const VectorXd &mu_u, const MatrixXd &Lambda_u, GlobalArrayD& users)
{
	int i = 0;
    MatrixXd MM(num_feat, num_feat); MM.setZero();
	VectorXd rr(num_feat); rr.setZero();

    //TODO je n'ai plus besoin de parcourir toute la matrice je vais parcourir juste mon out_link que ce soit local ou remote

    for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it, ++i)
    {
    	if ( (users.domain().local().lower[1] > it.row()) || (it.row() >= users.domain().local().upper[1]) )
    	{
			 coords_range missing_user;

			 missing_user.lower[1] = it.row();
			 missing_user.upper[1] = it.row() + 1;

		     missing_user.lower[0] = 0;
		     missing_user.upper[0] = num_feat;

		     //#pragma omp critical
		     users.get(missing_user, communication_buffer.data());

			 MM.noalias() += communication_buffer * communication_buffer.transpose();
			 rr.noalias() += communication_buffer * ((it.value() - mean_rating) * alpha);
		 }
		 else
		 {
			 auto col = samples.col(it.row() - users.domain().local().lower[1]);

			 MM.noalias() += col * col.transpose();
			 rr.noalias() += col * ((it.value()- mean_rating) * alpha);
		 }
	}

	Eigen::LLT<MatrixXd> chol = (Lambda_u + alpha * MM).llt();

	VectorXd tmp = rr + Lambda_u * mu_u;
	chol.matrixL().solveInPlace(tmp);
	tmp += nrandn(num_feat);
	chol.matrixU().solveInPlace(tmp);
	s.col(mm) = tmp;
}

void sample_users(auto &s, int mm, const SparseMatrixD &mat, double mean_rating,
    const MatrixXd &samples, int alpha, const VectorXd &mu_u, const MatrixXd &Lambda_u, int idx	)
{
    int i = 0;
    MatrixXd MM(num_feat, num_feat); MM.setZero();
    VectorXd rr(num_feat); rr.setZero();

    for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it, ++i)
    {
    	 auto col = samples.col(it.row());

    	 MM.noalias() += col * col.transpose();
    	 rr.noalias() += col * (( it.value() - mean_rating) * alpha);
    }

    Eigen::LLT<MatrixXd> chol = (Lambda_u + alpha * MM).llt();

    if(chol.info() != Eigen::Success)
    {
      throw std::runtime_error("Cholesky Decomposition failed!");
    }

    VectorXd tmp = rr + Lambda_u * mu_u;
    chol.matrixL().solveInPlace(tmp);
    tmp += nrandn(num_feat);
    chol.matrixU().solveInPlace(tmp);
    s.col(idx) = tmp;
}

int main(int argc, char *argv[])
{
	Init(&argc, &argv);

	bool random_assignment = false;
	bool block = false;
    int ch;
    string fname;
    string probename;

    while((ch = getopt(argc, argv, "n:t:b:r:p:i:")) != -1)
    {
    	switch(ch)
    	{
    		case 'i':
				{
					istringstream iss(optarg);
					iss >> nsims;
				}
				break;
    	    case 't':
    			{
    				istringstream iss(optarg);
    				iss >> nthrds;
    			}
    			break;
    		case 'n':
    			{
    				istringstream iss(optarg);
    				iss >> fname;
    			}
    			break;
    		case 'p':
    			{
    				istringstream iss(optarg);
    				iss >> probename;
    			}
    			break;
    		case 'r':
    			{
    				istringstream iss(optarg);
    				iss >> random_assignment;
    			}
    			break;
    		case 'b':
    				block = true;
    				break;
    			case '?':
    			default:
    				cout << "Usage: " << argv[0] << " [-t <threads>] [-p <problem>] [-r <random_assignement>] \n" << endl;
    				Abort(1);
    		}
    	}

    assert(fname.c_str() && "filename missing");

    loadMarket(M, fname);

    Mt = M.transpose();

    loadMarket(P, probename);

    assert(M.nonZeros() > P.nonZeros());

    num_m = M.cols();
    num_p = M.rows();

	init();

	SetupThreads();

	{
		typename GlobalArrayD::bounds bd = {{ BoundaryD::constant(0.0), BoundaryD::constant(0.0) }};

		const array<int,2> pcoords = {{ block ? 0 : 1, 0}};
		coords size = { {num_feat, num_p} };
		Domain d(world(), size, pcoords);

		GlobalArrayD users(d);

		VectorXd predictions;

		predictions = VectorXd::Zero( P.nonZeros() );

		long double  average_sampling_sec =0;

		if(world().procid == 0)
		{
			cout << "num_feat: " << num_feat<<endl;
			cout << "sched: " << sched << endl;
			cout << "block: " << boolalpha << block << noboolalpha << endl;
			cout << "nprocs: " << world().nprocs << endl;
			cout << "nthrds: " << nthrds << endl;

			d.outputDistribution(cout);
		}

		int my_users_min = users.domain().local().lower[1];
		int my_users_max = users.domain().local().upper[1];

		std::vector<int> movie_bids(world().nprocs,0); //contains the indices of the movies that this process will sample
		std::vector<int> assigned_movies(num_m, 0); //contains the indice of the process that will sample the movie assigned_movies[0] = 1 the process number 1 will sample the movies number 0 ...

		vector<double> recv(num_feat); //first element is the movie index to update

		if (world().nprocs != 0)
		if (random_assignment == 0) //assign a movie to the process that owns the biggest number of users that rated that movie
		{
			for (int mm = 0; mm < num_m; mm++)
			{
				int max = 0;

				for(int k = 0; k < world().nprocs; k++) movie_bids[k] = 0;

				for (SparseMatrixD::InnerIterator it(M,mm); it; ++it)
				{
					for(int k = 0; k < world().nprocs; k++)
					{
						if ( ( d.local(k).lower[1] <= it.row() ) && ( it.row() <= d.local(k).upper[1] ) ) movie_bids[k]++;
						if ( movie_bids[k] > movie_bids[max] ) max = k;
					}

					assigned_movies[mm] = max;
				}
			}
		}
		else //random assignment of the movies to the processes
		{
			int ratio = num_m / world().nprocs;

			for (int u = 0; u < num_m; u++)
			{
				for(int k = 0; k < world().nprocs; k++)
					if ( (int) (u/ratio) == k )
						assigned_movies[u] = k;

				if ( (int) (u/ratio) == world().nprocs ) assigned_movies[u] = 0;
			}
		}

		auto start = tick();

		int movies_assigned_to_me = 0;

		for (int m = 0; m < num_m; m++)
			if(assigned_movies[m] == world().procid)
				movies_assigned_to_me++;

    	cout << "I am process " << world().procid << " I will sample " << movies_assigned_to_me << " movies " <<endl;

        SparseMatrixD Mt = M.transpose();

        auto sample_u = Map<MatrixXd>(users.ptr, num_feat, (my_users_max - my_users_min));

        for(int i=0; i<nsims; ++i)
        {
    	  // Sample from user hyperparams
          tie(mu_u, Lambda_u) = CondNormalWishart(sample_u, mu0_u, b0_u, WI_u, df_u);

          // Sample from movie hyperparams
          tie(mu_m, Lambda_m) = CondNormalWishart(sample_m, mu0_m, b0_m, WI_m, df_m);

		  #pragma omp parallel for
          	  for(int uu = my_users_min; uu < my_users_max - 1; uu++)
          	  {
          		  sample_users(sample_u, uu, Mt, mean_rating, sample_m, alpha, mu_u, Lambda_u, uu - my_users_min);
          	  }

          Map<MatrixXd>(users.ptr, sample_u.rows(), sample_u.cols() ) = sample_u;

          MPI_Status status;

          vector<int> flagsm(num_m, 0); //receive info about a movie only once, this is used in the case this process have more than one user that have rated that movie

          MPI_Request request[num_m];

          for(int mm = 0; mm < num_m; mm++)
          {
        	  if (assigned_movies[mm] == world().procid) //this movie is assigned to me so I sample it and then I send the new movie data to all the users that are concerned
        	  {
        		  sample_movies(sample_m, mm, M, mean_rating, sample_u, alpha, mu_m, Lambda_m, users);

        		  vector<int> flagsp(world().nprocs, 0);

        		  for (SparseMatrixD::InnerIterator it(M,mm); it; ++it)
       				  for(int k = 0; k < world().nprocs; k++)
       					  if ( ( users.domain().local(k).lower[1] <= it.row() ) && ( it.row()< users.domain().local(k).upper[1] ) && flagsp[k] == 0 && k != world().procid )
       					  {
       						  flagsp[k] = 1;
           					  MPI_Isend(sample_m.col(mm).data(), num_feat, MPI_DOUBLE, k ,0 , MPI_COMM_WORLD, &request[mm]);
       					  }
        	  }
          }

          for(int mm = 0; mm < num_m; mm++)
          {
        	  if (assigned_movies[mm] != world().procid) //the movie is not assigned to me but I may have users that rated it
        	  {
        		  for (SparseMatrixD::InnerIterator it(M,mm); it; ++it)
        		  {
        			  if ( (users.domain().local().lower[1] <= it.row()) && (it.row() < users.domain().local().upper[1]) && flagsm[mm] == 0 )
        			  {
        				  flagsm[mm] = 1;

        				  MPI_Recv(recv.data(), num_feat, MPI_DOUBLE, assigned_movies[mm] , 0, MPI_COMM_WORLD, &status);

        				  sample_m.col(mm) = Map<VectorXd> (recv.data(), num_feat);
        				}
        		  }
        	  }
          }

           auto eval = eval_probe_vec( (i < burnin) ? 0 : (i - burnin), predictions, sample_m, sample_u, mean_rating, users);
           double norm_u = sample_u.norm();
           double norm_m = sample_m.norm();
           auto end = tick();
           auto elapsed = end - start;

           double samples_per_sec = (i + 1) * (M.rows() + M.cols()) / elapsed;

           if (world().procid == 0)  printf("Iteration %d:\t RMSE: %3.2f\tavg RMSE: %3.2f\tFU(%6.2f)\tFM(%6.2f)\tSamples/sec: %6.2f\n",
                   i, eval.first, eval.second, norm_u, norm_m, samples_per_sec);

           average_sampling_sec += samples_per_sec;
      }

      auto end = tick();
      auto elapsed = end - start;

      if (world().procid == 0)
      {
    	  cout << "Total time: " << elapsed <<endl <<flush;
		  cout << "Average Samples/sec: " << average_sampling_sec / nsims << endl <<flush;
	  }
	 }

      Finalize();

      return 0;
}
