#
# Copyright (c) 2014-2016, imec
# All rights reserved.
#

using MAT, Distributions, MatrixMarket

macro default_value(sym, value)
  quote
    symsym = $(QuoteNode(sym))
    v = $value
    if ! isdefined(symsym)
      $(esc(sym)) = v
      println("Setting default value for $symsym = $v")
    end
  end
end

@default_value(num_latent, 10)
@default_value(alpha, 2) # observation noise (precision)
@default_value(skip_load, false)
@default_value(nsims, 10)
@default_value(burnin, 5)
@default_value(clamp_lo, 1)
@default_value(clamp_hi, 5)

if length(ARGS) < 2
  println("Usage bpmf.jl <matrix_train.mtx> <matrix_test.mtx>")
  exit()
end

if ! skip_load
  println("Loading data")
  println(ARGS[1])
  Am = MatrixMarket.mmread(ARGS[1])
  P = MatrixMarket.mmread(ARGS[2])
  (I, J, V) = findnz(P)
  P = sparse(I, J, V)
  num_p, num_m = size(Am)
  Au = Am'
  mean_rating = sum(Am)/nnz(Am)
end

sample_u = zeros(num_p, num_latent)
sample_m = zeros(num_m, num_latent)

# Initialize hierarchical priors
mu_u = zeros(num_latent,1)
mu_m = zeros(num_latent,1)
Lambda_u = eye(num_latent)
Lambda_m = eye(num_latent)

# parameters of Inv-Whishart distribution (see paper for details)
WI_u = eye(num_latent)
b0_u = 2
df_u = num_latent
mu0_u = zeros(num_latent,1)

WI_m = eye(num_latent)
b0_m = 2
df_m = num_latent
mu0_m = zeros(num_latent,1)

function pred(P, sample_m, sample_u, mean_rating)
  (I, J, Vin) = findnz(P)
  pr = sum(sample_m[J,:].*sample_u[I,:],2) + mean_rating;
  Vout = clamp(pr[:,1], clamp_lo, clamp_hi)
  sparse(I, J, Vout)
end

function ConditionalNormalWishart(U::Matrix{Float64}, mu::Vector{Float64}, kappa::Real, T::Matrix{Float64}, nu::Real)
  N = size(U, 1)
  Ū = mean(U,1)
  S = cov(U, mean=Ū)
  Ū = Ū'

  mu_c = (kappa*mu + N*Ū) / (kappa + N)
  kappa_c = kappa + N
  T_c = inv( inv(T) + N * S + (kappa * N)/(kappa + N) * (mu - Ū) * (mu - Ū)' )
  nu_c = nu + N

  NormalWishart(vec(mu_c), kappa_c, T_c, nu_c)
end

function grab_col{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}, col::Integer)
  r = A.colptr[col]:A.colptr[col+1]-1
  A.rowval[r], A.nzval[r]
end

function sample_movie(mm, Am, mean_rating, sample_u, alpha, mu_m, Lambda_m)
  ff, v = grab_col(Am, mm)
  rr = v - mean_rating
  MM = sample_u[ff,:]

  # Sample from normal distribution
  L = chol(Lambda_m + alpha * MM'*MM, :L)
  L' \ (randn(num_latent) + (L \ (alpha * MM'*rr + Lambda_m * mu_m)))
end

function sample_user(uu, Au, mean_rating, sample_m, alpha, mu_u, Lambda_u)
  ff, v = grab_col(Au, uu)
  rr = v - mean_rating
  MM = sample_m[ff,:]

  # Sample from normal distribution
  L = chol(Lambda_u + alpha * MM'*MM, :L)
  L' \ (randn(num_latent) + (L \ (alpha * MM'*rr + Lambda_u * mu_u)))
end

tic()
println("Sampling")
for i in 1:nsims
  # Sample from movie hyperparams
  mu_m, Lambda_m = rand( ConditionalNormalWishart(sample_m, vec(mu0_m), b0_m, WI_m, df_m) )

  # Sample from user hyperparams
  mu_u, Lambda_u = rand( ConditionalNormalWishart(sample_u, vec(mu0_u), b0_u, WI_u, df_u) )

  for mm = 1:num_m
    sample_m[mm, :] = sample_movie(mm, Am, mean_rating, sample_u, alpha, mu_m, Lambda_m)
  end

  for uu = 1:num_p
    sample_u[uu, :] = sample_user(uu, Au, mean_rating, sample_m, alpha, mu_u, Lambda_u)
  end

  probe_rat = pred(P, sample_m, sample_u, mean_rating)

  if i > burnin
    probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1)
    counter_prob = counter_prob + 1
  else
    probe_rat_all = probe_rat
    counter_prob = 1
  end

  err_avg = sqrt( sum((P .- probe_rat_all).^2) / nnz(P) );
  err = sqrt( sum((P .- probe_rat).^2) / nnz(P) );

  @printf("Iteration %d:\t avg RMSE %6.4f RMSE %6.4f FU(%6.4f) FM(%6.4f)\n", i, err_avg, err, vecnorm(sample_u), vecnorm(sample_m))
end
toc()
