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

macro enum(T,syms...)
    blk = quote
        immutable $(esc(T))
            n::Int32
            $(esc(T))(n::Integer) = new(n)
        end
        Base.show(io::IO, x::$(esc(T))) = print(io, $syms[x.n+1])
        Base.show(io::IO, x::Type{$(esc(T))}) = print(io, $(string("enum ", T, ' ', '(', join(syms, ", "), ')')))
    end
    for (i,sym) in enumerate(syms)
        push!(blk.args, :(const $(esc(sym)) = $(esc(T))($(i-1))))
    end
    push!(blk.args, :nothing)
    blk.head = :toplevel
    return blk
end

@enum Problem TinyFlix Netflix Chemo

@default_value(num_feat, 30)
@default_value(alpha, 2) # observation noise (precision)
@default_value(skip_load, false)
@default_value(skip_initial, true)
@default_value(nsims, 300)
@default_value(burnin, 200)
@default_value(problem, Netflix)

using MAT, Distributions

isdefined(:DataSet) || @everywhere begin

include("replicated_data.jl")
immutable DataSet{T}
  Am::SparseMatrixCSC{T,Int64}
  Au::SparseMatrixCSC{T,Int64}
  mean_rating::Float64
end

function DataSet{T}(A::SparseMatrixCSC{T,Int64})
  DataSet(A, A', sum(A)/nnz(A))
end

end

if ! skip_load
  println("Loading data")

  if problem == Netflix
    using SQLite, DataFrames
    db = SQLite.connect("netflix.sqlite")

    println("Getting users")
    I = array( query("SELECT ID FROM Users", db) )
    J = full(sparsevec(vec(I), [1:maximum(I)]))

    println("Getting ratings")
    res = query("SELECT User, Movie, Rating FROM Ratings", db)
    Am = sparse( J[ res[:User] ], res[:Movie], res[:Rating] )
    res = nothing
    gc()

    println("Getting probe")
    probe_vec = array( query("SELECT User, Movie, Rating FROM Probe", db) )
    probe_vec[:,1] = J[ probe_vec[:,1] ]
    ratings_test = probe_vec[:,3]

    close(db)
    num_p, num_m = size(Am)
  elseif problem == TinyFlix
    m = matopen("BPMF/moviedata.mat")
      train_vec = read(m, "train_vec")
      probe_vec = read(m, "probe_vec")
      num_m = 3952
      num_p = 6040
    close(m)

    Am = sparse( int(train_vec[:,1]), int(train_vec[:, 2]), train_vec[:,3], num_p, num_m)
    ratings_test = probe_vec[:,3]
  elseif problem == Chemo
    using DataFrames
    X = readtable("chembl_19_mf1/chembl-IC50-360targets.csv", header=true)
    rename!(X, [:row, :col], [:compound, :target])

    X[:, :value] = log10(X[:, :value])
    idx = sample(1:size(X,1), int(floor(20/100 * size(X,1))); replace=false)
    probe_vec = array(X[idx,:])
    X = X[setdiff(1:size(X,1), idx), :]

    Am = sparse( X[:compound], X[:target], X[:value])
    num_p, num_m = size(Am)
    ratings_test = probe_vec[:,3] .< log10(200)
  end

  ds = DataSet(Am)
  mean_rating = ds.mean_rating

  if nprocs() > 1
    ds = ReplicatedData(ds)
  end
end


import Base: shmem_fill
#sample_u = zeros(num_p, num_feat)
sample_u = shmem_fill(0.0, (num_p, num_feat))
#sample_m = zeros(num_m, num_feat)
sample_m = shmem_fill(0.0, (num_m, num_feat))

# Initialize hierarchical priors
mu_u = zeros(num_feat,1)
mu_m = zeros(num_feat,1)
Lambda_u = eye(num_feat)
Lambda_m = eye(num_feat)

# parameters of Inv-Whishart distribution (see paper for details)
WI_u = eye(num_feat)
b0_u = 2
df_u = num_feat
mu0_u = zeros(num_feat,1)

WI_m = eye(num_feat)
b0_m = 2
df_m = num_feat
mu0_m = zeros(num_feat,1)

if ! skip_initial
  if problem == TinyFlix
    println("Loading initial solution")
    m = matopen("BPMF/pmf_weight.mat")
      sample_u = read(m, "w1_P1")
      sample_m = read(m, "w1_M1")
    close(m)

    mu_u = mean(sample_u,1)'
    mu_m = mean(sample_m,1)'
    Lambda_u = inv(cov(sample_u))
    Lambda_m = inv(cov(sample_m))
  end
end

function pred_clamp(probe_vec, sample_m, sample_u, mean_rating)
  out = sum(sample_m[probe_vec[:,2],:].*sample_u[probe_vec[:,1],:],2) + mean_rating
  out[ out .< 1 ] = 1
  out[ out .> 5 ] = 5
  out
end

function pred(probe_vec, sample_m, sample_u, mean_rating)
  sum(sample_m[probe_vec[:,2],:].*sample_u[probe_vec[:,1],:],2) + mean_rating
end

function ConditionalNormalWishart(U, mu::Vector{Float64}, kappa::Real, T::Matrix{Float64}, nu::Real)
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

isdefined(:sample_movie) || @everywhere begin

function grab_col{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}, col::Integer)
  r = A.colptr[col]:A.colptr[col+1]-1
  A.rowval[r], A.nzval[r]
end

sample_movie{T}(mm, rd::ReplicatedData{DataSet{T}}, sample_m, alpha, mu_u, Lambda_u) =
  sample_movie(mm, rd.val, sample_m, alpha, mu_u, Lambda_u)

sample_movie{T}(mm, ds::DataSet{T}, sample_u, alpha, mu_m, Lambda_m) =
  sample_movie(mm, ds.Am, ds.mean_rating, sample_u, alpha, mu_m, Lambda_m)

function sample_movie(mm, Am, mean_rating, sample_u, alpha, mu_m, Lambda_m)
  ff, v = grab_col(Am, mm)
  rr = v - mean_rating
  MM = sample_u[ff,:]

  covar = inv(Lambda_m + alpha * MM'*MM)
  mu = covar * (alpha * MM'*rr + Lambda_m * mu_m)

  # Sample from normal distribution
  num_feat = length(mu_m)
  chol(covar)' * randn(num_feat) + mu
end

sample_user{T}(uu, rd::ReplicatedData{DataSet{T}}, sample_m, alpha, mu_u, Lambda_u) =
  sample_user(uu, rd.val, sample_m, alpha, mu_u, Lambda_u)

sample_user{T}(uu, ds::DataSet{T}, sample_m, alpha, mu_u, Lambda_u) =
  sample_user(uu, ds.Au, ds.mean_rating, sample_m, alpha, mu_u, Lambda_u)

function sample_user(uu, Au, mean_rating, sample_m, alpha, mu_u, Lambda_u)
  ff, v = grab_col(Au, uu)
  rr = v - mean_rating
  MM = sample_m[ff,:]

  covar = inv(Lambda_u + alpha * MM'*MM)
  mu = covar * (alpha * MM'*rr + Lambda_u * mu_u)

  # Sample from normal distribution
  num_feat = length(mu_u)
  chol(covar)' * randn(num_feat) + mu
end

end

#=
println("Sampling")
for i in 1:nsims
  # Sample from movie hyperparams
  mu_m, Lambda_m = rand( ConditionalNormalWishart(sample_m, vec(mu0_m), b0_m, WI_m, df_m) )

  # Sample from user hyperparams
  mu_u, Lambda_u = rand( ConditionalNormalWishart(sample_u, vec(mu0_u), b0_u, WI_u, df_u) )

  tic()
  @sync begin
    @parallel for mm = 1:num_m
      sample_m[mm, :] = sample_movie(mm, ds, sample_u, alpha, mu_m, Lambda_m)
    end
  end

  @sync begin
    @parallel for uu = 1:num_p
      sample_u[uu, :] = sample_user(uu, ds, sample_m, alpha, mu_u, Lambda_u)
    end
  end
  toc()

  if problem == Chemo
    probe_rat = pred(probe_vec, sample_m, sample_u, mean_rating)
  else
    probe_rat = pred_clamp(probe_vec, sample_m, sample_u, mean_rating)
  end

  if i > burnin
    probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1)
    counter_prob = counter_prob + 1
  else
    probe_rat_all = probe_rat
    counter_prob = 1
  end

  if problem == Chemo
    err_avg = mean(ratings_test .== (probe_rat_all .< log10(200)))
    err = mean(ratings_test .== (probe_rat .< log10(200)))
  else
    err_avg = sqrt( sum((ratings_test - probe_rat_all).^2) / size(probe_vec,1) );
    err = sqrt( sum((ratings_test - probe_rat).^2) / size(probe_vec,1) );
  end

  @printf("Iteration %d:\t avg RMSE %6.4f RMSE %6.4f FU(%6.4f) FM(%6.4f)\n", i, err_avg, err, vecnorm(sample_u), vecnorm(sample_m))
end
=#
