using MAT, Distributions
using SQLite, DataFrames

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

@default_value(num_feat, 3)
@default_value(alpha, 2) # observation noise (precision)
@default_value(skip_load, false)
@default_value(skip_initial, true)
@default_value(nsims, 10)
@default_value(burnin, 5)
@default_value(problem, Chemo)

srand(1234)


# compute U * V for probe vector

function pred_clamp(probe_vec, sample_m, sample_u, mean_rating)
  out = sum(sample_m[probe_vec[:,2],:].*sample_u[probe_vec[:,1],:],2) + mean_rating
  out[ out .< 1 ] = 1
  out[ out .> 5 ] = 5
  out
end

function pred(probe_vec, sample_m, sample_u, mean_rating)
  sum(sample_m[probe_vec[:,2],:].*sample_u[probe_vec[:,1],:],2) + mean_rating
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
@show mean_rating
@show ff, rr, MM

  covar = inv(Lambda_m + alpha * MM'*MM)
  mu = covar * (alpha * MM'*rr + Lambda_m * mu_m)

@show covar, mu

  # Sample from normal distribution
  s = chol(covar)' * fill(0.25, num_feat) + mu
@show chol(covar)', s

  s
end

function sample_user(uu, Au, mean_rating, sample_m, alpha, mu_u, Lambda_u)
  ff, v = grab_col(Au, uu)
  rr = v - mean_rating
  MM = sample_m[ff,:]

  covar = inv(Lambda_u + alpha * MM'*MM)
  mu = covar * (alpha * MM'*rr + Lambda_u * mu_u)

  # Sample from normal distribution
  chol(covar)' * randn(num_feat) + mu
end


function main() 
      println("Loading data")

        X = readtable("chembl_19_mf1/chembl-IC50-360targets.csv", header=true)
        rename!(X, [:row, :col], [:compound, :target])

        # take log10 of each value
        X[:, :value] = log10(X[:, :value])
        # randomly sample 20% of all entries
        idx = sample(1:size(X,1), int(floor(20/100 * size(X,1))); replace=false)
        # put these in the probe_vec
        probe_vec = array(X[idx,:])
        # remove them from X
        # X = X[setdiff(1:size(X,1), idx), :]

        # create sparse matrix
        Am = sparse( X[:compound], X[:target], X[:value])
        num_p, num_m = size(Am)

        # boolean vector with ratings smaller than 200
        ratings_test = probe_vec[:,3] .< log10(200)

      Au = Am'
      mean_rating = sum(Am)/nnz(Am)

    sample_u = fill(2.0, num_p, num_feat)
    sample_m = zeros(num_m, num_feat)

    # Initialize hierarchical priors
    mu_u = zeros(num_feat,1)
    mu_m = zeros(num_feat,1)
    Lambda_u = eye(num_feat)
    Lambda_m = eye(num_feat) * 0.5

    # parameters of Inv-Whishart distribution (see paper for details)
    WI_u = eye(num_feat)
    b0_u = 2
    df_u = num_feat
    mu0_u = zeros(num_feat,1)

    WI_m = eye(num_feat)
    b0_m = 2
    df_m = num_feat
    mu0_m = zeros(num_feat,1)

      for mm = 1:1
        sample_m[mm, :] = sample_movie(mm, Am, mean_rating, sample_u, alpha, mu_m, Lambda_m)
      end
end

main()
