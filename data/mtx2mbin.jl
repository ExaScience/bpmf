using MatrixMarket;
using BayesianDataFusion;

in = ARGS[1];
out = replace(ARGS[1], ".mtx", ".mbin");
U = MatrixMarket.mmread(in);
write_sparse_float64(out, U);
