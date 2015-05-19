A = csvread("chembl-IC50-360targets.csv");
B = [ A(2:end,1), A(2:end,2), A(2:end,3) ];
B(1:end, 3) = log10(B(1:end, 3));
P = randperm(size(B,1)); 
s = round(size(B,1) / 5);

C_probe = B(P(1:s), :);
D_probe = spconvert(C_probe);
mmwrite("chembl-IC50-360targets_probe.mtx", D_probe);

C_sample = B(P(s+1:end), :);
D_sample = spconvert(C_sample);
mmwrite("chembl-IC50-360targets_sample.mtx", D_sample);
