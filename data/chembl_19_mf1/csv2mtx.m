A = csvread("chembl-IC50-360targets.csv");
B = [ A(2:end,1), A(2:end,2), A(2:end,3) ];
B(1:end, 3) = log10(B(1:end, 3));
P = randperm(size(B,1)); 
s = round(size(B,1) / 5);

D_all = spconvert(B); 
E_all = D_all';
F_all = E_all(:, colamd(E_all))';
mmwrite("chembl-IC50-360targets.mtx", F_all);

C_probe = B(P(1:s), :);
D_probe = spconvert(C_probe);
E_probe = D_probe';
F_probe = E_probe(:, colamd(E_probe))';
mmwrite("chembl-IC50-360targets_probe.mtx", F_probe);

C_sample = B(P(s+1:end), :);
D_sample = spconvert(C_sample);
E_sample = D_sample';
F_sample = E_sample(:, colamd(E_sample))';
mmwrite("chembl-IC50-360targets_sample.mtx", F_sample);
