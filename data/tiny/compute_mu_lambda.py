#!/usr/bin/python

import matrix_io as mio
import numpy as np
import math
from glob import glob

# collect U for all samples
Us = [ mio.read_matrix(m) for m in glob("output/U-[0123456789].ddm") ]
print("samples:\n", Us)

# stack them and compute mean
Ustacked = np.stack(Us)
mu1 = np.mean(Ustacked, axis = 0)
print("python mu:\n", mu1)
mu2 = mio.read_matrix("output/U-mu.ddm")
print("bpmf mu:\n", mu2)
print("norm mu1 - mu2: %.4f" % np.linalg.norm(mu1 - mu2))

# Compute covariance and precision, first unstack in different way
Uunstacked = np.squeeze(np.split(Ustacked, Ustacked.shape[2], axis = 2))
Ucov = [ np.cov(u, rowvar = False) for u in Uunstacked ]
Uprec = [ np.linalg.inv(np.cov(u, rowvar = False)) for u in Uunstacked ]
# restack
Ucovstacked = np.stack(Ucov, axis = 2)
Lambda1 = np.stack(Uprec, axis = 2)
# reshape correctly

print("python: precision user 0\n", Lambda1[:,:,0])

Lambda2_flat = mio.read_matrix("output/U-Lambda.ddm")
num_latent = int(math.sqrt(Lambda2_flat.shape[0]))
Lambda2 = Lambda2_flat.reshape(num_latent, num_latent, Lambda2_flat.shape[1])
print("bpmf: precision user 0\n", Lambda2[:,:,0])

print("norm Lambda1 - Lambda: %.4f" % np.linalg.norm(Lambda2 - Lambda2))
