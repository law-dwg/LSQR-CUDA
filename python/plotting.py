from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
from scipy.sparse.linalg import lsqr
from numpy import linalg as LA
import datetime
import csv
import numpy as np
import os 
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
print(np.finfo(np.float64).eps)
import time
import pandas as pd
from os import listdir
from os.path import isfile, join
from os import walk

########################
### Plotting results ###
########################

# LSQR-CUDA data
outpath = "./output/"
outputspath = "../examples/2021109/output"

now = datetime.datetime.now()
now = now.strftime('%Y-%m-%dT%H%M')
inpath = "../examples/2021109/input"
inputs = listdir(inpath)
inputs.sort()
mats = [m for m in inputs if ".mat" in m]
vecs = [v for v in inputs if ".vec" in v]

# Sparse plotting
LSQRCUDA = pd.read_csv(outputspath+"/2021-11-09T2101_LSQR-CUDA.csv")
DEVICE = pd.read_csv(outputspath+"/deviceProps.csv")

# python data
scipylsqr = pd.read_csv(outpath + "/2021-11-10T0956_LSQR_python.csv")

name = DEVICE['DEVICE_NAME'][0]

CUDASPARSE = LSQRCUDA[LSQRCUDA['IMPLEMENTATION']=='CUDA-SPARSE'].drop(columns='IMPLEMENTATION').drop(columns='A_COLUMNS').drop(columns='SPARSITY')
CUDASPARSE=CUDASPARSE.rename(columns={"TIME(ms)":"CUDA-SPARSE"})
CUDASPARSE["CUDA-SPARSE"] = CUDASPARSE["CUDA-SPARSE"].mul(1/1000)

CUSPARSE = LSQRCUDA[LSQRCUDA['IMPLEMENTATION']=='CUSPARSE-SPARSE'].drop(columns='IMPLEMENTATION').drop(columns='A_COLUMNS').drop(columns='SPARSITY')
CUSPARSE=CUSPARSE.rename(columns={"TIME(ms)":"CUSPARSE-SPARSE"})
CUSPARSE["CUSPARSE-SPARSE"] = CUSPARSE["CUSPARSE-SPARSE"].mul(1/1000)

BASELINE = scipylsqr[scipylsqr['IMPLEMENTATION']=='scipy-lsqr'].drop(columns='IMPLEMENTATION').drop(columns='A_COLUMNS').drop(columns='SPARSITY')
BASELINE = BASELINE.rename(columns={"TIME(ms)":"scipy-lsqr"})
BASELINE["scipy-lsqr"] = BASELINE["scipy-lsqr"].mul(1/1000)
CUDASPARSE = CUDASPARSE[:15]
CUSPARSE = CUSPARSE[:15]

all = pd.merge(CUSPARSE,CUDASPARSE,on="A_ROWS")
all = pd.merge(BASELINE,all,on="A_ROWS")
fig = all.plot(x='A_ROWS', ylabel="TIME(s)",title=name,grid=True).get_figure()
fig.savefig("../images/"+now+"_1000-8000_SPARSESOLUTION.png")

###############################################
### Calculation of root mean squared error (rmse) ###
###############################################

# Baseline data
PYTHONOUTS = []
for (dirpath, dirnames, filenames) in walk(outpath):
    PYTHONOUTS.extend(filenames)
    break
PYTHONOUTS.sort()
PYTHONOUTS = [v for v in PYTHONOUTS if ".vec" in v]


LSQROUTS = []
for (dirpath, dirnames, filenames) in walk(outputspath):
    LSQROUTS.extend(filenames)
    break
LSQROUTS.sort()
LSQROUTS = [v for v in LSQROUTS if ".vec" in v]

csvpath = "../images/" + now + "_RMSE.csv"
f = open(csvpath, 'w')
writer = csv.writer(f)
writer.writerow(['IMPLEMENTATION','A_ROWS','RMSE'])
f.close()
maxRmse=0
for pyout in PYTHONOUTS:
    rows = int(float(pyout.split("_")[0]))
    rowsStr = str(rows)
    outs = [v for v in LSQROUTS if rowsStr in v]
    p = np.loadtxt(outpath+"/"+pyout,dtype=np.double)
    for o in outs:
        l = np.loadtxt(outputspath+"/"+o,dtype=np.double)
        rmse = np.sqrt(np.mean((l-p)**2))
        if (rmse>maxRmse):
            maxRmse=rmse
        # write to csv
        implementation = o.split(".")[0].split("_")[3]
        rowcsv = [implementation,rowsStr,rmse]
        with open(csvpath, 'a') as fd:
            writer=csv.writer(fd)
            writer.writerow(rowcsv)

print(maxRmse)