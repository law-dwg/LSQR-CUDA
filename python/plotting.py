from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
from scipy.sparse.linalg import lsqr
from numpy import linalg as LA
import datetime
import csv
import numpy as np
import os 
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
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
outputspath = "../results/2021109/output"

now = datetime.datetime.now()
now = now.strftime('%Y-%m-%dT%H%M')
inpath = "../results/2021109/input"
inputs = listdir(inpath)
inputs.sort()
mats = [m for m in inputs if ".mat" in m]
vecs = [v for v in inputs if ".vec" in v]

# Sparse plotting
LSQRCUDA = pd.read_csv(outputspath+"/2021-11-09T2101_LSQR-CUDA.csv")
DEVICE = pd.read_csv(outputspath+"/deviceProps.csv")

# python data
scipylsqr = pd.read_csv(outpath + "/2021-11-10T0956_LSQR_python.csv")

# Load and parse LSQR-CUDA data
name = DEVICE['DEVICE_NAME'][0]
implementations = ['Cpp-DENSE','CUDA-DENSE','CUDA-SPARSE','CUBLAS-DENSE','CUSPARSE-SPARSE']
dfs = []
for i in implementations:
    temp = LSQRCUDA[LSQRCUDA['IMPLEMENTATION']==i].drop(columns='IMPLEMENTATION').drop(columns='A_COLUMNS').drop(columns='SPARSITY')
    temp = temp.rename(columns={"TIME(ms)":i})
    temp[i] = temp[i].mul(1/1000)
    dfs.append(temp)


BASELINE = scipylsqr[scipylsqr['IMPLEMENTATION']=='scipy-lsqr'].drop(columns='IMPLEMENTATION').drop(columns='A_COLUMNS').drop(columns='SPARSITY')
BASELINE = BASELINE.rename(columns={"TIME(ms)":"scipy-lsqr"})
BASELINE["scipy-lsqr"] = BASELINE["scipy-lsqr"].mul(1/1000)
CPPDENSE = dfs[0]
CUDADENSE = dfs[1]
CUDASPARSE = dfs[2]
CUBLASDENSE = dfs[3]
CUSPARSE = dfs[4]

all = pd.merge(CUBLASDENSE,CUSPARSE,on="A_ROWS")
all = pd.merge(CUDASPARSE,all,on="A_ROWS")
all = pd.merge(CUDADENSE,all,on="A_ROWS")
all = pd.merge(CPPDENSE,all,how='right',on="A_ROWS")
all = pd.merge(BASELINE,all,how='right',on="A_ROWS")
all.to_csv("../results/"+now+"_TIMES.csv",index=False,float_format='%.5f')

# Sparse implementation plots
sparse = pd.merge(CUSPARSE,CUDASPARSE,on="A_ROWS")
sparse = pd.merge(BASELINE,sparse,on="A_ROWS")
fig = sparse.plot(x='A_ROWS', ylabel="TIME(s)",title=name+" - sparse input implementations",grid=True).get_figure()
#fig.savefig("../images/"+now+"_1000-8000_SPARSESOLUTION.png")

#csvtimes = "../results/"+now+"_SPARSETIMES.csv"
#sparse.to_csv(csvtimes,index=False,float_format='%.8f')

# Sparse speedups

CUDASPARSE_SPEEDUP = (sparse["scipy-lsqr"]/sparse["CUDA-SPARSE"])
CUDASPARSE_SPEEDUP = CUDASPARSE_SPEEDUP.rename("CUDA-SPARSE")

CUSPARSE_SPEEDUP = (sparse["scipy-lsqr"]/sparse["CUSPARSE-SPARSE"])
CUSPARSE_SPEEDUP = CUSPARSE_SPEEDUP.rename("CUSPARSE-SPARSE")

ROWS_SPEEDUP = sparse["A_ROWS"]
ROWS_SPEEDUP = ROWS_SPEEDUP.rename("A_ROWS")

csvsparsespeedups = "../results/"+now+"_SPARSE-SPEEDUPS.csv"
SPEEDUPS = pd.concat([ROWS_SPEEDUP,CUDASPARSE_SPEEDUP,CUSPARSE_SPEEDUP], axis=1)
#SPEEDUPS.to_csv(csvsparsespeedups,index=False,float_format='%.8f')

## Dense implementation plots
dense = pd.merge(CUBLASDENSE,CUDADENSE,on="A_ROWS")
dense = pd.merge(CPPDENSE,dense,how='right',on="A_ROWS")
fig = dense.plot(x='A_ROWS', ylabel="TIME(s)",title=name+" - dense input implementations",grid=True).get_figure()
fig.savefig("../images/"+now+"_1000-8000_DENSESOLUTION.png")

#####################################################
### Calculation of root mean squared error (rmse) ###
#####################################################

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

csvrsmepath = "../results/" + now + "_RMSE.csv"
f = open(csvrsmepath, 'w')
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
        rowrsmecsv = [implementation,rowsStr,f'{rmse:.8f}']
        with open(csvrsmepath, 'a') as fd:
            writer=csv.writer(fd)
            writer.writerow(rowrsmecsv)

errors = pd.read_csv(csvrsmepath)

I0 = errors[errors['IMPLEMENTATION']=='Cpp-DENSE'].drop(columns='IMPLEMENTATION')
I0=I0.rename(columns={"RMSE":"Cpp-DENSE"})

I1 = errors[errors['IMPLEMENTATION']=='CUDA-DENSE'].drop(columns='IMPLEMENTATION')
I1=I1.rename(columns={"RMSE":"CUDA-DENSE"})

I2 = errors[errors['IMPLEMENTATION']=='CUDA-SPARSE-80'].drop(columns='IMPLEMENTATION')
I2=I2.rename(columns={"RMSE":"CUDA-SPARSE"})

I3 = errors[errors['IMPLEMENTATION']=='CUBLAS-DENSE'].drop(columns='IMPLEMENTATION')
I3=I3.rename(columns={"RMSE":"CUBLAS-DENSE"})

I4 = errors[errors['IMPLEMENTATION']=='CUSPARSE-SPARSE-80'].drop(columns='IMPLEMENTATION')
I4=I4.rename(columns={"RMSE":"CUSPARSE-SPARSE"})

I = pd.merge(I3,I4,on="A_ROWS")
I = pd.merge(I2,I,on="A_ROWS")
I = pd.merge(I1,I,on="A_ROWS")
I = pd.merge(I0,I,how='right',on="A_ROWS")

I.to_csv(csvrsmepath,index=False,float_format='%.5f')

