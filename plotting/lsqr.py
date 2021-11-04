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

outpath = "./output/"
outputspath = "../source/output"
content = []
for (dirpath, dirnames, filenames) in walk(outputspath):
    content.extend(dirnames)
    break
content.sort()
print(content)

now = datetime.datetime.now()
now = now.strftime('%Y-%m-%dT%H%M')
inpath = "../source/input"


csvpath = outpath + "/" + now + "_LSQR_python.csv"
f = open(csvpath, 'w')
writer = csv.writer(f)
writer.writerow(['IMPLEMENTATION','A_ROWS','A_COLUMNS','SPARSITY','TIME(ms)'])
f.close()
inputs = listdir(inpath)
inputs.sort()
mats = [m for m in inputs if ".mat" in m]
vecs = [v for v in inputs if ".vec" in v]

LSQRCUDA = pd.read_csv("../source/sol/2021-11-3T2141/2021-11-3T2147_LSQR-CUDA.csv")
DEVICE = pd.read_csv("../source/sol/2021-11-3T2141/deviceProps.csv")
name = DEVICE['DEVICE_NAME'][0]
CUDASPARSE = LSQRCUDA[LSQRCUDA['IMPLEMENTATION']=='CUDA-SPARSE'].drop('IMPLEMENTATION',1).drop('A_COLUMNS',1).drop('SPARSITY',1)
CUDASPARSE=CUDASPARSE.rename(columns={"TIME(ms)":"CUDASPARSE"})
CUSPARSE = LSQRCUDA[LSQRCUDA['IMPLEMENTATION']=='CUSPARSE-SPARSE'].drop('IMPLEMENTATION',1).drop('A_COLUMNS',1).drop('SPARSITY',1)
CUSPARSE=CUSPARSE.rename(columns={"TIME(ms)":"CUSPARSE"})
#print(CUDASPARSE)
#print(CUSPARSE)
#all = pd.merge(CUDASPARSE,CUSPARSE,on="A_ROWS")
#print(all)
#all[:5].plot(x='A_ROWS',title=name,grid=True)
#all[5:].plot(x='A_ROWS',title=name,grid=True)
#plt.show()


for i in mats:
    A = np.loadtxt(inpath+"/"+i, dtype=np.double)
    A_props = (i.split(".")[0]).split("_")
    A_rows = float(A_props[0])
    A_cols = float(A_props[1])
    A_sp = (float(A_props[2]) / 100)
    print(A_rows, A_cols, A_sp)
    if ((A_props[0]+"_1_b.vec") in vecs):
        b_path = (A_props[0]+"_1_b.vec")
        print(b_path)
    else:
        raise Exception('b-file',A_props[0]+"_1_b.vec",'does not exist')

    b = np.loadtxt(inpath+"/"+b_path, dtype=np.double)
    x_py_path=outpath+str(A_cols)+"_1_x_python-lsqr.vec"
    start = datetime.datetime.now()
    x, istop, itn, normr = lsqr(A, b, show=True)[:4]
    end = datetime.datetime.now()
    elapsed = end - start
    elapsedms = elapsed.total_seconds()*1000
    print("elapsed time =",elapsedms,"ms")
    np.savetxt(x_py_path,x,delimiter=' ')
    row = ['scipy-lsqr', A_props[0], A_props[1] , str(A_sp) , str(elapsedms)]
    
    # write to csv
    with open(csvpath, 'a') as fd:
        writer=csv.writer(fd)
        writer.writerow(row)
    