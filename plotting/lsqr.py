from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
from scipy.sparse.linalg import lsqr
from numpy import linalg as LA
import numpy as np
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
print(np.finfo(np.float64).eps)
import time
import pandas as pd
from os import listdir
from os.path import isfile, join
mypath = "../source/output"
outputDirs = listdir(mypath)
outputDirs.sort()
onlyfiles = [f for f in listdir(mypath+"/"+outputDirs[-1]) if isfile(join(mypath+"/"+outputDirs[-1], f))]
print(outputDirs[-1])
print(onlyfiles)
#data= pd.read_csv("Salary_Data.csv")
#data

#for i in range(1000,1100,500):
#i = 100
#i_2= i
#i_s = str(i)
#i_s_2 = str(i_2)
#A_path="../source/input/"+i_s_2+"_"+i_s+"_0_A.txt"
#b_path="../source/input/"+i_s_2+"_1_0_b.txt"
##x_c_path="../source/output/"+i_s+"_1_x_CPU.txt"
#x_py_path=dir_path + "/output/"+i_s+"_1_x_python.txt"
#A = np.loadtxt(A_path, dtype=np.double)
#b = np.loadtxt(b_path, dtype=np.double)
##x_c= np.loadtxt(x_c_path, dtype=np.double)
#A=A.reshape((i_2,i))
#b=b.reshape((i_2,1))
##x_c=x_c.reshape((i,1))
#print("A=\n",A)
#print("b=\n",b)
##sM = csr_matrix(A)
##out = np.array(sM)
##print(out.shape)
##
##print(out)
#start = time.time()
#x, istop, itn, normr = lsqr(A, b, show=True)[:4]
#end = time.time()
#print("elapsed time =",end-start)
#np.savetxt(x_py_path,x,delimiter=' ')
#same = True
##for j in range(i):
##    if x[j]==x_c[j]:
##        continue
##    else:
##        print("results do not match index ",j, "has difference of", x[j]-x_c[j])
##        same=False
##        break
##
##if(not same):
##    
##    with open(x_py_path, "a") as myfile:
##        myfile.write("\nresults do not match")
#