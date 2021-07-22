import pickle
import glob
import numpy as np
from parameters_toni import getParameters
import os

params = getParameters(print_par=False)
step = float(params["problem"]["ini_stepsize"])
 
dirListing = os.listdir("results/")
file_number = len(dirListing)

for i in range(file_number): 
    f = open("results/%d.pkl"%(i),"rb") 
    res = pickle.load(f)
    print("t = %f\n"%(1.+i*step))
    print("Y=\n",res[0],"\n")
    print("X=\n",res[1],"\n")
    print("lambda=\n",res[2],"\n")
    print("--------------------------------------------")
    f.close()
    
 
 
 