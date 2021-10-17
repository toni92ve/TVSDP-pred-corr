import pickle
import numpy as np 
import os

np.set_printoptions(edgeitems=30, linewidth=100000
    # ,formatter=dict(float=lambda x: "%.5g" % x)
    )
    
def visualize_sol(params):

    t = float(params["problem"]["initial_time"])

    res_tol = float(params["problem"]["res_tol"])
    
    dirListing = os.listdir("results/")
    file_number = len(dirListing)
    # each file named res contains:
    # res[0] = Y solution
    # res[1] = X solution
    # res[2] = lambda solution
    # res[3] = res
    # res[4] = dt
    # res[5] = run time 
    # res[6] = reduction steps
    
    print("============================================")
    for i in range(file_number): 
        f = open("results/%d.pkl"%(i),"rb") 
        res = pickle.load(f)
        dt = res[4]
        t += dt
        print("ITERATION", i)
        print("reduction steps: ", res[6])
        print("t = %f\n"%t) 
        print("Y=\n",res[0],"\n")
        print("X=\n",res[1],"\n")
        print("lambda=\n",res[2],"\n") 
        print("residual threshold: ", res_tol)
        print("residual: ", res[3])
        print("============================================")
        f.close()
    for f in dirListing:
        os.remove(os.path.join("results/", f))



    
 
 
 