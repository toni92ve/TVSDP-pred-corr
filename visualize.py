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
    # res[2] = actual X solution
    # res[3] = lambda solution
    # res[4] = res
    # res[5] = dt
    # res[6] = run time QP
    # res[7] = run time SDP
    # res[8] = reduction steps
    # res[9] = frobenius error
    
    print("============================================")
    for i in range(file_number): 
        f = open("results/%d.pkl"%(i),"rb") 
        res = pickle.load(f)
        dt = res[5]
        t += dt
        print("ITERATION", i)
        print("reduction steps: ", res[8])
        print("t = %f\n"%t) 
        # print("Y=\n",res[0],"\n")
        print("X=\n",res[1],"\n")
        print("actual X=\n",res[2],"\n")
        # print("lambda=\n",res[3],"\n") 
        print("residual threshold: ", res_tol)
        print("residual: ", res[4])
        print("frobenius error", res[9])
        print("run time QP: ", res[6])
        print("run time SDP: ", res[7])
        print("============================================")
        f.close()