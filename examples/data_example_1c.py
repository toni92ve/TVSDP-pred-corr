import numpy as np

n = 3
m = 4
rank = 2

A0 = [[1,0,0],
      [0,0,0],
      [0,0,0]]
A1 = [[0,0.5,0],
      [0.5,0,0],
      [0,0,0]]
A2 = [[0,0,0],
      [0,0,0.5],
      [0,0.5,0]]
A3 = [[0,0,0.5],
      [0,1,0],
      [0.5,0,0]]

A = np.array([A0,A1,A2,A3], dtype=np.float64)
Z = np.zeros((3,3))
A_lin = np.array([Z,Z,Z,Z], dtype=np.float64)

b = np.array([1., 0., 0., 1.])  
b_lin = np.array([ 0., 0., 0., 1.])  
