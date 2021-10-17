import numpy as np

class _Examples:

    def __init__(self, example: str):
        self._example_name = example 

    def _choose_example(self):

        if self._example_name == "example_1a":
             
            n = 3
            m = 3
            rank = 2

            A0 = [[0,.5,0],
                [.5,0,0],
                [0,0,0]]
            A1 = [[0,0,.5],
                [0,0,0],
                [.5,0,0]]
            A2 = [[0,0,0],
                [0,0,.5],
                [0,.5,0]]
            Z = np.zeros((3,3))

            A = np.array([A0,A1,A2], dtype=np.float64)
            A_lin = np.array([Z,Z,Z], dtype=np.float64)

            b = np.array([1., 1., 1.])  
            b_lin = np.array([1., 1., 1.]) 
        
        if self._example_name == "example_1b":

            n = 3
            m = 3
            rank = 2

            A0 = [[1,1,0],
                [1,0,0],
                [0,0,0]]
            A1 = [[0,0,1],
                [0,0,0],
                [1,0,0]]
            A2 = [[0,0,0],
                [0,0,1],
                [0,1,1]]
            Z = np.zeros((3,3))

            A = np.array([A0,A1,A2], dtype=np.float64)
            A_lin = np.array([Z,Z,Z], dtype=np.float64)

            b = np.array([1., 1., 1.])  
            b_lin = np.array([1., 1., 1.])  
        
        if self._example_name == "example_1c":

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
            Z = np.zeros((3,3))

            A = np.array([A0,A1,A2,A3], dtype=np.float64)
            A_lin = np.array([Z,Z,Z,Z], dtype=np.float64)

            b = np.array([1., 0., 0., 1.])  
            b_lin = np.array([ 0., 0., 0., 1.])  

        return n, m, rank, A, A_lin, b, b_lin 

     
        