import numpy as np

def _choose_example(example_name: str):

    if example_name == "example_1a":
            
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
    
    if example_name == "example_1b":

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
    
    if example_name == "example_1c":

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
    
    if example_name == "example_rand1":

        n = 3
        m = 1
        rank = 2

        rand_A0 = np.random.randint(5, size=(3, 3)) 
        A0 = (rand_A0 + rand_A0.T)/5 
        Z = np.zeros((3,3))

        A = np.array([A0], dtype=np.float64)
        A_lin = np.array([Z], dtype=np.float64)

        b = np.array([1.])  
        b_lin = np.random.randint(100, size=(1,))/100 
    
    if example_name == "example_rand2":

        n = 3
        m = 2
        rank = 2

        rand_A0 = np.random.randint(5, size=(3, 3))
        rand_A1 = np.random.randint(5, size=(3, 3))
        A0 = (rand_A0 + rand_A0.T)/5
        A1 = (rand_A1 + rand_A1.T)/5
        Z = np.zeros((3,3))

        A = np.array([A0,A1], dtype=np.float64)
        A_lin = np.array([Z,Z], dtype=np.float64)

        b = np.array([1., 1.])  
        b_lin = np.random.randint(100, size=(2,))/100 

    if example_name == "example_rand3":

        n = 3
        m = 3
        rank = 2

        rand_A0 = np.random.randint(5, size=(3, 3))
        rand_A1 = np.random.randint(5, size=(3, 3))
        rand_A2 = np.random.randint(5, size=(3, 3))
        A0 = (rand_A0 + rand_A0.T)/5
        A1 = (rand_A1 + rand_A1.T)/5
        A2 = (rand_A2 + rand_A2.T)/5
        Z = np.zeros((3,3))

        A = np.array([A0,A1,A2], dtype=np.float64)
        A_lin = np.array([Z,Z,Z], dtype=np.float64)

        b = np.array([1., 1., 1.])  
        b_lin = np.random.randint(100, size=(3,))/100 

    if example_name == "example_rand4":

        n = 3
        m = 4
        rank = 2

        rand_A0 = np.random.randint(5, size=(3, 3))
        rand_A1 = np.random.randint(5, size=(3, 3))
        rand_A2 = np.random.randint(5, size=(3, 3))
        rand_A3 = np.random.randint(5, size=(3, 3))
        A0 = (rand_A0 + rand_A0.T)/5
        A1 = (rand_A1 + rand_A1.T)/5
        A2 = (rand_A2 + rand_A2.T)/5
        A3 = (rand_A3 + rand_A3.T)/5
        Z = np.zeros((3,3))

        A = np.array([A0,A1,A2,A3], dtype=np.float64)
        A_lin = np.array([Z,Z,Z,Z], dtype=np.float64)

        b = np.array([1., 1., 1., 1.])  
        b_lin = np.array([1., 1., 1., 1.])   

    return n, m, rank, A, A_lin, b, b_lin 

    
    