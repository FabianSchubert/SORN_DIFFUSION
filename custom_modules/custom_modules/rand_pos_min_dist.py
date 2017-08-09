import numpy as np

def rand_pos_min_dist(n,L,min_dist):

    r = np.ndarray((n,2))
    
    ##avoid placing two exc. neurons at the same position on the grid
    for k in xrange(n):
        
        while True:
            
            rand_pos = np.random.rand(2)*L
            
            passed = True
            
            for l in xrange(k):
                
                if np.linalg.norm(r[l,:] - rand_pos) < min_dist:
                    passed = False
            if passed:
                r[k,:] = rand_pos
                break
    return r

def regular_square_grid(n,L):
    h = L/n
    r = np.ndarray((n**2,2))

    for k in xrange(n):
        for l in xrange(n):
            r[k*n+l,:] = h*(np.array([k,l]) + np.array([0.5,0.5]))

    return r