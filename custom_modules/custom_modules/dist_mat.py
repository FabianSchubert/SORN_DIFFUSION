import numpy as np

def dist_mat_from_pos(r,n_grid_x,n_grid_y,L,bound_cond):

    dist_mat = np.zeros(((2*n_grid_x+1)*(2*n_grid_y+1),r.shape[0],r.shape[0])) 

    grid_ind = 0


    if bound_cond == "neumann":

        for nx in xrange(-n_grid_x,n_grid_x+1):

            for ny in xrange(-n_grid_y,n_grid_y+1):

                X1,X2 = np.meshgrid(r[:,0]*(-1.)**nx+(abs(nx)%2+nx)*L,r[:,0])

                Y1,Y2 = np.meshgrid(r[:,1]*(-1.)**ny+(abs(ny)%2+ny)*L,r[:,1])

                dist_mat[grid_ind,:,:] = np.sqrt((X1-X2)**2+(Y1-Y2)**2)

                #M = np.array([[(-1.)**nx,0.],[0.,(-1.)**ny]])
                #shiftvec = L*np.array([abs(nx)%2+nx,abs(ny)%2+ny])

                #for k in xrange(400):                

                #   for l in xrange(400):#(m+1):



                #       dist_mat[grid_ind,k,l] = np.linalg.norm(r[k,:]-(np.dot(M,r[l,:])+shiftvec))


                grid_ind += 1

    elif bound_cond == "periodic":

        for nx in xrange(-n_grid_x,n_grid_x+1):

            for ny in xrange(-n_grid_y,n_grid_y+1):

                shiftvec = L*np.array([nx,ny])
                
                X1,X2 = np.meshgrid(r[:,0]+nx*L,r[:,0])

                Y1,Y2 = np.meshgrid(r[:,1]+ny*L,r[:,1])

                dist_mat[grid_ind,:,:] = np.sqrt((X1-X2)**2+(Y1-Y2)**2)
                

                #for k in xrange(400):                

                 #   for l in xrange(400):#(m+1):

                #       dist_mat[grid_ind,k,l] = np.linalg.norm(r[k,:]-(r[l,:]+shiftvec))


                grid_ind += 1

    return dist_mat