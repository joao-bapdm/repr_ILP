""" A TOPOLOGY OPTIMIZATION OF BINARY STRUCTURES PYTHON CODE, 2020 """
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from tobs import TOBS

def tobs101_cplex(nelx, nely, gbar, epsilons, beta, rmin):
    # PREPARE PLOTTING
    plt.ion()
    fig, ax = plt.subplots()
    fig.tight_layout(); ax.axis('equal'); ax.axis('off')
    # MATERIAL PROPERTIES
    E0, Emin, nu, penal = 1, 1e-9, 0.3, 3
    ########## PREPARE FINITE ELEMENT ANALYSIS ##########
    A11 = np.array([[12,  3, -6, -3],
                    [ 3, 12,  3,  0],
                    [-6,  3, 12, -3],
                    [-3,  0, -3, 12]])
    A12 = np.array([[-6, -3,  0,  3],
                    [-3, -6, -3, -6],
                    [ 0, -3, -6,  3],
                    [ 3, -6,  3, -6]])
    B11 = np.array([[-4,  3, -2,  9],
                    [ 3, -4, -9,  4],
                    [-2, -9, -4, -3],
                    [ 9,  4, -3, -4]])
    B12 = np.array([[ 2, -3,  4, -9],
                    [-3,  2,  9, -2],
                    [ 4,  9,  2,  3],
                    [-9, -2,  3,  2]])
    KE = 1/(1-nu**2)/24*(np.block([[A11, A12], [A12.T, A11]])
                        +nu*np.block([[B11, B12], [B12.T, B11]]))
    nodenrs = np.arange(1,(1+nelx)*(1+nely)+1).reshape(1+nely, 1+nelx,
                                                       order='F')
    edofVec = (2*nodenrs[:-1,:-1] + 1).reshape((nelx*nely, 1), order='F')
    edofMat =  ( np.tile(edofVec, (1, 8))
               + np.tile(np.block([np.array([ 0,  1]),
                                  2*nely + np.array([2, 3, 0, 1]),
                                  np.array([-2, -1])]), (nelx*nely, 1)))
    iK = np.kron(edofMat, np.ones((8,1))).T.reshape(64*nelx*nely, 1, order='F')
    jK = np.kron(edofMat, np.ones((1,8))).T.reshape(64*nelx*nely, 1, order='F')
    iK, jK = iK.flatten(), jK.flatten()
    # DEFINE LOADS AND SUPPORTS (HALD MBB-BEAM)
    F = coo_matrix(([-1], ([2*(nely+1)-1], [0])),
                   shape=(2*(nelx+1)*(nely+1), 1)).tolil()
    U = np.zeros((2*(nely+1)*(nelx+1),))
    fixeddofs = np.union1d(np.arange(1, 2*(nely+1)+1, 2),
                           [2*(nelx+1)*(nely+1)])
    alldofs   = np.arange(1, 2*(nelx+1)*(nely+1)+1)
    freedofs  = np.setdiff1d(alldofs, fixeddofs)
    ########## PREPARE FILTER ##########
    iH = np.ones((np.int(nelx*nely*(2*(np.ceil(rmin)-1)+1)**2),))
    jH, sH, k = np.ones(iH.shape), np.zeros(iH.shape), 0
    for i1 in range(1, nelx+1):
        for j1 in range(1, nely+1):
            e1 = (i1-1)*nely+j1
            for i2 in range(np.maximum(i1-(int(np.ceil(rmin))-1),1),
                            np.minimum(i1+(int(np.ceil(rmin))-1),nelx)+1):
                for j2 in range(np.maximum(j1-(int(np.ceil(rmin))-1),1),
                                np.minimum(j1+(int(np.ceil(rmin))-1),nely)+1):
                    e2 = (i2-1)*nely+j2
                    iH[k], jH[k] = e1, e2
                    sH[k] = np.maximum(0, rmin-np.sqrt((i1-i2)**2+(j1-j2)**2))
                    k = k+1
    H = coo_matrix((sH, (iH.astype(int), jH.astype(int)))).tocsr()[1:,1:]
    Hs = H.sum(axis=1)
    ########## INITIALIZE ITERATION ##########
    # x = np.ones((nely, nelx))
    x = np.zeros((nely, nelx))
    x[5:,:] = 1
    loop, change, obj, gi = 0, 1, np.array([0]), np.array([0])
    # START ITERATION
    while change >  1e-4:
        loop+=1
        # FE-ANALYSIS
        sK = KE.reshape((KE.size,1))*(Emin+x.reshape((1,x.size),
                        order='F')**penal)
        sK = np.reshape(sK, (64*nelx*nely,1), order='F')
        K = coo_matrix((sK.flatten(),
                        (iK.astype(int), jK.astype(int)))).tolil()[1:,1:]
        U[freedofs-1] = spsolve(K[:,freedofs-1][freedofs-1,:].tocsr(),
                                F[freedofs-1].tocsr())
        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        ce = (U[edofMat-1]@KE*U[edofMat-1]).sum(axis=1).reshape(nelx, nely).T
        c = (Emin + x**penal*(E0-Emin)*ce).sum()
        dc = -penal*(E0-Emin)*x**(penal-1)*ce
        dv = np.ones((nely, nelx))/(nelx*nely)
        # FILTERING/STABILIZATION
        dc = H @ (dc.reshape((dc.size,1), order='F')/Hs)
        dc = np.reshape(dc, x.shape, order='F')
        if loop > 1:
            dc = (dc+olddc)/2
            olddc = dc
        else:
            olddc = dc
        # TOBS UPDATE
        obj = np.append(obj, c)
        gi = np.append(gi, x.mean())
        ###################################
        # import IPython; IPython.embed()
        x = TOBS(dc, dv, gbar, gi[-1], epsilons, beta, x)
        ###################################
        # PRINT RESULTS
        if loop > 10:
            change = (np.abs(obj[loop-10:loop-5].sum()
                     -obj[loop-5:loop].sum()) / obj[loop-5:loop].sum())
        print(f' It.:{loop:5d} Obj.:{c:11.4f}'
              f' Vol.:{x.mean():7.3f} ch.:{change:7.3f}')
        # PLOT DENSITIES
        ax.pcolormesh(np.flipud(x),vmin=0,vmax=1,cmap='binary'); plt.show(); plt.pause(0.001)
    print('Stopping criteria met.')
    ax.pcolormesh(np.flipud(x), vmin=0, vmax=1, cmap='binary'); plt.show(); plt.pause(2); H = plt.gcf(); plt.close()

    return obj[-2], H

if __name__=='__main__':
    rmin = 2
    nelx, nely = 60, 20
    gbar, epsilons, beta = 0.5, 0.02, 0.1
    tobs101_cplex(nelx, nely, gbar, epsilons, beta, rmin)
