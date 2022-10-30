import numpy as np
import scipy.linalg as scla
import sys
sys.path.append('../')

from utils import cheb
from utils.timer import timer
from utils import finite_diff as fd
from utils import barycentricMatrix as bary

if __name__ == '__main__':
    print('# Start')

    N = 150
    M = 180
    DN, sN = cheb.cheb(N - 1)
    DM, sM = cheb.cheb(M - 1) 

    fN = np.exp(-sN**2 / 2.0) / np.sqrt(2.0 * np.pi)
    fM = np.exp(-sM**2 / 2.0) / np.sqrt(2.0 * np.pi)

    timer('P')
    P = bary.barycentricMatrix(sN, sM)
    timer('P')
    timer('P_loops')
    P_loops = bary.barycentricMatrix_loops(sN, sM)
    timer('P_loops')

    print('error_P = ', np.linalg.norm(P - P_loops))
    print('P.size = ', P.shape)
    # print('P = \n', P, '\n\n')
    
    fM_interp = np.dot(P, fN)

    error = fM - fM_interp
    print('error          = ', np.linalg.norm(error))
    print('error relative = ', np.linalg.norm(error) / np.linalg.norm(fM))

    resultN = np.zeros((N,2))
    resultN[:,0] = sN
    resultN[:,1] = fN
    np.savetxt('kkN.dat', resultN)

    resultM = np.zeros((M,4))
    resultM[:,0] = sM
    resultM[:,1] = fM
    resultM[:,2] = fM_interp
    resultM[:,3] = error
    np.savetxt('kkM.dat', resultM)



    timer(' ', print_all=True)
    print('# End')
    


