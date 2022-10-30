from __future__ import print_function
import numpy as np
import sys
sys.path.append('../')
from utils import cheb
from utils import timer

N = 128
print(N)

y = np.linspace(0,N-1,N)*np.pi/N
v = np.cos(y)

# FIRST DERIVATIVE WITH CHEB MAT
timer.timer('first derivative w/ CHEB-MAT')
D, alpha = cheb.cheb(N-1)
D_1 = np.flipud(np.flipud(D.T).T)
vs = np.dot(D_1,v)
timer.timer('first derivative w/ CHEB-MAT')

# FIRST DERIVATIVE WITH CHEB-FFT
timer.timer('first derivative w/ CHEB-FFT')
vsFFT = cheb.chebfft(v)
timer.timer('first derivative w/ CHEB-FFT')


# FOURTH DERIVATIVE WITH CHEB MAT
timer.timer('fourth derivative w/ CHEB-MAT')
D,alpha = cheb.cheb(N-1)
D_1 = np.flipud(np.flipud(D.T).T)
D_2 = np.dot(D_1, D_1)
D_2 -= np.diag(np.sum(D_2.T,axis=0))
D_4 = np.dot(D_2, D_2)
D_4 -= np.diag(np.sum(D_4.T,axis=0))
vssss = np.dot(D_4,v)
timer.timer('fourth derivative w/ CHEB-MAT')

# FOURTH DERIVATIVE WITH CHEB-FFT
timer.timer('fourth derivative w/ CHEB-FFT')
vsFFT = cheb.chebfft(v)
vssFFT = cheb.chebfft(vsFFT)
vsssFFT = cheb.chebfft(vssFFT)
vssssFFT = cheb.chebfft(vsssFFT)
timer.timer('fourth derivative w/ CHEB-FFT')

timer.timer(' ', print_all = True)

err1st = np.sqrt(np.sum((vs-vsFFT)**2))
err4th = np.sqrt(np.sum((vssss-vssssFFT)**2))
print('Error in 1st:',err1st)
print('Error in 4th:',err4th)

