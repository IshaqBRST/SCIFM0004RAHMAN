# energy_functions.pyx
cimport cython
from libc.math cimport cos

cpdef initdat(int nmax):

    cdef cnp.ndarray[cnp.float64_t, ndim=2] arr
    arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    return arr

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double one_energy(double[:, :] arr, int ix, int iy, int nmax):
    cdef:
        double en = 0.0
        int ixp, ixm, iyp, iym
        double ang
    
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax
    
    # Calculate energies considering periodic boundary conditions
    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    return en

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double all_energy(double[:, :] arr, int nmax):
    cdef:
        double enall = 0.0
        int i, j
    
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr, i, j, nmax)
    
    return enall

# cython: language_level=3
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin
from scipy.linalg.cython_lapack cimport dsyevd

# Necessary Cython and numpy imports
cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin

# Ensuring numpy arrays can be used
cnp.import_array()

# Declaration of the function
cpdef double get_order(double[:, :] arr, int nmax):
    cdef:
        double[:, :] Qab = np.zeros((3, 3), dtype=np.float64)
        double[:, :] delta = np.eye(3, dtype=np.float64)
        double[:, :, :] lab = np.zeros((3, nmax, nmax), dtype=np.float64)
        int a, b, i, j
        double max_eigenvalue

    # Populate the lab tensor
    for i in range(nmax):
        for j in range(nmax):
            lab[0, i, j] = cos(arr[i, j])
            lab[1, i, j] = sin(arr[i, j])
            # Z-component is zero for 2D problem
            lab[2, i, j] = 0.0
    
    # Calculate Q tensor
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a, b] += 3.0 * lab[a, i, j] * lab[b, i, j] - delta[a, b]

    # Normalize Qab
    for a in range(3):
        for b in range(3):
            Qab[a, b] /= (2.0 * nmax * nmax)
    
    # Call to numpy for eigenvalues since Cython lacks direct eigen solver
    eigenvalues = np.linalg.eigvalsh(Qab)
    max_eigenvalue = np.max(eigenvalues)
    
    return max_eigenvalue


# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time
from libc.math cimport exp

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double MC_step(double[:, :] arr, double Ts, int nmax):
    # Seed random number generator
    srand(time(NULL))
    
    cdef:
        int accept = 0
        int i, j, ix, iy
        double en0, en1, deltaE, boltz, ang, randVal
        double scale = 0.1 + Ts
    
    for i in range(nmax):
        for j in range(nmax):
            ix = rand() % nmax
            iy = rand() % nmax
            
            # Assuming a normal distribution with mean 0 and std dev `scale`
            # NumPy's random.normal is used here as C doesn't have a direct equivalent
            ang = np.random.normal(0, scale)
            
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix, iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            
            if en1 <= en0:
                accept += 1
            else:
                deltaE = en1 - en0
                boltz = exp(-deltaE / Ts)
                
                # Generate a random float between 0 and 1
                randVal = rand() / float(RAND_MAX)
                
                if boltz >= randVal:
                    accept += 1
                else:
                    arr[ix, iy] -= ang
    
    return accept / (nmax * nmax)




